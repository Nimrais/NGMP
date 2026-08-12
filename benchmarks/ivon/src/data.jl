function pge_root()
    root = Base.pkgdir(ProbabilisticEnsembling)
    root === nothing && error("Cannot locate pinned ProbabilisticEnsembling package")
    return root
end

function session_path(dataset::String, horizon::Int)
    return joinpath(pge_root(), "sessions", "neural_ensemble",
        "neural_ensemble_$(dataset)_$(horizon).yaml")
end

function configured_session(dataset::String, horizon::Int)
    dataset in DATASETS || error("Unsupported dataset: $dataset")
    horizon in HORIZONS || error("Unsupported horizon: $horizon")
    path = session_path(dataset, horizon)
    isfile(path) || error("Missing pinned PGE session: $path")
    raw = YAML.load_file(path)
    params = raw["params"]
    params["dataset"] == dataset || error("Dataset mismatch in $path")
    params["column"] == "OT" || error("Expected the OT target in $path")
    params["train_set"] == false || error("Expected train_set=false in $path")
    params["feature_type"] == "vae" || error("Expected frozen VAE features in $path")
    Float64.(params["quantiles"]) == [10.0, 90.0] ||
        error("Expected q10 and q90 in $path")
    params["gating"]["n_epochs"] == 100 || error("Expected a 100-epoch budget")

    params["dataset_path"] = joinpath(REPOSITORY_ROOT, "data", "$dataset.csv")
    params["experts"] = [joinpath(pge_root(), String(p)) for p in params["experts"]]
    return raw
end

function checkpoint_files(dataset::String, horizon::Int)
    models = joinpath(pge_root(), "models")
    prefix = "$(dataset)_h$(horizon)_s96_"
    return Dict(
        "CNN" => joinpath(models, prefix * "CNN_enzyme.jld2"),
        "NLinear" => joinpath(models, prefix * "MLP_enzyme.jld2"),
        "LSTM" => joinpath(models, prefix * "LSTM_enzyme.jld2"),
        "DLinear" => joinpath(models, prefix * "DLinear_enzyme.jld2"),
        "NConv" => joinpath(models, prefix * "NConv_enzyme.jld2"),
        "VAE" => joinpath(models, "$(dataset)_s96_VAE_enzyme.jld2"),
    )
end

file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

function expected_frozen_hashes(dataset::String, horizon::Int)
    hashes = TOML.parsefile(joinpath(BENCHMARK_ROOT, "frozen_hashes.toml"))
    result = Dict{String,String}("VAE" => hashes[dataset]["VAE"])
    for (name, digest) in hashes[dataset]["h$horizon"]
        result[name] = digest
    end
    return result
end

function verify_frozen_hashes(dataset::String, horizon::Int)
    expected = expected_frozen_hashes(dataset, horizon)
    files = checkpoint_files(dataset, horizon)
    actual = Dict{String,String}()
    for name in ("CNN", "NLinear", "LSTM", "DLinear", "NConv", "VAE")
        path = files[name]
        isfile(path) || error("Missing frozen $name checkpoint: $path")
        digest = file_sha256(path)
        digest == expected[name] || error(
            "Frozen $name checkpoint changed: expected $(expected[name]), got $digest",
        )
        actual[name] = digest
    end
    return actual
end

function parse_spec(dataset::String, horizon::Int)
    raw = configured_session(dataset, horizon)
    spec = PE._parse_neural_ensemble_spec(raw)
    spec.horizon == horizon || error("Session checkpoint horizon mismatch")
    return spec, raw
end

function prepare_full_data(dataset::String, horizon::Int)
    spec, raw = parse_spec(dataset, horizon)
    data = cd(pge_root()) do
        PE.before_neural_ensemble(spec)
    end
    data.n_features == 65 || error("Expected 64 VAE coordinates plus bias")
    data.n_total == 7 || error("Expected five neural and two quantile experts")
    return spec, raw, data
end

"""Prepare a tiny, protocol-faithful real-data slice for the smoke command."""
function prepare_smoke_data(dataset::String = "ETTh1", horizon::Int = 96;
    observations::Int = 4)
    spec, raw = parse_spec(dataset, horizon)
    data = cd(pge_root()) do
        experts = map(PE.load_jld2_model, spec.experts)
        base_meta = first(experts).meta
        Xmat, feat_cols = PE.load_dataset(spec.dataset, spec.dataset_path)
        col_idx = PE.find_column_index(feat_cols, spec.column)
        seq_len = Int(base_meta.seq_len)
        X3, Y2 = PE.make_sequences(Xmat; seq_len, horizon)
        split = base_meta.split
        Xtr, Ytr, Xval, Yval, Xte, Yte = PE.train_val_test_split(
            X3, Y2; ratios = (split.train, split.val, split.test))
        n = observations
        all(size(X, 3) >= n for X in (Xtr, Xval, Xte)) ||
            error("Smoke slice exceeds a chronological partition")
        scaler = base_meta.scaler
        Xtr_s = PE.scale_inputs(scaler, @view Xtr[:, :, 1:n])
        Xval_s = PE.scale_inputs(scaler, @view Xval[:, :, 1:n])
        Xte_s = PE.scale_inputs(scaler, @view Xte[:, :, 1:n])
        # Quantiles intentionally use all upstream training targets, as in PGE.
        Ytr_all_scaled = PE.scale_targets(scaler, Ytr)
        Ytr_s = @view Ytr_all_scaled[:, 1:n]
        Yval_s = PE.scale_targets(scaler, @view Yval[:, 1:n])
        Yte_s = PE.scale_targets(scaler, @view Yte[:, 1:n])
        pred_train, pred_val, pred_test =
            PE.generate_expert_predictions_three_splits(experts, Xtr_s, Xval_s, Xte_s)
        pred_train, pred_val, pred_test = PE.add_quantile_baselines!(
            pred_train, pred_val, pred_test, Ytr_all_scaled, spec.selected_quantiles)
        predictions_train_vec = PE.to_predictions_vec(pred_train)
        predictions_val_vec = PE.to_predictions_vec(pred_val)
        predictions_test_vec = PE.to_predictions_vec(pred_test)
        y_train = PE.to_target_vecs(Float64.(Ytr_s))
        y_val = PE.to_target_vecs(Float64.(Yval_s))
        predictions_train_vec_moe, y_train_moe =
            PE.restrict_to_column(predictions_train_vec, y_train, col_idx)
        predictions_val_vec_moe, y_val_moe =
            PE.restrict_to_column(predictions_val_vec, y_val, col_idx)
        features_train = PE._neural_ensemble_features(
            spec.feature_type, Xtr_s, spec.dataset, col_idx)
        features_val = PE._neural_ensemble_features(
            spec.feature_type, Xval_s, spec.dataset, col_idx)
        features_test = PE._neural_ensemble_features(
            spec.feature_type, Xte_s, spec.dataset, col_idx)
        return (
            n_total = size(pred_train, 1),
            d = size(pred_train, 2),
            col_idx = col_idx,
            n_features = length(first(features_train)),
            predictions_train_vec_moe = predictions_train_vec_moe,
            predictions_val_vec_moe = predictions_val_vec_moe,
            predictions_test_vec = predictions_test_vec,
            y_train_moe = y_train_moe,
            y_val_moe = y_val_moe,
            y_test_mat = Float64.(Yte_s),
            features_train = features_train,
            features_val = features_val,
            features_test = features_test,
            scaler = scaler,
        )
    end
    data.n_features == 65 || error("Unexpected smoke VAE feature dimension")
    data.n_total == 7 || error("Unexpected smoke expert count")
    return spec, raw, data
end

# Pilot-only preparation deliberately truncates the raw series at the final
# validation target before constructing sequences. No test target is created,
# scaled, passed to an expert, or returned from this function.
function prepare_pilot_data(dataset::String = "ETTh1", horizon::Int = 96)
    dataset == "ETTh1" && horizon == 96 || error("Pilot is restricted to ETTh1/H96")
    spec, raw = parse_spec(dataset, horizon)
    result = cd(pge_root()) do
        experts = map(PE.load_jld2_model, spec.experts)
        base_meta = first(experts).meta
        Xmat, feat_cols = PE.load_dataset(spec.dataset, spec.dataset_path)
        col_idx = PE.find_column_index(feat_cols, spec.column)
        seq_len = Int(base_meta.seq_len)
        total_sequences = size(Xmat, 2) - seq_len - horizon + 1
        n_train = round(Int, total_sequences * Float64(base_meta.split.train))
        n_validation = round(Int, total_sequences * Float64(base_meta.split.val))
        last_validation_sequence = n_train + n_validation
        raw_cutoff = last_validation_sequence + seq_len + horizon - 1
        X_prefix = @view Xmat[:, 1:raw_cutoff]
        X3, Y2 = PE.make_sequences(X_prefix; seq_len, horizon)
        size(X3, 3) == last_validation_sequence || error("Pilot prefix split mismatch")

        X_train = @view X3[:, :, 1:n_train]
        Y_train = @view Y2[:, 1:n_train]
        X_validation = @view X3[:, :, (n_train + 1):last_validation_sequence]
        Y_validation = @view Y2[:, (n_train + 1):last_validation_sequence]
        scaler = base_meta.scaler
        X_validation_scaled = PE.scale_inputs(scaler, X_validation)
        Y_train_scaled = PE.scale_targets(scaler, Y_train)
        Y_validation_scaled = PE.scale_targets(scaler, Y_validation)

        n_models = length(experts)
        d = size(Xmat, 1)
        predictions = Array{Float64}(undef, n_models + 2, d, n_validation)
        for (i, saved) in enumerate(experts)
            model = PE.build_model(saved.model_type, saved.config)
            yhat = PE.predict_unscaled(
                model, saved.parameters, saved.states, X_validation_scaled)
            predictions[i, :, :] = Float64.(yhat)
        end
        quantile_vectors = Vector{Vector{Float64}}()
        for (offset, q) in enumerate((0.10, 0.90))
            qvec = [quantile(Float64.(view(Y_train_scaled, i, :)), q) for i = 1:d]
            push!(quantile_vectors, qvec)
            for j = 1:n_validation
                predictions[n_models + offset, :, j] = qvec
            end
        end

        predictions_vec = PE.to_predictions_vec(predictions)
        targets = PE.to_target_vecs(Float64.(Y_validation_scaled))
        restricted_predictions, restricted_targets =
            PE.restrict_to_column(predictions_vec, targets, col_idx)
        features = PE._neural_ensemble_features(
            spec.feature_type, X_validation_scaled, spec.dataset, col_idx)
        return (
            predictions = restricted_predictions,
            features = features,
            targets = restricted_targets,
            scaler = scaler,
            col_idx = col_idx,
            n_train = n_train,
            n_validation = n_validation,
            total_sequences = total_sequences,
            test_targets_materialized = false,
            quantile_vectors = quantile_vectors,
        )
    end
    length(first(result.features)) == 65 || error("Unexpected pilot feature dimension")
    return spec, raw, result
end

function subset_gate_data(predictions, features, targets, indices)
    ids = collect(indices)
    return (
        predictions = predictions[:, ids],
        features = features[ids],
        targets = targets[ids],
    )
end

function restrict_full_test(data; max_observations = nothing)
    n = size(data.predictions_test_vec, 2)
    m = max_observations === nothing ? n : min(n, Int(max_observations))
    ids = 1:m
    predictions = Array{Vector{Float64}}(undef, data.n_total, m)
    for i = 1:data.n_total, (out, j) in enumerate(ids)
        predictions[i, out] = [Float64(data.predictions_test_vec[i, j][data.col_idx])]
    end
    return (
        predictions = predictions,
        features = data.features_test[ids],
        targets = [[Float64(data.y_test_mat[data.col_idx, j])] for j in ids],
    )
end

function quantile_hashes(gate_data)
    result = Dict{String,String}()
    for (name, row) in (("q10", 6), ("q90", 7))
        values = Float64[gate_data.predictions[row, j][1] for j in axes(gate_data.predictions, 2)]
        result[name] = bytes2hex(sha256(reinterpret(UInt8, values)))
    end
    return result
end

function split_metadata(data, mode::Symbol)
    if mode === :pilot
        n = length(data.features)
        n_fit = floor(Int, 0.8 * n)
        return (
            source_partition = "upstream_validation",
            chronological = true,
            fit_range = (1, n_fit),
            selection_range = (n_fit + 1, n),
            n_fit = n_fit,
            n_selection = n - n_fit,
            test_targets_read = false,
        )
    end
    return (
        source_partition = "upstream_train_set_false",
        chronological = true,
        fit_partition = "validation",
        monitor_partition = "training",
        test_targets_read_during_training = false,
    )
end
