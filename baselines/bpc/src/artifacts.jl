const BPC_POSTERIOR_SCHEMA_VERSION = "bpc-uci-posterior-v1"
const CONFIG_RUNTIME_ONLY_FIELDS = Set(["output_dir", "resume", "show_progress"])
const SCALAR_METRIC_NAMES = (
    :lpd_standardized,
    :nll_standardized,
    :expected_log_likelihood_standardized,
    :lpd_original,
    :nll_original,
    :expected_log_likelihood_original,
    :rmse_original,
    :coverage_50,
    :coverage_80,
    :coverage_95,
    :mean_total_variance_original,
    :mean_epistemic_variance_original,
    :mean_aleatoric_variance_original,
)

scalar_metrics(metrics) =
    (; (name => getfield(metrics, name) for name in SCALAR_METRIC_NAMES)...)

configuration_stem(dataset::String, split_id::Int) =
    @sprintf("%s_split%02d_homoscedastic", dataset, split_id)

function posterior_checkpoint_path(
    config::BPCConfig,
    dataset::String,
    split_id::Int,
)
    return joinpath(
        config.output_dir,
        "checkpoints",
        configuration_stem(dataset, split_id) * ".jld2",
    )
end

function split_spec_record(spec::UCISplitSpec)
    return (
        protocol_version = spec.protocol_version,
        split_id = spec.split_id,
        n_observations = spec.n_observations,
        outer_seed = spec.outer_seed,
        inner_seed = spec.inner_seed,
        test_fraction = spec.test_fraction,
        validation_fraction = spec.validation_fraction,
        train_indices = spec.train_indices,
        test_indices = spec.test_indices,
        inner_train_indices = spec.inner_train_indices,
        validation_indices = spec.validation_indices,
    )
end

function atomic_jld2_write(path::AbstractString, record::NamedTuple)
    mkpath(dirname(path))
    temporary = path * ".tmp-" * string(getpid())
    try
        JLD2.jldsave(temporary; record...)
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return path
end

function atomic_csv_write(path::AbstractString, table)
    mkpath(dirname(path))
    temporary = path * ".tmp-" * string(getpid())
    CSV.write(temporary, table)
    mv(temporary, path; force = true)
    return path
end

function ensure_output_directories(config::BPCConfig)
    mkpath(config.output_dir)
    mkpath(joinpath(config.output_dir, "histories"))
    config.save_checkpoints && mkpath(joinpath(config.output_dir, "checkpoints"))
    return config.output_dir
end

function validate_resume_config(path::AbstractString, config::BPCConfig)
    existing = TOML.parsefile(path)
    requested = config_dictionary(config)
    for key in keys(requested)
        key in CONFIG_RUNTIME_ONLY_FIELDS && continue
        haskey(existing, key) ||
            throw(ArgumentError("existing config is missing '$key': $path"))
        isequal(existing[key], requested[key]) || throw(ArgumentError(
            "result-affecting config '$key' differs in $path",
        ))
    end
    return nothing
end

function save_config(config::BPCConfig)
    path = joinpath(config.output_dir, "config.toml")
    isfile(path) && validate_resume_config(path, config)
    temporary = path * ".tmp-" * string(getpid())
    open(temporary, "w") do io
        TOML.print(io, config_dictionary(config); sorted = true)
    end
    mv(temporary, path; force = true)
    return path
end

function write_split_manifest(config::BPCConfig)
    datasets = Dict{String, Any}()
    for dataset_key in config.datasets
        dataset = load_dataset(dataset_key)
        datasets[dataset_key] = [
            split_spec_record(spec)
            for spec in uci_regression_splits(
                size(dataset.features, 1), config.n_splits;
                base_seed = config.split_seed,
                test_fraction = config.test_fraction,
                validation_fraction = config.validation_fraction,
            )
        ]
    end
    return atomic_jld2_write(
        joinpath(config.output_dir, "split_manifest.jld2"),
        (
            schema_version = "uci-split-manifest-v1",
            protocol_version = UCI_SPLIT_PROTOCOL_VERSION,
            base_seed = config.split_seed,
            n_splits = config.n_splits,
            test_fraction = config.test_fraction,
            validation_fraction = config.validation_fraction,
            datasets = datasets,
        ),
    )
end

function posterior_record(
    dataset,
    prepared,
    model::BPCModel,
    selection,
    seeds,
    metrics,
    config::BPCConfig,
)
    return (
        schema_version = BPC_POSTERIOR_SCHEMA_VERSION,
        split_protocol_version = UCI_SPLIT_PROTOCOL_VERSION,
        method = "Bayesian Predictive Coding (Matrix-Normal-Wishart)",
        method_reference = "Tschantz et al., arXiv:2503.24016 (2025)",
        dataset = dataset.key,
        dataset_name = dataset.display_name,
        n_observations = size(dataset.features, 1),
        n_features = size(dataset.features, 2),
        split_id = prepared.split_spec.split_id,
        split_spec = split_spec_record(prepared.split_spec),
        likelihood = "homoscedastic",
        posterior_family = "matrix_normal_wishart",
        posterior_model = host_model(model),
        standardizer = prepared.outer.standardizer,
        training_config = config_dictionary(config),
        best_epoch = selection.best_epoch,
        model_seed = seeds.model,
        test_seed = seeds.test,
        test_metrics = scalar_metrics(metrics),
    )
end

function load_posterior_checkpoint(path::AbstractString)
    isfile(path) || throw(ArgumentError("posterior checkpoint does not exist: $path"))
    data = JLD2.load(path)
    required = (
        "schema_version", "split_protocol_version", "dataset", "dataset_name",
        "n_observations", "n_features", "split_id", "split_spec",
        "likelihood", "posterior_family", "posterior_model", "standardizer", "training_config",
        "best_epoch", "model_seed", "test_seed", "test_metrics",
    )
    missing = filter(key -> !haskey(data, key), required)
    isempty(missing) || throw(ArgumentError(
        "BPC checkpoint is missing keys: $(join(missing, ", "))",
    ))
    data["schema_version"] == BPC_POSTERIOR_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported BPC checkpoint schema"))
    data["split_protocol_version"] == UCI_SPLIT_PROTOCOL_VERSION ||
        throw(ArgumentError("unsupported UCI split protocol"))
    return (
        path = abspath(path),
        schema_version = data["schema_version"],
        split_protocol_version = data["split_protocol_version"],
        dataset = String(data["dataset"]),
        dataset_name = String(data["dataset_name"]),
        n_observations = Int(data["n_observations"]),
        n_features = Int(data["n_features"]),
        split_id = Int(data["split_id"]),
        split_spec = data["split_spec"],
        likelihood = String(data["likelihood"]),
        posterior_family = String(data["posterior_family"]),
        posterior_model = data["posterior_model"],
        standardizer = data["standardizer"],
        training_config = data["training_config"],
        best_epoch = Int(data["best_epoch"]),
        model_seed = Int(data["model_seed"]),
        test_seed = Int(data["test_seed"]),
        test_metrics = data["test_metrics"],
    )
end

function checkpoint_prediction_config(checkpoint, n_samples::Int)
    stored = checkpoint.training_config
    return BPCConfig(
        datasets = [checkpoint.dataset],
        n_splits = 1,
        hidden_units = Int(stored["hidden_units"]),
        posterior_jitter = Float64(stored["posterior_jitter"]),
        activate_input = Bool(stored["activate_input"]),
        eval_samples = n_samples,
        backend = "cpu",
        save_checkpoints = false,
        show_progress = false,
    )
end

"""Replay the exact saved outer holdout prediction and metrics."""
function predict_holdout(
    checkpoint_or_path;
    n_samples::Union{Nothing, Int} = nothing,
    seed::Union{Nothing, Int} = nothing,
)
    checkpoint = checkpoint_or_path isa AbstractString ?
        load_posterior_checkpoint(checkpoint_or_path) : checkpoint_or_path
    dataset = load_dataset(checkpoint.dataset)
    size(dataset.features) ==
        (checkpoint.n_observations, checkpoint.n_features) ||
        throw(DimensionMismatch("dataset and BPC checkpoint disagree"))
    test_indices = Int.(checkpoint.split_spec.test_indices)
    features = dataset.features[test_indices, :]
    targets_original = dataset.targets[test_indices]
    standardized_features = transform_uci_features(
        features, checkpoint.standardizer,
    )
    targets_standardized = transform_uci_targets(
        targets_original, checkpoint.standardizer,
    )
    sample_count = isnothing(n_samples) ?
        Int(checkpoint.training_config["eval_samples"]) : n_samples
    prediction_seed = isnothing(seed) ? checkpoint.test_seed : seed
    config = checkpoint_prediction_config(checkpoint, sample_count)
    samples = predictive_samples(
        checkpoint.posterior_model,
        standardized_features,
        config;
        seed = prediction_seed,
        n_samples = sample_count,
    )
    metrics = predictive_metrics(
        samples, targets_standardized, targets_original, checkpoint.standardizer,
    )
    return (
        dataset = checkpoint.dataset,
        split_id = checkpoint.split_id,
        likelihood = checkpoint.likelihood,
        seed = prediction_seed,
        n_samples = sample_count,
        test_indices = test_indices,
        samples_standardized = samples,
        targets_original = targets_original,
        metrics = metrics,
    )
end

function load_run_table(config::BPCConfig)
    path = joinpath(config.output_dir, "runs.csv")
    return isfile(path) ? CSV.read(path, DataFrame) : DataFrame()
end

function configuration_succeeded(
    runs::DataFrame,
    dataset::String,
    split_id::Int,
    config::BPCConfig,
)
    isempty(runs) && return false
    all(name -> name in propertynames(runs), (:dataset, :split, :status)) ||
        return false
    succeeded = any(eachrow(runs)) do row
        string(row.dataset) == dataset && Int(row.split) == split_id &&
            string(row.status) == "success"
    end
    succeeded || return false
    config.save_checkpoints || return true
    return try
        checkpoint = load_posterior_checkpoint(
            posterior_checkpoint_path(config, dataset, split_id),
        )
        checkpoint.dataset == dataset && checkpoint.split_id == split_id
    catch
        false
    end
end

function append_run!(runs::DataFrame, row::NamedTuple, config::BPCConfig)
    incoming = DataFrame([row])
    updated = isempty(runs) ? incoming : vcat(runs, incoming; cols = :union)
    atomic_csv_write(joinpath(config.output_dir, "runs.csv"), updated)
    return updated
end

function summary_table(runs::DataFrame)
    isempty(runs) && return DataFrame()
    successful = filter(row -> row.status == "success", runs)
    isempty(successful) && return DataFrame()
    rows = NamedTuple[]
    for group in groupby(successful, [:dataset, :dataset_name, :likelihood])
        summaries = Pair{Symbol, Any}[]
        for metric in SCALAR_METRIC_NAMES
            values = Float64.(group[!, metric])
            deviation = length(values) > 1 ? std(values; corrected = true) : 0.0
            push!(summaries, Symbol(metric, "_mean") => mean(values))
            push!(summaries, Symbol(metric, "_std") => deviation)
            push!(summaries, Symbol(metric, "_se") => deviation / sqrt(length(values)))
        end
        push!(rows, (; (
            dataset = string(first(group.dataset)),
            dataset_name = string(first(group.dataset_name)),
            likelihood = string(first(group.likelihood)),
            n = nrow(group),
            summaries...,
        )...))
    end
    return sort!(DataFrame(rows), [:dataset, :likelihood])
end

function refresh_artifacts(runs::DataFrame, config::BPCConfig)
    summary = summary_table(runs)
    isempty(summary) && return summary
    atomic_csv_write(joinpath(config.output_dir, "summary.csv"), summary)
    path = joinpath(config.output_dir, "table.md")
    open(path, "w") do io
        println(io, "| Dataset | Likelihood | Runs | LPD (original) | RMSE |")
        println(io, "|---|---:|---:|---:|---:|")
        for row in eachrow(summary)
            @printf(
                io,
                "| %s | %s | %d | %.4f ± %.4f | %.4f ± %.4f |\n",
                row.dataset_name, row.likelihood, row.n,
                row.lpd_original_mean, row.lpd_original_std,
                row.rmse_original_mean, row.rmse_original_std,
            )
        end
    end
    return summary
end
