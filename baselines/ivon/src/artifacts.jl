function write_protocol_config(; selected = nothing)
    mkpath(RESULTS_ROOT)
    path = joinpath(RESULTS_ROOT, "config.toml")
    if selected === nothing && isfile(path) && filesize(path) > 0
        try
            TOML.parsefile(path)
            return path
        catch
            # Replace a partial protocol file left by an interrupted write.
        end
    end
    protocol = Dict{String,Any}(
        "benchmark" => "IVON posterior-gate ensemble for PGE MoE",
        "benchmark_version" => 2,
        "created_at_utc" => string(now(UTC)),
        "output_root" => relpath(RESULTS_ROOT, REPOSITORY_ROOT),
        "datasets" => collect(DATASETS),
        "horizons" => collect(HORIZONS),
        "architectures" => collect(string.(ARCHITECTURES)),
        "posterior_sample_counts" => collect(POSTERIOR_SAMPLE_COUNTS),
        "posterior_subsampling" =>
            "cell-seeded permutation; K=10 nested in K=100; K=1000 is the full bank",
        "base_seed" => DEFAULT_SEED,
        "pge_commit" => PGE_COMMIT,
        "ivonrepro_commit" => IVON_COMMIT,
        "julia_version" => string(VERSION),
        "benchmark_project_sha256" => file_sha256(joinpath(BENCHMARK_ROOT, "Project.toml")),
        "benchmark_manifest_sha256" => file_sha256(joinpath(BENCHMARK_ROOT, "Manifest.toml")),
    )
    if selected !== nothing
        protocol["selected_ivon"] = Dict(
            "learning_rate" => Float64(selected.learning_rate),
            "ess_multiplier" => Int(selected.ess_multiplier),
            "selection_mean_nll" => Float64(selected.mean_nll),
            "selection_mean_mse" => Float64(selected.mean_mse),
        )
    end
    open(path, "w") do io
        TOML.print(io, protocol; sorted = true)
    end
    return path
end

function initialize_results_root(; selected = nothing)
    for path in (RESULTS_ROOT, CHECKPOINT_ROOT, PILOT_ROOT, SMOKE_ROOT)
        mkpath(path)
    end
    cp(joinpath(BENCHMARK_ROOT, "frozen_hashes.toml"),
        joinpath(RESULTS_ROOT, "frozen_hashes.toml"); force = true)
    write_protocol_config(; selected)
    return RESULTS_ROOT
end

function checkpoint_path(root::String, config)
    name = "$(config.dataset)_h$(config.horizon)_$(config.architecture)_ivon.jld2"
    return joinpath(root, name)
end

function checkpoint_complete(path::String, config)
    isfile(path) || return false
    try
        saved = JLD2.load(path)
        basic = get(saved, "complete", false) === true &&
            saved["config_fingerprint"] == config_fingerprint(config) &&
            saved["prediction_digest"] == component_digest(saved["posterior_components"])
        basic || return false
        indices = saved["posterior_sample_indices"]
        validate_posterior_subset_indices(indices, config.posterior_samples)
        sort!(Int.(collect(keys(indices)))) == sort!(collect(config.posterior_sample_counts)) ||
            return false
        saved["posterior_subset_digest"] == posterior_subset_digest(indices) || return false
        evaluations = saved["posterior_sample_evaluations"]
        sort!(Int.(collect(keys(evaluations)))) == sort!(collect(config.posterior_sample_counts)) ||
            return false
        return true
    catch error_value
        @warn "Ignoring invalid checkpoint" path exception =
            (error_value, catch_backtrace())
        return false
    end
end

function save_checkpoint(path::String; config, training, frozen_hashes,
    quantile_hashes_before, quantile_hashes_after, split_metadata, scaler,
    standardized, original_units, mean_standardized, mean_original_units,
    posterior_components, mean_components, prediction_digest,
    posterior_sample_indices, posterior_sample_evaluations)
    mkpath(dirname(path))
    JLD2.jldsave(
        path;
        complete = true,
        created_at = string(now(UTC)),
        effective_config = config,
        config_fingerprint = config_fingerprint(config),
        pge_commit = PGE_COMMIT,
        ivonrepro_commit = IVON_COMMIT,
        frozen_hashes = frozen_hashes,
        quantile_hashes_before = quantile_hashes_before,
        quantile_hashes_after = quantile_hashes_after,
        split_metadata = split_metadata,
        scaler = scaler,
        gate_parameters_mean = training.parameters,
        gate_states = training.states,
        optimizer_state = training.optimizer_state,
        optimizer = training.optimizer,
        training_step = training.step,
        best_epoch = training.best_epoch,
        best_monitor_loss = training.best_monitor_loss,
        history = training.history,
        standardized_metrics = standardized.metrics,
        standardized_traces = standardized.traces,
        original_unit_metrics = original_units,
        posterior_mean_standardized_metrics = mean_standardized.metrics,
        posterior_mean_standardized_traces = mean_standardized.traces,
        posterior_mean_original_unit_metrics = mean_original_units,
        gate_diagnostics = posterior_components.gate_diagnostics,
        gate_diagnostics_by_sample_count =
            posterior_components.gate_diagnostics_by_sample_count,
        posterior_components = posterior_components,
        posterior_sample_indices = posterior_sample_indices,
        posterior_subset_digest = posterior_subset_digest(posterior_sample_indices),
        posterior_sample_evaluations = posterior_sample_evaluations,
        posterior_mean_components = mean_components,
        prediction_digest = prediction_digest,
    )
    checkpoint_complete(path, config) || error("Checkpoint round-trip validation failed: $path")
    return path
end

function load_checkpoint(path::String)
    saved = JLD2.load(path)
    get(saved, "complete", false) === true || error("Incomplete checkpoint: $path")
    saved["prediction_digest"] == component_digest(saved["posterior_components"]) ||
        error("Prediction replay digest mismatch: $path")
    config = saved["effective_config"]
    indices = saved["posterior_sample_indices"]
    validate_posterior_subset_indices(indices, config.posterior_samples)
    saved["posterior_subset_digest"] == posterior_subset_digest(indices) ||
        error("Posterior subset replay digest mismatch: $path")
    return saved
end

csv_escape(value) = begin
    text = string(value)
    (occursin(',', text) || occursin('"', text) || occursin('\n', text)) ?
        "\"$(replace(text, '"' => "\"\""))\"" : text
end

function write_csv(path::String, rows, columns)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(columns), ','))
        for row in rows
            println(io, join((csv_escape(getproperty(row, c)) for c in columns), ','))
        end
    end
    return path
end

function flatten_metrics_row(saved, path; posterior_mean::Bool = false,
    posterior_samples = nothing)
    config = saved["effective_config"]
    sample_count = posterior_mean ? 1 :
        posterior_samples === nothing ? config.posterior_samples : Int(posterior_samples)
    if posterior_mean
        standardized = saved["posterior_mean_standardized_metrics"]
        original = saved["posterior_mean_original_unit_metrics"]
        diagnostics = saved["posterior_mean_components"].gate_diagnostics
    elseif haskey(saved, "posterior_sample_evaluations")
        evaluation = saved["posterior_sample_evaluations"][sample_count]
        standardized = evaluation.standardized.metrics
        original = evaluation.original_unit_metrics
        diagnostics = saved["gate_diagnostics_by_sample_count"][sample_count]
    else
        standardized = saved["standardized_metrics"]
        original = saved["original_unit_metrics"]
        diagnostics = saved["gate_diagnostics"]
    end
    return (
        dataset = config.dataset,
        horizon = config.horizon,
        architecture = config.architecture,
        estimator = posterior_mean ? "posterior_mean_head" : "posterior_ensemble",
        posterior_samples = sample_count,
        learning_rate = config.learning_rate,
        ess_multiplier = config.ess_multiplier,
        best_epoch = saved["best_epoch"],
        nll_std = standardized.nll,
        log_predictive_density_std = standardized.log_predictive_density,
        mse_std = standardized.mse,
        rmse_std = standardized.rmse,
        mae_std = standardized.mae,
        crps_std = standardized.crps,
        coverage95_std = standardized.coverage95,
        interval_width_std = standardized.interval_width,
        interval_score_std = standardized.interval_score,
        epistemic_variance_std = standardized.epistemic_variance,
        aleatoric_variance_std = standardized.aleatoric_variance,
        total_variance_std = standardized.total_variance,
        nll_ot = original.nll,
        log_predictive_density_ot = original.log_predictive_density,
        mse_ot = original.mse,
        rmse_ot = original.rmse,
        mae_ot = original.mae,
        crps_ot = original.crps,
        coverage95_ot = original.coverage95,
        interval_width_ot = original.interval_width,
        interval_score_ot = original.interval_score,
        epistemic_variance_ot = original.epistemic_variance,
        aleatoric_variance_ot = original.aleatoric_variance,
        total_variance_ot = original.total_variance,
        gate_entropy = diagnostics.entropy,
        max_top_expert_share = diagnostics.max_top_expert_share,
        switching_rate = diagnostics.switching_rate,
        checkpoint = path,
    )
end

const RESULT_COLUMNS = (
    :dataset, :horizon, :architecture, :estimator, :posterior_samples,
    :learning_rate, :ess_multiplier, :best_epoch, :nll_std,
    :log_predictive_density_std, :mse_std, :rmse_std, :mae_std, :crps_std,
    :coverage95_std, :interval_width_std, :interval_score_std,
    :epistemic_variance_std, :aleatoric_variance_std, :total_variance_std,
    :nll_ot, :log_predictive_density_ot, :mse_ot, :rmse_ot, :mae_ot, :crps_ot,
    :coverage95_ot, :interval_width_ot, :interval_score_ot,
    :epistemic_variance_ot, :aleatoric_variance_ot, :total_variance_ot,
    :gate_entropy, :max_top_expert_share, :switching_rate, :checkpoint,
)

function write_markdown_table(path::String, rows; title::String)
    open(path, "w") do io
        println(io, "# $title\n")
        println(io, "| Dataset | H | Gate | Samples | NLL | MSE | RMSE | MAE | CRPS | 95% coverage |")
        println(io, "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows
            @printf(io, "| %s | %d | %s | %d | %.6g | %.6g | %.6g | %.6g | %.6g | %.4f |\n",
                row.dataset, row.horizon, row.architecture, row.posterior_samples,
                row.nll_std, row.mse_std, row.rmse_std, row.mae_std, row.crps_std,
                row.coverage95_std)
        end
    end
    return path
end

function run_record(path::String, config, status::String)
    return (
        timestamp_utc = string(now(UTC)),
        phase = config.phase,
        dataset = config.dataset,
        horizon = config.horizon,
        architecture = config.architecture,
        status = status,
        seed = config.training_seed,
        learning_rate = config.learning_rate,
        ess_multiplier = config.ess_multiplier,
        epochs = config.n_epochs,
        posterior_samples = config.posterior_samples,
        checkpoint = path,
    )
end

function append_run_record(row)
    path = joinpath(RESULTS_ROOT, "runs.csv")
    columns = propertynames(row)
    existing = isfile(path) && filesize(path) > 0
    mkpath(dirname(path))
    open(path, "a") do io
        existing || println(io, join(string.(columns), ','))
        println(io, join((csv_escape(getproperty(row, c)) for c in columns), ','))
    end
    return path
end
