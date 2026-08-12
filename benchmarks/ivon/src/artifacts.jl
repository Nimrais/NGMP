function checkpoint_path(root::String, config)
    name = "$(config.dataset)_h$(config.horizon)_$(config.architecture)_ivon.jld2"
    return joinpath(root, name)
end

function checkpoint_complete(path::String, config)
    isfile(path) || return false
    try
        saved = JLD2.load(path)
        return get(saved, "complete", false) === true &&
            saved["config_fingerprint"] == config_fingerprint(config) &&
            saved["prediction_digest"] == component_digest(saved["posterior_components"])
    catch error_value
        @warn "Ignoring invalid checkpoint" path exception =
            (error_value, catch_backtrace())
        return false
    end
end

function save_checkpoint(path::String; config, training, frozen_hashes,
    quantile_hashes_before, quantile_hashes_after, split_metadata, scaler,
    standardized, original_units, mean_standardized, mean_original_units,
    posterior_components, mean_components, prediction_digest)
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
        posterior_components = posterior_components,
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

function flatten_metrics_row(saved, path; posterior_mean::Bool = false)
    config = saved["effective_config"]
    standardized = saved[posterior_mean ?
        "posterior_mean_standardized_metrics" : "standardized_metrics"]
    original = saved[posterior_mean ?
        "posterior_mean_original_unit_metrics" : "original_unit_metrics"]
    diagnostics = posterior_mean ?
        saved["posterior_mean_components"].gate_diagnostics : saved["gate_diagnostics"]
    return (
        dataset = config.dataset,
        horizon = config.horizon,
        architecture = config.architecture,
        estimator = posterior_mean ? "posterior_mean_head" : "posterior_ensemble",
        posterior_samples = posterior_mean ? 1 : config.posterior_samples,
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
