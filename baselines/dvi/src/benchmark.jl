function scalar_metrics(metrics)
    return (; (
        name => getfield(metrics, name)
        for name in DVI_SCALAR_METRIC_NAMES
    )...)
end

function method_name(config::DVIConfig)
    return config.propagation == "full" ?
        "Deterministic Variational Inference (DVI)" :
        "Diagonal Deterministic Variational Inference (dDVI)"
end

function run_configuration(
    dataset,
    split_id::Int,
    likelihood::String,
    config::DVIConfig,
)
    prepared = outer_and_inner_splits(dataset, split_id, config)
    seeds = model_and_batch_seeds(
        config, dataset.dataset_index, split_id, likelihood,
    )
    input_dimension = size(dataset.features, 2)
    history_name = configuration_stem(
        dataset.key, split_id, likelihood, config.propagation,
    ) * ".csv"
    history_path = joinpath(config.output_dir, "histories", history_name)

    selection_start = time()
    selection = train_with_validation(
        prepared.inner,
        input_dimension,
        likelihood,
        config;
        seed = seeds.model,
        history_path = history_path,
    )
    selection_seconds = time() - selection_start
    atomic_csv_write(history_path, selection.history)

    metric_values = (; (
        name => NaN for name in DVI_SCALAR_METRIC_NAMES
    )...)
    if config.selection_only
        return (
            method = method_name(config),
            method_reference = "Wu et al., ICLR 2019, arXiv:1810.03958",
            uci_protocol_reference = "Wu et al., ICLR 2019, Table 2",
            numerical_protocol = config.numerical_protocol,
            dataset = dataset.key,
            dataset_name = dataset.display_name,
            split = split_id,
            split_protocol = UCI_SPLIT_PROTOCOL_VERSION,
            likelihood = likelihood,
            propagation = config.propagation,
            execution_backend = config.execution_backend,
            execution_device = config.execution_device,
            implementation_version = config.implementation_version,
            full_selection_patience_steps =
                config.full_selection_patience_steps,
            training_budget_protocol = config.training_budget_protocol,
            selection_max_optimizer_steps =
                config.selection_max_optimizer_steps,
            refit_max_optimizer_steps = config.refit_max_optimizer_steps,
            status = "selection_success",
            error = "",
            failure_phase = "",
            failure_epoch = 0,
            failure_batch = 0,
            failure_optimizer_step = 0,
            first_nonfinite_tensor = "",
            n_observations = size(dataset.features, 1),
            n_features = input_dimension,
            n_train = length(prepared.outer.train_indices),
            n_test = length(prepared.outer.test_indices),
            split_seed = prepared.outer_seed,
            model_seed = seeds.model,
            best_epoch = selection.best_epoch,
            stopped_early = selection.stopped_early,
            selection_budget_limited = selection.budget_limited,
            refit_budget_limited = false,
            selection_seconds = selection_seconds,
            refit_seconds = NaN,
            evaluation_seconds = NaN,
            total_seconds = selection_seconds,
            selection_optimizer_steps = selection.optimizer_steps,
            refit_optimizer_steps = 0,
            selection_safe_exp_clamp_count =
                selection.numerical.clamp_count,
            selection_safe_exp_clamp_rate = selection.numerical.clamp_rate,
            refit_safe_exp_clamp_count = 0,
            refit_safe_exp_clamp_rate = 0.0,
            evaluation_safe_exp_clamp_count = 0,
            evaluation_safe_exp_clamp_rate = 0.0,
            y_center = prepared.outer.standardizer.y_center,
            y_scale = prepared.outer.standardizer.y_scale,
            eb_hidden_layer_variance = NaN,
            eb_output_layer_variance = NaN,
            metric_values...,
        )
    end

    refit_start = time()
    refit = refit_model(
        prepared.outer,
        input_dimension,
        likelihood,
        config;
        seed = seeds.model,
        epochs = selection.best_epoch,
    )
    refit_seconds = time() - refit_start

    evaluation_start = time()
    evaluation_tracker = NumericalTracker()
    metrics = evaluate_standardized_model(
        refit.params,
        prepared.outer.x_test_standardized,
        prepared.outer.y_test_standardized,
        prepared.outer.y_test,
        prepared.outer.standardizer,
        likelihood,
        config,
        evaluation_tracker,
    )
    evaluation_seconds = time() - evaluation_start
    prior_variances = empirical_bayes_prior_variances(
        refit.params, config,
    )
    evaluation_numerical = tracker_record(evaluation_tracker)

    all(isfinite, (
        metrics.lpd_original,
        metrics.rmse_original,
        metrics.mean_total_variance_original,
    )) || throw_numerical_error(
        "non-finite test metrics",
        refit.params,
        prepared.outer.x_test_standardized,
        likelihood,
        config;
        phase = "evaluation",
        epoch = selection.best_epoch,
        batch = 0,
        optimizer_step = refit.optimizer_steps,
        loss = metrics.lpd_original,
        tracker = evaluation_tracker,
    )
    metrics.mean_total_variance_original > 0 ||
        throw(ErrorException("non-positive predictive variance"))

    if config.save_checkpoints
        atomic_jld2_write(
            posterior_checkpoint_path(
                config, dataset.key, split_id, likelihood,
            ),
            (
                schema_version = DVI_POSTERIOR_SCHEMA_VERSION,
                split_protocol_version = UCI_SPLIT_PROTOCOL_VERSION,
                dataset = dataset.key,
                dataset_name = dataset.display_name,
                n_observations = size(dataset.features, 1),
                n_features = size(dataset.features, 2),
                split_id = split_id,
                split_spec = split_spec_record(prepared.split_spec),
                posterior_params = refit.params,
                standardizer = prepared.outer.standardizer,
                method = method_name(config),
                method_reference =
                    "Wu et al., ICLR 2019, arXiv:1810.03958",
                uci_protocol_reference =
                    "Wu et al., ICLR 2019, Table 2",
                best_epoch = selection.best_epoch,
                likelihood = likelihood,
                propagation = config.propagation,
                execution_backend = config.execution_backend,
                execution_device = config.execution_device,
                implementation_version = config.implementation_version,
                full_selection_patience_steps =
                    config.full_selection_patience_steps,
                training_budget_protocol = config.training_budget_protocol,
                selection_max_optimizer_steps =
                    config.selection_max_optimizer_steps,
                refit_max_optimizer_steps =
                    config.refit_max_optimizer_steps,
                selection_budget_limited = selection.budget_limited,
                refit_budget_limited = refit.budget_limited,
                model_seed = seeds.model,
                prediction_config = (
                    hidden_units = config.hidden_units,
                    homo_log_variance = config.homo_log_variance,
                ),
                training_config = config_dictionary(config),
                test_metrics = scalar_metrics(metrics),
            ),
        )
    end

    return (
        method = method_name(config),
        method_reference = "Wu et al., ICLR 2019, arXiv:1810.03958",
        uci_protocol_reference = "Wu et al., ICLR 2019, Table 2",
        numerical_protocol = config.numerical_protocol,
        dataset = dataset.key,
        dataset_name = dataset.display_name,
        split = split_id,
        split_protocol = UCI_SPLIT_PROTOCOL_VERSION,
        likelihood = likelihood,
        propagation = config.propagation,
        execution_backend = config.execution_backend,
        execution_device = config.execution_device,
        implementation_version = config.implementation_version,
        full_selection_patience_steps = config.full_selection_patience_steps,
        training_budget_protocol = config.training_budget_protocol,
        selection_max_optimizer_steps =
            config.selection_max_optimizer_steps,
        refit_max_optimizer_steps = config.refit_max_optimizer_steps,
        status = "success",
        error = "",
        failure_phase = "",
        failure_epoch = 0,
        failure_batch = 0,
        failure_optimizer_step = 0,
        first_nonfinite_tensor = "",
        n_observations = size(dataset.features, 1),
        n_features = input_dimension,
        n_train = length(prepared.outer.train_indices),
        n_test = length(prepared.outer.test_indices),
        split_seed = prepared.outer_seed,
        model_seed = seeds.model,
        best_epoch = selection.best_epoch,
        stopped_early = selection.stopped_early,
        selection_budget_limited = selection.budget_limited,
        refit_budget_limited = refit.budget_limited,
        selection_seconds = selection_seconds,
        refit_seconds = refit_seconds,
        evaluation_seconds = evaluation_seconds,
        total_seconds =
            selection_seconds + refit_seconds + evaluation_seconds,
        selection_optimizer_steps = selection.optimizer_steps,
        refit_optimizer_steps = refit.optimizer_steps,
        selection_safe_exp_clamp_count =
            selection.numerical.clamp_count,
        selection_safe_exp_clamp_rate = selection.numerical.clamp_rate,
        refit_safe_exp_clamp_count = refit.numerical.clamp_count,
        refit_safe_exp_clamp_rate = refit.numerical.clamp_rate,
        evaluation_safe_exp_clamp_count =
            evaluation_numerical.clamp_count,
        evaluation_safe_exp_clamp_rate = evaluation_numerical.clamp_rate,
        y_center = prepared.outer.standardizer.y_center,
        y_scale = prepared.outer.standardizer.y_scale,
        eb_hidden_layer_variance = prior_variances.hidden,
        eb_output_layer_variance = prior_variances.output,
        scalar_metrics(metrics)...,
    )
end

function failure_row(
    dataset,
    split_id::Int,
    likelihood::String,
    error,
    config::DVIConfig,
    backtrace;
    elapsed_seconds::Float64 = NaN,
)
    prepared = outer_and_inner_splits(dataset, split_id, config)
    seeds = model_and_batch_seeds(
        config, dataset.dataset_index, split_id, likelihood,
    )
    diagnostics = error isa DVINumericalError ? error.diagnostics : nothing
    failure_phase = diagnostics === nothing ? "unknown" : diagnostics.phase
    failure_optimizer_step = diagnostics === nothing ? 0 :
        diagnostics.optimizer_step
    clamp_count = diagnostics === nothing ? 0 :
        diagnostics.safe_exp_clamp_count
    clamp_rate = diagnostics === nothing ? NaN :
        diagnostics.safe_exp_clamp_rate
    metric_values = (; (
        name => NaN for name in DVI_SCALAR_METRIC_NAMES
    )...)
    return (
        method = method_name(config),
        method_reference = "Wu et al., ICLR 2019, arXiv:1810.03958",
        uci_protocol_reference = "Wu et al., ICLR 2019, Table 2",
        numerical_protocol = config.numerical_protocol,
        dataset = dataset.key,
        dataset_name = dataset.display_name,
        split = split_id,
        split_protocol = UCI_SPLIT_PROTOCOL_VERSION,
        likelihood = likelihood,
        propagation = config.propagation,
        execution_backend = config.execution_backend,
        execution_device = config.execution_device,
        implementation_version = config.implementation_version,
        full_selection_patience_steps = config.full_selection_patience_steps,
        training_budget_protocol = config.training_budget_protocol,
        selection_max_optimizer_steps =
            config.selection_max_optimizer_steps,
        refit_max_optimizer_steps = config.refit_max_optimizer_steps,
        status = "failure",
        error = sprint(showerror, error, backtrace),
        failure_phase = failure_phase,
        failure_epoch = diagnostics === nothing ? 0 : diagnostics.epoch,
        failure_batch = diagnostics === nothing ? 0 : diagnostics.batch,
        failure_optimizer_step = failure_optimizer_step,
        first_nonfinite_tensor = diagnostics === nothing ? "unknown" :
            diagnostics.first_nonfinite_tensor,
        n_observations = size(dataset.features, 1),
        n_features = size(dataset.features, 2),
        n_train = length(prepared.outer.train_indices),
        n_test = length(prepared.outer.test_indices),
        split_seed = prepared.outer_seed,
        model_seed = seeds.model,
        best_epoch = 0,
        stopped_early = false,
        selection_budget_limited = false,
        refit_budget_limited = false,
        selection_seconds = NaN,
        refit_seconds = NaN,
        evaluation_seconds = NaN,
        total_seconds = elapsed_seconds,
        selection_optimizer_steps = failure_phase == "selection" ?
            failure_optimizer_step : 0,
        refit_optimizer_steps = failure_phase == "refit" ?
            failure_optimizer_step : 0,
        selection_safe_exp_clamp_count = failure_phase == "selection" ?
            clamp_count : 0,
        selection_safe_exp_clamp_rate = failure_phase == "selection" ?
            clamp_rate : NaN,
        refit_safe_exp_clamp_count = failure_phase == "refit" ?
            clamp_count : 0,
        refit_safe_exp_clamp_rate = failure_phase == "refit" ?
            clamp_rate : NaN,
        evaluation_safe_exp_clamp_count = failure_phase == "evaluation" ?
            clamp_count : 0,
        evaluation_safe_exp_clamp_rate = failure_phase == "evaluation" ?
            clamp_rate : NaN,
        y_center = prepared.outer.standardizer.y_center,
        y_scale = prepared.outer.standardizer.y_scale,
        eb_hidden_layer_variance = NaN,
        eb_output_layer_variance = NaN,
        metric_values...,
    )
end

function write_failure_diagnostic(
    dataset,
    split_id::Int,
    likelihood::String,
    error,
    backtrace,
    config::DVIConfig,
)
    stem = configuration_stem(
        dataset.key, split_id, likelihood, config.propagation,
    )
    path = joinpath(
        config.output_dir,
        "failures",
        "$(stem)_$(time_ns()).jld2",
    )
    diagnostics = error isa DVINumericalError ? error.diagnostics : (
        phase = "unknown",
        epoch = 0,
        batch = 0,
        optimizer_step = 0,
        first_nonfinite_tensor = "unknown",
    )
    return atomic_jld2_write(path, (
        schema_version = "dvi-uci-failure-v1",
        dataset = dataset.key,
        split_id = split_id,
        likelihood = likelihood,
        propagation = config.propagation,
        execution_backend = config.execution_backend,
        execution_device = config.execution_device,
        implementation_version = config.implementation_version,
        full_selection_patience_steps = config.full_selection_patience_steps,
        training_budget_protocol = config.training_budget_protocol,
        selection_max_optimizer_steps =
            config.selection_max_optimizer_steps,
        refit_max_optimizer_steps = config.refit_max_optimizer_steps,
        numerical_protocol = config.numerical_protocol,
        error = sprint(showerror, error, backtrace),
        diagnostics = diagnostics,
        config = config_dictionary(config),
    ))
end

function run_benchmark(config::DVIConfig = load_config())
    validate_config(config)
    ensure_output_directories(config)
    save_config(config)
    write_split_manifest(config)
    runs = config.resume ? load_run_table(config) : DataFrame()
    total =
        length(config.datasets) *
        length(config.split_ids) *
        length(config.likelihoods)
    completed = 0

    for dataset_key in config.datasets
        dataset = load_dataset(dataset_key)
        for split_id in config.split_ids
            for likelihood in config.likelihoods
                completed += 1
                if config.resume && configuration_succeeded(
                    runs,
                    dataset_key,
                    split_id,
                    likelihood,
                    config.propagation,
                    config,
                )
                    println(
                        "[$completed/$total] skip $(dataset.display_name) " *
                        "split $split_id $likelihood $(config.propagation)",
                    )
                    continue
                end
                println(
                    "[$completed/$total] $(dataset.display_name) split " *
                    "$split_id $likelihood $(config.propagation)",
                )
                configuration_start = time()
                row = try
                    run_configuration(
                        dataset, split_id, likelihood, config,
                    )
                catch error
                    backtrace = catch_backtrace()
                    @error(
                        "DVI configuration failed",
                        dataset = dataset_key,
                        split = split_id,
                        likelihood = likelihood,
                        propagation = config.propagation,
                        exception = (error, backtrace),
                    )
                    write_failure_diagnostic(
                        dataset,
                        split_id,
                        likelihood,
                        error,
                        backtrace,
                        config,
                    )
                    failure_row(
                        dataset,
                        split_id,
                        likelihood,
                        error,
                        config,
                        backtrace;
                        elapsed_seconds = time() - configuration_start,
                    )
                end
                runs = append_run!(runs, row, config)
                if row.status in ("success", "selection_success")
                    if row.status == "selection_success"
                        @printf(
                            "    selection complete at epoch %d in %.1fs\n",
                            row.best_epoch,
                            row.total_seconds,
                        )
                        continue
                    end
                    @printf(
                        "    test LPD %.5f  RMSE %.5f  epoch %d  %.1fs\n",
                        row.lpd_original,
                        row.rmse_original,
                        row.best_epoch,
                        row.total_seconds,
                    )
                end
            end
        end
    end

    summary = refresh_artifacts(runs, config)
    successful_status = config.selection_only ?
        "selection_success" : "success"
    successful = isempty(runs) ? 0 :
        count(==(successful_status), string.(runs.status))
    failures = isempty(runs) ? 0 :
        count(==("failure"), string.(runs.status))
    println(
        "DVI benchmark complete: $successful successful rows, " *
        "$failures failure rows. Output: $(config.output_dir)",
    )
    return (
        runs = runs,
        summary = summary,
        output_dir = config.output_dir,
    )
end

main() = run_benchmark(load_config())
