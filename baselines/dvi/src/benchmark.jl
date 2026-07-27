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

    selection_start = time()
    selection = train_with_validation(
        prepared.inner,
        input_dimension,
        likelihood,
        config;
        seed = seeds.model,
    )
    selection_seconds = time() - selection_start
    history_name = @sprintf(
        "%s_split%02d_%s_%s.csv",
        dataset.key,
        split_id,
        likelihood,
        config.propagation,
    )
    atomic_csv_write(
        joinpath(config.output_dir, "histories", history_name),
        selection.history,
    )

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
    metrics = evaluate_standardized_model(
        refit.params,
        prepared.outer.x_test_standardized,
        prepared.outer.y_test_standardized,
        prepared.outer.y_test,
        prepared.outer.standardizer,
        likelihood,
        config,
    )
    evaluation_seconds = time() - evaluation_start
    prior_variances = empirical_bayes_prior_variances(
        refit.params, config,
    )

    all(isfinite, (
        metrics.lpd_original,
        metrics.rmse_original,
        metrics.mean_total_variance_original,
    )) || throw(ErrorException("non-finite test metrics"))
    metrics.mean_total_variance_original > 0 ||
        throw(ErrorException("non-positive predictive variance"))

    if config.save_checkpoints
        checkpoint_name = replace(history_name, ".csv" => ".jld2")
        JLD2.jldsave(
            joinpath(
                config.output_dir, "checkpoints", checkpoint_name,
            );
            params = refit.params,
            standardizer = prepared.outer.standardizer,
            test_indices = prepared.outer.test_indices,
            method = method_name(config),
            method_reference =
                "Wu et al., ICLR 2019, arXiv:1810.03958",
            best_epoch = selection.best_epoch,
            likelihood = likelihood,
            propagation = config.propagation,
            model_seed = seeds.model,
        )
    end

    return (
        method = method_name(config),
        method_reference = "Wu et al., ICLR 2019, arXiv:1810.03958",
        dataset = dataset.key,
        dataset_name = dataset.display_name,
        split = split_id,
        likelihood = likelihood,
        propagation = config.propagation,
        status = "success",
        error = "",
        n_observations = size(dataset.features, 1),
        n_features = input_dimension,
        n_train = length(prepared.outer.train_indices),
        n_test = length(prepared.outer.test_indices),
        split_seed = prepared.outer_seed,
        model_seed = seeds.model,
        best_epoch = selection.best_epoch,
        stopped_early = selection.stopped_early,
        selection_seconds = selection_seconds,
        refit_seconds = refit_seconds,
        evaluation_seconds = evaluation_seconds,
        total_seconds =
            selection_seconds + refit_seconds + evaluation_seconds,
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
)
    metric_values = (; (
        name => NaN for name in DVI_SCALAR_METRIC_NAMES
    )...)
    return (
        method = method_name(config),
        method_reference = "Wu et al., ICLR 2019, arXiv:1810.03958",
        dataset = dataset.key,
        dataset_name = dataset.display_name,
        split = split_id,
        likelihood = likelihood,
        propagation = config.propagation,
        status = "failure",
        error = sprint(showerror, error, catch_backtrace()),
        n_observations = size(dataset.features, 1),
        n_features = size(dataset.features, 2),
        n_train = 0,
        n_test = 0,
        split_seed = config.split_seed + split_id - 1,
        model_seed = 0,
        best_epoch = 0,
        stopped_early = false,
        selection_seconds = NaN,
        refit_seconds = NaN,
        evaluation_seconds = NaN,
        total_seconds = NaN,
        y_center = NaN,
        y_scale = NaN,
        eb_hidden_layer_variance = NaN,
        eb_output_layer_variance = NaN,
        metric_values...,
    )
end

function run_benchmark(config::DVIConfig = load_config())
    validate_config(config)
    ensure_output_directories(config)
    save_config(config)
    runs = config.resume ? load_run_table(config) : DataFrame()
    total =
        length(config.datasets) *
        config.n_splits *
        length(config.likelihoods)
    completed = 0

    for dataset_key in config.datasets
        dataset = load_dataset(dataset_key)
        for split_id in 1:config.n_splits
            for likelihood in config.likelihoods
                completed += 1
                if config.resume && configuration_succeeded(
                    runs,
                    dataset_key,
                    split_id,
                    likelihood,
                    config.propagation,
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
                row = try
                    run_configuration(
                        dataset, split_id, likelihood, config,
                    )
                catch error
                    @error(
                        "DVI configuration failed",
                        dataset = dataset_key,
                        split = split_id,
                        likelihood = likelihood,
                        propagation = config.propagation,
                        exception = (error, catch_backtrace()),
                    )
                    failure_row(
                        dataset, split_id, likelihood, error, config,
                    )
                end
                runs = append_run!(runs, row, config)
                if row.status == "success"
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
    successful = isempty(runs) ? 0 :
        count(==("success"), string.(runs.status))
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
