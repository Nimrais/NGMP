function run_configuration(
    dataset,
    split_id::Int,
    config::BPCConfig,
)
    prepared = outer_and_inner_splits(dataset, split_id, config)
    seeds = model_and_evaluation_seeds(
        config, dataset.dataset_index, split_id,
    )
    input_dimension = size(dataset.features, 2)

    selection_start = time()
    selection = train_with_validation(
        prepared.inner,
        input_dimension,
        config;
        seed = seeds.model,
        validation_seed = seeds.validation,
    )
    synchronize_backend(config)
    selection_seconds = time() - selection_start
    atomic_csv_write(
        joinpath(
            config.output_dir,
            "histories",
            configuration_stem(dataset.key, split_id) * ".csv",
        ),
        selection.history,
    )

    refit_start = time()
    model = refit_model(
        prepared.outer,
        input_dimension,
        config;
        seed = seeds.model,
        epochs = selection.best_epoch,
    )
    synchronize_backend(config)
    refit_seconds = time() - refit_start

    evaluation_start = time()
    metrics, _ = evaluate_standardized_model(
        model,
        prepared.outer.x_test_standardized,
        prepared.outer.y_test_standardized,
        prepared.outer.y_test,
        prepared.outer.standardizer,
        config;
        seed = seeds.test,
    )
    evaluation_seconds = time() - evaluation_start
    all(isfinite, (
        metrics.lpd_original,
        metrics.rmse_original,
        metrics.mean_total_variance_original,
    )) || throw(ErrorException("non-finite BPC test metrics"))
    metrics.mean_total_variance_original > 0 ||
        throw(ErrorException("non-positive BPC predictive variance"))

    if config.save_checkpoints
        atomic_jld2_write(
            posterior_checkpoint_path(config, dataset.key, split_id),
            posterior_record(
                dataset, prepared, model, selection, seeds, metrics, config,
            ),
        )
    end

    return (
        method = "Bayesian Predictive Coding (Matrix-Normal-Wishart)",
        method_reference = "Tschantz et al., arXiv:2503.24016 (2025)",
        uci_protocol_reference =
            "Tschantz et al., arXiv:2503.24016, Appendix F.3; shared repeated-holdout-v1",
        dataset = dataset.key,
        dataset_name = dataset.display_name,
        split = split_id,
        split_protocol = UCI_SPLIT_PROTOCOL_VERSION,
        likelihood = "homoscedastic",
        posterior_family = "matrix_normal_wishart",
        backend = config.backend,
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
        posterior_updates = model.update_step,
        latent_steps_per_update = config.latent_steps,
        selection_seconds = selection_seconds,
        refit_seconds = refit_seconds,
        evaluation_seconds = evaluation_seconds,
        total_seconds = selection_seconds + refit_seconds + evaluation_seconds,
        y_center = prepared.outer.standardizer.y_center,
        y_scale = prepared.outer.standardizer.y_scale,
        scalar_metrics(metrics)...,
    )
end

function failure_row(
    dataset,
    split_id::Int,
    error,
    config::BPCConfig,
)
    metric_values = (; (name => NaN for name in SCALAR_METRIC_NAMES)...)
    return (
        method = "Bayesian Predictive Coding (Matrix-Normal-Wishart)",
        method_reference = "Tschantz et al., arXiv:2503.24016 (2025)",
        uci_protocol_reference =
            "Tschantz et al., arXiv:2503.24016, Appendix F.3; shared repeated-holdout-v1",
        dataset = dataset.key,
        dataset_name = dataset.display_name,
        split = split_id,
        split_protocol = UCI_SPLIT_PROTOCOL_VERSION,
        likelihood = "homoscedastic",
        posterior_family = "matrix_normal_wishart",
        backend = config.backend,
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
        posterior_updates = 0,
        latent_steps_per_update = config.latent_steps,
        selection_seconds = NaN,
        refit_seconds = NaN,
        evaluation_seconds = NaN,
        total_seconds = NaN,
        y_center = NaN,
        y_scale = NaN,
        metric_values...,
    )
end

"""Run requested UCI dataset/split combinations and write each result immediately."""
function run_benchmark(config::BPCConfig = load_config())
    validate_config(config)
    ensure_output_directories(config)
    save_config(config)
    write_split_manifest(config)
    runs = config.resume ? load_run_table(config) : DataFrame()
    total = length(config.datasets) * config.n_splits
    completed = 0

    for dataset_key in config.datasets
        dataset = load_dataset(dataset_key)
        for split_id in 1:config.n_splits
            completed += 1
            if config.resume && configuration_succeeded(
                runs, dataset_key, split_id, config,
            )
                println(
                    "[$completed/$total] skip $(dataset.display_name) split $split_id " *
                    "(already successful)",
                )
                continue
            end
            println(
                "[$completed/$total] $(dataset.display_name) split $split_id " *
                "homoscedastic MNW on $(config.backend)",
            )
            row = try
                run_configuration(dataset, split_id, config)
            catch error
                @error(
                    "BPC configuration failed",
                    dataset = dataset_key,
                    split = split_id,
                    backend = config.backend,
                    exception = (error, catch_backtrace()),
                )
                failure_row(dataset, split_id, error, config)
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

    summary = refresh_artifacts(runs, config)
    successful = isempty(runs) ? 0 : count(==("success"), string.(runs.status))
    failures = isempty(runs) ? 0 : count(==("failure"), string.(runs.status))
    println(
        "BPC benchmark complete: $successful successful rows, $failures failure rows. " *
        "Output: $(config.output_dir)",
    )
    return (runs = runs, summary = summary, output_dir = config.output_dir)
end

main() = run_benchmark(load_config())
