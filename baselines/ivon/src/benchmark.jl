function full_training_partitions(data; max_observations = nothing)
    # Preserve upstream train_set=false: update on validation and monitor train.
    fit = (
        predictions = data.predictions_val_vec_moe,
        features = data.features_val,
        targets = data.y_val_moe,
    )
    monitor = (
        predictions = data.predictions_train_vec_moe,
        features = data.features_train,
        targets = data.y_train_moe,
    )
    if max_observations !== nothing
        fit_n = min(length(fit.features), Int(max_observations))
        monitor_n = min(length(monitor.features), Int(max_observations))
        fit = subset_gate_data(fit.predictions, fit.features, fit.targets, 1:fit_n)
        monitor = subset_gate_data(
            monitor.predictions, monitor.features, monitor.targets, 1:monitor_n)
    end
    return fit, monitor
end

function evaluate_training(training, test_data, scaler, col_idx, config)
    sample_indices = posterior_subset_indices(config.posterior_samples;
        sample_counts = config.posterior_sample_counts,
        seed = config.posterior_subset_seed)
    components = posterior_components(training, test_data;
        nsamples = config.posterior_samples, seed = config.prediction_seed,
        sample_indices)
    standardized_by_count = mixture_metrics_by_sample_count(
        components, test_data.targets, sample_indices;
        interval_seed = config.interval_seed)
    scale = Float64(scaler.σ[col_idx])
    sample_evaluations = Dict{Int,Any}()
    for count in config.posterior_sample_counts
        standardized_at_count = standardized_by_count[count]
        sample_evaluations[count] = (
            standardized = standardized_at_count,
            original_unit_metrics = original_unit_metrics(
                standardized_at_count.metrics, scale),
        )
    end
    primary = sample_evaluations[config.posterior_samples]

    mean_components = posterior_components(training, test_data;
        nsamples = 1, seed = config.prediction_seed, sample_posterior = false)
    mean_standardized = mixture_metrics(mean_components, test_data.targets;
        interval_seed = config.interval_seed)
    mean_original = original_unit_metrics(mean_standardized.metrics, scale)
    return (; components, sample_indices, sample_evaluations,
        standardized = primary.standardized,
        original = primary.original_unit_metrics,
        mean_components, mean_standardized, mean_original)
end

function fit_cell(dataset::String, horizon::Int, architecture::Symbol;
    learning_rate::Real, ess_multiplier::Integer, output_root::String = CHECKPOINT_ROOT,
    phase::Symbol = :full, n_epochs::Integer = 100,
    posterior_samples::Integer = 1000, max_train_observations = nothing,
    max_eval_observations = nothing, force::Bool = false, prepared_data = nothing)
    config = effective_config(; phase, dataset, horizon, architecture, learning_rate,
        ess_multiplier, n_epochs, posterior_samples, max_train_observations,
        max_eval_observations)
    path = checkpoint_path(output_root, config)
    if !force && checkpoint_complete(path, config)
        append_run_record(run_record(path, config, "resumed"))
        return path
    end

    frozen_before = verify_frozen_hashes(dataset, horizon)
    data = if prepared_data === nothing
        _, _, loaded = prepare_full_data(dataset, horizon)
        loaded
    else
        prepared_data
    end
    fit, monitor = full_training_partitions(data; max_observations = max_train_observations)
    quantile_before = quantile_hashes(fit)
    training = train_ivon_gate(fit, monitor, architecture;
        learning_rate, ess_multiplier, seed = config.training_seed, n_epochs,
        patience = 1, min_delta = 1.0e-3)
    quantile_after = quantile_hashes(fit)
    quantile_before == quantile_after || error("Frozen quantile experts changed")
    frozen_after = verify_frozen_hashes(dataset, horizon)
    frozen_before == frozen_after || error("Frozen checkpoint hashes changed")

    test_data = restrict_full_test(data; max_observations = max_eval_observations)
    evaluated = evaluate_training(training, test_data, data.scaler, data.col_idx, config)
    digest = component_digest(evaluated.components)
    save_checkpoint(path;
        config,
        training,
        frozen_hashes = frozen_after,
        quantile_hashes_before = quantile_before,
        quantile_hashes_after = quantile_after,
        split_metadata = split_metadata(data, :full),
        scaler = data.scaler,
        standardized = evaluated.standardized,
        original_units = evaluated.original,
        mean_standardized = evaluated.mean_standardized,
        mean_original_units = evaluated.mean_original,
        posterior_components = evaluated.components,
        posterior_sample_indices = evaluated.sample_indices,
        posterior_sample_evaluations = evaluated.sample_evaluations,
        mean_components = evaluated.mean_components,
        prediction_digest = digest,
    )
    append_run_record(run_record(path, config, "completed"))
    return path
end

function pilot_fit_data()
    _, _, data = prepare_pilot_data()
    metadata = split_metadata(data, :pilot)
    fit = subset_gate_data(data.predictions, data.features, data.targets,
        metadata.fit_range[1]:metadata.fit_range[2])
    selection = subset_gate_data(data.predictions, data.features, data.targets,
        metadata.selection_range[1]:metadata.selection_range[2])
    return data, fit, selection, metadata
end

function run_pilot(; n_epochs::Integer = 100, posterior_samples::Integer = 1000,
    max_train_observations = nothing, max_eval_observations = nothing,
    force::Bool = false)
    initialize_results_root()
    all_rows = NamedTuple[]
    data, fit_all, selection_all, metadata = pilot_fit_data()
    frozen_before = verify_frozen_hashes("ETTh1", 96)
    for learning_rate in LEARNING_RATES, ess_multiplier in ESS_MULTIPLIERS
        for architecture in ARCHITECTURES
            fit = fit_all
            selection = selection_all
            max_train_observations === nothing || begin
                n = min(length(fit.features), Int(max_train_observations))
                fit = subset_gate_data(fit.predictions, fit.features, fit.targets, 1:n)
            end
            max_eval_observations === nothing || begin
                n = min(length(selection.features), Int(max_eval_observations))
                selection = subset_gate_data(
                    selection.predictions, selection.features, selection.targets, 1:n)
            end
            config = effective_config(; phase = :pilot, dataset = "ETTh1", horizon = 96,
                architecture, learning_rate, ess_multiplier, n_epochs,
                posterior_samples, max_train_observations, max_eval_observations)
            rate_tag = replace(string(learning_rate), "." => "p")
            cell_path = joinpath(PILOT_ROOT,
                "ETTh1_h96_$(architecture_name(architecture))_lr$(rate_tag)_ess$(ess_multiplier).jld2")
            metrics = nothing
            if !force && isfile(cell_path)
                saved = JLD2.load(cell_path)
                if get(saved, "complete", false) === true &&
                   saved["config_fingerprint"] == config_fingerprint(config) &&
                   saved["prediction_digest"] == component_digest(saved["posterior_components"])
                    metrics = saved["selection_metrics"]
                    append_run_record(run_record(cell_path, config, "resumed"))
                end
            end
            if metrics === nothing
                quantile_before = quantile_hashes(fit)
                result = train_ivon_gate(fit, selection, architecture;
                    learning_rate, ess_multiplier, seed = config.training_seed,
                    n_epochs, patience = 1, min_delta = 1.0e-3)
                components = posterior_components(result, selection;
                    nsamples = posterior_samples, seed = config.prediction_seed)
                metrics = mixture_selection_metrics(components, selection.targets)
                quantile_after = quantile_hashes(fit)
                quantile_before == quantile_after || error("Frozen pilot quantiles changed")
                frozen_after = verify_frozen_hashes("ETTh1", 96)
                frozen_before == frozen_after || error("Frozen pilot checkpoints changed")
                prediction_digest = component_digest(components)
                JLD2.jldsave(cell_path;
                    complete = true,
                    created_at = string(now(UTC)),
                    effective_config = config,
                    config_fingerprint = config_fingerprint(config),
                    pge_commit = PGE_COMMIT,
                    ivonrepro_commit = IVON_COMMIT,
                    frozen_hashes = frozen_after,
                    quantile_hashes_before = quantile_before,
                    quantile_hashes_after = quantile_after,
                    split_metadata = metadata,
                    scaler = data.scaler,
                    gate_parameters_mean = result.parameters,
                    gate_states = result.states,
                    optimizer_state = result.optimizer_state,
                    optimizer = result.optimizer,
                    training_step = result.step,
                    history = result.history,
                    selection_metrics = metrics,
                    posterior_components = components,
                    prediction_digest = prediction_digest,
                    test_targets_read = false,
                )
                append_run_record(run_record(cell_path, config, "completed"))
            end
            row = (
                learning_rate = Float64(learning_rate),
                ess_multiplier = Int(ess_multiplier),
                architecture = architecture_name(architecture),
                nll = metrics.nll,
                mse = metrics.mse,
                test_targets_read = metadata.test_targets_read,
            )
            push!(all_rows, row)
        end
    end
    grouped = NamedTuple[]
    for learning_rate in LEARNING_RATES, ess_multiplier in ESS_MULTIPLIERS
        rows = filter(r -> r.learning_rate == learning_rate &&
            r.ess_multiplier == ess_multiplier, all_rows)
        length(rows) == 2 || error("Incomplete two-head pilot cell")
        push!(grouped, (
            learning_rate = Float64(learning_rate),
            ess_multiplier = Int(ess_multiplier),
            mean_nll = mean(r.nll for r in rows),
            mean_mse = mean(r.mse for r in rows),
        ))
    end
    sort!(grouped; by = r -> (r.mean_nll, r.mean_mse))
    selected = first(grouped)
    write_csv(joinpath(PILOT_ROOT, "pilot_heads.csv"), all_rows,
        (:learning_rate, :ess_multiplier, :architecture, :nll, :mse, :test_targets_read))
    write_csv(joinpath(PILOT_ROOT, "pilot_grid.csv"), grouped,
        (:learning_rate, :ess_multiplier, :mean_nll, :mean_mse))
    JLD2.jldsave(joinpath(PILOT_ROOT, "selection.jld2");
        complete = true, selected = selected, rows = all_rows, grouped = grouped,
        selection_rule = "min mean posterior-predictive NLL over both heads; MSE tie-break",
        test_targets_read = false)
    write_protocol_config(; selected)
    return selected
end

function selected_config()
    path = joinpath(PILOT_ROOT, "selection.jld2")
    isfile(path) || error("Pilot selection missing. Run `julia --project=. run.jl pilot` first.")
    saved = JLD2.load(path)
    get(saved, "complete", false) === true || error("Incomplete pilot selection")
    saved["test_targets_read"] == false || error("Pilot selection accessed test targets")
    return saved["selected"]
end

function run_full(; force::Bool = false)
    selected = selected_config()
    initialize_results_root(; selected)
    paths = String[]
    for dataset in DATASETS, horizon in HORIZONS
        cell_configs = [effective_config(; phase = :full, dataset, horizon,
            architecture, learning_rate = selected.learning_rate,
            ess_multiplier = selected.ess_multiplier) for architecture in ARCHITECTURES]
        cell_paths = [checkpoint_path(CHECKPOINT_ROOT, config) for config in cell_configs]
        if !force && all(checkpoint_complete(path, config) for
                         (path, config) in zip(cell_paths, cell_configs))
            for (path, config) in zip(cell_paths, cell_configs)
                append_run_record(run_record(path, config, "resumed"))
                push!(paths, path)
            end
            continue
        end
        _, _, data = prepare_full_data(dataset, horizon)
        for architecture in ARCHITECTURES
            push!(paths, fit_cell(dataset, horizon, architecture;
                learning_rate = selected.learning_rate,
                ess_multiplier = selected.ess_multiplier,
                prepared_data = data,
                force))
        end
    end
    length(paths) == 16 || error("Expected 16 final benchmark cells")
    all(isfile, paths) || error("A final checkpoint is missing")
    summarize()
    return paths
end

function run_smoke(; force::Bool = false)
    initialize_results_root()
    paths = String[]
    configs = [effective_config(; phase = :smoke, dataset = "ETTh1", horizon = 96,
        architecture, learning_rate = 0.01, ess_multiplier = 1, n_epochs = 1,
        posterior_samples = 1000, max_train_observations = 4,
        max_eval_observations = 4) for architecture in ARCHITECTURES]
    prior_paths = [checkpoint_path(SMOKE_ROOT, config) for config in configs]
    if !force && all(checkpoint_complete(path, config) for
                     (path, config) in zip(prior_paths, configs))
        for (path, config) in zip(prior_paths, configs)
            append_run_record(run_record(path, config, "resumed"))
        end
        return prior_paths
    end
    _, _, data = prepare_smoke_data("ETTh1", 96; observations = 4)
    for architecture in ARCHITECTURES
        push!(paths, fit_cell("ETTh1", 96, architecture;
            learning_rate = 0.01, ess_multiplier = 1,
            output_root = SMOKE_ROOT, phase = :smoke, n_epochs = 1,
            posterior_samples = 1000, max_train_observations = 4,
            max_eval_observations = 4, prepared_data = data, force))
    end
    return paths
end

function summarize()
    paths = sort(filter(p -> endswith(p, ".jld2"),
        isdir(CHECKPOINT_ROOT) ? readdir(CHECKPOINT_ROOT; join = true) : String[]))
    isempty(paths) && error("No final checkpoints to summarize")
    main_rows = NamedTuple[]
    sensitivity_rows = NamedTuple[]
    mean_rows = NamedTuple[]
    for path in paths
        saved = load_checkpoint(path)
        push!(main_rows, flatten_metrics_row(saved, path))
        for count in sort!(Int.(collect(keys(saved["posterior_sample_indices"]))))
            push!(sensitivity_rows,
                flatten_metrics_row(saved, path; posterior_samples = count))
        end
        push!(mean_rows, flatten_metrics_row(saved, path; posterior_mean = true))
    end
    sort!(main_rows; by = r -> (r.dataset, r.horizon, r.architecture))
    sort!(sensitivity_rows;
        by = r -> (r.dataset, r.horizon, r.architecture, r.posterior_samples))
    sort!(mean_rows; by = r -> (r.dataset, r.horizon, r.architecture))
    write_csv(joinpath(RESULTS_ROOT, "summary.csv"), main_rows, RESULT_COLUMNS)
    write_csv(joinpath(RESULTS_ROOT, "main_ensemble_table.csv"), main_rows, RESULT_COLUMNS)
    write_csv(joinpath(RESULTS_ROOT, "posterior_sample_sensitivity_table.csv"),
        sensitivity_rows, RESULT_COLUMNS)
    write_csv(joinpath(RESULTS_ROOT, "appendix_mean_head_table.csv"), mean_rows,
        RESULT_COLUMNS)
    write_markdown_table(joinpath(RESULTS_ROOT, "main_ensemble_table.md"), main_rows;
        title = "IVON posterior-gate ensemble")
    write_markdown_table(
        joinpath(RESULTS_ROOT, "posterior_sample_sensitivity_table.md"),
        sensitivity_rows; title = "IVON posterior-sample sensitivity")
    write_markdown_table(joinpath(RESULTS_ROOT, "appendix_mean_head_table.md"), mean_rows;
        title = "Appendix: IVON posterior-mean gate")
    return (; main_rows, sensitivity_rows, mean_rows)
end
