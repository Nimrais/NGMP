# Clean-label-only focused tuning for the width-4, 14-coefficient Probit model.
#
# Full search, two unseen final evaluations, and five-panel surfaces:
#   OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_ctransition_mvsoftplus_probit_low_rank_tuning.jl
#
# End-to-end smoke:
#   XOR_CT_PROBIT_LOW_RANK_TUNING_SMOKE=true OPENBLAS_NUM_THREADS=1 \
#     julia --project=. \
#     experiments/xor_ctransition_mvsoftplus_probit_low_rank_tuning.jl

const LOW_RANK_TUNING_SMOKE = lowercase(get(
    ENV,
    "XOR_CT_PROBIT_LOW_RANK_TUNING_SMOKE",
    "false",
)) in ("1", "true", "yes", "on")

ENV["XOR_CT_PROBIT_SMOKE"] = string(LOW_RANK_TUNING_SMOKE)
include(joinpath(@__DIR__, "xor_ctransition_mvsoftplus_probit_low_rank.jl"))

const LOW_RANK_TUNING_SEEDS = (
    (data_seed = 2_026, split_seed = 2_027, prior_seed = 42),
    (data_seed = 2_028, split_seed = 2_029, prior_seed = 43),
)

const LOW_RANK_FINAL_SEEDS = (
    (data_seed = 2_030, split_seed = 2_031, prior_seed = 44),
    (data_seed = 2_032, split_seed = 2_033, prior_seed = 45),
)

const LOW_RANK_BASIS_CANDIDATES = [
    (
        name = "map_$(map_parameters)_pred_$(14 - map_parameters)_basis_$basis_seed",
        map_parameters = map_parameters,
        pred_parameters = 14 - map_parameters,
        basis_seed = basis_seed,
    )
    for map_parameters in (5, 6, 7, 8), basis_seed in 1:3
] |> vec

const LOW_RANK_HYPER_CANDIDATES = [
    (
        name = "score_$(score_mean)_scale_$(atom_scale)",
        score_precision_mean = Float64(score_mean),
        score_precision_concentration = 10.0,
        atom_scale_multiplier = Float64(atom_scale),
    )
    for score_mean in (3, 10, 30), atom_scale in (0.5, 1.0)
] |> vec

const LOW_RANK_TUNING_OUTPUT_PREFIX = get(
    ENV,
    "TUNING_OUTPUT_PREFIX",
    joinpath(
        @__DIR__,
        "..",
        "viz",
        "xor_ctransition_mvsoftplus_probit_low_rank_tuning",
    ),
)

function low_rank_candidate_config(base, seeds; n_samples, iterations, prediction_iterations, hyper)
    return merge(
        base,
        seeds,
        (;
            n_samples,
            iterations,
            prediction_iterations,
            score_precision_mean = hyper.score_precision_mean,
            score_precision_concentration = hyper.score_precision_concentration,
            save_outputs = false,
            show_progress = false,
            require_clean_baseline = false,
        ),
    )
end

function fit_low_rank_clean(config, basis, hyper; iterations = config.iterations)
    dataset = make_clean_xor_dataset(n = config.n_samples, seed = config.data_seed)
    train_data, test_data = split_dataset(
        dataset;
        train_fraction = config.train_fraction,
        seed = config.split_seed,
    )
    train_features = build_features(train_data)
    test_features = build_features(test_data)
    setup = make_probit_low_rank_setup(
        config;
        map_parameters = basis.map_parameters,
        pred_parameters = basis.pred_parameters,
        basis_seed = basis.basis_seed,
        atom_scale_multiplier = hyper.atom_scale_multiplier,
    )
    fit = fit_probit_arm(
        config,
        train_features,
        Float64.(train_data.label);
        iterations,
        priors = setup.priors,
        meta_map = setup.meta_map,
        meta_pred = setup.meta_pred,
        compute_free_energy = false,
    )
    return (; fit, setup, train_data, test_data, train_features, test_features)
end

function low_rank_metrics(run, config; checkpoint, prediction_iterations)
    prediction_config = merge(config, (; prediction_iterations))
    statistics = score_statistics(predict_score_marginals(
        run.fit,
        run.test_features;
        config = prediction_config,
        iteration = checkpoint,
    ); verify_native = false)
    labels = Float64.(run.test_data.label)
    metrics = classification_metrics(labels, statistics.probability)
    return merge(metrics, (;
        brier = mean(abs2, statistics.probability .- labels),
        mean_latent_variance = mean(statistics.score_variance),
    ))
end

function safe_low_rank_fit(config, basis, hyper; iterations)
    try
        return (
            status = "ok",
            run = fit_low_rank_clean(config, basis, hyper; iterations),
            error = missing,
        )
    catch exception
        return (
            status = "failed",
            run = nothing,
            error = sprint(showerror, exception, catch_backtrace()),
        )
    end
end

function final_low_rank_evaluation(config, basis, hyper, output_prefix; save_surface)
    run = fit_low_rank_clean(config, basis, hyper; iterations = config.iterations)
    prediction_elapsed = @elapsed statistics = score_statistics(
        predict_score_marginals(run.fit, run.test_features; config = config)
    )
    labels = Float64.(run.test_data.label)
    metrics = classification_metrics(labels, statistics.probability)
    brier = mean(abs2, statistics.probability .- labels)
    baseline = class_prior_baseline(Float64.(run.train_data.label), labels)

    surface_path = nothing
    if save_surface
        grid = make_prediction_grid(config.grid_size)
        grid_statistics = reshape_grid_statistics(score_statistics(
            predict_score_marginals(run.fit, grid.features; config = config);
            verify_native = false,
        ), grid)
        surface_path = save_five_panel_surface(
            :clean,
            grid,
            grid_statistics,
            output_prefix;
            score_mean_limits = nondegenerate_limits(
                minimum(grid_statistics.score_mean),
                maximum(grid_statistics.score_mean),
            ),
            score_variance_limits = nondegenerate_limits(
                minimum(grid_statistics.score_variance),
                maximum(grid_statistics.score_variance),
            ),
        )
    end

    return (;
        run,
        statistics,
        metrics,
        brier,
        baseline,
        prediction_elapsed,
        surface_path,
    )
end

function write_low_rank_tuning_summary(path, winner, final, elapsed)
    open(path, "w") do io
        println(io, "# Width-4 CT/MvSoftplus Probit low-rank tuning")
        println(io)
        println(io, "All selection used clean tuning seeds only; final seeds were unseen.")
        println(io)
        println(io, "- Transition coefficients: `14` (`$(winner.map_parameters)` map + `$(winner.pred_parameters)` predictor)")
        println(io, "- Orthogonal basis seed: `$(winner.basis_seed)`")
        println(io, "- Atom-scale multiplier: `$(winner.atom_scale_multiplier)`")
        println(io, "- Score precision mean/concentration: `$(winner.score_precision_mean)` / `$(winner.score_precision_concentration)`")
        println(io, "- Training checkpoint: `$(winner.checkpoint)`")
        println(io, "- Prediction iterations: `$(winner.prediction_iterations)`")
        println(io, "- Mean refinement NLL: `$(winner.mean_nll)`")
        println(io, "- Mean final NLL: `$(mean(final.nll))`")
        println(io, "- Mean final accuracy: `$(mean(final.accuracy))`")
        println(io, "- Total wall time: `$(round(elapsed, digits = 1))` seconds")
    end
end

function run_low_rank_focused_tuning()
    smoke = LOW_RANK_TUNING_SMOKE
    basis_candidates = smoke ? LOW_RANK_BASIS_CANDIDATES[1:1] : LOW_RANK_BASIS_CANDIDATES
    hyper_candidates = smoke ? LOW_RANK_HYPER_CANDIDATES[1:1] : LOW_RANK_HYPER_CANDIDATES
    tuning_seeds = smoke ? LOW_RANK_TUNING_SEEDS[1:1] : LOW_RANK_TUNING_SEEDS
    final_seeds = smoke ? LOW_RANK_FINAL_SEEDS[1:1] : LOW_RANK_FINAL_SEEDS
    screen_samples = smoke ? 24 : 400
    screen_iterations = smoke ? 1 : 80
    screen_prediction_iterations = smoke ? 1 : 3
    refine_samples = smoke ? 24 : 400
    checkpoints = smoke ? (1,) : (120, 160)
    prediction_iterations = smoke ? (1,) : (5, 10)
    final_samples = smoke ? 24 : 2_000
    save_outputs = !smoke
    fixed_hyper = (
        score_precision_mean = 10.0,
        score_precision_concentration = 10.0,
        atom_scale_multiplier = 1.0,
    )

    total_elapsed = @elapsed begin
        println("=== low-rank orthogonal-basis screening")
        screen_rows = NamedTuple[]
        for basis in basis_candidates
            config = low_rank_candidate_config(
                CONFIG,
                first(LOW_RANK_TUNING_SEEDS);
                n_samples = screen_samples,
                iterations = screen_iterations,
                prediction_iterations = screen_prediction_iterations,
                hyper = fixed_hyper,
            )
            fitted = safe_low_rank_fit(
                config,
                basis,
                fixed_hyper;
                iterations = screen_iterations,
            )
            if fitted.status == "ok"
                metrics = low_rank_metrics(
                    fitted.run,
                    config;
                    checkpoint = screen_iterations,
                    prediction_iterations = screen_prediction_iterations,
                )
                push!(screen_rows, merge(basis, (;
                    status = "ok",
                    nll = metrics.nll,
                    accuracy = metrics.accuracy,
                    brier = metrics.brier,
                    mean_latent_variance = metrics.mean_latent_variance,
                    error = missing,
                )))
                println(basis.name, ": NLL=", round(metrics.nll, digits = 5),
                        ", accuracy=", round(metrics.accuracy, digits = 4))
            else
                push!(screen_rows, merge(basis, (;
                    status = "failed",
                    nll = Inf,
                    accuracy = NaN,
                    brier = NaN,
                    mean_latent_variance = NaN,
                    error = fitted.error,
                )))
                println(basis.name, ": failed")
            end
        end
        screen = sort(DataFrame(screen_rows), :nll)
        successful = filter(:status => ==("ok"), screen)
        nrow(successful) > 0 || error("every low-rank basis candidate failed")
        selected_count = min(2, nrow(successful))
        selected_basis = [
            only(filter(
                candidate -> candidate.name == successful.name[index],
                LOW_RANK_BASIS_CANDIDATES,
            ))
            for index in 1:selected_count
        ]

        println("\n=== two-seed hyperparameter/checkpoint refinement")
        refinement_rows = NamedTuple[]
        max_checkpoint = maximum(checkpoints)
        for basis in selected_basis, hyper in hyper_candidates, (seed_index, seeds) in enumerate(tuning_seeds)
            config = low_rank_candidate_config(
                CONFIG,
                seeds;
                n_samples = refine_samples,
                iterations = max_checkpoint,
                prediction_iterations = maximum(prediction_iterations),
                hyper,
            )
            fitted = safe_low_rank_fit(
                config,
                basis,
                hyper;
                iterations = max_checkpoint,
            )
            if fitted.status == "ok"
                for checkpoint in checkpoints, pred_iterations in prediction_iterations
                    metrics = low_rank_metrics(
                        fitted.run,
                        config;
                        checkpoint,
                        prediction_iterations = pred_iterations,
                    )
                    push!(refinement_rows, merge(basis, hyper, (;
                        seed_index,
                        checkpoint,
                        prediction_iterations = pred_iterations,
                        status = "ok",
                        nll = metrics.nll,
                        accuracy = metrics.accuracy,
                        brier = metrics.brier,
                        mean_latent_variance = metrics.mean_latent_variance,
                        error = missing,
                    )))
                end
            else
                push!(refinement_rows, merge(basis, hyper, (;
                    seed_index,
                    checkpoint = max_checkpoint,
                    prediction_iterations = maximum(prediction_iterations),
                    status = "failed",
                    nll = Inf,
                    accuracy = NaN,
                    brier = NaN,
                    mean_latent_variance = NaN,
                    error = fitted.error,
                )))
            end
        end
        refinement = DataFrame(refinement_rows)
        successful_refinement = filter(:status => ==("ok"), refinement)
        nrow(successful_refinement) > 0 || error("every refinement fit failed")
        refinement_summary = combine(
            groupby(successful_refinement, [
                :name,
                :map_parameters,
                :pred_parameters,
                :basis_seed,
                :score_precision_mean,
                :score_precision_concentration,
                :atom_scale_multiplier,
                :checkpoint,
                :prediction_iterations,
            ]),
            :nll => mean => :mean_nll,
            :accuracy => mean => :mean_accuracy,
            :brier => mean => :mean_brier,
            :mean_latent_variance => mean => :mean_latent_variance,
        )
        sort!(refinement_summary, [:mean_nll, :mean_brier])
        winner = NamedTuple(first(eachrow(refinement_summary)))
        println(
            "winner: ", winner.name,
            ", map/pred=", winner.map_parameters, "/", winner.pred_parameters,
            ", basis seed=", winner.basis_seed,
            ", score mean=", winner.score_precision_mean,
            ", scale=", winner.atom_scale_multiplier,
            ", checkpoint=", winner.checkpoint,
            ", prediction iterations=", winner.prediction_iterations,
            ", mean NLL=", round(winner.mean_nll, digits = 5),
        )

        winner_basis = (
            name = winner.name,
            map_parameters = winner.map_parameters,
            pred_parameters = winner.pred_parameters,
            basis_seed = winner.basis_seed,
        )
        winner_hyper = (
            name = "winner",
            score_precision_mean = winner.score_precision_mean,
            score_precision_concentration = winner.score_precision_concentration,
            atom_scale_multiplier = winner.atom_scale_multiplier,
        )

        if save_outputs
            mkpath(dirname(LOW_RANK_TUNING_OUTPUT_PREFIX))
            CSV.write(LOW_RANK_TUNING_OUTPUT_PREFIX * "_screen.csv", screen)
            CSV.write(LOW_RANK_TUNING_OUTPUT_PREFIX * "_refinement.csv", refinement)
            CSV.write(
                LOW_RANK_TUNING_OUTPUT_PREFIX * "_refinement_summary.csv",
                refinement_summary,
            )
        end

        println("\n=== frozen winner on unseen clean datasets")
        final_rows = NamedTuple[]
        for (seed_index, seeds) in enumerate(final_seeds)
            config = low_rank_candidate_config(
                CONFIG,
                seeds;
                n_samples = final_samples,
                iterations = winner.checkpoint,
                prediction_iterations = winner.prediction_iterations,
                hyper = winner_hyper,
            )
            output_prefix = LOW_RANK_TUNING_OUTPUT_PREFIX * "_final_seed_$(seed_index)"
            final_run = final_low_rank_evaluation(
                config,
                winner_basis,
                winner_hyper,
                output_prefix;
                save_surface = save_outputs,
            )
            push!(final_rows, (;
                seed_index,
                accuracy = final_run.metrics.accuracy,
                nll = final_run.metrics.nll,
                brier = final_run.brier,
                baseline_accuracy = final_run.baseline.accuracy,
                baseline_nll = final_run.baseline.nll,
                mean_latent_variance = mean(final_run.statistics.score_variance),
                training_elapsed = final_run.run.fit.elapsed,
                prediction_elapsed = final_run.prediction_elapsed,
                surface_path = final_run.surface_path,
            ))
            println(
                "seed ", seed_index,
                ": accuracy/NLL/Brier = ",
                round(final_run.metrics.accuracy, digits = 4), " / ",
                round(final_run.metrics.nll, digits = 5), " / ",
                round(final_run.brier, digits = 5),
            )
        end
        final = DataFrame(final_rows)

        if save_outputs
            CSV.write(LOW_RANK_TUNING_OUTPUT_PREFIX * "_final.csv", final)
        end
    end

    if save_outputs
        write_low_rank_tuning_summary(
            LOW_RANK_TUNING_OUTPUT_PREFIX * "_summary.md",
            winner,
            final,
            total_elapsed,
        )
    end
    return (; screen, refinement, refinement_summary, winner, final, elapsed = total_elapsed)
end

if abspath(PROGRAM_FILE) == @__FILE__
    low_rank_tuning_report = run_low_rank_focused_tuning()
end
