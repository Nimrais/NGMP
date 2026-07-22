# Clean-label-only focused tuning for xor_ctransition_mvsoftplus_probit.jl.
#
# The search is intentionally narrow: width, Unscented MvSoftplus projection,
# promoted CT/MvSoftplus priors, and both damping policies remain fixed.  Only
# the Gamma prior on the softdot score precision is selected.  No corrupted
# labels are used until the winner has been frozen.
#
# Full search and two unseen paired confirmations:
#   OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_ctransition_mvsoftplus_probit_tuning.jl
#
# Reduced graph smoke (same control flow, smaller fits):
#   XOR_CT_PROBIT_TUNING_SMOKE=true OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_ctransition_mvsoftplus_probit_tuning.jl

const XOR_CT_PROBIT_TUNING_SMOKE =
    lowercase(get(ENV, "XOR_CT_PROBIT_TUNING_SMOKE", "false")) in
    ("1", "true", "yes", "on")

ENV["XOR_CT_PROBIT_SMOKE"] = string(XOR_CT_PROBIT_TUNING_SMOKE)
ENV["SAVE_OUTPUTS"] = "false"
ENV["SHOW_PROGRESS"] = "false"
ENV["REQUIRE_CLEAN_BASELINE"] = "false"

include(joinpath(@__DIR__, "xor_ctransition_mvsoftplus_probit.jl"))

const SCORE_PRECISION_CANDIDATES = [
    (
        name = "mean_$(mean_value)_concentration_$(concentration)",
        score_precision_mean = Float64(mean_value),
        score_precision_concentration = Float64(concentration),
    ) for mean_value in (10, 100, 300, 1_000), concentration in (10, 100)
] |> vec

const CLEAN_TUNING_SEEDS = (
    (data_seed = 2_026, split_seed = 2_027, prior_seed = 42),
    (data_seed = 2_028, split_seed = 2_029, prior_seed = 43),
)

const UNSEEN_FINAL_SEEDS = (
    (data_seed = 2_030, split_seed = 2_031, prior_seed = 44, flip_seed = 9_031),
    (data_seed = 2_032, split_seed = 2_033, prior_seed = 45, flip_seed = 9_033),
)

const TUNING_OUTPUT_PREFIX = get(
    ENV,
    "TUNING_OUTPUT_PREFIX",
    joinpath(@__DIR__, "..", "viz", "xor_ctransition_mvsoftplus_probit_tuning"),
)
const TUNING_SAVE_OUTPUTS = env_bool(
    "TUNING_SAVE_OUTPUTS",
    !XOR_CT_PROBIT_TUNING_SMOKE,
)

candidate_config(base, candidate) = merge(
    base,
    (
        score_precision_mean = candidate.score_precision_mean,
        score_precision_concentration = candidate.score_precision_concentration,
        d_hidden = 4,
        save_outputs = false,
        show_progress = false,
        require_clean_baseline = false,
    ),
)

function clean_dataset_fit(config; iterations)
    dataset = make_clean_xor_dataset(n = config.n_samples, seed = config.data_seed)
    train_data, test_data = split_dataset(
        dataset;
        train_fraction = config.train_fraction,
        seed = config.split_seed,
    )
    train_features = build_features(train_data)
    test_features = build_features(test_data)
    fit = fit_probit_arm(
        config,
        train_features,
        Float64.(train_data.label);
        iterations = iterations,
    )
    return (
        fit = fit,
        train_data = train_data,
        test_data = test_data,
        train_features = train_features,
        test_features = test_features,
    )
end

function clean_predictive_metrics(run, config; checkpoint, prediction_iterations)
    prediction_config = merge(
        config,
        (
            prediction_iterations = prediction_iterations,
            show_progress = false,
        ),
    )
    statistics = score_statistics(predict_score_marginals(
        run.fit,
        run.test_features;
        config = prediction_config,
        iteration = checkpoint,
    ); verify_native = false)
    metrics = classification_metrics(
        Float64.(run.test_data.label),
        statistics.probability,
    )
    return merge(metrics, (mean_latent_variance = mean(statistics.score_variance),))
end

function safe_clean_fit(config; iterations)
    try
        return (status = "ok", run = clean_dataset_fit(config; iterations = iterations))
    catch exception
        return (
            status = "failed",
            error = sprint(showerror, exception, catch_backtrace()),
        )
    end
end

function write_tuning_summary(path, winner, screen, refinement, final_rows, elapsed)
    open(path, "w") do io
        println(io, "# Width-4 CT/MvSoftplus Probit focused tuning")
        println(io)
        println(io, "All hyperparameter selection used clean XOR labels only.")
        println(io)
        println(io, "- Score-precision mean: `$(winner.score_precision_mean)`")
        println(
            io,
            "- Score-precision concentration: `$(winner.score_precision_concentration)`",
        )
        println(io, "- Training checkpoint: `$(winner.checkpoint)`")
        println(io, "- Prediction iterations: `$(winner.prediction_iterations)`")
        println(io, "- Mean clean refinement NLL: `$(winner.mean_nll)`")
        println(io, "- Total wall time: `$(round(elapsed, digits = 1))` seconds")
        println(io)
        println(io, "The frozen winner was then applied unchanged to both paired arms on " *
                    "$(length(unique(final_rows.seed_index))) unseen dataset seed(s).")
        println(io, "Variance localization is reported descriptively and is not an acceptance test.")
        println(io)
        println(io, "Screening rows: $(nrow(screen)); refinement rows: $(nrow(refinement)); " *
                    "final rows: $(nrow(final_rows)).")
    end
end

function run_focused_tuning()
    smoke = XOR_CT_PROBIT_TUNING_SMOKE
    screen_samples = smoke ? 24 : 400
    screen_iterations = smoke ? 1 : 80
    screen_prediction_iterations = smoke ? 1 : 3
    refine_samples = smoke ? 24 : 400
    checkpoints = smoke ? (1,) : (80, 120, 160)
    prediction_iterations = smoke ? (1,) : (3, 5, 10)
    tuning_seeds = smoke ? CLEAN_TUNING_SEEDS[1:1] : CLEAN_TUNING_SEEDS
    final_samples = smoke ? 24 : 2_000
    final_seeds = smoke ? UNSEEN_FINAL_SEEDS[1:1] : UNSEEN_FINAL_SEEDS

    total_elapsed = @elapsed begin
        println("=== clean-label score-precision screening")
        screen_rows = NamedTuple[]
        for candidate in SCORE_PRECISION_CANDIDATES
            config = merge(
                candidate_config(CONFIG, candidate),
                first(CLEAN_TUNING_SEEDS),
                (
                    n_samples = screen_samples,
                    iterations = screen_iterations,
                    prediction_iterations = screen_prediction_iterations,
                ),
            )
            fitted = safe_clean_fit(config; iterations = screen_iterations)
            if fitted.status == "ok"
                metrics = clean_predictive_metrics(
                    fitted.run,
                    config;
                    checkpoint = screen_iterations,
                    prediction_iterations = screen_prediction_iterations,
                )
                push!(screen_rows, (
                    candidate = candidate.name,
                    score_precision_mean = candidate.score_precision_mean,
                    score_precision_concentration = candidate.score_precision_concentration,
                    status = "ok",
                    nll = metrics.nll,
                    accuracy = metrics.accuracy,
                    mean_latent_variance = metrics.mean_latent_variance,
                    error = missing,
                ))
                println(candidate.name, ": NLL=", round(metrics.nll, digits = 5))
            else
                push!(screen_rows, (
                    candidate = candidate.name,
                    score_precision_mean = candidate.score_precision_mean,
                    score_precision_concentration = candidate.score_precision_concentration,
                    status = "failed",
                    nll = Inf,
                    accuracy = NaN,
                    mean_latent_variance = NaN,
                    error = fitted.error,
                ))
                println(candidate.name, ": failed")
            end
        end
        screen = sort(DataFrame(screen_rows), :nll)
        successful_screen = filter(:status => ==("ok"), screen)
        nrow(successful_screen) >= 3 || error(
            "fewer than three score-precision candidates completed screening",
        )
        selected_names = successful_screen.candidate[1:3]
        selected = [
            only(filter(candidate -> candidate.name == name, SCORE_PRECISION_CANDIDATES))
            for name in selected_names
        ]

        println("\n=== two-seed clean checkpoint/prediction refinement")
        refinement_rows = NamedTuple[]
        max_checkpoint = maximum(checkpoints)
        for candidate in selected, (seed_index, seeds) in enumerate(tuning_seeds)
            config = merge(
                candidate_config(CONFIG, candidate),
                seeds,
                (n_samples = refine_samples, iterations = max_checkpoint),
            )
            run = clean_dataset_fit(config; iterations = max_checkpoint)
            for checkpoint in checkpoints, pred_iterations in prediction_iterations
                metrics = clean_predictive_metrics(
                    run,
                    config;
                    checkpoint = checkpoint,
                    prediction_iterations = pred_iterations,
                )
                push!(refinement_rows, (
                    candidate = candidate.name,
                    score_precision_mean = candidate.score_precision_mean,
                    score_precision_concentration = candidate.score_precision_concentration,
                    seed_index = seed_index,
                    checkpoint = checkpoint,
                    prediction_iterations = pred_iterations,
                    nll = metrics.nll,
                    accuracy = metrics.accuracy,
                    mean_latent_variance = metrics.mean_latent_variance,
                ))
            end
        end
        refinement = DataFrame(refinement_rows)
        summary = combine(
            groupby(
                refinement,
                [
                    :candidate,
                    :score_precision_mean,
                    :score_precision_concentration,
                    :checkpoint,
                    :prediction_iterations,
                ],
            ),
            :nll => mean => :mean_nll,
            :accuracy => mean => :mean_accuracy,
            :mean_latent_variance => mean => :mean_latent_variance,
        )
        sort!(summary, [:mean_nll, :checkpoint, :prediction_iterations])
        winner = NamedTuple(first(eachrow(summary)))
        println(
            "winner: mean=", winner.score_precision_mean,
            ", concentration=", winner.score_precision_concentration,
            ", checkpoint=", winner.checkpoint,
            ", prediction iterations=", winner.prediction_iterations,
            ", clean NLL=", round(winner.mean_nll, digits = 5),
        )

        println("\n=== frozen winner on unseen paired datasets")
        final_rows = NamedTuple[]
        for (seed_index, seeds) in enumerate(final_seeds)
            final_prefix = TUNING_OUTPUT_PREFIX * "_final_seed_$(seed_index)"
            final_config = merge(
                CONFIG,
                seeds,
                (
                    n_samples = final_samples,
                    iterations = winner.checkpoint,
                    prediction_iterations = winner.prediction_iterations,
                    score_precision_mean = winner.score_precision_mean,
                    score_precision_concentration = winner.score_precision_concentration,
                    save_outputs = TUNING_SAVE_OUTPUTS,
                    output_prefix = final_prefix,
                    show_progress = false,
                    require_clean_baseline = !smoke,
                ),
            )
            report = run_paired_experiment(final_config)
            for arm in (:clean, :corrupted)
                diagnostics = getproperty(report.arms, arm).diagnostics
                overall = only(eachrow(diagnostics[diagnostics.quadrant .== "overall", :]))
                localization = getproperty(report.arms, arm).localization
                push!(final_rows, (
                    seed_index = seed_index,
                    arm = string(arm),
                    accuracy = overall.accuracy,
                    nll = overall.nll,
                    untrusted_latent_variance = localization.untrusted_latent_variance,
                    reliable_latent_variance = localization.reliable_latent_variance,
                    latent_variance_ratio = localization.latent_variance_ratio,
                    latent_variance_difference = localization.latent_variance_difference,
                    difference_in_differences =
                        report.localization.latent_variance_difference_in_differences,
                ))
            end
        end
        final = DataFrame(final_rows)

        if TUNING_SAVE_OUTPUTS
            mkpath(dirname(TUNING_OUTPUT_PREFIX))
            CSV.write(TUNING_OUTPUT_PREFIX * "_screen.csv", screen)
            CSV.write(TUNING_OUTPUT_PREFIX * "_refinement.csv", refinement)
            CSV.write(TUNING_OUTPUT_PREFIX * "_refinement_summary.csv", summary)
            CSV.write(TUNING_OUTPUT_PREFIX * "_final.csv", final)
        end
    end

    if TUNING_SAVE_OUTPUTS
        write_tuning_summary(
            TUNING_OUTPUT_PREFIX * "_summary.md",
            winner,
            screen,
            refinement,
            final,
            total_elapsed,
        )
    end
    return (
        screen = screen,
        refinement = refinement,
        refinement_summary = summary,
        winner = winner,
        final = final,
        elapsed = total_elapsed,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    tuning_report = run_focused_tuning()
end
