# Fast, reproducible width-4 tuning for xor_ctransition_mvsoftplus.jl.
#
# Full quick search (roughly 45-65 minutes on the development machine):
#   OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_ctransition_mvsoftplus_tuning.jl
#
# Reduced end-to-end graph smoke:
#   XOR_CT_TUNING_SMOKE=true OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_ctransition_mvsoftplus_tuning.jl

const XOR_CT_TUNING_SMOKE =
    lowercase(get(ENV, "XOR_CT_TUNING_SMOKE", "false")) in ("1", "true", "yes", "on")
ENV["XOR_CT_SMOKE"] = string(XOR_CT_TUNING_SMOKE)
ENV["SAVE_OUTPUTS"] = "false"
ENV["SHOW_PROGRESS"] = "false"

include(joinpath(@__DIR__, "xor_ctransition_mvsoftplus.jl"))

using CSV
using LinearAlgebra: norm

const TUNING_SEED = env_int("TUNING_SEED", 7_301)
const TUNING_SAVE_OUTPUTS = env_bool("TUNING_SAVE_OUTPUTS", !XOR_CT_TUNING_SMOKE)
const TUNING_OUTPUT_PREFIX = get(
    ENV,
    "TUNING_OUTPUT_PREFIX",
    joinpath(@__DIR__, "..", "viz", "xor_ctransition_mvsoftplus_tuning"),
)

const LEGACY_SETTINGS = (
    ct_precision_mean = 10.0,
    a_prior_mean_scale = 0.5,
    a_prior_variance = 0.1,
    theta_prior_mean_scale = 0.0,
    theta_prior_variance = 1.0,
    gamma_obs_mean = 1e6,
    gamma_obs_concentration = 1e6,
    ngmp_alpha = 0.2,
    ngmp_beta = 0.0,
    ngmp_max_step = 1.0,
    ct_a_alpha = 0.3,
    ct_a_beta = 0.3,
    ct_a_max_step = Inf,
)

const PRIOR_LEVELS = (
    ct_precision_mean = (10.0, 30.0, 100.0, 300.0),
    a_prior_mean_scale = (0.25, 0.5, 1.0),
    a_prior_variance = (0.1, 0.3, 1.0),
    theta_prior_mean_scale = (0.0, 0.1, 0.3),
    theta_prior_variance = (0.3, 1.0, 3.0),
    gamma_obs_mean = (30.0, 100.0, 300.0, 1_000.0),
    gamma_obs_concentration = (1.0, 10.0, 100.0),
)

const DAMPING_POLICIES = [
    (
        name = "ct02_00_sp02_00",
        ct_a_alpha = 0.2,
        ct_a_beta = 0.0,
        ngmp_alpha = 0.2,
        ngmp_beta = 0.0,
    ),
    (
        name = "ct02_00_sp04_02",
        ct_a_alpha = 0.2,
        ct_a_beta = 0.0,
        ngmp_alpha = 0.4,
        ngmp_beta = 0.2,
    ),
    (
        name = "ct03_03_sp02_00",
        ct_a_alpha = 0.3,
        ct_a_beta = 0.3,
        ngmp_alpha = 0.2,
        ngmp_beta = 0.0,
    ),
    (
        name = "ct03_03_sp04_02",
        ct_a_alpha = 0.3,
        ct_a_beta = 0.3,
        ngmp_alpha = 0.4,
        ngmp_beta = 0.2,
    ),
    (
        name = "ct05_02_sp02_00",
        ct_a_alpha = 0.5,
        ct_a_beta = 0.2,
        ngmp_alpha = 0.2,
        ngmp_beta = 0.0,
    ),
    (
        name = "ct05_02_sp04_02",
        ct_a_alpha = 0.5,
        ct_a_beta = 0.2,
        ngmp_alpha = 0.4,
        ngmp_beta = 0.2,
    ),
]

const TUNING_SEEDS = (
    (data_seed = 2_026, split_seed = 2_027, prior_seed = 42),
    (data_seed = 2_028, split_seed = 2_029, prior_seed = 43),
)

const HELDOUT_SEEDS = (
    (data_seed = 2_030, split_seed = 2_031, prior_seed = 44),
    (data_seed = 2_032, split_seed = 2_033, prior_seed = 45),
    (data_seed = 2_034, split_seed = 2_035, prior_seed = 46),
)

candidate_key(candidate) = Tuple(values(candidate))

function balanced_column(rng, levels, count)
    values = repeat(collect(levels), outer = ceil(Int, count / length(levels)))[1:count]
    return values[randperm(rng, count)]
end

function prior_candidates(count = 24; seed = TUNING_SEED)
    count >= 2 || throw(ArgumentError("prior candidate count must be at least two"))
    legacy = merge((name = "legacy",), NamedTuple{keys(PRIOR_LEVELS)}(LEGACY_SETTINGS))
    anchor = (
        name = "noise_matched_anchor",
        ct_precision_mean = 100.0,
        a_prior_mean_scale = 0.5,
        a_prior_variance = 1.0,
        theta_prior_mean_scale = 0.1,
        theta_prior_variance = 1.0,
        gamma_obs_mean = 100.0,
        gamma_obs_concentration = 10.0,
    )
    count == 2 && return [legacy, anchor]

    rng = StableRNG(seed)
    remaining = count - 2
    columns = map(levels -> balanced_column(rng, levels, remaining), values(PRIOR_LEVELS))
    candidates = [legacy, anchor]
    seen = Set((candidate_key(Base.structdiff(candidate, NamedTuple{(:name,)})) for candidate in candidates))
    for index in 1:remaining
        values_at_index = ntuple(column -> columns[column][index], length(columns))
        settings = NamedTuple{keys(PRIOR_LEVELS)}(values_at_index)
        while candidate_key(settings) in seen
            settings = NamedTuple{keys(PRIOR_LEVELS)}(
                map(levels -> rand(rng, levels), values(PRIOR_LEVELS)),
            )
        end
        push!(seen, candidate_key(settings))
        push!(candidates, merge((name = "prior_$(lpad(index, 2, '0'))",), settings))
    end
    return candidates
end

function candidate_config(base, prior, damping = nothing)
    prior_settings = Base.structdiff(prior, NamedTuple{(:name,)})
    damping_settings = isnothing(damping) ? NamedTuple() : (
        ct_a_alpha = damping.ct_a_alpha,
        ct_a_beta = damping.ct_a_beta,
        ngmp_alpha = damping.ngmp_alpha,
        ngmp_beta = damping.ngmp_beta,
    )
    return merge(
        base,
        prior_settings,
        damping_settings,
        (
            d_hidden = 4,
            ct_a_max_step = Inf,
            ngmp_max_step = 1.0,
            mvsoftplus_projection = "unscented",
            feature_jitter = 1e-4,
            save_outputs = false,
            diagnostics = false,
            show_progress = false,
        ),
    )
end

function legacy_config(base)
    return merge(
        base,
        LEGACY_SETTINGS,
        (
            d_hidden = 4,
            mvsoftplus_projection = "unscented",
            feature_jitter = 1e-4,
            save_outputs = false,
            diagnostics = false,
            show_progress = false,
        ),
    )
end

function seeded_config(base, seeds; n_samples, iterations)
    return merge(
        base,
        seeds,
        (
            n_samples = n_samples,
            iterations = iterations,
            train_fraction = 0.8,
            d_hidden = 4,
            save_outputs = false,
            diagnostics = false,
            show_progress = false,
        ),
    )
end

function global_priors(config, d_f = 3)
    return make_priors(
        d_h = 4,
        d_f = d_f,
        seed = config.prior_seed,
        ct_precision_mean = config.ct_precision_mean,
        a_prior_mean_scale = config.a_prior_mean_scale,
        a_prior_variance = config.a_prior_variance,
        theta_prior_mean_scale = config.theta_prior_mean_scale,
        theta_prior_variance = config.theta_prior_variance,
        gamma_obs_mean = config.gamma_obs_mean,
        gamma_obs_concentration = config.gamma_obs_concentration,
    )
end

function plugin_prediction_at(result, features, iteration, d_h = 4, d_f = 3)
    A_map = reshape(mean(result.posteriors[:a_map][iteration]), d_h, d_f)
    A_pred = reshape(mean(result.posteriors[:a_pred][iteration]), d_h, d_h)
    theta = mean(result.posteriors[:theta][iteration])
    return [dot(theta, A_pred * _softplus.(A_map * feature)) for feature in features]
end

function fit_candidate(config)
    d_h, d_f = 4, 3
    dataset = make_checkerboard_dataset(
        n = config.n_samples,
        noise_std = config.noise_std,
        seed = config.data_seed,
    )
    train_data, test_data = split_dataset(
        dataset;
        train_fraction = config.train_fraction,
        seed = config.split_seed,
    )
    train_features = build_features(train_data)
    test_features = build_features(test_data)
    priors = global_priors(config, d_f)
    ct_a_deps = NGMPDependencies(
        a = nothing,
        damping = DampingMeta(
            alpha = config.ct_a_alpha,
            beta = config.ct_a_beta,
            max_step = config.ct_a_max_step,
        ),
    )
    ct2_deps = make_ct2_dependencies(config)
    sp_deps = make_mvsoftplus_dependencies(config)
    sp_damping = DampingMeta(
        alpha = config.ngmp_alpha,
        beta = config.ngmp_beta,
        max_step = config.ngmp_max_step,
    )

    timed = @timed infer(
        model = xor_ct_mvsoftplus(
            priors = priors,
            feature_cov = Matrix(Diagonal(fill(config.feature_jitter, d_f))),
            meta_map = LinearReshapeMeta(d_h, d_f),
            meta_pred = LinearReshapeMeta(d_h, d_h),
            ct_a_deps = ct_a_deps,
            ct2_deps = ct2_deps,
            sp_deps = sp_deps,
            sp_damping = sp_damping,
        ),
        data = (y = train_data.OT, features = train_features),
        constraints = xor_ct_constraints(),
        initialization = make_initialization(priors, d_h, make_s_initial(config, d_h)),
        iterations = config.iterations,
        free_energy = true,
        showprogress = false,
        returnvars = (
            a_map = KeepEach(),
            a_pred = KeepEach(),
            theta = KeepEach(),
            P = KeepEach(),
            Gamma2 = KeepEach(),
            gamma_obs = KeepEach(),
        ),
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )

    result = timed.value
    all(isfinite, result.free_energy) || error("training produced non-finite free energy")
    expected_states = 2 * nrow(train_data)
    state_count = length(ct_a_deps.states) + length(ct2_deps.states)
    state_count == expected_states || error(
        "created $state_count CT states; expected $expected_states",
    )
    firings = vcat(
        getproperty.(ct_a_deps.states, :nfired),
        getproperty.(ct2_deps.states, :nfired),
    )
    all(==(config.iterations), firings) || error("CT sites did not fire once per iteration")

    prediction = plugin_prediction_at(result, test_features, config.iterations, d_h, d_f)
    all(isfinite, prediction) || error("plug-in prediction was non-finite")
    baseline = mean(abs2, mean(train_data.OT) .- test_data.OT)
    plugin_mse = mean(abs2, prediction .- test_data.OT)

    return (
        config = config,
        result = result,
        priors = priors,
        train_data = train_data,
        test_data = test_data,
        train_features = train_features,
        test_features = test_features,
        baseline_mse = baseline,
        plugin_mse = plugin_mse,
        normalized_plugin_mse = plugin_mse / baseline,
        seconds = timed.time,
        allocated_gib = timed.bytes / 2.0^30,
        free_energy_first = first(result.free_energy),
        free_energy_last = last(result.free_energy),
        free_energy_decreases = count(<(0), diff(result.free_energy)),
        ct_state_count = length(firings),
        ct_minimum_firings = minimum(firings),
        ct_maximum_firings = maximum(firings),
    )
end

function safe_fit(config)
    try
        return (status = "ok", fit = fit_candidate(config), error = missing)
    catch exception
        return (
            status = "failed",
            fit = nothing,
            error = sprint(showerror, exception, catch_backtrace()),
        )
    end
end

function plugin_checkpoint_metrics(fit, checkpoints)
    return [
        begin
            prediction = plugin_prediction_at(
                fit.result,
                fit.test_features,
                checkpoint,
            )
            mse = mean(abs2, prediction .- fit.test_data.OT)
            (
                checkpoint = checkpoint,
                plugin_mse = mse,
                normalized_plugin_mse = mse / fit.baseline_mse,
            )
        end for checkpoint in checkpoints
    ]
end

function predictive_test_metrics(fit, checkpoint, prediction_iterations)
    prediction_config = merge(
        fit.config,
        (prediction_iterations = prediction_iterations, show_progress = false),
    )
    marginals = predict_marginals(
        prediction_priors(fit.result, checkpoint),
        fit.test_features;
        batch_size = prediction_config.prediction_batch_size,
        config = prediction_config,
        d_h = 4,
        d_f = 3,
        output_mean = mean(fit.train_data.OT),
    )
    prediction = predictive_statistics(marginals)
    mse = mean(abs2, prediction.mean .- fit.test_data.OT)
    return (
        test_mse = mse,
        normalized_test_mse = mse / fit.baseline_mse,
        predictive_variance_mean = mean(prediction.variance),
    )
end

function clean_grid_metrics(fit, checkpoint, prediction_iterations, grid_size)
    prediction_config = merge(
        fit.config,
        (
            prediction_iterations = prediction_iterations,
            grid_size = grid_size,
            show_progress = false,
        ),
    )
    grid = make_prediction_grid(grid_size)
    marginals = predict_marginals(
        prediction_priors(fit.result, checkpoint),
        grid.features;
        batch_size = prediction_config.prediction_batch_size,
        config = prediction_config,
        d_h = 4,
        d_f = 3,
        output_mean = mean(fit.train_data.OT),
    )
    prediction = predictive_statistics(marginals)
    return (
        clean_grid_mse = mean(abs2, prediction.mean .- vec(grid.actual)),
        predictive_variance_mean = mean(prediction.variance),
    )
end

function prior_row(candidate, run)
    common = (
        candidate = candidate.name,
        ct_precision_mean = candidate.ct_precision_mean,
        a_prior_mean_scale = candidate.a_prior_mean_scale,
        a_prior_variance = candidate.a_prior_variance,
        theta_prior_mean_scale = candidate.theta_prior_mean_scale,
        theta_prior_variance = candidate.theta_prior_variance,
        gamma_obs_mean = candidate.gamma_obs_mean,
        gamma_obs_concentration = candidate.gamma_obs_concentration,
        status = run.status,
    )
    return run.status == "ok" ? merge(
        common,
        (
            plugin_mse = run.fit.plugin_mse,
            normalized_plugin_mse = run.fit.normalized_plugin_mse,
            baseline_mse = run.fit.baseline_mse,
            seconds = run.fit.seconds,
            free_energy_last = run.fit.free_energy_last,
            error = missing,
        ),
    ) : merge(
        common,
        (
            plugin_mse = NaN,
            normalized_plugin_mse = Inf,
            baseline_mse = NaN,
            seconds = NaN,
            free_energy_last = NaN,
            error = run.error,
        ),
    )
end

function fit_row(name, prior, damping, run)
    common = merge(
        (
            candidate = name,
            prior_candidate = prior.name,
            damping_policy = damping.name,
            ct_a_alpha = damping.ct_a_alpha,
            ct_a_beta = damping.ct_a_beta,
            ngmp_alpha = damping.ngmp_alpha,
            ngmp_beta = damping.ngmp_beta,
        ),
        Base.structdiff(prior, NamedTuple{(:name,)}),
        (status = run.status,),
    )
    return run.status == "ok" ? merge(
        common,
        (
            plugin_mse = run.fit.plugin_mse,
            normalized_plugin_mse = run.fit.normalized_plugin_mse,
            baseline_mse = run.fit.baseline_mse,
            seconds = run.fit.seconds,
            allocated_gib = run.fit.allocated_gib,
            free_energy_first = run.fit.free_energy_first,
            free_energy_last = run.fit.free_energy_last,
            error = missing,
        ),
    ) : merge(
        common,
        (
            plugin_mse = NaN,
            normalized_plugin_mse = Inf,
            baseline_mse = NaN,
            seconds = NaN,
            allocated_gib = NaN,
            free_energy_first = NaN,
            free_energy_last = NaN,
            error = run.error,
        ),
    )
end

function save_tuning_plot(heldout, output_prefix)
    successful = filter(:status => ==("ok"), heldout)
    isempty(successful) && return nothing
    grouped = combine(
        groupby(successful, :arm),
        :test_mse => mean => :test_mse,
        :clean_grid_mse => mean => :clean_grid_mse,
    )
    figure = groupedbar(
        string.(grouped.arm),
        Matrix(select(grouped, :test_mse, :clean_grid_mse));
        bar_position = :dodge,
        label = ["noisy test" "clean grid"],
        ylabel = "MSE",
        title = "Width-4 shallow CT held-out comparison",
    )
    path = output_prefix * "_heldout_mse.png"
    mkpath(dirname(path))
    savefig(figure, path)
    return path
end

function write_summary(path, winner, validation, heldout, accepted, elapsed)
    winner_rows = filter(:arm => ==("winner"), heldout)
    legacy_rows = filter(:arm => ==("legacy"), heldout)
    open(path, "w") do io
        println(io, "# Width-4 ContinuousTransition / MvSoftplus tuning")
        println(io)
        println(io, "- Accepted for default promotion: `$(accepted)`")
        println(io, "- Total wall time: $(round(elapsed, digits = 1)) seconds")
        println(io, "- Training iterations: $(winner.training_iterations)")
        println(io, "- Prediction iterations: $(winner.prediction_iterations)")
        println(io, "- Mean winner test MSE: $(mean(winner_rows.test_mse))")
        println(io, "- Mean legacy test MSE: $(mean(legacy_rows.test_mse))")
        println(io, "- Mean winner clean-grid MSE: $(mean(winner_rows.clean_grid_mse))")
        println(io, "- Mean legacy clean-grid MSE: $(mean(legacy_rows.clean_grid_mse))")
        println(io)
        println(io, "## Winning settings")
        println(io)
        for field in keys(winner.config)
            println(io, "- `$(field)=$(getproperty(winner.config, field))`")
        end
        println(io)
        println(io, "Validation rows: $(nrow(validation)); held-out rows: $(nrow(heldout)).")
    end
end

function run_tuning()
    total_timed = @timed begin
        prior_count = XOR_CT_TUNING_SMOKE ? 4 : 24
        stage_a_samples = XOR_CT_TUNING_SMOKE ? 60 : 240
        stage_a_iterations = XOR_CT_TUNING_SMOKE ? 2 : 40
        stage_b_samples = XOR_CT_TUNING_SMOKE ? 60 : 400
        stage_b_iterations = XOR_CT_TUNING_SMOKE ? 2 : 80
        stage_c_samples = XOR_CT_TUNING_SMOKE ? 60 : 400
        stage_c_iterations = XOR_CT_TUNING_SMOKE ? 3 : 160
        checkpoints = XOR_CT_TUNING_SMOKE ? (2, 3) : (80, 120, 160)
        prediction_iterations = XOR_CT_TUNING_SMOKE ? (1,) : (1, 3, 5, 10)
        heldout_samples = XOR_CT_TUNING_SMOKE ? 60 : 2_000
        heldout_grid_size = XOR_CT_TUNING_SMOKE ? 8 : 40
        tuning_seeds = XOR_CT_TUNING_SMOKE ? TUNING_SEEDS[1:1] : TUNING_SEEDS
        heldout_seeds = XOR_CT_TUNING_SMOKE ? HELDOUT_SEEDS[1:1] : HELDOUT_SEEDS
        prior_keep = XOR_CT_TUNING_SMOKE ? 2 : 3
        refine_keep = XOR_CT_TUNING_SMOKE ? 2 : 4
        damping_policies = XOR_CT_TUNING_SMOKE ? DAMPING_POLICIES[1:2] : DAMPING_POLICIES

        println("=== stage A: prior screening")
        stage_a_runs = Dict{String, Any}()
        stage_a_rows = NamedTuple[]
        for candidate in prior_candidates(prior_count)
            config = seeded_config(
                candidate_config(CONFIG, candidate),
                first(tuning_seeds);
                n_samples = stage_a_samples,
                iterations = stage_a_iterations,
            )
            run = safe_fit(config)
            stage_a_runs[candidate.name] = (candidate = candidate, run = run)
            push!(stage_a_rows, prior_row(candidate, run))
            score = run.status == "ok" ? run.fit.normalized_plugin_mse : Inf
            println("$(candidate.name): normalized plugin MSE=$(round(score, digits = 4))")
        end
        stage_a = sort(DataFrame(stage_a_rows), :normalized_plugin_mse)
        successful_a = filter(:status => ==("ok"), stage_a)
        nrow(successful_a) >= prior_keep || error("too few successful prior candidates")
        selected_prior_names = successful_a.candidate[1:prior_keep]
        selected_priors = [stage_a_runs[name].candidate for name in selected_prior_names]

        println("\n=== stage B: damping refinement")
        stage_b_runs = Dict{String, Any}()
        stage_b_rows = NamedTuple[]
        for prior in selected_priors, damping in damping_policies
            name = "$(prior.name)__$(damping.name)"
            config = seeded_config(
                candidate_config(CONFIG, prior, damping),
                first(tuning_seeds);
                n_samples = stage_b_samples,
                iterations = stage_b_iterations,
            )
            run = safe_fit(config)
            stage_b_runs[name] = (
                name = name,
                prior = prior,
                damping = damping,
                run = run,
            )
            push!(stage_b_rows, fit_row(name, prior, damping, run))
            score = run.status == "ok" ? run.fit.normalized_plugin_mse : Inf
            println("$name: normalized plugin MSE=$(round(score, digits = 4))")
        end
        stage_b = sort(DataFrame(stage_b_rows), :normalized_plugin_mse)
        successful_b = filter(:status => ==("ok"), stage_b)
        nrow(successful_b) >= refine_keep || error("too few successful damping candidates")
        finalist_names = successful_b.candidate[1:refine_keep]
        finalists = [stage_b_runs[name] for name in finalist_names]

        println("\n=== stage C: checkpoint and prediction refinement")
        validation_rows = NamedTuple[]
        finalist_summaries = NamedTuple[]
        for finalist in finalists
            fits = Any[]
            for seeds in tuning_seeds
                config = seeded_config(
                    candidate_config(CONFIG, finalist.prior, finalist.damping),
                    seeds;
                    n_samples = stage_c_samples,
                    iterations = stage_c_iterations,
                )
                run = safe_fit(config)
                run.status == "ok" || error(
                    "validation fit failed for $(finalist.name): $(run.error)",
                )
                push!(fits, run.fit)
            end

            checkpoint_scores = Dict(
                checkpoint => mean(
                    only(filter(metric -> metric.checkpoint == checkpoint,
                                plugin_checkpoint_metrics(fit, checkpoints))).normalized_plugin_mse
                    for fit in fits
                ) for checkpoint in checkpoints
            )
            checkpoint = findmin(checkpoint_scores)[2]
            prediction_scores = Dict{Int, Float64}()
            for prediction_iteration in prediction_iterations
                seed_scores = Float64[]
                for (seed_index, fit) in enumerate(fits)
                    metrics = predictive_test_metrics(fit, checkpoint, prediction_iteration)
                    push!(seed_scores, metrics.normalized_test_mse)
                    push!(validation_rows, (
                        candidate = finalist.name,
                        seed_index = seed_index,
                        training_iterations = checkpoint,
                        prediction_iterations = prediction_iteration,
                        plugin_mse = only(
                            filter(
                                metric -> metric.checkpoint == checkpoint,
                                plugin_checkpoint_metrics(fit, checkpoints),
                            ),
                        ).plugin_mse,
                        test_mse = metrics.test_mse,
                        normalized_test_mse = metrics.normalized_test_mse,
                        predictive_variance_mean = metrics.predictive_variance_mean,
                    ))
                end
                prediction_scores[prediction_iteration] = mean(seed_scores)
            end
            best_prediction_iterations = findmin(prediction_scores)[2]
            push!(finalist_summaries, (
                name = finalist.name,
                config = merge(
                    candidate_config(CONFIG, finalist.prior, finalist.damping),
                    (
                        iterations = checkpoint,
                        prediction_iterations = best_prediction_iterations,
                    ),
                ),
                training_iterations = checkpoint,
                prediction_iterations = best_prediction_iterations,
                normalized_test_mse = prediction_scores[best_prediction_iterations],
            ))
            println(
                "$(finalist.name): checkpoint=$checkpoint, " *
                "prediction_iterations=$best_prediction_iterations, " *
                "normalized predictive MSE=" *
                "$(round(prediction_scores[best_prediction_iterations], digits = 4))",
            )
        end
        validation = DataFrame(validation_rows)
        sort!(finalist_summaries, by = summary -> (
            summary.normalized_test_mse,
            summary.training_iterations,
            summary.prediction_iterations,
        ))
        winner = first(finalist_summaries)

        println("\n=== held-out paired confirmation")
        heldout_rows = NamedTuple[]
        for (seed_index, seeds) in enumerate(heldout_seeds)
            winner_config = seeded_config(
                winner.config,
                seeds;
                n_samples = heldout_samples,
                iterations = winner.training_iterations,
            )
            baseline_config = seeded_config(
                legacy_config(CONFIG),
                seeds;
                n_samples = heldout_samples,
                iterations = XOR_CT_TUNING_SMOKE ? 2 : 100,
            )
            for (arm, config, training_iteration, pred_iteration) in (
                (
                    "legacy",
                    baseline_config,
                    baseline_config.iterations,
                    XOR_CT_TUNING_SMOKE ? 1 : 5,
                ),
                (
                    "winner",
                    winner_config,
                    winner.training_iterations,
                    winner.prediction_iterations,
                ),
            )
                run = safe_fit(config)
                if run.status == "ok"
                    test = predictive_test_metrics(run.fit, training_iteration, pred_iteration)
                    grid = clean_grid_metrics(
                        run.fit,
                        training_iteration,
                        pred_iteration,
                        heldout_grid_size,
                    )
                    prior_a_map = mean(run.fit.priors[:a_map])
                    prior_a_pred = mean(run.fit.priors[:a_pred])
                    posterior_a_map = mean(
                        run.fit.result.posteriors[:a_map][training_iteration],
                    )
                    posterior_a_pred = mean(
                        run.fit.result.posteriors[:a_pred][training_iteration],
                    )
                    push!(heldout_rows, (
                        arm = arm,
                        seed_index = seed_index,
                        status = "ok",
                        training_iterations = training_iteration,
                        prediction_iterations = pred_iteration,
                        plugin_mse = run.fit.plugin_mse,
                        test_mse = test.test_mse,
                        normalized_test_mse = test.normalized_test_mse,
                        baseline_mse = run.fit.baseline_mse,
                        clean_grid_mse = grid.clean_grid_mse,
                        predictive_variance_mean = test.predictive_variance_mean,
                        free_energy_first = run.fit.free_energy_first,
                        free_energy_last = run.fit.free_energy_last,
                        free_energy_decreases = run.fit.free_energy_decreases,
                        a_map_movement = norm(posterior_a_map .- prior_a_map) /
                            norm(prior_a_map),
                        a_pred_movement = norm(posterior_a_pred .- prior_a_pred) /
                            norm(prior_a_pred),
                        seconds = run.fit.seconds,
                        allocated_gib = run.fit.allocated_gib,
                        process_peak_rss_gib = Sys.maxrss() / 2.0^30,
                        error = missing,
                    ))
                    println(
                        "seed $seed_index $arm: test=$(round(test.test_mse, digits = 4)), " *
                        "clean=$(round(grid.clean_grid_mse, digits = 4))",
                    )
                else
                    push!(heldout_rows, (
                        arm = arm,
                        seed_index = seed_index,
                        status = "failed",
                        training_iterations = training_iteration,
                        prediction_iterations = pred_iteration,
                        plugin_mse = NaN,
                        test_mse = NaN,
                        normalized_test_mse = Inf,
                        baseline_mse = NaN,
                        clean_grid_mse = NaN,
                        predictive_variance_mean = NaN,
                        free_energy_first = NaN,
                        free_energy_last = NaN,
                        free_energy_decreases = 0,
                        a_map_movement = NaN,
                        a_pred_movement = NaN,
                        seconds = NaN,
                        allocated_gib = NaN,
                        process_peak_rss_gib = Sys.maxrss() / 2.0^30,
                        error = run.error,
                    ))
                end
            end
        end
        heldout = DataFrame(heldout_rows)
        winner_rows = filter(:arm => ==("winner"), heldout)
        legacy_rows = filter(:arm => ==("legacy"), heldout)
        successful = all(heldout.status .== "ok")
        paired_wins = successful ? count(winner_rows.test_mse .< legacy_rows.test_mse) : 0
        required_wins = XOR_CT_TUNING_SMOKE ? 1 : 2
        accepted = successful &&
            mean(winner_rows.test_mse) < mean(legacy_rows.test_mse) &&
            paired_wins >= required_wins &&
            mean(winner_rows.clean_grid_mse) <= 1.01 * mean(legacy_rows.clean_grid_mse)

        (
            stage_a = stage_a,
            stage_b = stage_b,
            validation = validation,
            heldout = heldout,
            winner = winner,
            accepted = accepted,
        )
    end

    outputs = total_timed.value
    println("\n=== tuning summary")
    println("accepted for promotion : ", outputs.accepted)
    println("winner                  : ", outputs.winner.name)
    println("training iterations     : ", outputs.winner.training_iterations)
    println("prediction iterations   : ", outputs.winner.prediction_iterations)
    println("validation norm. MSE    : ", outputs.winner.normalized_test_mse)
    println("wall time                : ", round(total_timed.time, digits = 1), "s")
    show(outputs.heldout; allrows = true, allcols = true, truncate = 80)
    println()

    if TUNING_SAVE_OUTPUTS
        mkpath(dirname(TUNING_OUTPUT_PREFIX))
        CSV.write(TUNING_OUTPUT_PREFIX * "_prior_screen.csv", outputs.stage_a)
        CSV.write(TUNING_OUTPUT_PREFIX * "_damping_refinement.csv", outputs.stage_b)
        CSV.write(TUNING_OUTPUT_PREFIX * "_validation.csv", outputs.validation)
        CSV.write(TUNING_OUTPUT_PREFIX * "_heldout.csv", outputs.heldout)
        plot_path = save_tuning_plot(outputs.heldout, TUNING_OUTPUT_PREFIX)
        summary_path = TUNING_OUTPUT_PREFIX * "_summary.md"
        write_summary(
            summary_path,
            outputs.winner,
            outputs.validation,
            outputs.heldout,
            outputs.accepted,
            total_timed.time,
        )
        println("summary                  : ", summary_path)
        println("MSE plot                 : ", plot_path)
    end
    return outputs
end

if abspath(PROGRAM_FILE) == @__FILE__
    tuning_results = run_tuning()
end
