# Ablation for xor_ctransition_mvsoftplus_probit.jl: remove the learned global
# gamma_score variable and use a fixed, effectively deterministic score map.
#
#   h2 -> softdot(theta, h2, FIXED_SCORE_PRECISION) -> z -> Probit -> y
#
# FIXED_SCORE_PRECISION defaults to 1e6.  This leaves a residual softdot
# variance of 1e-6: small enough to be negligible relative to posterior weight
# and hidden-state uncertainty, while remaining finite for Gaussian message
# arithmetic.  All data, splits, priors other than gamma_score, damping, and
# training/prediction iteration counts match the clean-selected final run.
#
# Smoke:
#   XOR_CT_FIXED_SCORE_SMOKE=true julia --project=. \
#     experiments/xor_ctransition_mvsoftplus_probit_fixed_score_precision.jl
#
# Two full paired final seeds with saved plots:
#   OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_ctransition_mvsoftplus_probit_fixed_score_precision.jl

const XOR_CT_FIXED_SCORE_SMOKE =
    lowercase(get(ENV, "XOR_CT_FIXED_SCORE_SMOKE", "false")) in
    ("1", "true", "yes", "on")
const FIXED_SAVE_OUTPUTS = lowercase(
    get(ENV, "SAVE_OUTPUTS", string(!XOR_CT_FIXED_SCORE_SMOKE)),
) in ("1", "true", "yes", "on")
const FIXED_SHOW_PROGRESS =
    lowercase(get(ENV, "SHOW_PROGRESS", "false")) in ("1", "true", "yes", "on")
const FIXED_REQUIRE_CLEAN_BASELINE =
    lowercase(get(ENV, "REQUIRE_CLEAN_BASELINE", "false")) in
    ("1", "true", "yes", "on")

ENV["XOR_CT_PROBIT_SMOKE"] = string(XOR_CT_FIXED_SCORE_SMOKE)
ENV["SAVE_OUTPUTS"] = "false"
ENV["SHOW_PROGRESS"] = "false"
ENV["REQUIRE_CLEAN_BASELINE"] = "false"

include(joinpath(@__DIR__, "xor_ctransition_mvsoftplus_probit.jl"))

const FIXED_FINAL_SEEDS = (
    (data_seed = 2_030, split_seed = 2_031, prior_seed = 44, flip_seed = 9_031),
    (data_seed = 2_032, split_seed = 2_033, prior_seed = 45, flip_seed = 9_033),
)

const FIXED_CONFIG = merge(
    CONFIG,
    (
        fixed_score_precision = env_float("FIXED_SCORE_PRECISION", 1e12),
        n_samples = env_int("N_SAMPLES", XOR_CT_FIXED_SCORE_SMOKE ? 24 : 2_000),
        iterations = env_int("N_ITERATIONS", XOR_CT_FIXED_SCORE_SMOKE ? 1 : 160),
        prediction_iterations = env_int(
            "PREDICTION_ITERATIONS",
            XOR_CT_FIXED_SCORE_SMOKE ? 1 : 10,
        ),
        grid_size = env_int("GRID_SIZE", XOR_CT_FIXED_SCORE_SMOKE ? 8 : 60),
        save_outputs = FIXED_SAVE_OUTPUTS,
        output_prefix = get(
            ENV,
            "OUTPUT_PREFIX",
            joinpath(
                @__DIR__,
                "..",
                "viz",
                "xor_ctransition_mvsoftplus_probit_fixed_score_precision",
            ),
        ),
        show_progress = FIXED_SHOW_PROGRESS,
        require_clean_baseline = FIXED_REQUIRE_CLEAN_BASELINE,
    ),
)

function validate_fixed_config(config)
    validate_config(config)
    isfinite(config.fixed_score_precision) && config.fixed_score_precision > 0 ||
        throw(ArgumentError("FIXED_SCORE_PRECISION must be finite and positive"))
    return config
end

validate_fixed_config(FIXED_CONFIG)

@model function xor_ct_mvsoftplus_probit_fixed_score(
    y,
    features,
    priors,
    fixed_score_precision,
    feature_cov,
    meta_map,
    meta_pred,
    ct_a_deps,
    ct2_deps,
    sp_deps,
    sp_damping,
)
    a_map ~ priors[:a_map]
    a_pred ~ priors[:a_pred]
    theta ~ priors[:theta]
    P ~ priors[:P]
    Gamma2 ~ priors[:Gamma2]
    for i in eachindex(y)
        x_f[i] ~ MvNormalMeanCovariance(features[i], feature_cov)
        h1[i] ~ ContinuousTransition(x_f[i], a_map, P) where {
            dependencies = ct_a_deps, meta = meta_map
        }
        s[i] ~ MvSoftplus(h1[i]) where {dependencies = sp_deps, meta = sp_damping}
        h2[i] ~ ContinuousTransition(s[i], a_pred, Gamma2) where {
            dependencies = ct2_deps, meta = meta_pred
        }
        z[i] ~ softdot(theta, h2[i], fixed_score_precision)
        y[i] ~ Probit(z[i])
    end
end

@constraints function xor_ct_mvsoftplus_probit_fixed_score_constraints()
    q(x_f, h1, s, h2, z, a_map, a_pred, P, Gamma2, theta) =
        q(x_f, h1)q(s, h2, z)q(a_map)q(a_pred)q(P)q(Gamma2)q(theta)
end

@model function xor_ct_mvsoftplus_fixed_score_prediction(
    features,
    priors,
    fixed_score_precision,
    feature_cov,
    meta_map,
    meta_pred,
    sp_deps,
    sp_damping,
    score_prior_variance,
)
    local x_f, h1, s, h2, z

    a_map ~ priors[:a_map]
    a_pred ~ priors[:a_pred]
    theta ~ priors[:theta]
    P ~ priors[:P]
    Gamma2 ~ priors[:Gamma2]
    for i in eachindex(features)
        x_f[i] ~ MvNormalMeanCovariance(features[i], feature_cov)
        h1[i] ~ ContinuousTransition(x_f[i], a_map, P) where {meta = meta_map}
        s[i] ~ MvSoftplus(h1[i]) where {dependencies = sp_deps, meta = sp_damping}
        h2[i] ~ ContinuousTransition(s[i], a_pred, Gamma2) where {meta = meta_pred}
        z[i] ~ softdot(theta, h2[i], fixed_score_precision)
        z[i] ~ NormalMeanVariance(0.0, score_prior_variance)
    end
end

@constraints function xor_ct_mvsoftplus_fixed_score_prediction_constraints(priors)
    q(x_f, h1, s, h2, z, a_map, a_pred, P, Gamma2, theta) =
        q(x_f, h1)q(s, h2, z)q(a_map)q(a_pred)q(P)q(Gamma2)q(theta)

    q(a_map)::RxInfer.FixedMarginalFormConstraint(priors[:a_map])
    q(a_pred)::RxInfer.FixedMarginalFormConstraint(priors[:a_pred])
    q(theta)::RxInfer.FixedMarginalFormConstraint(priors[:theta])
    q(P)::RxInfer.FixedMarginalFormConstraint(priors[:P])
    q(Gamma2)::RxInfer.FixedMarginalFormConstraint(priors[:Gamma2])
end

const FIXED_GLOBAL_KEYS = (:a_map, :a_pred, :theta, :P, :Gamma2)

function make_fixed_priors(config; d_f = 3)
    priors = make_priors(config; d_f = d_f)
    delete!(priors, :gamma_score)
    return priors
end

function make_fixed_initialization(config, priors)
    d_h = config.d_hidden
    return @initialization begin
        q(a_map) = priors[:a_map]
        q(a_pred) = priors[:a_pred]
        q(theta) = priors[:theta]
        q(P) = priors[:P]
        q(Gamma2) = priors[:Gamma2]
        q(h1) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        q(s) = MvNormalMeanCovariance(
            fill(config.softplus_output_initial_mean, d_h),
            Diagonal(fill(config.softplus_output_initial_variance, d_h)),
        )
        q(h2) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        q(z) = NormalMeanVariance(0.0, config.score_initial_variance)
    end
end

function make_fixed_prediction_initialization(config, priors)
    d_h = config.d_hidden
    return @initialization begin
        q(a_map) = priors[:a_map]
        q(a_pred) = priors[:a_pred]
        q(theta) = priors[:theta]
        q(P) = priors[:P]
        q(Gamma2) = priors[:Gamma2]
        q(h1) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        q(s) = MvNormalMeanCovariance(
            fill(config.softplus_output_initial_mean, d_h),
            Diagonal(fill(config.softplus_output_initial_variance, d_h)),
        )
        q(h2) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        q(z) = NormalMeanVariance(0.0, config.prediction_prior_variance)
        μ(z) = NormalMeanVariance(0.0, config.prediction_prior_variance)
    end
end

function validate_fixed_fit(fit, n_train)
    all(isfinite, fit.result.free_energy) || error("training free energy is non-finite")
    for key in FIXED_GLOBAL_KEYS
        all(isfinite, mean(last(fit.result.posteriors[key]))) ||
            error("posterior mean for $key is non-finite")
    end
    length(fit.ct_a_deps.states) == n_train ||
        error("first CT layer has the wrong damping-state count")
    length(fit.ct2_deps.states) == n_train ||
        error("second CT layer has the wrong damping-state count")
    all(state -> state.nfired == fit.iterations, fit.ct_a_deps.states) ||
        error("first CT layer did not fire exactly once per iteration")
    all(state -> state.nfired == fit.iterations, fit.ct2_deps.states) ||
        error("second CT layer did not fire exactly once per iteration")
    return true
end

function fit_fixed_score_arm(config, features, labels)
    validate_fixed_config(config)
    all(label -> label == 0.0 || label == 1.0, labels) ||
        error("Probit observations must be exact binary labels")
    d_f = length(first(features))
    priors = make_fixed_priors(config; d_f = d_f)
    ct_a_deps = make_ct_dependencies(config)
    ct2_deps = make_ct_dependencies(config)
    sp_deps = make_mvsoftplus_dependencies()
    sp_damping = DampingMeta(
        alpha = config.ngmp_alpha,
        beta = config.ngmp_beta,
        max_step = config.ngmp_max_step,
    )

    elapsed = @elapsed result = infer(
        model = xor_ct_mvsoftplus_probit_fixed_score(
            priors = priors,
            fixed_score_precision = config.fixed_score_precision,
            feature_cov = Matrix(Diagonal(fill(config.feature_jitter, d_f))),
            meta_map = LinearReshapeMeta(config.d_hidden, d_f),
            meta_pred = LinearReshapeMeta(config.d_hidden, config.d_hidden),
            ct_a_deps = ct_a_deps,
            ct2_deps = ct2_deps,
            sp_deps = sp_deps,
            sp_damping = sp_damping,
        ),
        data = (y = labels, features = features),
        constraints = xor_ct_mvsoftplus_probit_fixed_score_constraints(),
        initialization = make_fixed_initialization(config, priors),
        iterations = config.iterations,
        free_energy = true,
        showprogress = config.show_progress,
        returnvars = (
            a_map = KeepEach(),
            a_pred = KeepEach(),
            theta = KeepEach(),
            P = KeepEach(),
            Gamma2 = KeepEach(),
        ),
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    fit = (
        result = result,
        priors = priors,
        ct_a_deps = ct_a_deps,
        ct2_deps = ct2_deps,
        iterations = config.iterations,
        elapsed = elapsed,
    )
    validate_fixed_fit(fit, length(labels))
    return fit
end

function fixed_prediction_priors(fit)
    return Dict{Symbol, Any}(
        key => deepcopy(last(fit.result.posteriors[key])) for key in FIXED_GLOBAL_KEYS
    )
end

function run_fixed_score_prediction_batch(priors, features; config)
    isempty(features) && return Any[]
    d_f = length(first(features))
    sp_deps = make_mvsoftplus_dependencies()
    sp_damping = DampingMeta(
        alpha = config.ngmp_alpha,
        beta = config.ngmp_beta,
        max_step = config.ngmp_max_step,
    )
    result = infer(
        model = xor_ct_mvsoftplus_fixed_score_prediction(
            priors = priors,
            fixed_score_precision = config.fixed_score_precision,
            feature_cov = Matrix(Diagonal(fill(config.feature_jitter, d_f))),
            meta_map = LinearReshapeMeta(config.d_hidden, d_f),
            meta_pred = LinearReshapeMeta(config.d_hidden, config.d_hidden),
            sp_deps = sp_deps,
            sp_damping = sp_damping,
            score_prior_variance = config.prediction_prior_variance,
        ),
        data = (features = features,),
        constraints = xor_ct_mvsoftplus_fixed_score_prediction_constraints(priors),
        initialization = make_fixed_prediction_initialization(config, priors),
        iterations = config.prediction_iterations,
        free_energy = false,
        showprogress = false,
        returnvars = (z = KeepLast(),),
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    marginals = collect(vec(result.posteriors[:z]))
    length(marginals) == length(features) ||
        error("prediction graph returned the wrong number of score marginals")
    return marginals
end

function predict_fixed_score_marginals(fit, features; config)
    priors = fixed_prediction_priors(fit)
    marginals = Vector{Any}(undef, length(features))
    for first_index in 1:config.prediction_batch_size:length(features)
        indices = first_index:min(
            first_index + config.prediction_batch_size - 1,
            length(features),
        )
        marginals[indices] = run_fixed_score_prediction_batch(
            priors,
            features[indices];
            config = config,
        )
    end
    return marginals
end

function run_fixed_score_pair(config)
    validate_fixed_config(config)
    verify_native_probit_probability()
    dataset = make_clean_xor_dataset(n = config.n_samples, seed = config.data_seed)
    train_data, test_data = split_dataset(
        dataset;
        train_fraction = config.train_fraction,
        seed = config.split_seed,
    )
    labels = make_paired_training_labels(
        train_data;
        quadrant = config.untrusted_quadrant,
        flip_probability = config.quadrant_flip_prob,
        seed = config.flip_seed,
    )
    train_features = build_features(train_data)
    test_features = build_features(test_data)

    clean_fit = fit_fixed_score_arm(config, train_features, labels.clean)
    corrupted_fit = fit_fixed_score_arm(config, train_features, labels.corrupted)
    for key in FIXED_GLOBAL_KEYS
        mean(clean_fit.priors[key]) == mean(corrupted_fit.priors[key]) ||
            error("paired arms do not share the same prior mean for $key")
    end

    clean_statistics = score_statistics(predict_fixed_score_marginals(
        clean_fit,
        test_features;
        config = config,
    ))
    corrupted_statistics = score_statistics(predict_fixed_score_marginals(
        corrupted_fit,
        test_features;
        config = config,
    ))
    clean_diagnostics = quadrant_diagnostics(:clean, test_data, clean_statistics)
    corrupted_diagnostics = quadrant_diagnostics(
        :corrupted,
        test_data,
        corrupted_statistics,
    )
    diagnostics = vcat(clean_diagnostics, corrupted_diagnostics)
    clean_localization = localization_metrics(
        clean_statistics,
        test_data,
        config.untrusted_quadrant,
    )
    corrupted_localization = localization_metrics(
        corrupted_statistics,
        test_data,
        config.untrusted_quadrant,
    )
    did = corrupted_localization.latent_variance_difference -
        clean_localization.latent_variance_difference
    localization = (
        fixed_score_precision = config.fixed_score_precision,
        untrusted_quadrant = string(config.untrusted_quadrant),
        clean_untrusted_latent_variance = clean_localization.untrusted_latent_variance,
        clean_reliable_latent_variance = clean_localization.reliable_latent_variance,
        clean_latent_variance_ratio = clean_localization.latent_variance_ratio,
        clean_latent_variance_difference = clean_localization.latent_variance_difference,
        corrupted_untrusted_latent_variance = corrupted_localization.untrusted_latent_variance,
        corrupted_reliable_latent_variance = corrupted_localization.reliable_latent_variance,
        corrupted_latent_variance_ratio = corrupted_localization.latent_variance_ratio,
        corrupted_latent_variance_difference =
            corrupted_localization.latent_variance_difference,
        latent_variance_difference_in_differences = did,
    )

    baseline = class_prior_baseline(labels.clean, Float64.(test_data.label))
    clean_overall = only(eachrow(clean_diagnostics[clean_diagnostics.quadrant .== "overall", :]))
    clean_beats_baseline_accuracy = clean_overall.accuracy > baseline.accuracy
    clean_beats_baseline_nll = clean_overall.nll < baseline.nll
    if config.require_clean_baseline
        clean_beats_baseline_accuracy ||
            error("fixed-score clean accuracy did not beat the class-prior baseline")
        clean_beats_baseline_nll ||
            error("fixed-score clean NLL did not beat the class-prior baseline")
    end

    surface_paths = nothing
    if config.save_outputs
        grid = make_prediction_grid(config.grid_size)
        clean_grid = reshape_grid_statistics(score_statistics(
            predict_fixed_score_marginals(clean_fit, grid.features; config = config);
            verify_native = false,
        ), grid)
        corrupted_grid = reshape_grid_statistics(score_statistics(
            predict_fixed_score_marginals(corrupted_fit, grid.features; config = config);
            verify_native = false,
        ), grid)
        surface_paths = save_matched_surfaces(
            grid,
            clean_grid,
            corrupted_grid,
            config.output_prefix,
        )
        mkpath(dirname(config.output_prefix))
        CSV.write(config.output_prefix * "_quadrant_metrics.csv", diagnostics)
        CSV.write(config.output_prefix * "_localization.csv", DataFrame([localization]))
    end

    println()
    println(
        "=== fixed-score-precision Probit (precision = $(config.fixed_score_precision), " *
        "n_train = $(nrow(train_data)), n_test = $(nrow(test_data)))",
    )
    println("selected/flipped training labels: ", count(labels.selected), "/", count(labels.flipped))
    println(
        "clean baseline accuracy/NLL: ",
        round(baseline.accuracy, digits = 4), " / ", round(baseline.nll, digits = 4),
        "; beaten = ", clean_beats_baseline_accuracy, " / ", clean_beats_baseline_nll,
    )
    for (arm, fit, arm_diagnostics) in (
        (:clean, clean_fit, clean_diagnostics),
        (:corrupted, corrupted_fit, corrupted_diagnostics),
    )
        overall = only(eachrow(arm_diagnostics[arm_diagnostics.quadrant .== "overall", :]))
        println(
            rpad(string(arm), 10),
            " accuracy/NLL = ", round(overall.accuracy, digits = 4),
            " / ", round(overall.nll, digits = 4),
            "; FE first/last = ", first(fit.result.free_energy),
            " / ", last(fit.result.free_energy),
            "; elapsed = ", round(fit.elapsed, digits = 1), "s",
        )
    end
    println("localization: ", localization)
    isnothing(surface_paths) || println("surface files: ", surface_paths)

    return (
        config = config,
        train_data = train_data,
        test_data = test_data,
        labels = labels,
        arms = (
            clean = (
                fit = clean_fit,
                statistics = clean_statistics,
                diagnostics = clean_diagnostics,
                localization = clean_localization,
            ),
            corrupted = (
                fit = corrupted_fit,
                statistics = corrupted_statistics,
                diagnostics = corrupted_diagnostics,
                localization = corrupted_localization,
            ),
        ),
        localization = localization,
        baseline = baseline,
        clean_beats_baseline_accuracy = clean_beats_baseline_accuracy,
        clean_beats_baseline_nll = clean_beats_baseline_nll,
        surface_paths = surface_paths,
    )
end

function fixed_result_rows(report, seed_index)
    rows = NamedTuple[]
    for arm in (:clean, :corrupted)
        arm_report = getproperty(report.arms, arm)
        overall = only(eachrow(
            arm_report.diagnostics[arm_report.diagnostics.quadrant .== "overall", :],
        ))
        push!(rows, (
            seed_index = seed_index,
            arm = string(arm),
            fixed_score_precision = report.config.fixed_score_precision,
            accuracy = overall.accuracy,
            nll = overall.nll,
            untrusted_latent_variance = arm_report.localization.untrusted_latent_variance,
            reliable_latent_variance = arm_report.localization.reliable_latent_variance,
            latent_variance_ratio = arm_report.localization.latent_variance_ratio,
            latent_variance_difference = arm_report.localization.latent_variance_difference,
            difference_in_differences =
                report.localization.latent_variance_difference_in_differences,
            clean_beats_baseline_accuracy = report.clean_beats_baseline_accuracy,
            clean_beats_baseline_nll = report.clean_beats_baseline_nll,
        ))
    end
    return rows
end

function compare_with_learned_gamma(fixed, learned_path)
    isfile(learned_path) || return DataFrame()
    learned = CSV.read(learned_path, DataFrame)
    rows = NamedTuple[]
    for fixed_row in eachrow(fixed)
        match = learned[
            (learned.seed_index .== fixed_row.seed_index) .&
            (learned.arm .== fixed_row.arm),
            :,
        ]
        nrow(match) == 1 || error("expected one learned-gamma comparison row")
        learned_row = first(eachrow(match))
        push!(rows, (
            seed_index = fixed_row.seed_index,
            arm = fixed_row.arm,
            learned_accuracy = learned_row.accuracy,
            fixed_accuracy = fixed_row.accuracy,
            accuracy_change = fixed_row.accuracy - learned_row.accuracy,
            learned_nll = learned_row.nll,
            fixed_nll = fixed_row.nll,
            nll_change = fixed_row.nll - learned_row.nll,
            learned_untrusted_latent_variance = learned_row.untrusted_latent_variance,
            fixed_untrusted_latent_variance = fixed_row.untrusted_latent_variance,
            learned_reliable_latent_variance = learned_row.reliable_latent_variance,
            fixed_reliable_latent_variance = fixed_row.reliable_latent_variance,
            learned_latent_variance_ratio = learned_row.latent_variance_ratio,
            fixed_latent_variance_ratio = fixed_row.latent_variance_ratio,
            latent_variance_ratio_change =
                fixed_row.latent_variance_ratio - learned_row.latent_variance_ratio,
        ))
    end
    return DataFrame(rows)
end

function write_fixed_summary(path, fixed, comparison, elapsed, precision)
    open(path, "w") do io
        println(io, "# Fixed score-precision Probit ablation")
        println(io)
        println(io, "- Fixed score precision: `$(precision)` (residual variance `$(inv(precision))`)")
        println(io, "- Total wall time: `$(round(elapsed, digits = 1))` seconds")
        println(io, "- Final datasets: `$(length(unique(fixed.seed_index)))`")
        println(io)
        if !isempty(comparison)
            for arm in ("clean", "corrupted")
                subset = comparison[comparison.arm .== arm, :]
                println(io, "## $(uppercasefirst(arm)) arm, fixed minus learned gamma")
                println(io)
                println(io, "- Accuracy change: `$(mean(subset.accuracy_change))`")
                println(io, "- NLL change: `$(mean(subset.nll_change))`")
                println(
                    io,
                    "- Latent-variance-ratio change: `$(mean(subset.latent_variance_ratio_change))`",
                )
                println(io)
            end
        end
        println(io, "Localization remains descriptive, not a pass/fail criterion.")
    end
end

function run_fixed_score_final(config = FIXED_CONFIG)
    seeds = XOR_CT_FIXED_SCORE_SMOKE ? FIXED_FINAL_SEEDS[1:1] : FIXED_FINAL_SEEDS
    rows = NamedTuple[]
    elapsed = @elapsed for (seed_index, seed_config) in enumerate(seeds)
        run_config = merge(
            config,
            seed_config,
            (
                output_prefix = config.output_prefix * "_final_seed_$(seed_index)",
                require_clean_baseline = false,
            ),
        )
        report = run_fixed_score_pair(run_config)
        append!(rows, fixed_result_rows(report, seed_index))
    end
    fixed = DataFrame(rows)
    learned_path = joinpath(
        @__DIR__,
        "..",
        "viz",
        "xor_ctransition_mvsoftplus_probit_tuning_final.csv",
    )
    comparison = compare_with_learned_gamma(fixed, learned_path)
    if config.save_outputs
        mkpath(dirname(config.output_prefix))
        CSV.write(config.output_prefix * "_final.csv", fixed)
        isempty(comparison) ||
            CSV.write(config.output_prefix * "_vs_learned_gamma.csv", comparison)
        write_fixed_summary(
            config.output_prefix * "_summary.md",
            fixed,
            comparison,
            elapsed,
            config.fixed_score_precision,
        )
    end
    return (fixed = fixed, comparison = comparison, elapsed = elapsed)
end

if abspath(PROGRAM_FILE) == @__FILE__
    fixed_score_report = run_fixed_score_final(FIXED_CONFIG)
end
