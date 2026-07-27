# Stage 3 — Arm B: dense covariance over ALL weights, input layer included.
#
# Arm A (experiments/joint_output_layer_bnn.jl) made the OUTPUT weight vector one
# dense-covariance variable and showed that this is the necessary and sufficient
# change for GP-like contraction. Arm B asks the follow-up question: does making
# the INPUT layer joint as well buy anything?
#
# Structure, per observation:
#
#     x_f[o]  ~ MvNormal(features[o], jitter)          # pseudo-observed features
#     za[o]   ~ ContinuousTransition(x_f[o], a, P)     # a = vec(W), DENSE posterior
#     h[o]    ~ MvResidualSine(za[o])                  # exact JOINT covariance
#     y[o]    ~ softdot(h[o], v, precision[o])         # v dense, softdot IS the likelihood
#
# Everything here is tested machinery:
#   * `ContinuousTransition` + `LinearReshapeMeta(H, d_f)` gives a dense posterior
#     over `vec(W)` -- including cross-neuron covariance, which Arm A's per-neuron
#     `w[k]` cannot represent -- and maintains a structured joint `q(x_f, za)`
#     through the layer (src/nodes/ContinuousTransition/linear_reshape_meta.jl,
#     496 lines of tests in test/ngmp/linear_reshape_meta_tests.jl).
#   * `MvResidualSine` propagates the EXACT joint mean and covariance of the
#     activation (`_mv_residual_sine_mean_cov`, src/nodes/mv_residual_sine/node.jl:250-270),
#     so cross-neuron correlation survives the nonlinearity instead of being
#     discarded by H independent scalar activations.
#   * `ContinuousTransition` has no PointMass rules on its `x` interface, so the
#     observed features enter through a tight pseudo-observed latent -- the pattern
#     used by every `experiments/xor_ctransition_*` script (CTProgress.md D6).
#
# Two deliberate differences from Arm A, both forced by the architecture:
#
# 1. No constant/linear stack coordinates. `MvResidualSine` returns a vector edge,
#    and there is no node that concatenates a vector with scalars, so the intercept
#    and linear trend cannot be appended as extra coordinates here. They are not
#    lost: `phi(z) = z + (rho/omega) sin(omega z)` contains an identity component, so
#    `v' h` already contains `sum_k v_k w_k' [1, x]` -- a learned affine function of
#    x -- and the predictive variance still grows like `x^2` outside the data
#    through that component. It is a less well-conditioned path than a dedicated
#    coordinate with its own prior, which is worth remembering when comparing the
#    two arms' extrapolation numbers.
#
# 2. The transition precision `P` is a genuine modelling choice, not a numerical
#    device, and it must stay MODERATE. `a` is mean-field-split from `q(x_f, za)`,
#    so by the same argument that made the production arm's `tau_c = 1e4` pin the
#    output weights, a large `P` would pin `vec(W)`: the message toward `a` carries
#    site precision proportional to `E[P]`. A moderate `P` means the hidden layer is
#    genuinely stochastic, which is a defensible model, but it is NOT "Arm A with a
#    joint input layer" -- read the comparison with that in mind.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 julia --project=. experiments/joint_all_weights_bnn.jl
#
# Env: ARM_B_NEURONS, ARM_B_ITERATIONS, ARM_B_PREDICTION_ITERATIONS, ARM_B_GRID,
#      ARM_B_BENCHMARKS (cubic,sine), ARM_B_OUTPUT, ARM_B_RESULTS

using LinearAlgebra
using Printf
using Random
using Serialization
using Statistics

using Plots
using RxInfer
using StableRNGs
using SurrogateModelling

import ProbabilisticEnsembling: Exp

include(joinpath(@__DIR__, "uq_benchmarks.jl"))

env_integer(key, default) = parse(Int, get(ENV, key, string(default)))
env_float(key, default) = parse(Float64, get(ENV, key, string(default)))

const ARM_B_CONFIG = (
    n_neurons = env_integer("ARM_B_NEURONS", 12),
    iterations = env_integer("ARM_B_ITERATIONS", 100),
    prediction_iterations = env_integer("ARM_B_PREDICTION_ITERATIONS", 100),
    grid_points = env_integer("ARM_B_GRID", 121),
    feature_jitter = env_float("ARM_B_FEATURE_JITTER", 1e-6),
    # Transition precision. Moderate on purpose -- see the header note.
    transition_precision_mean = env_float("ARM_B_TRANSITION_PRECISION", 20.0),
    score_carrier_precision = env_float("ARM_B_SCORE_CARRIER", 10.0),
    weight_prior_variance = env_float("ARM_B_WEIGHT_VARIANCE", 0.25),
    frequency_scale = env_float("ARM_B_FREQUENCY_SCALE", 2.2),
    frequency_low = env_float("ARM_B_FREQUENCY_LOW", 0.35),
    frequency_high = env_float("ARM_B_FREQUENCY_HIGH", 5.0),
    output_signal_sd = env_float("ARM_B_OUTPUT_SIGNAL_SD", 1.0),
    noise_signal_sd = env_float("ARM_B_NOISE_SIGNAL_SD", 0.5),
    anchor_variance = env_float("ARM_B_ANCHOR_VARIANCE", 1.0),
    phi_rho = env_float("ARM_B_RHO", 0.9),
    phi_omega = env_float("ARM_B_OMEGA", 1.0),
    activation_alpha = env_float("ARM_B_ACTIVATION_ALPHA", 0.05),
    link_alpha = env_float("ARM_B_LINK_ALPHA", 0.05),
    ngmp_max_step = env_float("ARM_B_MAX_STEP", 0.1),
    prediction_activation_alpha = env_float("ARM_B_PREDICTION_ACTIVATION_ALPHA", 0.5),
    prediction_link_alpha = env_float("ARM_B_PREDICTION_LINK_ALPHA", 0.5),
    prediction_max_step = env_float("ARM_B_PREDICTION_MAX_STEP", 1.0),
    prior_seed = env_integer("ARM_B_PRIOR_SEED", 42),
    data_seed = env_integer("ARM_B_DATA_SEED", 2027),
    output_path = get(ENV, "ARM_B_OUTPUT", "/tmp/joint_all_weights_bnn.png"),
    results_path = get(ENV, "ARM_B_RESULTS", "/tmp/joint_all_weights_bnn.jls"),
)

const ARM_B_FEATURE_DIM = 2  # [1, x]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Each head gets its OWN pseudo-observed feature latent. `x_f` can belong to only
# one factorization cluster, so the mean and noise heads cannot share it while both
# keep a structured joint through their transition.
@model function joint_all_weights_model(
    y, features, priors, feature_cov, reshape_meta,
    ct_deps, activation_meta, activation_deps, link_meta, link_deps,
)
    local x_f, za, h, noise_x_f, noise_za, noise_h
    local log_precision, shifted_log_precision, precision

    a ~ priors[:a]              # vec(W) for the mean head, dense posterior
    noise_a ~ priors[:noise_a]
    P ~ priors[:P]              # Wishart transition precision
    noise_P ~ priors[:noise_P]
    v ~ priors[:v]              # dense readout weights
    noise_v ~ priors[:noise_v]

    for observation in eachindex(features)
        x_f[observation] ~ MvNormalMeanCovariance(features[observation], feature_cov)
        za[observation] ~ ContinuousTransition(x_f[observation], a, P) where {
            dependencies = ct_deps, meta = reshape_meta,
        }
        h[observation] ~ MvResidualSine(za[observation]) where {
            dependencies = activation_deps, meta = activation_meta,
        }

        noise_x_f[observation] ~
            MvNormalMeanCovariance(features[observation], feature_cov)
        noise_za[observation] ~
            ContinuousTransition(noise_x_f[observation], noise_a, noise_P) where {
                dependencies = ct_deps, meta = reshape_meta,
            }
        noise_h[observation] ~ MvResidualSine(noise_za[observation]) where {
            dependencies = activation_deps, meta = activation_meta,
        }

        log_precision[observation] ~ softdot(
            noise_h[observation], noise_v, priors[:score_carrier],
        )
        # Deterministic shift by a KNOWN constant, not an extra latent: the noise
        # head learns zero-mean deviations from the ridge-fit log precision. Arm A
        # gets this from a constant stack coordinate (where E[h] = 1 exactly);
        # here there is no such coordinate, and spreading the anchor across the
        # neuron coordinates would be wrong because sum_k E[h_k] is not n.
        shifted_log_precision[observation] :=
            log_precision[observation] + priors[:log_precision_anchor]
        precision[observation] ~ Exp(shifted_log_precision[observation]) where {
            dependencies = link_deps, meta = link_meta,
        }

        y[observation] ~ softdot(h[observation], v, precision[observation])
    end
end

@constraints function joint_all_weights_constraints()
    q(x_f, za, h, noise_x_f, noise_za, noise_h, a, noise_a, P, noise_P,
      v, noise_v, log_precision, shifted_log_precision, precision, y) =
        q(v, y) * q(x_f, za) * q(h) * q(a) * q(P) *
        q(noise_v, log_precision, shifted_log_precision, precision) *
        q(noise_x_f, noise_za) * q(noise_h) * q(noise_a) * q(noise_P)
    q(a)::MomentForm()
    q(noise_a)::MomentForm()
    q(v)::MomentForm()
    q(noise_v)::MomentForm()
end

activation_meta_for(config) =
    ResidualSineMeta(rho = config.phi_rho, omega = config.phi_omega)

activation_deps_for(config, alpha) = NGMPDependencies(
    out = nothing, in = nothing;
    projection = TangentProjection(type = ClosedForm),
    damping = DampingMeta(alpha = alpha, beta = 0.0, max_step = config.ngmp_max_step),
)

link_deps_for(_config) = NGMPDependencies(
    out = nothing, in = nothing; projection = TangentProjection(type = ClosedForm),
)

link_meta_for(config, alpha) =
    DampingMeta(alpha = alpha, beta = 0.0, max_step = config.ngmp_max_step)

# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

"""
    arm_b_weight_prior(n_neurons, config, rng)

Prior over `a = vec(W)` for a `H x 2` layer acting on `[1, x]`. `LinearReshapeMeta`
uses column-major `reshape`, so entries `1..H` are the bias column and `H+1..2H` the
slope column. The bias means alternate in sign and the slope means are LOG-spaced
over `[frequency_low, frequency_high] * frequency_scale`, matching Arm A so the
comparison isolates the joint structure rather than the initialization.
"""
function arm_b_weight_prior(n_neurons::Int, config, rng)
    pairs = max(n_neurons ÷ 2, 1)
    slope_centers = pairs == 1 ?
        [config.frequency_scale] :
        exp.(collect(range(
            log(config.frequency_low * config.frequency_scale),
            log(config.frequency_high * config.frequency_scale);
            length = pairs,
        )))
    biases = map(1:n_neurons) do neuron
        pair = cld(neuron, 2)
        sign_value = (isodd(pair) ? 1.0 : -1.0) * (isodd(neuron) ? 1.0 : -1.0)
        sign_value * (pi / 4) * (1 + 0.1 * randn(rng))
    end
    slopes = map(1:n_neurons) do neuron
        slope_centers[cld(neuron, 2)] * (1 + 0.03 * randn(rng))
    end
    return MvNormalMeanCovariance(
        vcat(biases, slopes),
        diagm(fill(config.weight_prior_variance, n_neurons * ARM_B_FEATURE_DIM)),
    )
end

function ridge_log_precision_anchor(design, targets; ridge::Real = 1e-3)
    coefficients = (design' * design + ridge * I(size(design, 2))) \ (design' * targets)
    residual_variance = max(mean(abs2.(targets .- design * coefficients)), 1e-8)
    return -log(residual_variance)
end

function arm_b_priors(config, design, targets)
    rng = StableRNG(config.prior_seed)
    n_neurons = config.n_neurons
    anchor = ridge_log_precision_anchor(design, targets)

    # Wishart with degrees of freedom just above the dimension: weakly informative
    # about the transition precision while keeping the mean at the configured level.
    degrees = n_neurons + 2.0
    scale = diagm(fill(config.transition_precision_mean / degrees, n_neurons))

    return Dict{Symbol, Any}(
        :a => arm_b_weight_prior(n_neurons, config, rng),
        :noise_a => arm_b_weight_prior(n_neurons, config, rng),
        :P => Wishart(degrees, scale),
        :noise_P => Wishart(degrees, scale),
        :v => MvNormalMeanCovariance(
            zeros(n_neurons),
            diagm(fill(abs2(config.output_signal_sd) / n_neurons, n_neurons)),
        ),
        # Zero mean: the anchor is applied as a deterministic shift in the model,
        # so this head learns only deviations from the ridge-fit log precision.
        :noise_v => MvNormalMeanCovariance(
            zeros(n_neurons),
            diagm(fill(abs2(config.noise_signal_sd) / n_neurons, n_neurons)),
        ),
        :score_carrier => config.score_carrier_precision,
        :log_precision_anchor => anchor,
        :ridge_anchor => anchor,
    )
end

"""
    arm_b_initial_states(priors, features, config)

Push the prior weights forward through the layer and the activation so every
NGMP-constrained edge starts at a sensible expansion point. `_mv_residual_sine_mean_cov`
gives the exact joint moments of the activation, including cross-neuron terms.
"""
function arm_b_initial_states(priors, features, config)
    n_neurons = config.n_neurons
    activation = activation_meta_for(config)
    transition_covariance = diagm(fill(inv(config.transition_precision_mean), n_neurons))

    forward(weight_prior) = map(features) do feature
        m_a, V_a = mean_cov(weight_prior)
        matrix = reshape(m_a, n_neurons, ARM_B_FEATURE_DIM)
        mean_za = matrix * feature
        # Var(W f) for a dense vec(W): sum over the Kronecker structure.
        kron_feature = kron(feature, Matrix{Float64}(I, n_neurons, n_neurons))
        covariance_za =
            kron_feature' * V_a * kron_feature + transition_covariance
        MvNormalMeanCovariance(mean_za, Matrix(Symmetric(covariance_za)))
    end

    za = forward(priors[:a])
    noise_za = forward(priors[:noise_a])
    activate(marginals) = map(marginals) do marginal
        m, V = mean_cov(marginal)
        MvNormalMeanCovariance(
            SurrogateModelling._mv_residual_sine_mean_cov(m, V, activation)...,
        )
    end
    h = activate(za)
    noise_h = activate(noise_za)

    m_noise_v, V_noise_v = mean_cov(priors[:noise_v])
    log_precision = map(noise_h) do marginal
        m_h, V_h = mean_cov(marginal)
        NormalMeanVariance(
            dot(m_noise_v, m_h),
            dot(m_noise_v, V_h * m_noise_v) + dot(m_h, V_noise_v * m_h) +
                tr(V_noise_v * V_h) + inv(priors[:score_carrier]),
        )
    end
    shifted_log_precision = map(log_precision) do marginal
        mean_log, variance_log = mean_var(marginal)
        NormalMeanVariance(mean_log + priors[:log_precision_anchor], variance_log)
    end
    precision = map(shifted_log_precision) do marginal
        mean_log, variance_log = mean_var(marginal)
        shape = 1 + inv(max(variance_log, 1e-6))
        GammaShapeRate(shape, shape * exp(-mean_log))
    end

    return (; za, h, noise_za, noise_h, log_precision, shifted_log_precision, precision)
end

@initialization function joint_all_weights_initialization(priors, states)
    q(a) = deepcopy(priors[:a])
    q(noise_a) = deepcopy(priors[:noise_a])
    q(P) = deepcopy(priors[:P])
    q(noise_P) = deepcopy(priors[:noise_P])
    q(v) = deepcopy(priors[:v])
    q(noise_v) = deepcopy(priors[:noise_v])
    q(za) = states.za
    q(h) = states.h
    q(noise_za) = states.noise_za
    q(noise_h) = states.noise_h
    q(log_precision) = states.log_precision
    q(shifted_log_precision) = states.shifted_log_precision
    q(precision) = states.precision
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

build_model(priors, config, alphas) = joint_all_weights_model(
    priors = priors,
    feature_cov = diagm(fill(config.feature_jitter, ARM_B_FEATURE_DIM)),
    reshape_meta = LinearReshapeMeta(config.n_neurons, ARM_B_FEATURE_DIM),
    # `a` is NGMP-constrained: a custom dependencies policy replaces
    # ContinuousTransition's default q_a-injecting policy, and the NGMP :a adapter
    # restores that injection.
    ct_deps = NGMPDependencies(
        a = nothing, x = nothing;
        projection = TangentProjection(type = ClosedForm),
        damping = DampingMeta(
            alpha = alphas.activation, beta = 0.0, max_step = config.ngmp_max_step,
        ),
    ),
    activation_meta = activation_meta_for(config),
    activation_deps = activation_deps_for(config, alphas.activation),
    link_meta = link_meta_for(config, alphas.link),
    link_deps = link_deps_for(config),
)

function arm_b_run_inference(priors, features, targets, config, alphas, iterations)
    states = arm_b_initial_states(priors, features, config)
    infer(
        model = build_model(priors, config, alphas),
        data = (y = targets, features = features),
        constraints = joint_all_weights_constraints(),
        initialization = joint_all_weights_initialization(priors, states),
        returnvars = (
            a = KeepLast(), noise_a = KeepLast(), P = KeepLast(),
            noise_P = KeepLast(), v = KeepLast(), noise_v = KeepLast(),
            h = KeepLast(), precision = KeepLast(),
        ),
        iterations = iterations,
        free_energy = false,
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
end

function arm_b_train(targets, features, config)
    priors = arm_b_priors(config, reduce(hcat, features)', targets)
    started = time()
    result = arm_b_run_inference(
        priors, features, targets, config,
        (activation = config.activation_alpha, link = config.link_alpha),
        config.iterations,
    )
    elapsed = time() - started

    learned = copy(priors)
    for key in (:a, :noise_a, :P, :noise_P, :v, :noise_v)
        learned[key] = deepcopy(result.posteriors[key])
    end

    covariance = cov(learned[:v])
    off_diagonal = covariance - Diagonal(diag(covariance))
    weight_covariance = cov(learned[:a])
    weight_off_diagonal = weight_covariance - Diagonal(diag(weight_covariance))
    @printf("  fit: %d iterations in %.1f s\n", config.iterations, elapsed)
    @printf("  q(v): diag %.3e..%.3e, max|offdiag| %.3e\n",
        minimum(diag(covariance)), maximum(diag(covariance)),
        maximum(abs, off_diagonal))
    @printf("  q(vec W): diag %.3e..%.3e, max|offdiag| %.3e  (Arm A cannot represent this)\n",
        minimum(diag(weight_covariance)), maximum(diag(weight_covariance)),
        maximum(abs, weight_off_diagonal))
    return (; learned, elapsed)
end

function arm_b_predict(priors, features, config)
    prediction_config = merge(config, (ngmp_max_step = config.prediction_max_step,))
    result = arm_b_run_inference(
        priors, features, fill(missing, length(features)), prediction_config,
        (activation = config.prediction_activation_alpha,
         link = config.prediction_link_alpha),
        config.prediction_iterations,
    )
    return (;
        q_h = collect(vec(result.posteriors[:h])),
        q_v = result.posteriors[:v],
        q_precision = collect(vec(result.posteriors[:precision])),
    )
end

"""`E[v'h]` and `Var(v'h)` for independent Gaussian `v` and `h`."""
function mean_function_moments(q_v, q_h)
    m_v, V_v = mean_cov(q_v)
    means = zeros(length(q_h))
    variances = zeros(length(q_h))
    for (index, marginal) in enumerate(q_h)
        m_h, V_h = mean_cov(marginal)
        means[index] = dot(m_v, m_h)
        variances[index] = dot(m_v, V_h * m_v) + dot(m_h, V_v * m_h) + tr(V_v * V_h)
    end
    return means, variances
end

function run_arm_b(benchmark, config)
    x_center, x_scale = mean(benchmark.x), std(benchmark.x)
    y_center, y_scale = mean(benchmark.y), std(benchmark.y)
    standardize_x(values) = (values .- x_center) ./ x_scale
    targets = (benchmark.y .- y_center) ./ y_scale

    train_features = [[1.0, xi] for xi in standardize_x(benchmark.x)]
    grid_features = [[1.0, xi] for xi in standardize_x(benchmark.grid)]

    println("\n### ", benchmark.name)
    fit = arm_b_train(targets, train_features, config)
    prediction = arm_b_predict(fit.learned, grid_features, config)

    standardized_mean, standardized_function_variance =
        mean_function_moments(prediction.q_v, prediction.q_h)
    standardized_aleatoric = inv.(mean.(prediction.q_precision))

    predicted_mean = y_center .+ y_scale .* standardized_mean
    function_variance = abs2(y_scale) .* standardized_function_variance
    aleatoric_variance = abs2(y_scale) .* standardized_aleatoric
    total_variance = function_variance .+ aleatoric_variance

    return (;
        benchmark_name = benchmark.name,
        grid = benchmark.grid,
        x_train = benchmark.x, y_train = benchmark.y,
        predicted_mean, total_variance, function_variance, aleatoric_variance,
        learned_parameter_posteriors = fit.learned, elapsed = fit.elapsed, y_scale,
        contraction = contraction_metrics(
            benchmark, predicted_mean, function_variance, total_variance,
        ),
        calibration = grid_calibration(
            benchmark, predicted_mean, total_variance, aleatoric_variance,
        ),
    )
end

requested = Symbol.(strip.(split(get(ENV, "ARM_B_BENCHMARKS", "cubic,sine"), ",")))
selected_benchmarks = map(requested) do name
    if name === :cubic
        cubic_gap_benchmark(
            seed = ARM_B_CONFIG.data_seed, grid_points = ARM_B_CONFIG.grid_points,
        )
    elseif name === :sine
        sine_benchmark(
            seed = ARM_B_CONFIG.data_seed, grid_points = ARM_B_CONFIG.grid_points,
        )
    else
        throw(ArgumentError("ARM_B_BENCHMARKS entries must be cubic or sine"))
    end
end

println("\n", "="^98)
println("Stage 3 — Arm B: dense over all weights  (H=$(ARM_B_CONFIG.n_neurons), ",
    "CTransition + LinearReshapeMeta input layer, MvResidualSine activation)")
println("="^98)

arm_b_results = map(benchmark -> run_arm_b(benchmark, ARM_B_CONFIG), selected_benchmarks)

println("\n", "-"^98)
@printf("%-22s %9s %9s %11s %11s %11s %10s %10s\n",
    "benchmark", "obs RMSE", "gap RMSE", "Var obs", "Var gap", "Var outer",
    "gap/obs", "out/obs")
for result in arm_b_results
    c = result.contraction
    @printf("%-22s %9.3f %9.3f %11.3f %11.3f %11.3f %10.3f %10.3f\n",
        result.benchmark_name, c.observed_rmse, c.gap_rmse,
        c.observed_function_variance, c.gap_function_variance,
        c.outer_function_variance, c.gap_over_observed, c.outer_over_observed)
end

println()
@printf("%-22s %12s %12s %12s %12s %12s\n",
    "benchmark", "aleatoric", "truth", "log ratio", "excess NLL", "aleat corr")
for result in arm_b_results
    k = result.calibration
    @printf("%-22s %12.4f %12.4f %12.3f %12.3f %12.3f\n",
        result.benchmark_name, k.aleatoric_mean, k.aleatoric_truth,
        k.aleatoric_log_ratio, k.excess_nll, k.aleatoric_correlation)
end

serialize(ARM_B_CONFIG.results_path, (; config = ARM_B_CONFIG, results = arm_b_results))
@info "saved" ARM_B_CONFIG.results_path

panels = []
for result in arm_b_results
    interval = 1.96 .* sqrt.(max.(result.total_variance, 0.0))
    fit_panel = plot(
        result.grid, result.predicted_mean;
        ribbon = interval, fillalpha = 0.2, color = :teal, linewidth = 2,
        label = "mean ± 1.96 SD", xlabel = "x", ylabel = "y",
        title = "Arm B: $(result.benchmark_name)", legend = :topleft,
    )
    scatter!(fit_panel, result.x_train, result.y_train;
        color = :gray50, markersize = 3, markeralpha = 0.55,
        markerstrokewidth = 0, label = "observations")
    variance_panel = plot(
        result.grid, result.function_variance;
        color = :royalblue, linewidth = 2, label = "Var(q(mean))",
        xlabel = "x", ylabel = "variance", yscale = :log10,
        title = "contraction", legend = :topleft,
    )
    plot!(variance_panel, result.grid, result.total_variance;
        color = :darkorange, linewidth = 2, label = "Var(q(y*))")
    plot!(variance_panel, result.grid, result.aleatoric_variance;
        color = :purple, linestyle = :dash, linewidth = 2, label = "1/E[q(precision)]")
    push!(panels, fit_panel, variance_panel)
end
plot(panels...; layout = (length(arm_b_results), 2),
    size = (1180, 420 * length(arm_b_results)))
savefig(ARM_B_CONFIG.output_path)
@info "saved" ARM_B_CONFIG.output_path
