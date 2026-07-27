# Stage 1 — a known-correct, fully parametric GP reference arm.
#
# Nothing in this repository currently produces a predictive variance that is
# known to be right, so every uncertainty claim so far has been scored against
# nothing. This arm fixes that. It is deliberately the simplest model that has
# GP behaviour exactly:
#
#     phi(x) in R^H           fixed feature map (random Fourier, or a frozen
#                             draw of the ResidualSine input layer)
#     v ~ MvNormal(0, I)      DENSE H-dimensional weight posterior
#     y_o ~ softdot(phi(x_o), v, gamma)          i.e. N(phi(x_o)' v, 1/gamma)
#
# With the features deterministic, `phi` a PointMass on the theta edge and a
# dense Gaussian on the x edge, this is exact Bayesian linear regression, and
# the info-form fast path at src/nodes/softdot/rules/structured_info_form.jl:34-62
# is the selected dispatch. Weight-space/function-space duality then makes it
# *identically* a GP with kernel k(x, x') = phi(x)' phi(x'):
#
#     Var(f(x*)) = phi(x*)' Sigma_v phi(x*)
#                = k(x*,x*) - k(x*,X) (K + sigma^2 I)^-1 k(X,x*)
#
# and Sigma_v = (I + gamma Phi'Phi)^-1 is DENSE. That density is the whole point.
# Stage 0.1 showed the production arm's output-weight variances collapse to ~5e-7
# because, under mean-field, each Var(v_k) is driven by the diagonal of
# gamma Phi'Phi -- which is large in every coordinate. The dense inverse instead
# retains prior variance in the directions the data does not span, and those
# retained directions are exactly what makes the variance grow in a gap.
#
# Two independent checks:
#   * exactness   -- message-passing q(v) vs the analytic BLR posterior vs the
#                    function-space GP, at fixed noise. Must agree to ~1e-8.
#   * fidelity    -- max |phi'phi - RBF| for the RFF draw. This is a Monte Carlo
#                    error that shrinks like 1/sqrt(H); it is reported, not
#                    asserted, and it is NOT part of the exactness claim.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 julia --project=. experiments/parametric_gp_reference.jl
#
# Env:
#   GP_REF_FEATURES    rff | residual_sine      (default rff)
#   GP_REF_H           number of features       (default 64)
#   GP_REF_LENGTHSCALE RBF lengthscale, standardized x units (default 0.35)
#   GP_REF_SIGNAL_SD   prior signal sd          (default 1.0)
#   GP_REF_LINEAR      append an uncertain linear feature (default true)
#   GP_REF_LEARN_NOISE learn gamma by VMP as well as fixing it (default true)
#   GP_REF_OUTPUT / GP_REF_RESULTS  artifact paths

using LinearAlgebra
using Printf
using Random
using Serialization
using Statistics

using Plots
using RxInfer
using StableRNGs
using SurrogateModelling

include(joinpath(@__DIR__, "uq_benchmarks.jl"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

env_string(key, default) = get(ENV, key, default)
env_integer(key, default) = parse(Int, get(ENV, key, string(default)))
env_float(key, default) = parse(Float64, get(ENV, key, string(default)))
env_flag(key, default) = lowercase(get(ENV, key, string(default))) in ("1", "true", "yes")

const GP_REF_CONFIG = (
    feature_kind = Symbol(env_string("GP_REF_FEATURES", "rff")),
    n_features = env_integer("GP_REF_H", 64),
    lengthscale = env_float("GP_REF_LENGTHSCALE", 0.35),
    signal_sd = env_float("GP_REF_SIGNAL_SD", 1.0),
    include_linear = env_flag("GP_REF_LINEAR", true),
    learn_noise = env_flag("GP_REF_LEARN_NOISE", true),
    feature_seed = env_integer("GP_REF_FEATURE_SEED", 20260725),
    data_seed = env_integer("GP_REF_DATA_SEED", 2027),
    grid_points = env_integer("GP_REF_GRID", 241),
    iterations = env_integer("GP_REF_ITERATIONS", 20),
    noise_prior_shape = env_float("GP_REF_NOISE_SHAPE", 2.0),
    noise_prior_rate = env_float("GP_REF_NOISE_RATE", 0.2),
    output_path = env_string("GP_REF_OUTPUT", "/tmp/parametric_gp_reference.png"),
    results_path = env_string("GP_REF_RESULTS", "/tmp/parametric_gp_reference.jls"),
)

# ---------------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------------

"""
    RandomFourierFeatures(n_features, lengthscale, signal_sd; rng, include_linear)

`phi_j(x) = sqrt(2 sigma_f^2 / H) cos(omega_j x + b_j)` with
`omega_j ~ N(0, 1/l^2)` and `b_j ~ U(0, 2pi)`, the standard RBF random-feature
map. The `sqrt(2 sigma_f^2 / H)` scaling is where width normalization belongs:
it keeps the prior predictive variance at `sigma_f^2` independent of `H`, which
is the invariance the production arm lacks (its `ManyPlus` sum makes prior
variance grow with width).

`include_linear` appends a raw `x` feature. Bounded cosines revert to the prior
envelope away from the data; a linear feature with a Gaussian weight contributes
variance growing like `x^2`, which is the linear-kernel component an
extrapolating benchmark such as `y = x^3` needs.
"""
struct RandomFourierFeatures
    frequencies::Vector{Float64}
    phases::Vector{Float64}
    scale::Float64
    signal_sd::Float64
    lengthscale::Float64
    include_linear::Bool
end

function RandomFourierFeatures(
    n_features::Int, lengthscale::Real, signal_sd::Real;
    rng, include_linear::Bool = true,
)
    n_cosines = include_linear ? n_features - 1 : n_features
    n_cosines >= 1 || throw(ArgumentError("need at least one cosine feature"))
    return RandomFourierFeatures(
        randn(rng, n_cosines) ./ lengthscale,
        2pi .* rand(rng, n_cosines),
        sqrt(2 * abs2(signal_sd) / n_cosines),
        float(signal_sd),
        float(lengthscale),
        include_linear,
    )
end

function (map::RandomFourierFeatures)(x::Real)
    cosines = map.scale .* cos.(map.frequencies .* x .+ map.phases)
    return map.include_linear ? vcat(cosines, float(x)) : cosines
end

"""
    FrozenResidualSineFeatures(n_features, activation; rng, ...)

A frozen draw of the production arm's own input layer:
`phi_k(x) = phi_act(w_k' [1, x])` with `w_k` sampled once from the same style of
prior. Keeping the *feature family* comparable isolates the effect of the dense
output layer from the effect of changing basis, so the GP reference cannot be
dismissed as "a different model that happens to work".
"""
struct FrozenResidualSineFeatures
    biases::Vector{Float64}
    slopes::Vector{Float64}
    activation::ResidualSineMeta
    scale::Float64
    include_linear::Bool
end

function FrozenResidualSineFeatures(
    n_features::Int, activation::ResidualSineMeta;
    rng, signal_sd::Real = 1.0, slope_scale::Real = 2.2, include_linear::Bool = true,
)
    n_units = include_linear ? n_features - 1 : n_features
    n_units >= 1 || throw(ArgumentError("need at least one unit"))
    return FrozenResidualSineFeatures(
        (pi / 4) .* sign.(randn(rng, n_units)) .* (1 .+ 0.3 .* randn(rng, n_units)),
        slope_scale .* (0.75 .+ 0.5 .* rand(rng, n_units)),
        activation,
        sqrt(abs2(signal_sd) / n_units),
        include_linear,
    )
end

function (map::FrozenResidualSineFeatures)(x::Real)
    preactivation = map.biases .+ map.slopes .* x
    units = map.scale .* (
        preactivation .+
        (map.activation.rho / map.activation.omega) .*
        sin.(map.activation.omega .* preactivation)
    )
    return map.include_linear ? vcat(units, float(x)) : units
end

build_feature_map(config) =
    if config.feature_kind === :rff
        RandomFourierFeatures(
            config.n_features, config.lengthscale, config.signal_sd;
            rng = StableRNG(config.feature_seed), include_linear = config.include_linear,
        )
    elseif config.feature_kind === :residual_sine
        FrozenResidualSineFeatures(
            config.n_features, ResidualSineMeta(rho = 0.9, omega = 1.0);
            rng = StableRNG(config.feature_seed), signal_sd = config.signal_sd,
            include_linear = config.include_linear,
        )
    else
        throw(ArgumentError("GP_REF_FEATURES must be rff or residual_sine"))
    end

design_matrix(feature_map, x_values) =
    reduce(hcat, (feature_map(x) for x in x_values))'  # rows = observations

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Fixed noise: with gamma known this graph is exactly conjugate, so q(v) is the
# exact BLR posterior and the predictive is closed form. This is the arm the
# exactness test uses.
@model function gp_reference_fixed_noise(y, features, n_features, noise_precision)
    v ~ MvNormalMeanCovariance(zeros(n_features), diageye(n_features))
    for observation in eachindex(y)
        y[observation] ~ softdot(features[observation], v, noise_precision)
    end
end

# Learned noise: gamma gets a Gamma prior and ordinary mean-field VMP. The
# weight posterior stays dense; only the coupling to gamma is relaxed.
@model function gp_reference_learned_noise(y, features, n_features, shape, rate)
    gamma ~ GammaShapeRate(shape, rate)
    v ~ MvNormalMeanCovariance(zeros(n_features), diageye(n_features))
    for observation in eachindex(y)
        y[observation] ~ softdot(features[observation], v, gamma)
    end
end

@constraints function gp_reference_constraints()
    q(v)::MomentForm()
end

# With gamma latent the joint over (v, gamma) is Normal-Gamma and not conjugate
# to the softdot factor, so the noise precision is split off by mean field. The
# WEIGHT vector stays a single dense cluster -- that is the part that must never
# be factorized, and the whole point of this arm.
@constraints function gp_reference_learned_noise_constraints()
    q(v, gamma) = q(v)q(gamma)
    q(v)::MomentForm()
end

"""
    fit_gp_reference(targets, feature_rows, config; noise_precision)

Run message passing. Passing `noise_precision` fixes gamma (exact conjugate
arm); omitting it learns gamma by VMP. Returns the dense `q(v)` either way.
"""
function fit_gp_reference(targets, feature_rows, config; noise_precision = nothing)
    n_features = length(first(feature_rows))
    if isnothing(noise_precision)
        result = infer(
            model = gp_reference_learned_noise(
                features = feature_rows,
                n_features = n_features,
                shape = config.noise_prior_shape,
                rate = config.noise_prior_rate,
            ),
            data = (y = targets,),
            constraints = gp_reference_learned_noise_constraints(),
            initialization = @initialization(begin
                q(gamma) = GammaShapeRate(config.noise_prior_shape, config.noise_prior_rate)
                q(v) = MvNormalMeanCovariance(zeros(n_features), diageye(n_features))
            end),
            returnvars = (v = KeepLast(), gamma = KeepLast()),
            iterations = config.iterations,
            free_energy = false,
            showprogress = false,
            options = (limit_stack_depth = 100,),
        )
        return (;
            q_v = result.posteriors[:v],
            q_gamma = result.posteriors[:gamma],
            noise_precision = mean(result.posteriors[:gamma]),
        )
    end

    result = infer(
        model = gp_reference_fixed_noise(
            features = feature_rows,
            n_features = n_features,
            noise_precision = noise_precision,
        ),
        data = (y = targets,),
        constraints = gp_reference_constraints(),
        returnvars = (v = KeepLast(),),
        iterations = config.iterations,
        free_energy = false,
        showprogress = false,
        options = (limit_stack_depth = 100,),
    )
    return (; q_v = result.posteriors[:v], q_gamma = nothing, noise_precision)
end

# ---------------------------------------------------------------------------
# Analytic ground truth
# ---------------------------------------------------------------------------

"""
    analytic_blr(design, targets, noise_precision)

Weight-space posterior with an `N(0, I)` prior:
`Sigma = (I + gamma Phi'Phi)^-1`, `m = gamma Sigma Phi' y`.
"""
function analytic_blr(design, targets, noise_precision)
    n_features = size(design, 2)
    precision = Matrix{Float64}(I, n_features, n_features) +
        noise_precision .* (design' * design)
    covariance = inv(Symmetric(precision))
    mean_vector = noise_precision .* (covariance * (design' * targets))
    return mean_vector, Matrix(covariance)
end

"""
    analytic_gp(design, targets, design_star, noise_precision)

Function-space posterior with the *same* kernel `k = phi' phi`. Algebraically
identical to `analytic_blr`, computed by an independent route (an `n x n` solve
instead of an `H x H` one) so agreement is a real check rather than a tautology.
"""
function analytic_gp(design, targets, design_star, noise_precision)
    noise_variance = inv(noise_precision)
    gram = design * design'
    cross = design_star * design'
    prior_variance = vec(sum(abs2, design_star; dims = 2))
    factorization = cholesky(Symmetric(gram + noise_variance * I))
    predictive_mean = cross * (factorization \ targets)
    predictive_variance =
        prior_variance .- vec(sum(cross .* (factorization \ cross')'; dims = 2))
    return predictive_mean, predictive_variance
end

"""
    message_passing_predictive(q_v, design_star, noise_precision)

`mean = phi' m_v`, `Var(f) = phi' Sigma_v phi`, `Var(y) = Var(f) + 1/gamma`.
One forward evaluation -- no iteration, nothing to converge.
"""
function message_passing_predictive(q_v, design_star, noise_precision)
    m_v, V_v = mean_cov(q_v)
    predictive_mean = design_star * m_v
    function_variance = vec(sum((design_star * V_v) .* design_star; dims = 2))
    return predictive_mean, function_variance, function_variance .+ inv(noise_precision)
end

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

# Benchmarks come from `experiments/uq_benchmarks.jl`, included above. They used to
# be duplicated here, and the copies drifted: the local sine grid ran only over
# [-2, 2] while the data reaches |x| ~ 3.3, and its `outer` mask sat inside the data.
# One definition, no drift.

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"""
    run_benchmark(benchmark, config)

Standardize, build the fixed feature map, fit both the exact and the
learned-noise arm, and compare message passing against both analytic routes.
"""
function run_benchmark(benchmark, config)
    x_center, x_scale = mean(benchmark.x), std(benchmark.x)
    y_center, y_scale = mean(benchmark.y), std(benchmark.y)
    standardize_x(values) = (values .- x_center) ./ x_scale
    targets = (benchmark.y .- y_center) ./ y_scale

    feature_map = build_feature_map(config)
    design = design_matrix(feature_map, standardize_x(benchmark.x))
    design_star = design_matrix(feature_map, standardize_x(benchmark.grid))
    n_features = size(design, 2)

    # The exactness test needs a known gamma. Use the benchmark's own noise
    # level expressed in standardized target units.
    true_noise_variance = mean(benchmark.noise_variance(benchmark.x)) / abs2(y_scale)
    fixed_precision = inv(true_noise_variance)

    exact_fit = fit_gp_reference(
        targets, collect(eachrow(design)), config; noise_precision = fixed_precision,
    )
    mp_mean, mp_function_var, mp_total_var =
        message_passing_predictive(exact_fit.q_v, design_star, fixed_precision)

    blr_mean, blr_cov = analytic_blr(design, targets, fixed_precision)
    blr_predictive_mean = design_star * blr_mean
    blr_function_var = vec(sum((design_star * blr_cov) .* design_star; dims = 2))

    gp_mean, gp_function_var =
        analytic_gp(design, targets, design_star, fixed_precision)

    exactness = (;
        weight_mean_max_abs_error = maximum(abs, mean(exact_fit.q_v) .- blr_mean),
        weight_cov_max_abs_error = maximum(abs, cov(exact_fit.q_v) .- blr_cov),
        predictive_mean_max_abs_error = maximum(abs, mp_mean .- blr_predictive_mean),
        predictive_var_max_abs_error = maximum(abs, mp_function_var .- blr_function_var),
        gp_mean_max_abs_error = maximum(abs, blr_predictive_mean .- gp_mean),
        gp_var_max_abs_error = maximum(abs, blr_function_var .- gp_function_var),
    )

    learned_fit = config.learn_noise ?
        fit_gp_reference(targets, collect(eachrow(design)), config) : exact_fit
    learned_mean, learned_function_var, learned_total_var =
        message_passing_predictive(
            learned_fit.q_v, design_star, learned_fit.noise_precision,
        )

    to_data_mean(values) = y_center .+ y_scale .* values
    to_data_variance(values) = abs2(y_scale) .* values

    arms = Dict(
        :fixed_noise => (;
            predicted_mean = to_data_mean(mp_mean),
            function_variance = to_data_variance(mp_function_var),
            total_variance = to_data_variance(mp_total_var),
            noise_precision = fixed_precision,
        ),
        :learned_noise => (;
            predicted_mean = to_data_mean(learned_mean),
            function_variance = to_data_variance(learned_function_var),
            total_variance = to_data_variance(learned_total_var),
            noise_precision = learned_fit.noise_precision,
        ),
    )

    return (;
        benchmark_name = benchmark.name,
        grid = benchmark.grid,
        x_train = benchmark.x,
        y_train = benchmark.y,
        n_features,
        arms,
        exactness,
        kernel_fidelity = kernel_fidelity(feature_map, standardize_x(benchmark.grid), config),
        y_scale,
        benchmark.true_mean, benchmark.true_variance, benchmark.masks,
    )
end

"""
    kernel_fidelity(feature_map, x_values, config)

How close the *finite* random-feature kernel is to the RBF kernel it
approximates. A Monte Carlo error of order `1/sqrt(H)`, reported for context.
Not applicable to the frozen-ResidualSine map, which targets no closed-form
kernel.
"""
function kernel_fidelity(feature_map::RandomFourierFeatures, x_values, config)
    # Only the cosine block approximates the RBF kernel. The appended linear
    # feature contributes an exact `x * x'` linear-kernel term, so including it
    # here would report a spurious error of order max(x)^2.
    n_cosines = length(feature_map.frequencies)
    design = design_matrix(feature_map, x_values)[:, 1:n_cosines]
    approximate = design * design'
    exact = [
        abs2(config.signal_sd) * exp(-abs2(a - b) / (2 * abs2(config.lengthscale)))
        for a in x_values, b in x_values
    ]
    return (;
        max_abs_error = maximum(abs, approximate .- exact),
        relative_error = maximum(abs, approximate .- exact) / abs2(config.signal_sd),
    )
end
kernel_fidelity(::FrozenResidualSineFeatures, _, _) =
    (; max_abs_error = NaN, relative_error = NaN)

"""Metrics, including the contraction-ratio triple that encodes GP-likeness."""
function arm_metrics(result, arm_key)
    arm = result.arms[arm_key]
    masks = result.masks
    observed_function = mean(arm.function_variance[masks.observed])
    rmse(mask) = sqrt(mean(abs2.(arm.predicted_mean[mask] .- result.true_mean[mask])))
    coverage(mask) = mean(
        abs.(arm.predicted_mean[mask] .- result.true_mean[mask]) .<=
        1.96 .* sqrt.(arm.total_variance[mask]),
    )
    return (;
        arm = arm_key,
        observed_rmse = rmse(masks.observed),
        gap_rmse = any(masks.gap) ? rmse(masks.gap) : NaN,
        observed_function_variance = observed_function,
        gap_function_variance = any(masks.gap) ?
            mean(arm.function_variance[masks.gap]) : NaN,
        outer_function_variance = any(masks.outer) ?
            mean(arm.function_variance[masks.outer]) : NaN,
        gap_over_observed = any(masks.gap) ?
            mean(arm.function_variance[masks.gap]) / observed_function : NaN,
        outer_over_observed = any(masks.outer) ?
            mean(arm.function_variance[masks.outer]) / observed_function : NaN,
        aleatoric_variance = abs2(result.y_scale) / arm.noise_precision,
        true_aleatoric = mean(result.true_variance[masks.observed]),
        observed_coverage = coverage(masks.observed),
    )
end

benchmarks = [
    cubic_gap_benchmark(seed = GP_REF_CONFIG.data_seed,
        grid_points = GP_REF_CONFIG.grid_points),
    sine_benchmark(seed = GP_REF_CONFIG.data_seed,
        grid_points = GP_REF_CONFIG.grid_points),
]

results = map(benchmark -> run_benchmark(benchmark, GP_REF_CONFIG), benchmarks)

println("\n", "="^94)
println("Stage 1 — parametric GP reference  (features=$(GP_REF_CONFIG.feature_kind), ",
    "H=$(GP_REF_CONFIG.n_features), lengthscale=$(GP_REF_CONFIG.lengthscale), ",
    "linear=$(GP_REF_CONFIG.include_linear))")
println("="^94)

for result in results
    println("\n### ", result.benchmark_name, "  (H = ", result.n_features, ")")
    println("\n  exactness: message passing vs analytic BLR vs function-space GP")
    for (name, value) in pairs(result.exactness)
        @printf("     %-32s %.3e\n", String(name), value)
    end
    if !isnan(result.kernel_fidelity.max_abs_error)
        @printf("     %-32s %.3e  (RFF Monte Carlo error, informational)\n",
            "kernel max_abs_error", result.kernel_fidelity.max_abs_error)
    end

    println("\n  predictive quality and contraction")
    @printf("     %-14s %9s %9s %11s %11s %11s %9s %9s\n",
        "arm", "obs RMSE", "gap RMSE", "Var obs", "Var gap", "Var outer",
        "gap/obs", "out/obs")
    for arm_key in (:fixed_noise, :learned_noise)
        m = arm_metrics(result, arm_key)
        @printf("     %-14s %9.3f %9.3f %11.3f %11.3f %11.3f %9.3f %9.3f\n",
            String(m.arm), m.observed_rmse, m.gap_rmse,
            m.observed_function_variance, m.gap_function_variance,
            m.outer_function_variance, m.gap_over_observed, m.outer_over_observed)
    end
    for arm_key in (:fixed_noise, :learned_noise)
        m = arm_metrics(result, arm_key)
        @printf("     %-14s aleatoric=%.4f (truth %.4f)  coverage95=%.3f\n",
            String(m.arm), m.aleatoric_variance, m.true_aleatoric, m.observed_coverage)
    end
end

println("""

Reference values this arm must be compared against
  * production cubic arm (Exp, 16+16, width-scaled): gap/observed = 2.079, with
    observed-domain epistemic variance 905.9 against a true noise variance of 9,
    i.e. no contraction at the data at all.
  * A GP contracts at the data, so `Var obs` should approach the noise level and
    `gap/obs` should be much larger than 2.
""")

serialize(GP_REF_CONFIG.results_path, (; config = GP_REF_CONFIG, results))
@info "saved" GP_REF_CONFIG.results_path

panels = []
for result in results
    arm = result.arms[:learned_noise]
    interval = 1.96 .* sqrt.(max.(arm.total_variance, 0.0))
    fit_panel = plot(
        result.grid, arm.predicted_mean;
        ribbon = interval, fillalpha = 0.2, color = :seagreen, linewidth = 2,
        label = "mean(q(y*)) ± 1.96 SD", xlabel = "x", ylabel = "y",
        title = result.benchmark_name, legend = :topleft,
    )
    plot!(fit_panel, result.grid, result.true_mean;
        color = :black, linewidth = 2, label = "truth")
    scatter!(fit_panel, result.x_train, result.y_train;
        color = :gray50, markersize = 3, markeralpha = 0.55,
        markerstrokewidth = 0, label = "observations")

    variance_panel = plot(
        result.grid, arm.function_variance;
        color = :royalblue, linewidth = 2, label = "Var(q(f))",
        xlabel = "x", ylabel = "variance", yscale = :log10,
        title = "$(result.benchmark_name): contraction", legend = :topleft,
    )
    plot!(variance_panel, result.grid, arm.total_variance;
        color = :darkorange, linewidth = 2, label = "Var(q(y*))")
    plot!(variance_panel, result.grid, max.(result.true_variance, 1e-6);
        color = :black, linestyle = :dashdot, linewidth = 2, label = "true aleatoric")
    push!(panels, fit_panel, variance_panel)
end
plot(panels...; layout = (length(results), 2), size = (1180, 420 * length(results)))
savefig(GP_REF_CONFIG.output_path)
@info "saved" GP_REF_CONFIG.output_path
