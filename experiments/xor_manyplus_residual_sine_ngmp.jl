# Additive-mean checkerboard (XOR) regression with explicit ManyPlus summation.
#
# This is the ManyPlus counterpart of notebooks/xor_softplus_ut_ngmp.jl. The
# precision-gated ensemble there combines K experts by attaching K
# NormalMeanPrecision factors to one shared `out` — a product of experts whose
# precision Σγ_k only ever accumulates confidence, which is why its predictive
# variance field comes out nearly flat. Here the combination is additive in the
# MEAN domain instead:
#
#     za[k, i] ~ softdot(features[i], w[k], τ)          linear pre-activation
#     h[k, i]  ~ ResidualSine(za[k, i])                  scalar Gaussian activation
#     c[k, i]  ~ softdot(v[k], h[k, i], τ_c)             scalar soft product v_k · h_k
#     out[i]   ~ ManyPlus(inputs = c[:, i])              explicit summation
#     y[i]     ~ NormalMeanPrecision(out[i], obs_noise)  LEARNED noise precision
#
# Variances now ADD across neurons (ManyPlus sum-product), so predictive
# uncertainty genuinely grows where the contributions are uncertain, and the
# Softplus→Gamma gate machinery (the failure class that blows up the log
# variant) is gone entirely. The smooth 2x2 checkerboard is a sum of two
# diagonal sinusoidal ridges — sin(u)sin(w) = ½[cos(u−w) − cos(u+w)] — so the
# additive residual-sine basis can represent it exactly.
#
# Run: julia --project=. experiments/xor_manyplus_residual_sine_ngmp.jl

ENV["GKSwstype"] = "100"

using SurrogateModelling
using RxInfer
using StableRNGs
using LinearAlgebra
using Random
using DataFrames
using CSV
using Plots

const OUTPUT_DIR = joinpath(@__DIR__, "xor_manyplus_residual_sine_output")
mkpath(OUTPUT_DIR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config = (
    n_samples = 1_600,
    n_neurons = 8,
    iterations = 500,
    train_fraction = 0.40,
    noise_std = 0.10,
    data_seed = 2_026,
    split_seed = 2_027,
    prior_seed = 42,
    # phi(x) = x + (rho/omega) sin(omega x); the effective ridge frequency in
    # input space is omega * |w|, so the prior weight radius targets the
    # diagonal checkerboard frequency pi/sqrt(2) * ... ≈ 2.2 at omega = 1.
    phi_rho = 0.9,
    phi_omega = 1.0,
    w_prior_scale = 2.2,
    w_prior_variance = 0.005, # HARD frequency anchor: the NGMP fixed point has a low-pass bias
                           # (attenuated sine moments under-reward high frequencies), so
                           # unanchored ridge norms drift toward zero. Verified in the
                           # single-pair probe: anchored precision 200 recovers the true
                           # solution to the noise floor, precision 10 shrinks it.
    bias_prior_variance = 0.5, # looser: the pair phases must be able to rotate
    v_prior_scale = 0.5,       # needed cosine amplitude 0.5 = v·2(ρ/ω)sin(ωb) → v ≈ 0.39
    v_prior_variance = 0.01,   # tight: a diffuse q(v) shrinks every backward target by E[v]/E[v²]
    tau_prior = (1e3, 1.0),        # za tracks wᵀx tightly
    tau_c_prior = (1e4, 1.0),      # soft product ≈ hard product
    obs_noise_prior = (100.0, 1.0),  # mean 100 (true), ~100 pseudo-obs inertia — learned, but cannot race to absorb structure
    ngmp_alpha = 0.05,
    ngmp_beta = 0.0,               # keep 0: damping-only updates stay proper
    ngmp_max_step = 1.0,
    backward_projection = :closed_form,   # :closed_form | :unscented ablation
    prediction_iterations = 20,
    prediction_batch_size = 1_024,
    prediction_prior_variance = 1e12,
    mse_snapshots = 8,
    mse_subsample = 240,
    mse_subsample_seed = 2_028,
    grid_size = 60,
    # Data lives on [-2, 2]²; the extended evaluation grid deliberately leaves
    # that box to show extrapolation behavior (periodic mean continuation of
    # the cosine basis, leverage-driven variance growth x'Σ_w x).
    eval_extent = 4.0,
    checkboard_size = (2, 2),
)

script_start_time = time()

# ---------------------------------------------------------------------------
# Data (identical to notebooks/xor_softplus_ut_ngmp.jl)
# ---------------------------------------------------------------------------

function checkerboard_label(x1, x2, checkerboard_size)
    nx, ny = checkerboard_size
    nx > 0 && ny > 0 ||
        throw(ArgumentError("checkerboard dimensions must be positive"))
    cell_x = clamp(floor(Int, nx * (x1 + 2) / 4), 0, nx - 1)
    cell_y = clamp(floor(Int, ny * (x2 + 2) / 4), 0, ny - 1)
    return Float64(isodd(cell_x + cell_y))
end

function make_checkerboard_dataset(;
    n::Int = 1_600,
    checkerboard_size::Tuple{Int, Int} = (2, 2),
    noise_std::Float64 = 0.10,
    seed::Int = 1011,
)
    rng = StableRNG(seed)
    x1 = 4 .* rand(rng, n) .- 2
    x2 = 4 .* rand(rng, n) .- 2
    clean = checkerboard_label.(x1, x2, Ref(checkerboard_size))
    target = clamp.(clean .+ noise_std .* randn(rng, n), 0.0, 1.0)
    return DataFrame(x1 = x1, x2 = x2, OT = target)
end

function split_dataset(df; train_fraction = 0.30, seed = 42)
    rng = StableRNG(seed)
    indices = randperm(rng, nrow(df))
    n_train = round(Int, train_fraction * nrow(df))
    return df[indices[1:n_train], :], df[indices[(n_train + 1):end], :]
end

build_features(df) = [[1.0, df.x1[index], df.x2[index]] for index in 1:nrow(df)]

# ---------------------------------------------------------------------------
# Priors: symmetry broken in the PRIORS, not the initialization, and arranged
# as PAIRED DIFFERENCE UNITS. phi passes za through linearly, so a lone unit
# leaks the huge linear term v·(w·x) into `out`; at any fan of directions the
# leaks do not cancel, the misfit dwarfs the [0,1] target, and the cheapest
# free-energy descent shrinks every |w| — the constant-collapse observed with
# unpaired priors. Two units sharing one ridge direction with v = (+1, −1) and
# opposite biases ±b give
#
#     phi(a·s + b) − phi(a·s − b) = 2b + 2(rho/omega)·cos(omega a·s)·sin(omega b),
#
# i.e. the x-linear parts cancel EXACTLY (independently of the biases), leaving
# a pure cosine ridge feature with phase/amplitude controlled by the biases and
# the pair's v magnitude. Alternating the bias sign across pairs cancels the
# constant leak as well. Misaligned pairs can then prune through v → 0 without
# touching w. The bias coordinate gets a looser prior than the ridge direction
# so phases can rotate while the direction stays anchored.
# ---------------------------------------------------------------------------

function make_manyplus_priors(config)
    rng = StableRNG(config.prior_seed)
    n_neurons = config.n_neurons
    iseven(n_neurons) || throw(ArgumentError("paired priors require even n_neurons"))
    n_pairs = n_neurons ÷ 2
    prior_precision = Diagonal([
        1 / config.bias_prior_variance,
        1 / config.w_prior_variance,
        1 / config.w_prior_variance,
    ])
    w = Vector{Any}(undef, n_neurons)
    v = Vector{Any}(undef, n_neurons)
    for pair in 1:n_pairs
        angle = π * (pair - 1) / n_pairs + 0.1 * randn(rng)
        radius = config.w_prior_scale * (1 + 0.05 * randn(rng))
        direction = [radius * cos(angle), radius * sin(angle)]
        bias = (isodd(pair) ? 1.0 : -1.0) * (π / 4) * (1 + 0.1 * randn(rng))
        for (slot, sign) in ((2pair - 1, 1.0), (2pair, -1.0))
            prior_mean = [sign * bias, direction[1], direction[2]]
            # MvNormalWeightedMeanPrecision expects ξ = Λμ, not μ itself.
            w[slot] = MvNormalWeightedMeanPrecision(
                prior_precision * prior_mean, prior_precision,
            )
            v[slot] = NormalMeanVariance(sign * config.v_prior_scale, config.v_prior_variance)
        end
    end
    return Dict{Symbol, Any}(
        :w => w,
        :v => v,
        :τ => GammaShapeRate(config.tau_prior...),
        :τ_c => GammaShapeRate(config.tau_c_prior...),
        :obs_noise => GammaShapeRate(config.obs_noise_prior...),
    )
end

backward_projection(config) =
    config.backward_projection === :unscented ?
        TangentProjection(type = Unscented) :
        TangentProjection(type = ClosedForm)

# Fresh per infer call: the edge states inside are mutable and belong to
# exactly one inference graph.
make_activation_dependencies(config) = NGMPDependencies(
    out = nothing,
    in = nothing,
    projection = backward_projection(config),
    damping = DampingMeta(
        alpha = config.ngmp_alpha,
        beta = config.ngmp_beta,
        max_step = config.ngmp_max_step,
    ),
)

# ---------------------------------------------------------------------------
# Training model
# ---------------------------------------------------------------------------

@model function xor_manyplus_residual_sine(
    n_neurons,
    features,
    y,
    priors,
    activation,
    activation_deps,
)
    local w, v, za, h, c, out

    τ ~ priors[:τ]
    τ_c ~ priors[:τ_c]
    obs_noise ~ priors[:obs_noise]

    for neuron in 1:n_neurons
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
    end

    for observation in eachindex(y)
        for neuron in 1:n_neurons
            za[neuron, observation] ~
                softdot(features[observation], w[neuron], τ)
            h[neuron, observation] ~ ResidualSine(za[neuron, observation]) where {
                dependencies = activation_deps,
                meta = activation,
            }
            c[neuron, observation] ~
                softdot(v[neuron], h[neuron, observation], τ_c)
        end
        out[observation] ~ ManyPlus(
            inputs = [c[neuron, observation] for neuron in 1:n_neurons],
        )
        y[observation] ~ NormalMeanPrecision(out[observation], obs_noise)
    end
end

# One structured cluster along the deterministic chain (deterministic nodes
# cannot straddle a factorization cut); the scalar soft product requires
# exactly q(c, h) joint with q(v) and q(τ_c) mean-field — the only shape the
# stock structured softdot rules support.
@constraints function xor_manyplus_constraints()
    q(w, v, za, h, c, out, τ, τ_c, obs_noise) =
        q(w, za, h, c, out)q(v)q(τ)q(τ_c)q(obs_noise)

    # softdot repeatedly consumes the same weight means and covariances.
    q(w)::MomentForm()
end

# Coupled initialization: every latent on the deterministic chain starts at the
# PRIOR PUSHFORWARD of the weight means, not at a generic N(0, 1). The NGMP
# backward sites are projected at the current marginals; projecting them at
# marginals inconsistent with the prior-mean ridges makes the first softdot
# least-squares snap w to zero (verified in the single-pair probe).
@initialization function xor_manyplus_initialization(priors, inits)
    q(v) = deepcopy(priors[:v])
    q(za) = inits.za
    q(h) = inits.h
    q(c) = inits.c
    q(out) = inits.out
    q(τ) = priors[:τ]
    q(τ_c) = priors[:τ_c]
    q(obs_noise) = priors[:obs_noise]
    μ(w) = deepcopy(priors[:w])
end

function pushforward_inits(priors, features, config)
    n_neurons = config.n_neurons
    n = length(features)
    activation = ResidualSineMeta(rho = config.phi_rho, omega = config.phi_omega)
    phi(x) = SurrogateModelling._residual_sine(x, activation)
    w_means = [mean(prior) for prior in priors[:w]]
    v_means = [mean(prior) for prior in priors[:v]]
    za = [
        NormalMeanVariance(dot(w_means[k], features[i]), 0.5)
        for k in 1:n_neurons, i in 1:n
    ]
    h = [
        NormalMeanVariance(phi(mean(za[k, i])), 1.0)
        for k in 1:n_neurons, i in 1:n
    ]
    c = [
        NormalMeanVariance(v_means[k] * mean(h[k, i]), 1.0)
        for k in 1:n_neurons, i in 1:n
    ]
    out = [
        NormalMeanVariance(sum(mean(c[k, i]) for k in 1:n_neurons), 1.0)
        for i in 1:n
    ]
    return (za = za, h = h, c = c, out = out)
end

function run_manyplus_training(observations, features, config; showprogress = true)
    priors = make_manyplus_priors(config)
    activation = ResidualSineMeta(rho = config.phi_rho, omega = config.phi_omega)
    result = infer(
        model = xor_manyplus_residual_sine(
            n_neurons = config.n_neurons,
            priors = priors,
            activation = activation,
            activation_deps = make_activation_dependencies(config),
        ),
        data = (y = observations, features = features),
        constraints = xor_manyplus_constraints(),
        initialization = xor_manyplus_initialization(
            priors, pushforward_inits(priors, features, config),
        ),
        iterations = config.iterations,
        free_energy = true,
        showprogress = showprogress,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    return (result = result, priors = priors)
end

# ---------------------------------------------------------------------------
# Prediction graph: same chain, learned globals clamped, y latent with the
# model likelihood plus a diffuse pseudo-prior (the baseline's pattern).
# ---------------------------------------------------------------------------

@model function xor_manyplus_prediction(
    n_neurons,
    features,
    priors,
    activation,
    activation_deps,
    y_prior_variance,
)
    local w, v, za, h, c, out, y

    τ ~ priors[:τ]
    τ_c ~ priors[:τ_c]
    obs_noise ~ priors[:obs_noise]

    for neuron in 1:n_neurons
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
    end

    for observation in eachindex(features)
        for neuron in 1:n_neurons
            za[neuron, observation] ~
                softdot(features[observation], w[neuron], τ)
            h[neuron, observation] ~ ResidualSine(za[neuron, observation]) where {
                dependencies = activation_deps,
                meta = activation,
            }
            c[neuron, observation] ~
                softdot(v[neuron], h[neuron, observation], τ_c)
        end
        out[observation] ~ ManyPlus(
            inputs = [c[neuron, observation] for neuron in 1:n_neurons],
        )
        y[observation] ~ NormalMeanPrecision(out[observation], obs_noise)
        y[observation] ~ NormalMeanVariance(0.0, y_prior_variance)
    end
end

@constraints function xor_manyplus_prediction_constraints(priors)
    q(w, v, za, h, c, out, τ, τ_c, obs_noise, y) =
        q(w)q(v)q(τ)q(τ_c)q(obs_noise)q(za, h, c, out, y)

    q(τ)::RxInfer.FixedMarginalFormConstraint(priors[:τ])
    q(τ_c)::RxInfer.FixedMarginalFormConstraint(priors[:τ_c])
    q(obs_noise)::RxInfer.FixedMarginalFormConstraint(priors[:obs_noise])

    for (neuron, prior) in enumerate(priors[:w])
        q(w[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    for (neuron, prior) in enumerate(priors[:v])
        q(v[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
end

@initialization function xor_manyplus_prediction_initialization(
    priors,
    inits,
    output_mean,
    y_prior_variance,
)
    q(w) = deepcopy(priors[:w])
    q(v) = deepcopy(priors[:v])
    q(za) = inits.za
    q(h) = inits.h
    q(c) = inits.c
    q(out) = inits.out
    q(τ) = priors[:τ]
    q(τ_c) = priors[:τ_c]
    q(obs_noise) = priors[:obs_noise]
    q(y) = NormalMeanVariance(output_mean, y_prior_variance)

    μ(out) = NormalMeanVariance(output_mean, 10.0)
    μ(y) = NormalMeanVariance(output_mean, y_prior_variance)
end

function manyplus_prediction_priors(result, iteration)
    checkbounds(result.posteriors[:w], iteration)
    return Dict{Symbol, Any}(
        :w => deepcopy(result.posteriors[:w][iteration]),
        :v => deepcopy(result.posteriors[:v][iteration]),
        :τ => deepcopy(result.posteriors[:τ][iteration]),
        :τ_c => deepcopy(result.posteriors[:τ_c][iteration]),
        :obs_noise => deepcopy(result.posteriors[:obs_noise][iteration]),
    )
end

function run_manyplus_prediction_batch(priors, features, config; output_mean)
    isempty(features) && return Any[]
    activation = ResidualSineMeta(rho = config.phi_rho, omega = config.phi_omega)
    result = infer(
        model = xor_manyplus_prediction(
            n_neurons = config.n_neurons,
            priors = priors,
            activation = activation,
            activation_deps = make_activation_dependencies(config),
            y_prior_variance = config.prediction_prior_variance,
        ),
        data = (features = features,),
        constraints = xor_manyplus_prediction_constraints(priors),
        initialization = xor_manyplus_prediction_initialization(
            priors,
            pushforward_inits(priors, features, config),
            output_mean,
            config.prediction_prior_variance,
        ),
        iterations = config.prediction_iterations,
        free_energy = false,
        showprogress = false,
        returnvars = (y = KeepLast(),),
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    marginals = collect(vec(result.posteriors[:y]))
    length(marginals) == length(features) ||
        error("prediction graph returned the wrong number of y marginals")
    return marginals
end

function predict_manyplus_marginals(priors, features, config; output_mean)
    marginals = Vector{Any}(undef, length(features))
    batch_size = config.prediction_batch_size
    for first_index in 1:batch_size:length(features)
        indices = first_index:min(first_index + batch_size - 1, length(features))
        marginals[indices] = run_manyplus_prediction_batch(
            priors, features[indices], config; output_mean = output_mean,
        )
    end
    return marginals
end

function predictive_statistics(marginals)
    means = Float64.(mean.(marginals))
    variances = Float64.(var.(marginals))
    all(isfinite, means) || error("prediction graph produced a non-finite mean")
    all(v -> isfinite(v) && v > 0, variances) ||
        error("prediction graph produced a non-positive or non-finite variance")
    return (mean = means, variance = variances)
end

function thinned_iterations(n_iterations, n_frames)
    n_frames == 1 && return [n_iterations]
    return unique(round.(Int, range(1, n_iterations; length = min(n_iterations, n_frames))))
end

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

dataset = make_checkerboard_dataset(
    n = config.n_samples,
    noise_std = config.noise_std,
    seed = config.data_seed,
    checkerboard_size = config.checkboard_size,
)
train_data, test_data = split_dataset(
    dataset;
    train_fraction = config.train_fraction,
    seed = config.split_seed,
)
train_features = build_features(train_data)
test_features = build_features(test_data)

println("Training: $(nrow(train_data)) points, $(config.n_neurons) neurons, ",
    "$(config.iterations) iterations, backward = $(config.backward_projection)")
training_time = @elapsed ngmp_fit = run_manyplus_training(
    train_data.OT, train_features, config; showprogress = false,
)
println("Training finished in $(round(training_time; digits = 1)) s; ",
    "final free energy = $(round(ngmp_fit.result.free_energy[end]; digits = 3))")

output_mean = mean(train_data.OT)

# Learning curve on a fixed test subsample at thinned training iterations.
n_recorded = length(ngmp_fit.result.posteriors[:w])
snapshot_iterations = thinned_iterations(n_recorded, config.mse_snapshots)
mse_subsample_indices = let
    rng = StableRNG(config.mse_subsample_seed)
    n_subsample = min(config.mse_subsample, length(test_features))
    sort(randperm(rng, length(test_features))[1:n_subsample])
end
mse_features = test_features[mse_subsample_indices]
mse_targets = test_data.OT[mse_subsample_indices]

learning_curve_time = @elapsed mse_by_iteration = map(snapshot_iterations) do training_iteration
    priors = manyplus_prediction_priors(ngmp_fit.result, training_iteration)
    statistics = predictive_statistics(predict_manyplus_marginals(
        priors, mse_features, config; output_mean = output_mean,
    ))
    mse = mean(abs2, statistics.mean .- mse_targets)
    println("  iteration $training_iteration: subsample test MSE = ",
        round(mse; digits = 4))
    mse
end
println("Learning-curve predictions ($(length(snapshot_iterations)) snapshots × ",
    "$(length(mse_features)) points): $(round(learning_curve_time; digits = 1)) s")
constant_mse = mean(abs2, output_mean .- mse_targets)
constant_mse_full = mean(abs2, output_mean .- test_data.OT)

# Final evaluation on the full test set.
final_priors = manyplus_prediction_priors(ngmp_fit.result, n_recorded)
test_eval_time = @elapsed final_test = let
    statistics = predictive_statistics(predict_manyplus_marginals(
        final_priors, test_features, config; output_mean = output_mean,
    ))
    (
        mean = statistics.mean,
        variance = statistics.variance,
        mse = mean(abs2, statistics.mean .- test_data.OT),
    )
end
println("Full test-set prediction ($(length(test_features)) points): ",
    round(test_eval_time; digits = 1), " s")

# Predictive surfaces: the data-region grid and an extended grid that leaves
# the observed box to expose extrapolation behavior.
function grid_predictions(extent, final_priors, config; output_mean)
    axis = range(-extent, extent; length = config.grid_size)
    grid_features = [[1.0, x1, x2] for x2 in axis for x1 in axis]
    statistics = predictive_statistics(predict_manyplus_marginals(
        final_priors, grid_features, config; output_mean = output_mean,
    ))
    return (
        axis = axis,
        mean = reshape(statistics.mean, config.grid_size, config.grid_size),
        variance = reshape(statistics.variance, config.grid_size, config.grid_size),
    )
end

grid_time = @elapsed data_grid = grid_predictions(
    2.0, final_priors, config; output_mean = output_mean,
)
extended_grid_time = @elapsed extended_grid = grid_predictions(
    config.eval_extent, final_priors, config; output_mean = output_mean,
)
println("Grid predictions (2 × $(config.grid_size)²): ",
    round(grid_time; digits = 1), " + ", round(extended_grid_time; digits = 1), " s")

grid_axis = data_grid.axis
grid_mean = data_grid.mean
grid_variance = data_grid.variance
grid_truth = [
    checkerboard_label(x1, x2, config.checkboard_size)
    for x2 in grid_axis, x1 in grid_axis
]
# The checkerboard is only defined where data was generated; mask the outside
# so the extended truth panel does not extrapolate the clamped edge cells.
extended_truth = [
    max(abs(x1), abs(x2)) <= 2 ?
        checkerboard_label(x1, x2, config.checkboard_size) : NaN
    for x2 in extended_grid.axis, x1 in extended_grid.axis
]

# ---------------------------------------------------------------------------
# Summaries and figures
# ---------------------------------------------------------------------------

learned_weights = DataFrame(
    neuron = 1:config.n_neurons,
    v_mean = [mean(final_priors[:v][k]) for k in 1:config.n_neurons],
    v_std = [sqrt(var(final_priors[:v][k])) for k in 1:config.n_neurons],
    w_bias = [mean(final_priors[:w][k])[1] for k in 1:config.n_neurons],
    w_x1 = [mean(final_priors[:w][k])[2] for k in 1:config.n_neurons],
    w_x2 = [mean(final_priors[:w][k])[3] for k in 1:config.n_neurons],
    ridge_norm = [norm(mean(final_priors[:w][k])[2:3]) for k in 1:config.n_neurons],
)

run_summary = DataFrame(
    neurons = config.n_neurons,
    iterations = n_recorded,
    train_points = nrow(train_data),
    test_points = nrow(test_data),
    training_seconds = round(training_time; digits = 1),
    test_mse = final_test.mse,
    constant_mse = constant_mse_full,
    minimum_predictive_variance = minimum(final_test.variance),
    mean_predictive_variance = mean(final_test.variance),
    maximum_predictive_variance = maximum(final_test.variance),
    variance_ratio = maximum(final_test.variance) / minimum(final_test.variance),
    softdot_precision = mean(ngmp_fit.result.posteriors[:τ][end]),
    product_precision = mean(ngmp_fit.result.posteriors[:τ_c][end]),
    observation_precision = mean(ngmp_fit.result.posteriors[:obs_noise][end]),
)

println("\n=== learned weights ===")
println(learned_weights)
println("\n=== run summary ===")
println(run_summary)

CSV.write(joinpath(OUTPUT_DIR, "run_summary.csv"), run_summary)
CSV.write(joinpath(OUTPUT_DIR, "learned_weights.csv"), learned_weights)
CSV.write(
    joinpath(OUTPUT_DIR, "mse_history.csv"),
    DataFrame(iteration = snapshot_iterations, test_mse = mse_by_iteration),
)

learning_curve = plot(
    snapshot_iterations,
    mse_by_iteration;
    marker = :circle,
    linewidth = 2,
    color = :steelblue,
    xlabel = "Iteration",
    ylabel = "Test MSE ($(length(mse_targets))-point subsample)",
    title = "ManyPlus residual-sine NGMP learning",
    label = "ManyPlus additive",
)
hline!(learning_curve, [constant_mse];
    color = :black, linestyle = :dash, label = "constant baseline")
savefig(learning_curve, joinpath(OUTPUT_DIR, "learning_curve.png"))

free_energy_plot = plot(
    eachindex(ngmp_fit.result.free_energy),
    ngmp_fit.result.free_energy;
    linewidth = 2,
    color = :darkorange,
    xlabel = "Iteration",
    ylabel = "Bethe free energy",
    legend = false,
)
savefig(free_energy_plot, joinpath(OUTPUT_DIR, "free_energy.png"))

# Symmetric color limits around 0.5 wide enough for the smooth cosine basis'
# overshoot (Gibbs-style ringing past 0 and 1 at cell centers) — GR leaves
# out-of-clims regions unpainted, which previously showed as white holes.
function mean_clims(values...)
    lo = min(0.0, minimum(minimum, values))
    hi = max(1.0, maximum(maximum, values))
    margin = max(0.5 - lo, hi - 0.5)
    return (0.5 - margin, 0.5 + margin)
end

data_region!(panel) = plot!(
    panel,
    [-2, 2, 2, -2, -2],
    [-2, -2, 2, 2, -2];
    seriestype = :path,
    linecolor = :black,
    linestyle = :dash,
    linewidth = 1.5,
    label = "",
)

surface_plot = plot(
    contourf(grid_axis, grid_axis, grid_mean;
        color = :RdBu, clims = mean_clims(grid_mean), levels = 21,
        title = "Predictive mean"),
    contourf(grid_axis, grid_axis, grid_variance;
        color = :viridis, title = "Predictive variance"),
    heatmap(grid_axis, grid_axis, grid_truth;
        color = :RdBu, clims = (0, 1), title = "Clean target");
    layout = (1, 3),
    size = (1500, 420),
    xlabel = "x1",
    ylabel = "x2",
)
savefig(surface_plot, joinpath(OUTPUT_DIR, "final_surfaces.png"))

extended_mean_panel = contourf(
    extended_grid.axis, extended_grid.axis, extended_grid.mean;
    color = :RdBu, clims = mean_clims(extended_grid.mean), levels = 21,
    title = "Predictive mean (extrapolation)",
)
data_region!(extended_mean_panel)
extended_variance_panel = contourf(
    extended_grid.axis, extended_grid.axis, extended_grid.variance;
    color = :viridis, title = "Predictive variance (extrapolation)",
)
data_region!(extended_variance_panel)
extended_truth_panel = heatmap(
    extended_grid.axis, extended_grid.axis, extended_truth;
    color = :RdBu, clims = (0, 1), title = "Clean target (data region only)",
)
data_region!(extended_truth_panel)
extended_surface_plot = plot(
    extended_mean_panel,
    extended_variance_panel,
    extended_truth_panel;
    layout = (1, 3),
    size = (1500, 420),
    xlabel = "x1",
    ylabel = "x2",
)
savefig(extended_surface_plot, joinpath(OUTPUT_DIR, "final_surfaces_extended.png"))

println("\nSaved figures and CSVs to $(OUTPUT_DIR)")
println("Extended grid (±$(config.eval_extent)) variance range: ",
    round(minimum(extended_grid.variance); sigdigits = 3), " – ",
    round(maximum(extended_grid.variance); sigdigits = 3))
println("Total script time: $(round(time() - script_start_time; digits = 1)) s")
println("Predictive variance range: ",
    round(minimum(final_test.variance); sigdigits = 3), " – ",
    round(maximum(final_test.variance); sigdigits = 3),
    " (ratio ", round(run_summary.variance_ratio[1]; sigdigits = 3), ")")
