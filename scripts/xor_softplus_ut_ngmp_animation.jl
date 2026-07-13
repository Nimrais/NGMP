# Checkerboard Softplus UT NGMP — learning animation
#
# Self-contained companion to `notebooks/xor_softplus_ut_ngmp.jl`. The notebook
# keeps only the cheap diagnostics (subsampled MSE curve + one small final
# surface); this script does the expensive visualisation work:
#
#   1. trains the Softplus UT NGMP model (~1 min at defaults),
#   2. re-runs the RxInfer prediction graph at `animation_frames` thinned
#      training iterations over the test set plus an
#      `animation_grid_size`² grid (this is the dominant cost — roughly
#      15–20 min at defaults),
#   3. renders a GIF (predictive mean / variance / clean target / test MSE)
#      into `viz/`,
#   4. renders one final-iteration surface at `final_grid_size`² resolution
#      and saves it as a PNG into `viz/`.
#
# Knobs to play with live in `config` below. The cheap ones are
# `checkboard_size`, `n_neurons`, `iterations` and the NGMP damping
# parameters; the expensive ones are `animation_frames`,
# `animation_grid_size` and `final_grid_size` (cost scales with
# frames × grid² and grid² respectively). Run from anywhere:
#
#   julia scripts/xor_softplus_ut_ngmp_animation.jl

begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

begin
    using DataFrames
    using ExponentialFamily
    using LinearAlgebra: Diagonal
    using StatsPlots
    using ProbabilisticEnsembling
    using Random
    using StableRNGs
    using RxInfer
    using Statistics
    using SurrogateModelling
end

# ── Configuration ────────────────────────────────────────────────────────────

config = (
    # Data
    n_samples = 1_600,
    train_fraction = 0.40,
    noise_std = 0.10,
    checkboard_size = (1, 2),
    data_seed = 2_026,
    split_seed = 2_027,

    # Model / training
    n_neurons = 8,
    iterations = 100,
    prior_seed = 42,
    mean_prior_precision = 1e-4,
    gate_prior_precision = 1.0,
    gamma_rate_prior_shape = 10.0,
    gamma_rate_prior_rate = 10.0,

    # NGMP damping
    ngmp_alpha = 0.2,
    ngmp_beta = 0.0,
    ngmp_max_step = 1.0,

    # Prediction graph
    prediction_iterations = 20,
    prediction_batch_size = 1_024,
    prediction_prior_variance = 1e12,

    # Animation (the expensive part: cost ≈ frames × grid²)
    animation_frames = 12,
    animation_grid_size = 60,
    animation_fps = 5,

    # Final high-resolution surface (single prediction run, cost ≈ grid²)
    final_grid_size = 160,

    output_dir = joinpath(@__DIR__, "..", "viz"),
)

# ── Data ─────────────────────────────────────────────────────────────────────

function checkerboard_label(x1, x2, checkerboard_size)
    nx, ny = checkerboard_size
    nx > 0 && ny > 0 ||
        throw(ArgumentError("checkerboard dimensions must be positive"))
    cell_x = clamp(floor(Int, nx * (x1 + 2) / 4), 0, nx - 1)
    cell_y = clamp(floor(Int, ny * (x2 + 2) / 4), 0, ny - 1)
    return Float64(isodd(cell_x + cell_y))
end

function make_checkerboard_dataset(
    ;
    n::Int = 1_600,
    checkerboard_size::Tuple{Int, Int} = (2, 2),
    noise_std::Float64 = 0.10,
    seed::Int = 1011,
)
    nx, ny = checkerboard_size
    nx > 0 && ny > 0 ||
        throw(ArgumentError("checkerboard dimensions must be positive"))

    rng = StableRNG(seed)
    x1 = 4 .* rand(rng, n) .- 2
    x2 = 4 .* rand(rng, n) .- 2

    clean = checkerboard_label.(x1, x2, Ref(checkerboard_size))
    target = clamp.(clean .+ noise_std .* randn(rng, n), 0.0, 1.0)

    return DataFrame(x1 = x1, x2 = x2, OT = target)
end

function split_dataset(df; train_fraction = 0.30, seed = 42)
    0 < train_fraction < 1 ||
        throw(ArgumentError("train_fraction must be in (0, 1)"))
    rng = StableRNG(seed)
    indices = randperm(rng, nrow(df))
    n_train = round(Int, train_fraction * nrow(df))
    return df[indices[1:n_train], :], df[indices[(n_train + 1):end], :]
end

build_features(df) = [[1.0, df.x1[index], df.x2[index]] for index in 1:nrow(df)]

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

# ── Training model ───────────────────────────────────────────────────────────

@model function xor_softplus_ut_ngmp(
    n_neurons,
    features,
    y,
    priors,
    dependencies,
    damping,
    obs_dependencies,
    obs_damping
)
    local w_mean, w_a, z_mean, za, γ, τ, τ_mean, obs_noise, out, β

    τ ~ priors[:τ]
    τ_mean ~ priors[:τ_mean]
    obs_noise ~ priors[:obs_noise]
    β ~ priors[:β]

    for neuron in 1:n_neurons
        w_mean[neuron] ~ priors[:w_mean][neuron]
        w_a[neuron] ~ priors[:w_a][neuron]
    end

    for observation in eachindex(y)
        for neuron in 1:n_neurons
            z_mean[neuron, observation] ~
                softdot(features[observation], w_mean[neuron], τ_mean)
            za[neuron, observation] ~
                softdot(features[observation], w_a[neuron], τ) where {
                    meta = LowRankMeta(),
                }
            γ[neuron, observation] ~ GammaShapeRate(1.0, β)
            γ[neuron, observation] ~ Softplus(za[neuron, observation]) where {
                dependencies = dependencies,
                meta = damping,
            }
            out[observation] ~ NormalMeanPrecision(
                z_mean[neuron, observation],
                γ[neuron, observation],
            ) where {
                dependencies = obs_dependencies,
                meta = obs_damping
            }
        end
        y[observation] ~ NormalMeanPrecision(out[observation], obs_noise)
    end
end

@constraints function xor_softplus_ut_constraints()
    q(w_mean, w_a, z_mean, za, γ, τ, τ_mean, out, obs_noise, β) =
        q(w_mean, z_mean, out, za, γ)q(w_a)q(τ)q(τ_mean)q(obs_noise)q(β)

    # softdot repeatedly consumes the same weight means and covariances.
    q(w_mean)::MomentForm()
    q(w_a)::MomentForm()
end

@initialization function xor_softplus_ut_initialization(priors, output_mean)
    q(w_a) = deepcopy(priors[:w_a])
    q(z_mean) = NormalMeanVariance(output_mean, 1.0)
    q(out) = NormalMeanVariance(output_mean, 1.0)
    q(za) = NormalMeanVariance(0.0, 1.0)
    q(γ) = GammaShapeScale(2.0, 1.0)
    q(τ) = priors[:τ]
    q(obs_noise) = priors[:obs_noise]
    q(τ_mean) = priors[:τ_mean]
    q(β) = priors[:β]
    μ(w_mean) = deepcopy(priors[:w_mean])
end

function make_softplus_priors(
    ;
    n_neurons,
    n_features = 3,
    seed = 42,
    mean_prior_precision = 1e-4,
    gate_prior_precision = 1e-4,
    gamma_rate_prior_shape = 10.0,
    gamma_rate_prior_rate = 10.0,
)
    rng = MersenneTwister(seed)
    mean_precision = Diagonal(fill(mean_prior_precision, n_features))
    gate_precision = Diagonal(fill(gate_prior_precision, n_features))

    # MvNormalWeightedMeanPrecision expects ξ = Λμ, not μ itself.
    w_mean = [
        MvNormalWeightedMeanPrecision(
            mean_precision * randn(rng, n_features),
            mean_precision,
        ) for _ in 1:n_neurons
    ]
    w_a = [
        MvNormalWeightedMeanPrecision(
            gate_precision * randn(rng, n_features),
            gate_precision,
        ) for _ in 1:n_neurons
    ]

    return Dict{Symbol, Any}(
        :w_mean => w_mean,
        :w_a => w_a,
        :τ => GammaShapeRate(1e3, 1.0),
        :τ_mean => GammaShapeRate(1e4, 1.0),
        :obs_noise => GammaShapeRate(1e6, 1.0),
        :β => GammaShapeRate(
            gamma_rate_prior_shape,
            gamma_rate_prior_rate,
        ),
    )
end

function run_softplus_ut_ngmp(
    observations,
    features;
    n_neurons,
    iterations,
    prior_seed,
    mean_prior_precision,
    gate_prior_precision,
    gamma_rate_prior_shape,
    gamma_rate_prior_rate,
    ngmp_alpha,
    ngmp_beta,
    ngmp_max_step,
    showprogress = true,
)
    priors = make_softplus_priors(
        n_neurons = n_neurons,
        seed = prior_seed,
        mean_prior_precision = mean_prior_precision,
        gate_prior_precision = gate_prior_precision,
        gamma_rate_prior_shape = gamma_rate_prior_shape,
        gamma_rate_prior_rate = gamma_rate_prior_rate,
    )

    # These objects are deliberately fresh for every call. Their edge states
    # are mutable and belong to exactly one inference graph.
    dependencies = NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = TangentProjection(type = Unscented),
    )
    damping = DampingMeta(
        alpha = ngmp_alpha,
        beta = ngmp_beta,
        max_step = ngmp_max_step,
    )

    obs_dependencies = NGMPDependencies(
        out = nothing,
        μ = nothing,
        τ = nothing,
        projection = TangentProjection(type = Unscented),
    )

    obs_damping = DampingMeta(
        alpha = ngmp_alpha,
        beta = ngmp_beta,
        max_step = ngmp_max_step,
    )

    inference_model = xor_softplus_ut_ngmp(
        n_neurons = n_neurons,
        priors = priors,
        dependencies = dependencies,
        damping = damping,
        obs_dependencies = obs_dependencies,
        obs_damping = obs_damping
    )

    output_mean = mean(observations)

    result = infer(
        model = inference_model,
        data = (y = observations, features = features),
        constraints = xor_softplus_ut_constraints(),
        initialization = xor_softplus_ut_initialization(priors, output_mean),
        iterations = iterations,
        free_energy = true,
        showprogress = showprogress,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )

    return (
        result = result,
    )
end

# ── Prediction model ─────────────────────────────────────────────────────────

@model function xor_softplus_ut_ngmp_prediction(
    n_neurons,
    features,
    priors,
    dependencies,
    damping,
    obs_dependencies,
    obs_damping,
    y_prior_variance,
)
    local w_mean, w_a, z_mean, za, γ, τ, τ_mean, obs_noise, out, β, y

    τ ~ priors[:τ]
    τ_mean ~ priors[:τ_mean]
    obs_noise ~ priors[:obs_noise]
    β ~ priors[:β]

    for neuron in 1:n_neurons
        w_mean[neuron] ~ priors[:w_mean][neuron]
        w_a[neuron] ~ priors[:w_a][neuron]
    end

    for observation in eachindex(features)
        for neuron in 1:n_neurons
            z_mean[neuron, observation] ~
                softdot(features[observation], w_mean[neuron], τ_mean)
            za[neuron, observation] ~
                softdot(features[observation], w_a[neuron], τ) where {
                    meta = LowRankMeta(),
                }
            γ[neuron, observation] ~ GammaShapeRate(1.0, β)
            γ[neuron, observation] ~ Softplus(za[neuron, observation]) where {
                dependencies = dependencies,
                meta = damping,
            }
            out[observation] ~ NormalMeanPrecision(
                z_mean[neuron, observation],
                γ[neuron, observation],
            ) where {
                dependencies = obs_dependencies,
                meta = obs_damping,
            }
        end
        y[observation] ~ NormalMeanPrecision(out[observation], obs_noise)
        y[observation] ~ NormalMeanVariance(0.0, y_prior_variance)
    end
end

@constraints function xor_softplus_ut_prediction_constraints(priors)
    # Keep the predictive output chain Gaussian while updating each gate through
    # its own structured Softplus belief.
    q(w_mean, w_a, z_mean, za, γ, τ, τ_mean, out, obs_noise, β, y) =
        q(w_mean)q(w_a)q(τ)q(τ_mean)q(obs_noise)q(β)q(z_mean, out, y)q(za, γ)

    q(τ)::RxInfer.FixedMarginalFormConstraint(priors[:τ])
    q(τ_mean)::RxInfer.FixedMarginalFormConstraint(priors[:τ_mean])
    q(obs_noise)::RxInfer.FixedMarginalFormConstraint(priors[:obs_noise])
    q(β)::RxInfer.FixedMarginalFormConstraint(priors[:β])

    for (neuron, prior) in enumerate(priors[:w_mean])
        q(w_mean[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    for (neuron, prior) in enumerate(priors[:w_a])
        q(w_a[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
end

@initialization function xor_softplus_ut_prediction_initialization(
    priors,
    output_mean,
    y_prior_variance,
)
    q(w_mean) = deepcopy(priors[:w_mean])
    q(w_a) = deepcopy(priors[:w_a])
    q(z_mean) = NormalMeanVariance(output_mean, 1.0)
    q(out) = NormalMeanVariance(output_mean, 1.0)
    q(za) = NormalMeanVariance(0.0, 1.0)
    q(γ) = GammaShapeScale(2.0, 1.0)
    q(τ) = priors[:τ]
    q(τ_mean) = priors[:τ_mean]
    q(obs_noise) = priors[:obs_noise]
    q(β) = priors[:β]
    q(y) = NormalMeanVariance(output_mean, y_prior_variance)

    μ(z_mean) = NormalMeanVariance(output_mean, 10.0)
    μ(out) = NormalMeanVariance(output_mean, 10.0)
    μ(y) = NormalMeanVariance(output_mean, y_prior_variance)
end

function softplus_prediction_priors(result, iteration)
    checkbounds(result.posteriors[:w_mean], iteration)
    return Dict{Symbol, Any}(
        :w_mean => deepcopy(result.posteriors[:w_mean][iteration]),
        :w_a => deepcopy(result.posteriors[:w_a][iteration]),
        :τ => deepcopy(result.posteriors[:τ][iteration]),
        :τ_mean => deepcopy(result.posteriors[:τ_mean][iteration]),
        :obs_noise => deepcopy(result.posteriors[:obs_noise][iteration]),
        :β => deepcopy(result.posteriors[:β][iteration]),
    )
end

function run_softplus_prediction_batch(
    priors,
    features;
    n_neurons,
    iterations,
    output_mean,
    y_prior_variance,
    ngmp_alpha,
    ngmp_beta,
    ngmp_max_step,
)
    isempty(features) && return Any[]

    dependencies = NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = TangentProjection(type = Unscented),
    )
    obs_dependencies = NGMPDependencies(
        τ = nothing,
        projection = TangentProjection(type = Unscented),
    )
    damping = DampingMeta(
        alpha = ngmp_alpha,
        beta = ngmp_beta,
        max_step = ngmp_max_step,
    )
    obs_damping = DampingMeta(
        alpha = ngmp_alpha,
        beta = ngmp_beta,
        max_step = ngmp_max_step,
    )

    result = infer(
        model = xor_softplus_ut_ngmp_prediction(
            n_neurons = n_neurons,
            priors = priors,
            dependencies = dependencies,
            damping = damping,
            obs_dependencies = obs_dependencies,
            obs_damping = obs_damping,
            y_prior_variance = y_prior_variance,
        ),
        data = (features = features,),
        constraints = xor_softplus_ut_prediction_constraints(priors),
        initialization = xor_softplus_ut_prediction_initialization(
            priors,
            output_mean,
            y_prior_variance,
        ),
        iterations = iterations,
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

function predict_softplus_marginals(
    priors,
    features;
    batch_size,
    kwargs...,
)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    marginals = Vector{Any}(undef, length(features))

    for first_index in 1:batch_size:length(features)
        indices = first_index:min(first_index + batch_size - 1, length(features))
        marginals[indices] = run_softplus_prediction_batch(
            priors,
            features[indices];
            kwargs...,
        )
    end
    return marginals
end

function predictive_statistics(marginals)
    means = Float64.(mean.(marginals))
    variances = Float64.(var.(marginals))
    all(isfinite, means) || error("prediction graph produced a non-finite mean")
    all(variance -> isfinite(variance) && variance > 0, variances) ||
        error("prediction graph produced a non-positive or non-finite variance")
    return (mean = means, variance = variances)
end

function thinned_iterations(n_iterations, n_frames)
    n_iterations > 0 || throw(ArgumentError("n_iterations must be positive"))
    n_frames > 0 || throw(ArgumentError("n_frames must be positive"))
    n_frames == 1 && return [n_iterations]
    return unique(round.(Int, range(
        1,
        n_iterations;
        length = min(n_iterations, n_frames),
    )))
end

make_grid(grid_size, checkerboard_size) = let
    x = range(-2.0, 2.0; length = grid_size)
    y = range(-2.0, 2.0; length = grid_size)
    actual = [
        checkerboard_label(x_value, y_value, checkerboard_size) for
        y_value in y, x_value in x
    ]
    features = vec([
        [1.0, x_value, y_value] for y_value in y, x_value in x
    ])
    (x = x, y = y, actual = actual, features = features)
end

# ── Training ─────────────────────────────────────────────────────────────────

@info "Training Softplus UT NGMP model" config.n_neurons config.iterations
training_seconds = @elapsed ngmp_fit = run_softplus_ut_ngmp(
    train_data.OT,
    train_features;
    n_neurons = config.n_neurons,
    iterations = config.iterations,
    prior_seed = config.prior_seed,
    mean_prior_precision = config.mean_prior_precision,
    gate_prior_precision = config.gate_prior_precision,
    gamma_rate_prior_shape = config.gamma_rate_prior_shape,
    gamma_rate_prior_rate = config.gamma_rate_prior_rate,
    ngmp_alpha = config.ngmp_alpha,
    ngmp_beta = config.ngmp_beta,
    ngmp_max_step = config.ngmp_max_step,
)
@info "Training finished" training_seconds

# ── Snapshot predictions (the expensive part) ────────────────────────────────

animation_grid = make_grid(config.animation_grid_size, config.checkboard_size)

snapshot_iterations = thinned_iterations(
    length(ngmp_fit.result.posteriors[:w_mean]),
    config.animation_frames,
)
prediction_output_mean = mean(train_data.OT)
n_test_predictions = length(test_features)
snapshot_features = vcat(test_features, animation_grid.features)

@info "Predicting animation snapshots" length(snapshot_iterations) length(snapshot_features)
prediction_snapshots = map(snapshot_iterations) do training_iteration
    seconds = @elapsed snapshot = let
        priors = softplus_prediction_priors(ngmp_fit.result, training_iteration)
        marginals = predict_softplus_marginals(
            priors,
            snapshot_features;
            batch_size = config.prediction_batch_size,
            n_neurons = config.n_neurons,
            iterations = config.prediction_iterations,
            output_mean = prediction_output_mean,
            y_prior_variance = config.prediction_prior_variance,
            ngmp_alpha = config.ngmp_alpha,
            ngmp_beta = config.ngmp_beta,
            ngmp_max_step = config.ngmp_max_step,
        )
        test_statistics = predictive_statistics(
            @view(marginals[1:n_test_predictions]),
        )
        surface_statistics = predictive_statistics(
            @view(marginals[(n_test_predictions + 1):end]),
        )
        (
            iteration = training_iteration,
            test_mean = test_statistics.mean,
            test_variance = test_statistics.variance,
            test_mse = mean(abs2, test_statistics.mean .- test_data.OT),
            mean_surface = reshape(
                surface_statistics.mean,
                length(animation_grid.y),
                length(animation_grid.x),
            ),
            variance_surface = reshape(
                surface_statistics.variance,
                length(animation_grid.y),
                length(animation_grid.x),
            ),
        )
    end
    @info "Snapshot done" training_iteration snapshot.test_mse seconds
    snapshot
end

constant_prediction = prediction_output_mean
constant_mse = mean(abs2, constant_prediction .- test_data.OT)
mse_by_iteration = getproperty.(prediction_snapshots, :test_mse)

# ── Animation ────────────────────────────────────────────────────────────────

animation_output = let
    all_animation_variances = reduce(
        vcat,
        vec.(getproperty.(prediction_snapshots, :variance_surface)),
    )
    variance_lower = minimum(all_animation_variances)
    variance_upper = maximum(all_animation_variances)
    variance_limits = variance_lower == variance_upper ?
        (variance_lower, nextfloat(variance_upper)) :
        (variance_lower, variance_upper)
    mse_upper = max(constant_mse, maximum(mse_by_iteration))
    mse_limits = (0.0, mse_upper > 0 ? 1.1 * mse_upper : 1.0)

    animation = @animate for frame in eachindex(prediction_snapshots)
        snapshot = prediction_snapshots[frame]
        mean_panel = contourf(
            animation_grid.x,
            animation_grid.y,
            snapshot.mean_surface;
            color = :RdBu,
            levels = 20,
            clims = (0, 1),
            xlabel = "x1",
            ylabel = "x2",
            title = "Predictive mean",
            linewidth = 0,
            aspect_ratio = :equal,
        )
        variance_panel = contourf(
            animation_grid.x,
            animation_grid.y,
            snapshot.variance_surface;
            color = :viridis,
            levels = 20,
            clims = variance_limits,
            xlabel = "x1",
            ylabel = "x2",
            title = "Predictive variance q(y)",
            linewidth = 0,
            aspect_ratio = :equal,
        )
        actual_panel = heatmap(
            animation_grid.x,
            animation_grid.y,
            animation_grid.actual;
            color = :RdBu,
            clims = (0, 1),
            xlabel = "x1",
            ylabel = "x2",
            title = "Clean $(config.checkboard_size[1])x$(config.checkboard_size[2]) target",
            aspect_ratio = :equal,
        )
        mse_panel = plot(
            snapshot_iterations[1:frame],
            mse_by_iteration[1:frame];
            color = :steelblue,
            marker = :circle,
            linewidth = 2,
            xlabel = "Iteration",
            ylabel = "Test MSE",
            title = "MSE = $(round(snapshot.test_mse; digits = 4))",
            label = "q(y) mean",
            xlims = (0.5, length(ngmp_fit.result.posteriors[:w_mean]) + 0.5),
            ylims = mse_limits,
        )
        hline!(
            mse_panel,
            [constant_mse];
            color = :black,
            linestyle = :dash,
            label = "constant",
        )
        plot(
            mean_panel,
            variance_panel,
            actual_panel,
            mse_panel;
            layout = (2, 2),
            size = (1_100, 820),
            plot_title = "$(config.checkboard_size[1])x$(config.checkboard_size[2]) checkerboard - training iteration $(snapshot.iteration)",
        )
    end

    mkpath(config.output_dir)
    output_path = joinpath(
        config.output_dir,
        "checkerboard_$(config.checkboard_size[1])x$(config.checkboard_size[2])_softplus_ut_ngmp_learning.gif",
    )
    gif(animation, output_path; fps = config.animation_fps)
end
@info "Animation written" animation_output.filename

# ── Final high-resolution surface ────────────────────────────────────────────

final_grid = make_grid(config.final_grid_size, config.checkboard_size)

@info "Predicting final surface" config.final_grid_size
final_surface_seconds = @elapsed final_grid_prediction = let
    priors = softplus_prediction_priors(
        ngmp_fit.result,
        length(ngmp_fit.result.posteriors[:w_mean]),
    )
    marginals = predict_softplus_marginals(
        priors,
        final_grid.features;
        batch_size = config.prediction_batch_size,
        n_neurons = config.n_neurons,
        iterations = config.prediction_iterations,
        output_mean = prediction_output_mean,
        y_prior_variance = config.prediction_prior_variance,
        ngmp_alpha = config.ngmp_alpha,
        ngmp_beta = config.ngmp_beta,
        ngmp_max_step = config.ngmp_max_step,
    )
    statistics = predictive_statistics(marginals)
    (
        mean = reshape(statistics.mean, length(final_grid.y), length(final_grid.x)),
        variance = reshape(
            statistics.variance,
            length(final_grid.y),
            length(final_grid.x),
        ),
    )
end
@info "Final surface predicted" final_surface_seconds

final_surface_path = let
    variance_lower = minimum(final_grid_prediction.variance)
    variance_upper = maximum(final_grid_prediction.variance)
    variance_limits = variance_lower == variance_upper ?
        (variance_lower, nextfloat(variance_upper)) :
        (variance_lower, variance_upper)

    mean_panel = contourf(
        final_grid.x,
        final_grid.y,
        final_grid_prediction.mean;
        color = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Predictive mean",
        linewidth = 0,
        aspect_ratio = :equal,
    )
    variance_panel = contourf(
        final_grid.x,
        final_grid.y,
        final_grid_prediction.variance;
        color = :viridis,
        levels = 20,
        clims = variance_limits,
        xlabel = "x1",
        ylabel = "x2",
        title = "Predictive variance q(y)",
        linewidth = 0,
        aspect_ratio = :equal,
    )
    actual_panel = heatmap(
        final_grid.x,
        final_grid.y,
        final_grid.actual;
        color = :RdBu,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Clean $(config.checkboard_size[1])x$(config.checkboard_size[2]) target",
        aspect_ratio = :equal,
    )
    figure = plot(
        mean_panel,
        variance_panel,
        actual_panel;
        layout = (1, 3),
        size = (1_350, 420),
        plot_title = "$(config.checkboard_size[1])x$(config.checkboard_size[2]) checkerboard - posterior prediction",
    )

    output_path = joinpath(
        config.output_dir,
        "checkerboard_$(config.checkboard_size[1])x$(config.checkboard_size[2])_softplus_ut_ngmp_final_surface.png",
    )
    savefig(figure, output_path)
    output_path
end
@info "Final surface written" final_surface_path
