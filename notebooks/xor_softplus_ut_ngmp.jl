### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 01d63ba0-6582-4aa6-a102-81a361672512
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 755ef39b-b9eb-41e9-88c8-953cb16d15e4
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

# ╔═╡ 59216211-1c24-4c40-957a-d671c3dde95b
md"""
# Checkerboard Softplus UT NGMP

This notebook generalizes the successful XOR experiment to an ``N \times M``
checkerboard. The `(2, 2)` special case is XOR; `checkboard_size` in `config`
selects the active grid while keeping the inference model unchanged.

The positive gate is

```math
\gamma = \operatorname{softplus}(z_a) = \log(1 + e^{z_a}),
```

with a Gamma belief on ``\gamma`` and a Gaussian belief on ``z_a``. Both
Softplus messages use natural-gradient message passing. The tangent projection
uses the unscented transform; damping and a bounded natural-parameter step keep
the Gamma messages proper on this large factor graph.
"""

# ╔═╡ ba670c90-a7db-4f2b-8067-cf3e3c9b58d4
begin
    config = (
        n_samples =  1_600,
        n_neurons =  8,
        iterations = 100,
        train_fraction = 0.40,
        noise_std = 0.10,
        data_seed = 2_026,
        split_seed = 2_027,
        prior_seed = 42,
        mean_prior_precision = 1e-4,
        gate_prior_precision = 1.0,
        ngmp_alpha = 0.2,
        ngmp_beta = 0.0,
        ngmp_max_step = 1.0,
        gamma_rate_prior_shape = 10.0,
        gamma_rate_prior_rate = 10.0,
        prediction_iterations = 20,
        prediction_batch_size = 1_024,
        prediction_prior_variance = 1e12,
        animation_frames = 12,
        animation_grid_size = 60,
        grid_size = 160,
        animation_fps = 5,
        checkboard_size = (2, 2),
        output_dir = joinpath(@__DIR__, "..", "viz"),
    )
end

# ╔═╡ c4b9483d-789c-44a7-955d-1a3a35babff9
md"""
## Data

Inputs are uniform on ``[-2, 2]^2``. The clean target alternates between zero
and one across the configured cells, then receives clipped Gaussian noise.
The current relation is a **$(config.checkboard_size[1])x$(config.checkboard_size[2])
checkerboard**.
"""

# ╔═╡ bff85b72-63e8-4d31-a8db-6a84b7750985
function checkerboard_label(x1, x2, checkerboard_size)
    nx, ny = checkerboard_size
    nx > 0 && ny > 0 ||
        throw(ArgumentError("checkerboard dimensions must be positive"))
    cell_x = clamp(floor(Int, nx * (x1 + 2) / 4), 0, nx - 1)
    cell_y = clamp(floor(Int, ny * (x2 + 2) / 4), 0, ny - 1)
    return Float64(isodd(cell_x + cell_y))
end

# ╔═╡ 33781fa0-d3f8-442f-a596-027d27e62660
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

# ╔═╡ 923d3cc6-3db2-46d9-8948-390c7091451a
function split_dataset(df; train_fraction = 0.30, seed = 42)
    0 < train_fraction < 1 ||
        throw(ArgumentError("train_fraction must be in (0, 1)"))
    rng = StableRNG(seed)
    indices = randperm(rng, nrow(df))
    n_train = round(Int, train_fraction * nrow(df))
    return df[indices[1:n_train], :], df[indices[(n_train + 1):end], :]
end

# ╔═╡ 8b2279a7-3405-4518-af82-3bc3ddce721b
build_features(df) = [[1.0, df.x1[index], df.x2[index]] for index in 1:nrow(df)]

# ╔═╡ a6e1f2d3-08dc-4108-bc8d-9fd33547b06b
begin
    dataset = make_checkerboard_dataset(
        n = config.n_samples,
        noise_std = config.noise_std,
        seed = config.data_seed,
        checkerboard_size=config.checkboard_size
    )
    train_data, test_data = split_dataset(
        dataset;
        train_fraction = config.train_fraction,
        seed = config.split_seed,
    )
    train_features = build_features(train_data)
    test_features = build_features(test_data)
end

# ╔═╡ d47a3bc7-d040-4675-98d5-1497cdfaa673
scatter(
    dataset.x1,
    dataset.x2;
    marker_z = dataset.OT,
    color = :RdBu,
    clims = (0, 1),
    markersize = 3,
    markerstrokewidth = 0,
    xlabel = "x1",
    ylabel = "x2",
    title = "Noisy $(config.checkboard_size[1])x$(config.checkboard_size[2]) checkerboard data",
    aspect_ratio = :equal,
    legend = false,
)

# ╔═╡ 57bb6db3-53bb-4fde-96c6-ce26a33e8029
md"""
## Softplus model

Each neuron contributes a local linear mean and a positive input-dependent
precision. The shared output is therefore a precision-weighted ensemble. The
`NGMPDependencies` object below requests an unscented tangent projection on
both Softplus message edges.

Every local precision also has an exponential prior with one inferred global
rate:

```math
\beta \sim \operatorname{Gamma}(10, 10), \qquad
\gamma_{k,j} \sim \operatorname{Gamma}(1, \beta).
```

The shape-one factor contributes no extra `log(γ)` term, while its learned
positive rate regularizes the absolute gate scale.
"""

# ╔═╡ 222053f5-382d-418e-a649-045d59728513
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
                obs_damping = obs_damping
            }
        end
        y[observation] ~ NormalMeanPrecision(out[observation], obs_noise)
    end
end

# ╔═╡ 70d0a30d-54c8-4515-bfb5-3d470686bc99
@constraints function xor_softplus_ut_constraints()
    q(w_mean, w_a, z_mean, za, γ, τ, τ_mean, out, obs_noise, β) =
        q(w_mean, z_mean, out, za, γ)q(w_a)q(τ)q(τ_mean)q(obs_noise)q(β)

    # softdot repeatedly consumes the same weight means and covariances.
    q(w_mean)::MomentForm()
    q(w_a)::MomentForm()
end

# ╔═╡ 7f098370-c2fc-4d0c-86d2-5487a1527765
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

# ╔═╡ 9e3a784a-ea10-4edc-b5c8-583bc3df77c0
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

# ╔═╡ c9267a99-5dfc-4d89-b054-eab7a4c69484
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

# ╔═╡ 041edacb-5c47-487c-8117-0757e79d975f
fit = run_softplus_ut_ngmp(
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

# ╔═╡ db34de5d-f617-4236-aabb-3318863b8acf
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

# ╔═╡ df426399-1486-4640-86f6-dfc6d5542f08
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

# ╔═╡ 0f9e2758-1b87-4444-944b-dd12b33d05dc
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

# ╔═╡ 6f70997c-dc59-4295-a168-4b9623165c9e
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

# ╔═╡ f74c2668-9b51-4444-8c1e-e2027073d32d
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

# ╔═╡ b59e6867-147d-43d3-9db0-ed2a07460dc7
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

# ╔═╡ b3cf22f8-82e3-47a4-82a3-7dc98f9ffec8
begin
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
end

# ╔═╡ c3ec192b-2b65-4102-b417-eb8c1d042bb3
animation_grid = let
    x = range(-2.0, 2.0; length = config.animation_grid_size)
    y = range(-2.0, 2.0; length = config.animation_grid_size)
    actual = [
        checkerboard_label(x_value, y_value, config.checkboard_size) for
        y_value in y, x_value in x
    ]
    features = vec([
        [1.0, x_value, y_value] for y_value in y, x_value in x
    ])
    (x = x, y = y, actual = actual, features = features)
end

# ╔═╡ ae1f0398-f75c-4539-bfd2-3fbcc98d17f8
begin
    snapshot_iterations = thinned_iterations(
        length(fit.result.posteriors[:w_mean]),
        config.animation_frames,
    )
    prediction_output_mean = mean(train_data.OT)
    n_test_predictions = length(test_features)
    snapshot_features = vcat(test_features, animation_grid.features)

    prediction_snapshots = map(snapshot_iterations) do training_iteration
        priors = softplus_prediction_priors(fit.result, training_iteration)
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

    constant_prediction = prediction_output_mean
    constant_mse = mean(abs2, constant_prediction .- test_data.OT)
    mse_by_iteration = getproperty.(prediction_snapshots, :test_mse)
    mse_history = DataFrame(
        iteration = snapshot_iterations,
        test_mse = mse_by_iteration,
    )
end

# ╔═╡ 87f5a99a-6251-44b1-bcd2-bf562715a35f
begin
    final_gamma = vec(fit.result.posteriors[:γ][end])
    final_gamma_matrix = mean.(fit.result.posteriors[:γ][end])
    final_total_gamma = vec(sum(final_gamma_matrix; dims = 1))
    run_summary = DataFrame(
        neurons = config.n_neurons,
        iterations = length(fit.result.posteriors[:w_mean]),
        train_points = nrow(train_data),
        test_points = nrow(test_data),
        test_mse = last(mse_by_iteration),
        constant_mse = constant_mse,
        minimum_predictive_variance = minimum(prediction_snapshots[end].test_variance),
        mean_predictive_variance = mean(prediction_snapshots[end].test_variance),
        maximum_predictive_variance = maximum(prediction_snapshots[end].test_variance),
        minimum_gamma_shape = minimum(shape, final_gamma),
        minimum_gamma_rate = minimum(rate, final_gamma),
        maximum_total_gamma = maximum(final_total_gamma),
        global_gamma_rate = mean(fit.result.posteriors[:β][end]),
        gate_softdot_precision = mean(fit.result.posteriors[:τ][end]),
        mean_softdot_precision = mean(fit.result.posteriors[:τ_mean][end]),
        observation_precision = mean(fit.result.posteriors[:obs_noise][end]),
    )
end

# ╔═╡ 4758395c-51a4-43c4-b1e1-853678612c13
learning_curve = let
    curve = plot(
        snapshot_iterations,
        mse_by_iteration;
        marker = :circle,
        linewidth = 2,
        color = :steelblue,
        xlabel = "Iteration",
        ylabel = "Test MSE",
        title = "Checkerboard Softplus UT NGMP learning",
        label = "Softplus",
        xlims = (0.5, length(fit.result.posteriors[:w_mean]) + 0.5),
    )
    hline!(
        curve,
        [constant_mse];
        color = :black,
        linestyle = :dash,
        label = "constant baseline",
    )
    curve
end

# ╔═╡ 726541a4-9de7-468f-9ac5-2ebcdcc18644
begin
    start_from = 1
    plot(
        eachindex(fit.result.free_energy[start_from:end]),
        fit.result.free_energy[start_from:end];
        marker = :circle,
        linewidth = 2,
        color = :darkorange,
        xlabel = "Iteration",
        ylabel = "Bethe free energy",
        # title = ",
        legend = false,
    )
end

# ╔═╡ a6830b67-90bd-41bd-a5be-85a6ace9eccd
md"""
The free-energy values are finite and useful as a within-run diagnostic. Their
absolute scale includes the local deterministic-node entropy approximation, so
do not compare the values directly with another activation model.
"""

# ╔═╡ d45d778b-789a-4b66-a7c4-a2a42062007a
learned_weights = DataFrame(
    neuron = 1:config.n_neurons,
    mean_weights = mean.(fit.result.posteriors[:w_mean][end]),
    gate_weights = mean.(fit.result.posteriors[:w_a][end]),
)

# ╔═╡ 8d66c7f6-75b7-4db6-8350-05257d80d0bb
md"""
## Learned surface

Prediction uses a second RxInfer graph. Learned global marginals are fixed with
`FixedMarginalFormConstraint`; each predictive ``y`` receives a diffuse
``\mathcal{N}(0, 10^{12})`` factor. The resulting ``q(y)`` includes both latent
model uncertainty and learned observation noise.

The full-resolution final graph is used below. Animation frames use a smaller
grid and a thinned set of training iterations so that the notebook remains
interactive.
"""

# ╔═╡ 429bd3a0-8c79-4376-a6b6-eb0c63cffb77
grid = let
    x = range(-2.0, 2.0; length = config.grid_size)
    y = range(-2.0, 2.0; length = config.grid_size)
    actual = [
        checkerboard_label(x_value, y_value, config.checkboard_size) for
        y_value in y, x_value in x
    ]
    features = vec([
        [1.0, x_value, y_value] for y_value in y, x_value in x
    ])
    (x = x, y = y, actual = actual, features = features)
end

# ╔═╡ 685ca7c0-ccea-4f7c-8067-5e948f0da331
final_grid_prediction = let
    priors = softplus_prediction_priors(
        fit.result,
        length(fit.result.posteriors[:w_mean]),
    )
    marginals = predict_softplus_marginals(
        priors,
        grid.features;
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
        mean = reshape(statistics.mean, length(grid.y), length(grid.x)),
        variance = reshape(
            statistics.variance,
            length(grid.y),
            length(grid.x),
        ),
    )
end

# ╔═╡ 3395813c-c790-47c2-916e-75abb32355c4
final_variance_limits = let
    lower = minimum(final_grid_prediction.variance)
    upper = maximum(final_grid_prediction.variance)
    lower == upper ? (lower, nextfloat(upper)) : (lower, upper)
end

# ╔═╡ d47d2ef0-da37-4127-ad06-1bb602e655c4
final_surface_plot = let
    mean_panel = contourf(
        grid.x,
        grid.y,
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
        grid.x,
        grid.y,
        final_grid_prediction.variance;
        color = :viridis,
        levels = 20,
        clims = final_variance_limits,
        xlabel = "x1",
        ylabel = "x2",
        title = "Predictive variance q(y)",
        linewidth = 0,
        aspect_ratio = :equal,
    )
    actual_panel = heatmap(
        grid.x,
        grid.y,
        grid.actual;
        color = :RdBu,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Clean $(config.checkboard_size[1])x$(config.checkboard_size[2]) target",
        aspect_ratio = :equal,
    )
    plot(
        mean_panel,
        variance_panel,
        actual_panel;
        layout = (1, 3),
        size = (1_350, 420),
        plot_title = "$(config.checkboard_size[1])x$(config.checkboard_size[2]) checkerboard - posterior prediction",
    )
end

# ╔═╡ 5bc819f5-61a4-47d3-ac36-5c022f9aca99
md"""
## Learning animation

The final cell shows the GIF inline in Pluto and writes a file named for the
current checkerboard dimensions into `viz`. Each frame runs the prediction graph
against a saved training posterior and displays total ``q(y)`` variance.
"""

# ╔═╡ d8fa36a3-ac39-44e1-88e8-5adba1aa4ef1
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
            xlims = (0.5, length(fit.result.posteriors[:w_mean]) + 0.5),
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

# ╔═╡ Cell order:
# ╠═01d63ba0-6582-4aa6-a102-81a361672512
# ╠═755ef39b-b9eb-41e9-88c8-953cb16d15e4
# ╟─59216211-1c24-4c40-957a-d671c3dde95b
# ╠═ba670c90-a7db-4f2b-8067-cf3e3c9b58d4
# ╟─c4b9483d-789c-44a7-955d-1a3a35babff9
# ╠═bff85b72-63e8-4d31-a8db-6a84b7750985
# ╠═33781fa0-d3f8-442f-a596-027d27e62660
# ╠═923d3cc6-3db2-46d9-8948-390c7091451a
# ╠═8b2279a7-3405-4518-af82-3bc3ddce721b
# ╠═a6e1f2d3-08dc-4108-bc8d-9fd33547b06b
# ╠═d47a3bc7-d040-4675-98d5-1497cdfaa673
# ╟─57bb6db3-53bb-4fde-96c6-ce26a33e8029
# ╠═222053f5-382d-418e-a649-045d59728513
# ╠═70d0a30d-54c8-4515-bfb5-3d470686bc99
# ╠═7f098370-c2fc-4d0c-86d2-5487a1527765
# ╠═9e3a784a-ea10-4edc-b5c8-583bc3df77c0
# ╠═c9267a99-5dfc-4d89-b054-eab7a4c69484
# ╠═041edacb-5c47-487c-8117-0757e79d975f
# ╠═db34de5d-f617-4236-aabb-3318863b8acf
# ╠═df426399-1486-4640-86f6-dfc6d5542f08
# ╠═0f9e2758-1b87-4444-944b-dd12b33d05dc
# ╠═6f70997c-dc59-4295-a168-4b9623165c9e
# ╠═f74c2668-9b51-4444-8c1e-e2027073d32d
# ╠═b59e6867-147d-43d3-9db0-ed2a07460dc7
# ╠═b3cf22f8-82e3-47a4-82a3-7dc98f9ffec8
# ╠═c3ec192b-2b65-4102-b417-eb8c1d042bb3
# ╠═ae1f0398-f75c-4539-bfd2-3fbcc98d17f8
# ╠═87f5a99a-6251-44b1-bcd2-bf562715a35f
# ╠═4758395c-51a4-43c4-b1e1-853678612c13
# ╠═726541a4-9de7-468f-9ac5-2ebcdcc18644
# ╟─a6830b67-90bd-41bd-a5be-85a6ace9eccd
# ╠═d45d778b-789a-4b66-a7c4-a2a42062007a
# ╟─8d66c7f6-75b7-4db6-8350-05257d80d0bb
# ╠═429bd3a0-8c79-4376-a6b6-eb0c63cffb77
# ╠═685ca7c0-ccea-4f7c-8067-5e948f0da331
# ╠═3395813c-c790-47c2-916e-75abb32355c4
# ╠═d47d2ef0-da37-4127-ad06-1bb602e655c4
# ╟─5bc819f5-61a4-47d3-ac36-5c022f9aca99
# ╠═d8fa36a3-ac39-44e1-88e8-5adba1aa4ef1
