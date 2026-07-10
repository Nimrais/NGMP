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
    using LinearAlgebra: Diagonal, dot
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
# XOR Softplus UT NGMP

This notebook isolates the successful Softplus experiment from the activation
comparison. The default run reproduces the **16-neuron, 20-iteration** setup
that reached a test MSE of about **0.16346**.

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
        n_neurons =  16,
        iterations = 100,
        train_fraction = 0.30,
        noise_std = 0.10,
        data_seed = 2_026,
        split_seed = 2_027,
        prior_seed = 42,
        mean_prior_precision = 1e-4,
        gate_prior_precision = 1.0,
        ngmp_alpha = 0.2,
        ngmp_beta = 0.0,
        ngmp_max_step = 0.1,
        grid_size = 160,
        animation_fps = 5,
        checkboard_size = (2, 2),
        output_dir = joinpath(@__DIR__, "..", "viz"),
    )
end

# ╔═╡ c4b9483d-789c-44a7-955d-1a3a35babff9
md"""
## Data

The data and split seeds match the comparison script. Inputs are uniform on
``[-2, 2]^2`` and the target is a clipped noisy XOR response. With the default
configuration, 480 points are used for learning and 1,120 for evaluation.
"""

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

      cell_x = clamp.(floor.(Int, nx .* (x1 .+ 2) ./ 4), 0, nx - 1)
      cell_y = clamp.(floor.(Int, ny .* (x2 .+ 2) ./ 4), 0, ny - 1)

      clean = Float64.(isodd.(cell_x .+ cell_y))
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
    title = "Generated noisy XOR data",
    legend = false,
)

# ╔═╡ 57bb6db3-53bb-4fde-96c6-ce26a33e8029
md"""
## Softplus model

Each neuron contributes a local linear mean and a positive input-dependent
precision. The shared output is therefore a precision-weighted ensemble. The
`NGMPDependencies` object below requests an unscented tangent projection on
both Softplus message edges.
"""

# ╔═╡ 222053f5-382d-418e-a649-045d59728513
@model function xor_softplus_ut_ngmp(
    n_neurons,
    features,
    y,
    priors,
    dependencies,
    damping,
)
    local w_mean, w_a, z_mean, za, γ, τ, out

    τ ~ priors[:τ]
    τ_mean ~ priors[:τ_mean]
    obs_noise ~ priors[:obs_noise]

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
            γ[neuron, observation] ~ Softplus(za[neuron, observation]) where {
                dependencies = dependencies,
                meta = damping,
            }
            out[observation] ~ NormalMeanPrecision(
                z_mean[neuron, observation],
                γ[neuron, observation],
            )
        end
        y[observation] ~ NormalMeanPrecision(out[observation], obs_noise)
    end
end

# ╔═╡ 70d0a30d-54c8-4515-bfb5-3d470686bc99
@constraints function xor_softplus_ut_constraints()
    q(w_mean, w_a, z_mean, za, γ, τ, τ_mean, out, obs_noise) =
        q(w_mean, z_mean, out)q(w_a)q(za, γ)q(τ)q(τ_mean)q(obs_noise)

    # softdot repeatedly consumes the same weight means and covariances.
    q(w_mean)::MomentForm()
    q(w_a)::MomentForm()
end

# ╔═╡ 7f098370-c2fc-4d0c-86d2-5487a1527765
@initialization function xor_softplus_ut_initialization(priors)
    q(w_a) = deepcopy(priors[:w_a])
    q(za) = NormalMeanVariance(0.0, 1.0)
    q(γ) = GammaShapeScale(2.0, 1.0)
    q(τ) = priors[:τ]
    q(obs_noise) = priors[:obs_noise]
    q(τ_mean) = priors[:τ_mean]
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
        :τ_mean => GammaShapeRate(1e3, 1.0),
        :obs_noise => GammaShapeRate(1e6, 1.0)
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

    inference_model = xor_softplus_ut_ngmp(
        n_neurons = n_neurons,
        priors = priors,
        dependencies = dependencies,
        damping = damping,
    )
    result = infer(
        model = inference_model,
        data = (y = observations, features = features),
        constraints = xor_softplus_ut_constraints(),
        initialization = xor_softplus_ut_initialization(priors),
        iterations = iterations,
        free_energy = true,
        showprogress = true,
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
        ngmp_alpha = config.ngmp_alpha,
        ngmp_beta = config.ngmp_beta,
        ngmp_max_step = config.ngmp_max_step,
)

# ╔═╡ db34de5d-f617-4236-aabb-3318863b8acf
stable_softplus(x::Real) = max(x, zero(x)) + log1p(exp(-abs(x)))

# ╔═╡ df426399-1486-4640-86f6-dfc6d5542f08
function predict_softplus(feature, w_means, w_as)
    local_means = [dot(weight, feature) for weight in w_means]
    gate_scores = [dot(weight, feature) for weight in w_as]
    precisions = stable_softplus.(gate_scores)
    total_precision = sum(precisions)
    return total_precision > eps(Float64) ?
        dot(precisions, local_means) / total_precision : mean(local_means)
end

# ╔═╡ 0f9e2758-1b87-4444-944b-dd12b33d05dc
posterior_weights = [
    (
        w_means = mean.(fit.result.posteriors[:w_mean][iteration]),
        w_as = mean.(fit.result.posteriors[:w_a][iteration]),
    ) for iteration in eachindex(fit.result.posteriors[:w_mean])
]

# ╔═╡ 6f70997c-dc59-4295-a168-4b9623165c9e
begin
      predictions_by_iteration = [
          [
              predict_softplus(feature, weights.w_means, weights.w_as) for
              feature in test_features
          ] for weights in posterior_weights
      ]

      constant_prediction = mean(train_data.OT)
      constant_mse = mean(abs2, constant_prediction .- test_data.OT)

      mse_by_iteration = [
          mean(abs2, predictions .- test_data.OT) for
          predictions in predictions_by_iteration
      ]

      mse_history = DataFrame(
          iteration = eachindex(mse_by_iteration),
          test_mse = mse_by_iteration,
      );
  end

# ╔═╡ 87f5a99a-6251-44b1-bcd2-bf562715a35f
begin
    final_gamma = vec(fit.result.posteriors[:γ][end])
    run_summary = DataFrame(
        neurons = config.n_neurons,
        iterations = length(posterior_weights),
        train_points = nrow(train_data),
        test_points = nrow(test_data),
        test_mse = last(mse_by_iteration),
        constant_mse = constant_mse,
        minimum_gamma_shape = minimum(shape, final_gamma),
        minimum_gamma_rate = minimum(rate, final_gamma),
    )
end

# ╔═╡ 4758395c-51a4-43c4-b1e1-853678612c13
learning_curve = let
    curve = plot(
        eachindex(mse_by_iteration),
        mse_by_iteration;
        marker = :circle,
        linewidth = 2,
        color = :steelblue,
        xlabel = "Iteration",
        ylabel = "Test MSE",
        title = "Softplus UT NGMP learning",
        label = "Softplus",
        xlims = (0.5, length(mse_by_iteration) + 0.5),
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
    mean_weights = posterior_weights[end].w_means,
    gate_weights = posterior_weights[end].w_as,
)

# ╔═╡ 8d66c7f6-75b7-4db6-8350-05257d80d0bb
md"""
## Learned surface

All iteration surfaces are cached below. This makes the final animation a
rendering step rather than another inference run.
"""

# ╔═╡ 429bd3a0-8c79-4376-a6b6-eb0c63cffb77
grid = let
    margin = 0.5
    x = range(
        minimum(test_data.x1) - margin,
        maximum(test_data.x1) + margin;
        length = config.grid_size,
    )
    y = range(
        minimum(test_data.x2) - margin,
        maximum(test_data.x2) + margin;
        length = config.grid_size,
    )
    actual = [Float64(x_value * y_value < 0) for y_value in y, x_value in x]
    (x = x, y = y, actual = actual)
end

# ╔═╡ 685ca7c0-ccea-4f7c-8067-5e948f0da331
iteration_surfaces = [
    [
        predict_softplus(
            [1.0, x_value, y_value],
            weights.w_means,
            weights.w_as,
        ) for y_value in grid.y, x_value in grid.x
    ] for weights in posterior_weights
]

# ╔═╡ 3395813c-c790-47c2-916e-75abb32355c4
# Change this index to inspect a cached iteration without rerunning inference.
surface_iteration = length(iteration_surfaces)

# ╔═╡ d47d2ef0-da37-4127-ad06-1bb602e655c4
final_surface_plot = let
    checkbounds(iteration_surfaces, surface_iteration)
    predicted_plot = contourf(
        grid.x,
        grid.y,
        iteration_surfaces[surface_iteration];
        color = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Predicted",
        linewidth = 0,
    )
    actual_plot = contourf(
        grid.x,
        grid.y,
        grid.actual;
        color = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Actual XOR",
        linewidth = 0,
    )
    plot(
        predicted_plot,
        actual_plot;
        layout = (1, 2),
        size = (1_000, 400),
        plot_title = "XOR Softplus UT NGMP - iteration $surface_iteration",
    )
end

# ╔═╡ 5bc819f5-61a4-47d3-ac36-5c022f9aca99
md"""
## Learning animation

The final cell shows the GIF inline in Pluto and writes it to
`viz/xor_softplus_ut_ngmp_learning.gif`. Set `make_animation = false` while
tuning model settings when you do not need the GIF rebuilt after every run.
"""

# ╔═╡ d8fa36a3-ac39-44e1-88e8-5adba1aa4ef1
animation_output = let
    animation = @animate for iteration in eachindex(iteration_surfaces)
        predicted_panel = contourf(
            grid.x,
            grid.y,
            iteration_surfaces[iteration];
            color = :RdBu,
            levels = 20,
            clims = (0, 1),
            xlabel = "x1",
            ylabel = "x2",
            title = "Predicted (iteration $iteration)",
            linewidth = 0,
        )
        actual_panel = contourf(
            grid.x,
            grid.y,
            grid.actual;
            color = :RdBu,
            levels = 20,
            clims = (0, 1),
            xlabel = "x1",
            ylabel = "x2",
            title = "Actual XOR",
            linewidth = 0,
        )
        mse_panel = plot(
            1:iteration,
            mse_by_iteration[1:iteration];
            color = :steelblue,
            marker = :circle,
            linewidth = 2,
            xlabel = "Iteration",
            ylabel = "Test MSE",
            title = "MSE = $(round(mse_by_iteration[iteration]; digits = 4))",
            label = "Softplus",
            xlims = (0.5, length(mse_by_iteration) + 0.5),
            ylims = (
                0,
                1.1 * max(constant_mse, maximum(mse_by_iteration)),
            ),
        )
        hline!(
            mse_panel,
            [constant_mse];
            color = :black,
            linestyle = :dash,
            label = "constant",
        )
        plot(
            predicted_panel,
            actual_panel,
            mse_panel;
            layout = (1, 3),
            size = (1_350, 400),
            plot_title = "XOR Softplus UT NGMP - iteration $iteration",
        )
    end

    mkpath(config.output_dir)
    output_path = joinpath(
        config.output_dir,
        "xor_softplus_ut_ngmp_learning.gif",
    )
    gif(animation, output_path; fps = config.animation_fps)
end

# ╔═╡ Cell order:
# ╠═01d63ba0-6582-4aa6-a102-81a361672512
# ╠═755ef39b-b9eb-41e9-88c8-953cb16d15e4
# ╟─59216211-1c24-4c40-957a-d671c3dde95b
# ╠═ba670c90-a7db-4f2b-8067-cf3e3c9b58d4
# ╟─c4b9483d-789c-44a7-955d-1a3a35babff9
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
