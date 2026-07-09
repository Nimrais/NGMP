### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 2fd780b0-ff12-4f6c-bb02-9438380ce152
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 88e431ef-e913-4758-aaf6-aa7fc369207a
begin
    using ClosedFormExpectations: EnzymeBackend
    using DataFrames
    using ExponentialFamily
    using ExponentialFamilyProjection
    using ExponentialFamilyProjection: ClosedFormStrategy
    using LinearAlgebra: Diagonal, dot
    using Plots
    using ProbabilisticEnsembling
    using Random
    using RxInfer
    using Statistics
    using SurrogateModelling: MomentForm
end

# ╔═╡ e96b4886-0efc-4f15-8394-d1e7f293bf44
md"""
# XOR ReLU direct: VMP baseline

This is the original precision-weighted ReLU ensemble, kept as a plain VMP
baseline. The non-conjugate ReLU marginals still use closed-form Gamma
projection; no natural-gradient messages are used.

The only inference optimization is `MomentForm()` on both weight families. It
stores each Gaussian weight marginal in mean/covariance form once per update,
instead of repeatedly solving its information-form system at every `softdot`.
"""

# ╔═╡ 1117f0f8-24a3-4590-8027-d6a8cbb31c97
begin
    env_int(name, default) = parse(Int, get(ENV, name, string(default)))
    env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
    env_bool(name, default = false) =
        lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")

    smoke_test = env_bool("XOR_RELU_SMOKE")
    config = (
        n_samples = env_int("N_SAMPLES", smoke_test ? 60 : 1_600),
        n_neurons = env_int("N_NEURONS", smoke_test ? 2 : 16),
        iterations = 20,
        train_fraction = env_float("TRAIN_FRACTION", 0.30),
        noise_std = env_float("NOISE_STD", 0.10),
        grid_size = env_int("GRID_SIZE", smoke_test ? 30 : 200),
        compute_free_energy = env_bool("FREE_ENERGY", true),
        make_animation = env_bool("MAKE_ANIMATION"),
        save_outputs = env_bool("SAVE_OUTPUTS"),
    )
end

# ╔═╡ fa0334d4-bcb1-476f-bd30-cd4185fd097c
md"""
## Data

The previous script read a package-local CSV containing 1,600 points. Here the
same kind of data is generated in the notebook: uniform coordinates on
``[-2,2]^2`` and a clipped noisy XOR response.
"""

# ╔═╡ 45c39afd-d1fc-450c-b01e-60e908fd6a77
function make_xor_dataset(; n = 1_600, noise_std = 0.10, seed = 2_026)
    rng = MersenneTwister(seed)
    x1 = 4 .* rand(rng, n) .- 2
    x2 = 4 .* rand(rng, n) .- 2
    clean = Float64.(x1 .* x2 .< 0)
    target = clamp.(clean .+ noise_std .* randn(rng, n), 0.0, 1.0)
    return DataFrame(x1 = x1, x2 = x2, OT = target)
end

# ╔═╡ 083bd15a-8641-4bde-9cab-6d29e1313b28
function split_dataset(df; train_fraction = 0.30, seed = 2_027)
    0 < train_fraction < 1 || throw(ArgumentError("train_fraction must be in (0, 1)"))
    rng = MersenneTwister(seed)
    indices = randperm(rng, nrow(df))
    n_train = round(Int, train_fraction * nrow(df))
    return df[indices[1:n_train], :], df[indices[(n_train + 1):end], :]
end

# ╔═╡ ad754270-ba87-48f5-a3bc-fb0d9295b90b
begin
    dataset = make_xor_dataset(
        n = config.n_samples,
        noise_std = config.noise_std,
    )
    train_data, test_data = split_dataset(
        dataset;
        train_fraction = config.train_fraction,
    )
end

# ╔═╡ 6fec6bad-6498-40dc-99ed-80b9e6dded08
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

# ╔═╡ 6bc20914-405e-4a73-9364-411cce113e48
md"""
## Model

Each neuron predicts a local mean and an input-dependent non-negative precision.
All neurons attach Gaussian likelihood factors to the shared `out[j]`, producing
the same precision-weighted combination as the previous script.
"""

# ╔═╡ e81044a3-4312-44ad-b3d8-a08943aa6327
@model function xor_relu_direct(n_neurons, features, y, priors)
    local w_mean, w_a, z_mean, za, γ, τ, out

    τ ~ priors[:τ]

    for k in 1:n_neurons
        w_mean[k] ~ priors[:w_mean][k]
        w_a[k] ~ priors[:w_a][k]
    end

    for j in 1:length(y)
        for k in 1:n_neurons
            z_mean[k, j] ~ softdot(features[j], w_mean[k], τ)
            za[k, j] ~ softdot(features[j], w_a[k], τ) where {meta = LowRankMeta()}
            γ[k, j] ~ ReLU(za[k, j])
            out[j] ~ NormalMeanPrecision(z_mean[k, j], γ[k, j])
        end
        y[j] ~ NormalMeanPrecision(out[j], 1e6)
    end
end

# ╔═╡ a7e98b20-d114-43a2-824e-3873980fd3d3
@constraints function xor_relu_direct_constraints()
    q(w_mean, w_a, z_mean, za, γ, τ, out) =
        q(w_mean, z_mean)q(w_a)q(za, γ)q(τ)q(out)

    # softdot consumes weight means/covariances many times per update.
    q(w_mean)::MomentForm()
    q(w_a)::MomentForm()

    q(za)::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(EnzymeBackend()),
        ),
    )
    q(γ)::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(EnzymeBackend()),
        ),
    )
end

# ╔═╡ 917573e4-7061-42a1-bef0-62e1bbb8df30
@initialization function xor_relu_direct_init(priors)
    q(w_mean) = deepcopy(priors[:w_mean])
    q(w_a) = deepcopy(priors[:w_a])
    q(z_mean) = NormalMeanVariance(0.0, 1.0)
    q(za) = GammaShapeScale(2.0, 1.0)
    q(γ) = GammaShapeScale(2.0, 1.0)
    q(τ) = priors[:τ]
end

# ╔═╡ e941fc95-e9f1-465e-80f0-2440166ae330
function make_priors(; n_neurons = 4, n_features = 3, seed = 42)
    rng = MersenneTwister(seed)
    w_mean = [
        MvNormalWeightedMeanPrecision(
            randn(rng, n_features),
            Diagonal(fill(1e-4, n_features)),
        ) for _ in 1:n_neurons
    ]
    w_a = [
        MvNormalWeightedMeanPrecision(
            randn(rng, n_features),
            Diagonal(fill(1e-3, n_features)),
        ) for _ in 1:n_neurons
    ]

    return Dict{Symbol, Any}(
        :w_mean => w_mean,
        :w_a => w_a,
        :τ => GammaShapeRate(1e6, 1.0),
    )
end

# ╔═╡ 4db052cc-98b1-46bd-8de4-edb589ec9722
build_features(df) = [[1.0, df.x1[i], df.x2[i]] for i in 1:nrow(df)]

# ╔═╡ 17303253-c127-4f74-9132-bac1b7853de4
begin
    priors = make_priors(n_neurons = config.n_neurons)
    train_features = build_features(train_data)
    test_features = build_features(test_data)
end

# ╔═╡ 1b61c420-2397-463d-84f0-3a25b1e7f174
md"""
## Inference

Default run: **$(config.n_neurons) neurons**, **$(nrow(train_data)) training
points**, and **$(config.iterations) VMP iterations**.
"""

# ╔═╡ a6945d41-fe7d-492a-91af-b103ebd1372d
result= infer(
    model = xor_relu_direct(
        n_neurons = config.n_neurons,
        priors = priors,
    ),
    data = (y = train_data.OT, features = train_features),
    constraints = xor_relu_direct_constraints(),
    initialization = xor_relu_direct_init(priors),
    iterations = config.iterations,
    free_energy = config.compute_free_energy,
    showprogress = !smoke_test,
    options = (limit_stack_depth = 100,),
)

# ╔═╡ 994ceffb-9c16-40d5-a424-55b993da8e2d
if config.compute_free_energy
    plot(
        eachindex(result.free_energy),
        result.free_energy;
        marker = :circle,
        xlabel = "Iteration",
        ylabel = "Bethe free energy",
        title = "VMP convergence",
        legend = false,
    )
else
    md"Bethe free-energy scoring is disabled for this run."
end

# ╔═╡ cb6c3703-4730-42f4-afbd-c87c1b085f38
begin
    final_w_means = [
        mean(result.posteriors[:w_mean][end][k]) for k in 1:config.n_neurons
    ]
    final_w_a = [
        mean(result.posteriors[:w_a][end][k]) for k in 1:config.n_neurons
    ]
    learned_weights = DataFrame(
        neuron = 1:config.n_neurons,
        mean_weights = final_w_means,
        gate_weights = final_w_a,
    )
end

# ╔═╡ 94c14598-b59d-48ba-a7a6-c43f3267e803
function predict_point(x1, x2, w_means, w_as)
    feature = [1.0, x1, x2]
    local_means = [dot(w, feature) for w in w_means]
    precisions = [max(0.0, dot(w, feature)) for w in w_as]
    total_precision = sum(precisions)

    # A zero-precision point has no preferred neuron; use their unweighted mean.
    return total_precision > eps(Float64) ?
        dot(precisions, local_means) / total_precision : mean(local_means)
end

# ╔═╡ be240571-5073-468a-97f9-1b2caa499167
begin
    predictions = [
        predict_point(feature[2], feature[3], final_w_means, final_w_a) for
        feature in test_features
    ]
    test_mse = mean(abs2, predictions .- test_data.OT)
    constant_prediction = mean(train_data.OT)
    constant_mse = mean(abs2, constant_prediction .- test_data.OT)
    metrics = (
        test_mse = test_mse,
        constant_mse = constant_mse,
        constant_prediction = constant_prediction,
        # inference_seconds = timed_inference.time,
    )
end

# ╔═╡ 68a973e0-451d-4f8d-80a5-af477b67e744
md"""
**Test MSE:** $(round(metrics.test_mse; digits = 4))  
**Constant baseline MSE:** $(round(metrics.constant_mse; digits = 4))  
**Inference time:** $(round(metrics.inference_seconds; digits = 2)) seconds
"""

# ╔═╡ a67c3a82-a999-4332-b80a-ddc9ddfb756e
begin
    margin = 0.5
    grid_x = range(
        minimum(test_data.x1) - margin,
        maximum(test_data.x1) + margin;
        length = config.grid_size,
    )
    grid_y = range(
        minimum(test_data.x2) - margin,
        maximum(test_data.x2) + margin;
        length = config.grid_size,
    )
    predicted_surface = [
        predict_point(x, y, final_w_means, final_w_a) for y in grid_y, x in grid_x
    ]
    actual_surface = [Float64(x * y < 0) for y in grid_y, x in grid_x]
end

# ╔═╡ 4c9d487f-c271-486d-b9e7-c335b7959551
begin
    predicted_plot = contourf(
        grid_x,
        grid_y,
        predicted_surface;
        color = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Predicted",
        linewidth = 0,
    )
    actual_plot = contourf(
        grid_x,
        grid_y,
        actual_surface;
        color = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Actual XOR",
        linewidth = 0,
    )
    heatmap_plot = plot(
        predicted_plot,
        actual_plot;
        layout = (1, 2),
        size = (1_000, 400),
        plot_title = "XOR ReLU VMP baseline",
    )
end

# ╔═╡ 86837f53-8007-4c61-a39f-4108b2460de5
saved_heatmap = let
    if config.save_outputs
        output_dir = joinpath(@__DIR__, "..", "viz")
        mkpath(output_dir)
        output_path = joinpath(output_dir, "xor_relu_direct_baseline.png")
        savefig(heatmap_plot, output_path)
        output_path
    else
        nothing
    end
end

# ╔═╡ f3399882-5a76-488d-bc6f-5fd90e27c12a
animation_output = let
    animation = @animate for iteration in eachindex(result.posteriors[:w_mean])
        iteration_w_means = [
            mean(result.posteriors[:w_mean][iteration][k]) for
            k in 1:config.n_neurons
        ]
        iteration_w_a = [
            mean(result.posteriors[:w_a][iteration][k]) for
            k in 1:config.n_neurons
        ]
        iteration_surface = [
            predict_point(x, y, iteration_w_means, iteration_w_a) for
            y in grid_y, x in grid_x
        ]

        iteration_plot = contourf(
            grid_x,
            grid_y,
            iteration_surface;
            color = :RdBu,
            levels = 20,
            clims = (0, 1),
            xlabel = "x1",
            ylabel = "x2",
            title = "Predicted (iteration $iteration)",
            linewidth = 0,
        )
        reference_plot = contourf(
            grid_x,
            grid_y,
            actual_surface;
            color = :RdBu,
            levels = 20,
            clims = (0, 1),
            xlabel = "x1",
            ylabel = "x2",
            title = "Actual XOR",
            linewidth = 0,
        )
        plot(
            iteration_plot,
            reference_plot;
            layout = (1, 2),
            size = (1_000, 400),
            plot_title = "XOR ReLU VMP iteration $iteration",
        )
    end

    output_dir = joinpath(@__DIR__, "..", "viz")
    mkpath(output_dir)
    output_path = joinpath(output_dir, "xor_relu_direct_iterations.gif")
    gif(animation, output_path; fps = 1)
    output_path
end

# ╔═╡ Cell order:
# ╠═2fd780b0-ff12-4f6c-bb02-9438380ce152
# ╠═88e431ef-e913-4758-aaf6-aa7fc369207a
# ╟─e96b4886-0efc-4f15-8394-d1e7f293bf44
# ╠═1117f0f8-24a3-4590-8027-d6a8cbb31c97
# ╟─fa0334d4-bcb1-476f-bd30-cd4185fd097c
# ╠═45c39afd-d1fc-450c-b01e-60e908fd6a77
# ╠═083bd15a-8641-4bde-9cab-6d29e1313b28
# ╠═ad754270-ba87-48f5-a3bc-fb0d9295b90b
# ╠═6fec6bad-6498-40dc-99ed-80b9e6dded08
# ╟─6bc20914-405e-4a73-9364-411cce113e48
# ╠═e81044a3-4312-44ad-b3d8-a08943aa6327
# ╠═a7e98b20-d114-43a2-824e-3873980fd3d3
# ╠═917573e4-7061-42a1-bef0-62e1bbb8df30
# ╠═e941fc95-e9f1-465e-80f0-2440166ae330
# ╠═4db052cc-98b1-46bd-8de4-edb589ec9722
# ╠═17303253-c127-4f74-9132-bac1b7853de4
# ╟─1b61c420-2397-463d-84f0-3a25b1e7f174
# ╠═a6945d41-fe7d-492a-91af-b103ebd1372d
# ╠═994ceffb-9c16-40d5-a424-55b993da8e2d
# ╠═cb6c3703-4730-42f4-afbd-c87c1b085f38
# ╠═94c14598-b59d-48ba-a7a6-c43f3267e803
# ╠═be240571-5073-468a-97f9-1b2caa499167
# ╟─68a973e0-451d-4f8d-80a5-af477b67e744
# ╠═a67c3a82-a999-4332-b80a-ddc9ddfb756e
# ╠═4c9d487f-c271-486d-b9e7-c335b7959551
# ╠═86837f53-8007-4c61-a39f-4108b2460de5
# ╠═f3399882-5a76-488d-bc6f-5fd90e27c12a
