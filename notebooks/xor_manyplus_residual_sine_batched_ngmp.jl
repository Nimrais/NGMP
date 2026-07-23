### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ d2109b90-89be-422b-beac-f432a358838e
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 7cb71685-a8f2-46f3-a906-5a3f705f70d2
begin
    ENV["GKSwstype"] = "100"

    using DataFrames
    using LinearAlgebra: Diagonal, diag, norm
    using Plots
    using Random
    using RxInfer
    using StableRNGs
    using Statistics
    using SurrogateModelling
end

# ╔═╡ 6c01c288-2794-401a-902b-b8c049407e65
md"""
# Fast batched ManyPlus residual-sine NGMP

This notebook fits a Bayesian additive model to a configurable noisy
checkerboard regression problem; ``2\times2`` is the XOR special case.

For neuron ``k`` and observation ``i``, the model is

```math
z_{ki}=w_k^\top[1,x_{i1},x_{i2}],\qquad
h_{ki}=\phi(z_{ki}),\qquad
c_{ki}=v_k h_{ki},\qquad
o_i=\sum_k c_{ki},
```

where ``\phi(z)=z+(\rho/\omega)\sin(\omega z)``. The linear residual makes
local propagation stable, while the sine term supplies diagonal ridge bases
for the alternating pattern.
"""

# ╔═╡ 2997efb1-9a49-4368-b2c6-7ec381d3eeff
config = (
    n_samples = 800,
    n_neurons = 8,
    train_fraction = 0.60,
    # This is the only training-batch setting to tune. Any integer from 1 to
    # the number of training observations works; batch sizes are derived below.
    n_training_batches = 8,
    max_batch_iterations = 125,
    stop_after_iteration = 75,
    stop_atol = 0.0,
    stop_rtol = 1e-2,
    noise_std = 0.10,
    data_seed = 2_026,
    split_seed = 2_027,
    prior_seed = 42,
    phi_rho = 0.9,
    phi_omega = 1.0,
    w_prior_scale = 2.2,
    w_prior_variance = 0.005,
    bias_prior_variance = 0.5,
    v_prior_scale = 0.5,
    v_prior_variance = 0.01,
    tau_prior = (1e3, 1.0),
    tau_c_prior = (1e4, 1.0),
    obs_noise_prior = (100.0, 1.0),
    ngmp_alpha = 0.05,
    ngmp_beta = 0.0,
    ngmp_max_step = 1.0,
    prediction_iterations = 8,
    prediction_batch_size = 1_024,
    prediction_prior_variance = 1e12,
)

# ╔═╡ 9d8c448a-3d46-4179-9a4c-c21e86994f4d
begin
    pattern_config = (
        # Number of alternating cells along (x, y). Examples:
        # (2, 2) = XOR, (3, 3) = a 3×3 checkerboard, (4, 2) = rectangular.
        checkerboard_size = (3, 1),
    )
    checkboardsize1, checkboardsize2 = pattern_config.checkerboard_size[1], pattern_config.checkerboard_size[2]
end

# ╔═╡ a208ea36-e397-4ae9-8f3c-43f16dc53d7e
grid_config = (
    # Grid refinement and domain are independent in x and y. Increasing either
    # point count raises prediction time roughly in proportion to x * y.
    grid_points = (x = 64, y = 64),
    grid_limits = (x = (-2.0, 2.0), y = (-2.0, 2.0)),
)

# ╔═╡ 8b889cf2-96fe-4f50-b011-1509bdb10f8d
md"""
## Data

Inputs are uniform on ``[-2,2]^2``. The clean cell label is perturbed by
Gaussian noise and clipped to ``[0,1]``. A seeded permutation selects
$(round(Int, 100 * config.train_fraction))% of the observations for training.
Their fixed order is divided as evenly as possible into
`n_training_batches` adjacent batches; no power-of-two restriction applies.

The selected target is a $checkboardsize1 x $checkboardsize2 checkerboard. Edit `pattern_config.checkerboard_size = (x_cells, y_cells)`
above before running the notebook. More cells create a higher-frequency and
usually harder target for the fixed eight-neuron model.
"""

# ╔═╡ 5f97ef05-c7c8-4f39-94b9-77eda0c971da
begin
    function checkerboard_label(x1, x2, checkerboard_size)
        nx, ny = checkerboard_size
        nx > 0 && ny > 0 ||
            throw(ArgumentError("checkerboard dimensions must be positive"))
        cell_x = clamp(floor(Int, nx * (x1 + 2) / 4), 0, nx - 1)
        cell_y = clamp(floor(Int, ny * (x2 + 2) / 4), 0, ny - 1)
        return Float64(isodd(cell_x + cell_y))
    end

    function make_checkerboard_dataset(;
        n::Int,
        checkerboard_size::Tuple{Int, Int},
        noise_std::Float64,
        seed::Int,
    )
        rng = StableRNG(seed)
        x1 = 4 .* rand(rng, n) .- 2
        x2 = 4 .* rand(rng, n) .- 2
        clean = checkerboard_label.(x1, x2, Ref(checkerboard_size))
        target = clamp.(clean .+ noise_std .* randn(rng, n), 0.0, 1.0)
        return DataFrame(x1 = x1, x2 = x2, OT = target)
    end

    function split_dataset(df; train_fraction, seed)
        0 < train_fraction < 1 ||
            throw(ArgumentError("train_fraction must be in (0, 1)"))
        rng = StableRNG(seed)
        indices = randperm(rng, nrow(df))
        n_train = round(Int, train_fraction * nrow(df))
        return df[indices[1:n_train], :], df[indices[(n_train + 1):end], :]
    end

    function deterministic_batch_ranges(n_observations, n_batches)
        n_observations > 0 ||
            throw(ArgumentError("training set must not be empty"))
        1 <= n_batches <= n_observations || throw(ArgumentError(
            "n_training_batches must be between 1 and $n_observations",
        ))

        base_size, remainder = divrem(n_observations, n_batches)
        batch_sizes = fill(base_size, n_batches)
        batch_sizes[1:remainder] .+= 1

        ranges = Vector{UnitRange{Int}}(undef, n_batches)
        start_index = 1
        for batch in eachindex(batch_sizes)
            stop_index = start_index + batch_sizes[batch] - 1
            ranges[batch] = start_index:stop_index
            start_index = stop_index + 1
        end
        @assert start_index == n_observations + 1
        return ranges
    end

    build_features(df) =
        [[1.0, df.x1[index], df.x2[index]] for index in 1:nrow(df)]
end

# ╔═╡ c2d1af82-7dab-4fc4-9f77-fee10f59b557
begin
    dataset = make_checkerboard_dataset(
        n = config.n_samples,
        checkerboard_size = pattern_config.checkerboard_size,
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
    training_batches = deterministic_batch_ranges(
        nrow(train_data),
        config.n_training_batches,
    )

    @assert nrow(dataset) == config.n_samples
    @assert nrow(train_data) ==
        round(Int, config.train_fraction * config.n_samples)
    @assert nrow(test_data) == nrow(dataset) - nrow(train_data)
    @assert length(training_batches) == config.n_training_batches
    @assert vcat(collect.(training_batches)...) == collect(1:nrow(train_data))
    nothing
end

# ╔═╡ 6e8c94b2-c65f-4c4c-8877-4819ad5ae68d
data_summary = DataFrame(
    checkerboard_x_cells = pattern_config.checkerboard_size[1],
    checkerboard_y_cells = pattern_config.checkerboard_size[2],
    observations = nrow(dataset),
    training_observations = nrow(train_data),
    test_observations = nrow(test_data),
    deterministic_batches = length(training_batches),
    smallest_batch = minimum(length, training_batches),
    largest_batch = maximum(length, training_batches),
    batch_sizes = join(length.(training_batches), ", "),
)

# ╔═╡ 49efceb5-40ff-4740-9b8d-75cfb173e955
pattern_preview = scatter(
    dataset.x1,
    dataset.x2;
    marker_z = dataset.OT,
    color = :RdBu,
    clims = (0, 1),
    markersize = 3.6,
    markeralpha = 0.68,
    markerstrokecolor = "#36454F",
    markerstrokewidth = 0.3,
    label = "",
    xlabel = "x1",
    ylabel = "x2",
    title = "Noisy $(pattern_config.checkerboard_size[1])×$(pattern_config.checkerboard_size[2]) checkerboard",
    aspect_ratio = :equal,
    xlims = (-2, 2),
    ylims = (-2, 2),
)

# ╔═╡ ca2fb580-1aad-4019-b1fe-5e4f3f5415e7
md"""
## NGMP and the additive basis

Natural-gradient message passing projects a non-conjugate factor's outgoing
log-message onto the receiving exponential family at its current marginal.
Here `ResidualSine` uses a closed-form tangent projection. Damping updates the
natural parameters by only 5% of each proposed step, with no momentum.

The weight priors come in pairs. Two neurons share a ridge direction, have
opposite biases and opposite coefficient signs. Their linear residuals cancel,
leaving a cosine-like ridge; alternating the pair phases also controls the
constant contribution. This symmetry breaking is in the priors rather than in
arbitrary initial marginals.

`ManyPlus` is an explicit sum node. Its forward mean is the sum of component
means, and its uncertainty includes the component variances instead of adding
precisions as a product-of-experts construction would.
"""

# ╔═╡ c1d1a7eb-8d87-4035-8c34-71fe0396b679
begin
    function make_manyplus_priors(config)
        rng = StableRNG(config.prior_seed)
        iseven(config.n_neurons) ||
            throw(ArgumentError("paired priors require an even neuron count"))
        n_pairs = config.n_neurons ÷ 2
        prior_precision = Diagonal([
            1 / config.bias_prior_variance,
            1 / config.w_prior_variance,
            1 / config.w_prior_variance,
        ])
        w = Vector{Any}(undef, config.n_neurons)
        v = Vector{Any}(undef, config.n_neurons)

        for pair in 1:n_pairs
            angle = π * (pair - 1) / n_pairs + 0.1 * randn(rng)
            radius = config.w_prior_scale * (1 + 0.05 * randn(rng))
            direction = [radius * cos(angle), radius * sin(angle)]
            bias = (isodd(pair) ? 1.0 : -1.0) *
                   (π / 4) * (1 + 0.1 * randn(rng))

            for (slot, sign) in ((2pair - 1, 1.0), (2pair, -1.0))
                prior_mean = [sign * bias, direction[1], direction[2]]
                w[slot] = MvNormalWeightedMeanPrecision(
                    prior_precision * prior_mean,
                    prior_precision,
                )
                v[slot] = NormalMeanVariance(
                    sign * config.v_prior_scale,
                    config.v_prior_variance,
                )
            end
        end

        return Dict{Symbol, Any}(
            :intercept => NormalMeanVariance(0, 100),
            :w => w,
            :v => v,
            :τ => GammaShapeRate(config.tau_prior...),
            :τ_c => GammaShapeRate(config.tau_c_prior...),
            :obs_noise => GammaShapeRate(config.obs_noise_prior...),
        )
    end

    function make_activation_dependencies(config)
        return NGMPDependencies(
            out = nothing,
            in = nothing,
            projection = TangentProjection(type = ClosedForm),
            damping = DampingMeta(
                alpha = config.ngmp_alpha,
                beta = config.ngmp_beta,
                max_step = config.ngmp_max_step,
            ),
        )
    end
end

# ╔═╡ 9734f371-3af4-4906-b7ed-729371bd9dbc
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

    intercept ~ priors[:intercept]

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
        y[observation] ~ NormalMeanPrecision(out[observation]+intercept, obs_noise)
    end
end

# ╔═╡ 94c063e2-be4d-48eb-85f2-be656bec70c4
@constraints function xor_manyplus_constraints()
    q(w, v, za, h, c, out, intercept, τ, τ_c, obs_noise) =
        q(w, za, h, c, out, intercept)q(v)q(τ)q(τ_c)q(obs_noise)

    q(w)::MomentForm()
end

# ╔═╡ 5e2aba98-14c2-4a37-baaf-4973bf4fa82f
md"""
## Structured inference

Training preserves the factorization

```math
q(w,z_a,h,c,o)\,q(v)\,q(\tau)\,q(\tau_c)\,q(\tau_{\mathrm{obs}}).
```

In particular, ``w`` stays coupled to the complete local deterministic chain
``z_a\rightarrow h\rightarrow c\rightarrow o``. The
`q(w)::MomentForm()` constraint changes only the marginal representation used
for repeated moment access; it does **not** introduce a mean-field cut or
change this factorization.
"""

# ╔═╡ eb185ec0-130f-4299-96a1-ad5a8a037779
begin
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
        μ(intercept) = priors[:intercept]
    end

    function pushforward_inits(priors, features, config)
        activation = ResidualSineMeta(
            rho = config.phi_rho,
            omega = config.phi_omega,
        )
        phi(x) = SurrogateModelling._residual_sine(x, activation)
        w_means = mean.(priors[:w])
        v_means = mean.(priors[:v])
        n = length(features)

        za = [
            NormalMeanVariance(dot(w_means[k], features[i]), 0.5)
            for k in 1:config.n_neurons, i in 1:n
        ]
        h = [
            NormalMeanVariance(phi(mean(za[k, i])), 1.0)
            for k in 1:config.n_neurons, i in 1:n
        ]
        c = [
            NormalMeanVariance(v_means[k] * mean(h[k, i]), 1.0)
            for k in 1:config.n_neurons, i in 1:n
        ]
        out = [
            NormalMeanVariance(
                sum(mean(c[k, i]) for k in 1:config.n_neurons),
                1.0,
            )
            for i in 1:n
        ]
        return (za = za, h = h, c = c, out = out)
    end
end

# ╔═╡ 2a47b1cd-fe34-4bce-a895-ecf8e885e544
md"""
## One-pass posterior-as-prior batching

Each automatically sized batch gets a newly built factor graph, fresh local
marginals, fresh NGMP damping state, and a fresh early-stopping object. Only
the five global posteriors ``w``, ``v``, ``\tau``, ``\tau_c``, and
observation-noise precision are transported as the next batch's priors.

Every observation is processed exactly once. Repeating these batches as
"epochs" would multiply the same likelihood factors into the carried posterior
again, double-counting observations and producing unjustified certainty.
Early stopping compares consecutive free energies with `atol = 0` and
`rtol = 1e-6`, but the callback is enabled only after iteration 75.
"""

# ╔═╡ 888e3f6e-5f07-4aa7-ba33-480e52a79185
begin
    function global_posterior_priors(result)
        return Dict{Symbol, Any}(
            :intercept => deepcopy(result.posteriors[:intercept]),
            :w => deepcopy(collect(vec(result.posteriors[:w]))),
            :v => deepcopy(collect(vec(result.posteriors[:v]))),
            :τ => deepcopy(result.posteriors[:τ]),
            :τ_c => deepcopy(result.posteriors[:τ_c]),
            :obs_noise => deepcopy(result.posteriors[:obs_noise]),
        )
    end

    function run_training_batch(priors, observations, features, config)
        # Both objects below are deliberately constructed inside this function:
        # neither callback history nor mutable NGMP edge state crosses a batch.
        stopper = StopEarlyIterationStrategy(config.stop_atol, config.stop_rtol)
        delayed_stopper = function (event)
            if event.iteration > config.stop_after_iteration
                stopper(event)
            end
            return nothing
        end
        activation = ResidualSineMeta(
            rho = config.phi_rho,
            omega = config.phi_omega,
        )

        local result
        elapsed_seconds = @elapsed begin
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
                    priors,
                    pushforward_inits(priors, features, config),
                ),
                returnvars = (
                    w = KeepLast(),
                    v = KeepLast(),
                    τ = KeepLast(),
                    τ_c = KeepLast(),
                    obs_noise = KeepLast(),
                    intercept = KeepLast(),
                ),
                iterations = config.max_batch_iterations,
                free_energy = true,
                callbacks = (after_iteration = delayed_stopper,),
                showprogress = false,
                options = (limit_stack_depth = 100,),
                disable_inference_error_hint = true,
            )
        end

        iterations_run = length(result.free_energy)
        all(isfinite, result.free_energy) ||
            error("batch produced a non-finite free energy")

        updated_priors = global_posterior_priors(result)
        return (
            priors = updated_priors,
            iterations = iterations_run,
            elapsed_seconds = elapsed_seconds,
            free_energy = Float64.(result.free_energy),
            final_free_energy = Float64(last(result.free_energy)),
        )
    end

    function run_batched_training(
        train_data,
        train_features,
        training_batches,
        config,
    )
        carried_priors = make_manyplus_priors(config)
        reports = NamedTuple[]

        for (batch_number, indices) in enumerate(training_batches)
            batch_fit = run_training_batch(
                carried_priors,
                train_data.OT[indices],
                train_features[indices],
                config,
            )
            carried_priors = batch_fit.priors
            report = (
                batch = batch_number,
                observations = length(indices),
                stopping_iteration = batch_fit.iterations,
                elapsed_seconds = batch_fit.elapsed_seconds,
                free_energy = batch_fit.free_energy,
                final_free_energy = batch_fit.final_free_energy,
            )
            push!(reports, report)
            println(
                "batch $(report.batch)/$(length(training_batches)): ",
                "iteration $(report.stopping_iteration), ",
                "$(round(report.elapsed_seconds; digits = 2)) s, ",
                "final FE = $(round(report.final_free_energy; digits = 3))",
            )
        end

        return (priors = carried_priors, reports = reports)
    end
end

# ╔═╡ b5520663-afaa-4679-92a2-03c6f45e4987
begin
    training_wall_seconds = @elapsed begin
        batched_fit = run_batched_training(
            train_data,
            train_features,
            training_batches,
            config,
        )
    end
    final_priors = batched_fit.priors
    nothing
end

# ╔═╡ c3484d04-2b62-4f92-bdea-8e853f1424af
batch_summary = DataFrame(
    batch = getindex.(batched_fit.reports, :batch),
    observations = getindex.(batched_fit.reports, :observations),
    stopping_iteration = getindex.(batched_fit.reports, :stopping_iteration),
    elapsed_seconds = round.(
        getindex.(batched_fit.reports, :elapsed_seconds);
        digits = 2,
    ),
    final_free_energy = getindex.(batched_fit.reports, :final_free_energy),
)

# ╔═╡ 718d28dd-c22a-4dd3-b7bd-8d8e4feda416
md"""
Each row above is one complete and non-repeated batch. `stopping_iteration` is
the number actually executed, not the configured cap of 125.
"""

# ╔═╡ eddd638a-6812-44c7-832b-90c1523ae192
free_energy_plot = let
    panel = plot(
        xlabel = "Cumulative inference iteration",
        ylabel = "Bethe free energy",
        title = "Free energy through one-pass training batches",
        legend = :outerright,
        size = (1_050, 420),
    )
    iteration_offset = 0
    for (index, report) in enumerate(batched_fit.reports)
        batch_iterations = iteration_offset .+ collect(eachindex(report.free_energy))
        plot!(
            panel,
            batch_iterations,
            report.free_energy;
            linewidth = 2,
            label = "Batch $(report.batch) (n=$(report.observations))",
        )
        iteration_offset += length(report.free_energy)
        if index < length(batched_fit.reports)
            vline!(
                panel,
                [iteration_offset + 0.5];
                color = :gray45,
                linestyle = :dash,
                linewidth = 1,
                label = "",
            )
        end
    end
    panel
end

# ╔═╡ 3d4b2176-2ecf-48af-bd02-87ec8f11186f
md"""
The colored segments follow the actual batch order, and dashed lines mark
where one batch ends and the next fresh inference graph begins. Within a
segment, stabilization of free energy is the convergence diagnostic used by
early stopping.

The vertical level may jump at a batch boundary because both the observations
in the graph and the carried global prior have changed. The concatenated curve
is therefore not one continuously optimized global objective: interpret the
shape within each batch, rather than treating cross-boundary jumps as inference
instability. The table reports the final value and stopping iteration for each
segment.
"""

# ╔═╡ f2e910f5-43e5-4994-afbc-1d96ced290e2
md"""
## Prediction

Prediction uses a fresh graph too. The five learned globals are fixed while a
new local chain is inferred for each test or grid point. Eight iterations are
enough for this feed-forward prediction graph; observations are represented by
the model likelihood plus a numerically diffuse pseudo-prior.
"""

# ╔═╡ 9ef0ead1-97d0-470d-b6cf-dc79dce1dc34
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

# ╔═╡ 05a905c3-c3c5-4ba2-9b76-cd69c182fc3a
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

# ╔═╡ a567144d-a024-4af3-a4a7-c78dd270dab2
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

# ╔═╡ 02946b2f-2d6f-466c-9726-6ce7f80139b2
begin
    function run_manyplus_prediction_batch(
        priors,
        features,
        config;
        output_mean,
    )
        isempty(features) && return Any[]
        activation = ResidualSineMeta(
            rho = config.phi_rho,
            omega = config.phi_omega,
        )
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
            error("prediction graph returned the wrong number of marginals")
        return marginals
    end

    function predict_manyplus_marginals(
        priors,
        features,
        config;
        output_mean,
    )
        marginals = Vector{Any}(undef, length(features))
        for first_index in
            1:config.prediction_batch_size:length(features)
            indices = first_index:min(
                first_index + config.prediction_batch_size - 1,
                length(features),
            )
            marginals[indices] = run_manyplus_prediction_batch(
                priors,
                features[indices],
                config;
                output_mean = output_mean,
            )
        end
        return marginals
    end

    function predictive_statistics(marginals)
        predictive_means = Float64.(mean.(marginals))
        predictive_variances = Float64.(var.(marginals))
        all(isfinite, predictive_means) ||
            error("prediction graph produced a non-finite mean")
        all(
            value -> isfinite(value) && value > 0,
            predictive_variances,
        ) || error("prediction graph produced a non-positive variance")
        return (mean = predictive_means, variance = predictive_variances)
    end
end

# ╔═╡ 15688546-2645-45f9-89a9-d8b62e214169
begin
    prediction_output_mean = mean(train_data.OT)
    test_prediction_seconds = @elapsed begin
        test_prediction = predictive_statistics(predict_manyplus_marginals(
            final_priors,
            test_features,
            config;
            output_mean = prediction_output_mean,
        ))
    end
    full_test_mse = mean(abs2, test_prediction.mean .- test_data.OT)
    constant_predictor_mse = mean(abs2, prediction_output_mean .- test_data.OT)
    full_test_mse < constant_predictor_mse || error(
        "learned test MSE $full_test_mse did not beat constant MSE " *
        "$constant_predictor_mse",
    )
    nothing
end

# ╔═╡ 6a801c2a-ac06-49d7-aef5-c8b14db6eacb
run_summary = DataFrame(
    full_test_mse = full_test_mse,
    constant_predictor_mse = constant_predictor_mse,
    beats_constant = full_test_mse < constant_predictor_mse,
    training_seconds = round(training_wall_seconds; digits = 2),
    test_prediction_seconds = round(test_prediction_seconds; digits = 2),
    minimum_predictive_variance = minimum(test_prediction.variance),
    mean_predictive_variance = mean(test_prediction.variance),
    maximum_predictive_variance = maximum(test_prediction.variance),
    predictive_variance_ratio =
        maximum(test_prediction.variance) / minimum(test_prediction.variance),
)

# ╔═╡ e05b1025-3d1a-44fc-835d-57b8d228a34f
learned_weights = DataFrame(
    neuron = 1:config.n_neurons,
    v_mean = mean.(final_priors[:v]),
    v_std = sqrt.(var.(final_priors[:v])),
    w_bias = [mean(final_priors[:w][k])[1] for k in 1:config.n_neurons],
    w_x1 = [mean(final_priors[:w][k])[2] for k in 1:config.n_neurons],
    w_x2 = [mean(final_priors[:w][k])[3] for k in 1:config.n_neurons],
    ridge_norm = [
        norm(mean(final_priors[:w][k])[2:3]) for k in 1:config.n_neurons
    ],
)

# ╔═╡ f54b814e-c012-46dc-aa26-feb1959d20ed
md"""
## Predictive surfaces

The default $(grid_config.grid_points.x) × $(grid_config.grid_points.y) grid is
intentionally modest. Edit `grid_points = (x = ..., y = ...)` to control its
refinement, and edit `grid_limits = (x = (left, right), y = (bottom, top))` to
evaluate any axis-aligned rectangle. The x and y point counts need not match.

The third panel is a scatter plot of **all
800 noisy measurements**, colored by target value with fixed color limits
``(0,1)``. It stays on the full ``[-2,2]^2`` data domain even when the
prediction rectangle changes, so observations are never clipped from view.
""" 

# ╔═╡ 4f8e693a-12e2-4cf3-8c55-780bb84b674f
grid = let
    nx = grid_config.grid_points.x
    ny = grid_config.grid_points.y
    x_limits = grid_config.grid_limits.x
    y_limits = grid_config.grid_limits.y
    nx >= 2 && ny >= 2 ||
        throw(ArgumentError("grid point counts must both be at least 2"))
    x_limits[1] < x_limits[2] ||
        throw(ArgumentError("grid x limits must be strictly increasing"))
    y_limits[1] < y_limits[2] ||
        throw(ArgumentError("grid y limits must be strictly increasing"))

    x = range(x_limits...; length = nx)
    y = range(y_limits...; length = ny)
    features = [
        [1.0, x1, x2] for x2 in y for x1 in x
    ]
    (
        x = x,
        y = y,
        nx = nx,
        ny = ny,
        x_limits = x_limits,
        y_limits = y_limits,
        features = features,
    )
end

# ╔═╡ ec214b80-6871-4b34-b077-5189e7b10f54
begin
    grid_prediction_seconds = @elapsed begin
        grid_statistics = predictive_statistics(predict_manyplus_marginals(
            final_priors,
            grid.features,
            config;
            output_mean = prediction_output_mean,
        ))
        grid_prediction = (
            # Features have x varying fastest. Plots expects y rows by x
            # columns, hence the transpose after reshaping to x by y.
            mean = Matrix(permutedims(reshape(
                grid_statistics.mean,
                grid.nx,
                grid.ny,
            ))),
            variance = Matrix(permutedims(reshape(
                grid_statistics.variance,
                grid.nx,
                grid.ny,
            ))),
        )
    end
    nothing
end

# ╔═╡ 55c61034-cd4e-4e56-b9be-e27d85f4b73c
final_surface_plot = let
    variance_minimum = minimum(grid_prediction.variance)
    variance_maximum = maximum(grid_prediction.variance)
    variance_limits = variance_minimum == variance_maximum ?
                      (variance_minimum, nextfloat(variance_maximum)) :
                      (variance_minimum, variance_maximum)

    mean_panel = heatmap(
        grid.x,
        grid.y,
        grid_prediction.mean;
        color = :RdBu,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Predictive mean",
        aspect_ratio = :equal,
        xlims = grid.x_limits,
        ylims = grid.y_limits,
    )
    variance_panel = heatmap(
        grid.x,
        grid.y,
        grid_prediction.variance;
        color = :viridis,
        clims = variance_limits,
        xlabel = "x1",
        ylabel = "x2",
        title = "Predictive variance",
        aspect_ratio = :equal,
        xlims = grid.x_limits,
        ylims = grid.y_limits,
    )
    observed_panel = scatter(
        dataset.x1,
        dataset.x2;
        marker_z = dataset.OT,
        color = :RdBu,
        clims = (0, 1),
        markersize = 3.6,
        markeralpha = 0.68,
        markerstrokecolor = "#36454F",
        markerstrokewidth = 0.3,
        label = "",
        xlabel = "x1",
        ylabel = "x2",
        title = "Noisy $(pattern_config.checkerboard_size[1])×$(pattern_config.checkerboard_size[2]) targets (n = 800)",
        aspect_ratio = :equal,
        xlims = (-2, 2),
        ylims = (-2, 2),
    )

    figure = plot(
        mean_panel,
        variance_panel,
        observed_panel;
        layout = (1, 3),
        size = (1_350, 420),
    )
    if haskey(ENV, "MANYPLUS_NOTEBOOK_FIGURE")
        savefig(figure, ENV["MANYPLUS_NOTEBOOK_FIGURE"])
    end
    figure
end

# ╔═╡ e75018fd-f900-4552-866f-f454c8d4da73
md"""
Intermediate colors in the observed-target panel are noisy measured values,
not missing cells or an interpolated clean surface. Moderate transparency and
thin charcoal outlines keep those mid-valued observations visible where
points overlap.
"""

# ╔═╡ 152908ab-38f3-4eb1-a9af-13e781f0cb4f
md"""
## Interpretation

The learned model reaches full-test MSE **$(round(full_test_mse; digits = 4))**
versus **$(round(constant_predictor_mse; digits = 4))** for the training-mean
constant predictor. Test predictive variance ranges from
**$(round(minimum(test_prediction.variance); sigdigits = 4))** to
**$(round(maximum(test_prediction.variance); sigdigits = 4))**. Training took
**$(round(training_wall_seconds; digits = 2)) s**, while test and grid
prediction took **$(round(test_prediction_seconds + grid_prediction_seconds;
digits = 2)) s** together.

The paired rows in `learned_weights` show how the model retains ridge
directions while adjusting coefficients and biases. The variance panel is the
uncertainty of the additive predictive distribution, including learned
observation noise.
"""

# ╔═╡ Cell order:
# ╠═d2109b90-89be-422b-beac-f432a358838e
# ╠═7cb71685-a8f2-46f3-a906-5a3f705f70d2
# ╠═6c01c288-2794-401a-902b-b8c049407e65
# ╠═2997efb1-9a49-4368-b2c6-7ec381d3eeff
# ╠═9d8c448a-3d46-4179-9a4c-c21e86994f4d
# ╠═a208ea36-e397-4ae9-8f3c-43f16dc53d7e
# ╟─8b889cf2-96fe-4f50-b011-1509bdb10f8d
# ╠═5f97ef05-c7c8-4f39-94b9-77eda0c971da
# ╠═c2d1af82-7dab-4fc4-9f77-fee10f59b557
# ╠═6e8c94b2-c65f-4c4c-8877-4819ad5ae68d
# ╠═49efceb5-40ff-4740-9b8d-75cfb173e955
# ╟─ca2fb580-1aad-4019-b1fe-5e4f3f5415e7
# ╠═c1d1a7eb-8d87-4035-8c34-71fe0396b679
# ╠═9734f371-3af4-4906-b7ed-729371bd9dbc
# ╠═94c063e2-be4d-48eb-85f2-be656bec70c4
# ╟─5e2aba98-14c2-4a37-baaf-4973bf4fa82f
# ╠═eb185ec0-130f-4299-96a1-ad5a8a037779
# ╟─2a47b1cd-fe34-4bce-a895-ecf8e885e544
# ╠═888e3f6e-5f07-4aa7-ba33-480e52a79185
# ╠═b5520663-afaa-4679-92a2-03c6f45e4987
# ╠═c3484d04-2b62-4f92-bdea-8e853f1424af
# ╟─718d28dd-c22a-4dd3-b7bd-8d8e4feda416
# ╠═eddd638a-6812-44c7-832b-90c1523ae192
# ╟─3d4b2176-2ecf-48af-bd02-87ec8f11186f
# ╟─f2e910f5-43e5-4994-afbc-1d96ced290e2
# ╠═9ef0ead1-97d0-470d-b6cf-dc79dce1dc34
# ╠═05a905c3-c3c5-4ba2-9b76-cd69c182fc3a
# ╠═a567144d-a024-4af3-a4a7-c78dd270dab2
# ╠═02946b2f-2d6f-466c-9726-6ce7f80139b2
# ╠═15688546-2645-45f9-89a9-d8b62e214169
# ╠═6a801c2a-ac06-49d7-aef5-c8b14db6eacb
# ╠═e05b1025-3d1a-44fc-835d-57b8d228a34f
# ╟─f54b814e-c012-46dc-aa26-feb1959d20ed
# ╠═4f8e693a-12e2-4cf3-8c55-780bb84b674f
# ╠═ec214b80-6871-4b34-b077-5189e7b10f54
# ╠═55c61034-cd4e-4e56-b9be-e27d85f4b73c
# ╟─e75018fd-f900-4552-866f-f454c8d4da73
# ╟─152908ab-38f3-4eb1-a9af-13e781f0cb4f
