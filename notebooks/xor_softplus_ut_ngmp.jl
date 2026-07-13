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

This notebook trains a small Bayesian neural network on a noisy ``N \times M``
checkerboard — the ``(2, 2)`` special case is XOR; `checkboard_size` in
`config` selects the active grid — using **natural-gradient message passing
(NGMP)**, the algorithm from *Information Geometry of Message Passing*, as
implemented in the local `SurrogateModelling` package. Each neuron carries a
positive, input-dependent precision gate

```math
\gamma = \operatorname{softplus}(z_a) = \log(1 + e^{z_a}),
```

with a Gamma belief on ``\gamma`` and a Gaussian belief on ``z_a``. Both
messages through the Softplus factor are computed with NGMP; the tangent
projection is evaluated with the unscented transform.

## NGMP in a nutshell

Beliefs live in exponential families
``q_\lambda(z) = h(z)\exp\{\lambda^\top T(z) - A(\lambda)\}``, which carry two
coordinate systems: the natural parameters ``\lambda`` and the mean parameters
``\mu(\lambda) = \nabla_\lambda A(\lambda) = \mathbb{E}_{q_\lambda}[T(z)]``.
The Fisher information ``G(\lambda) = \nabla^2_\lambda A(\lambda)`` is the
Jacobian of the map between them, which gives the identity that powers
everything below: the *natural* gradient in ``\lambda`` is the *plain*
gradient in ``\mu``,

```math
\tilde\nabla_\lambda F = G(\lambda)^{-1} \nabla_\lambda F = \nabla_\mu F .
```

On a factor graph, the exact belief-propagation (BP) message
``\mu_{a \to i}(z_i)`` out of a non-conjugate factor such as Softplus has no
finite parameterization: its logarithm ``\ell_{a \to i} = \log \mu_{a \to i}``
is not a linear combination of the sufficient statistics ``T_i`` of the
receiving edge. NGMP sends instead the exponential-family message whose
natural parameter is the **natural gradient of the expected log-message**,
evaluated at the *current marginal of the receiving edge*:

```math
\eta_{a \to i}
  = \nabla_{\mu_i}\, \mathbb{E}_{q_{\lambda_i}}\!\bigl[ \ell_{a \to i}(z_i) \bigr],
\qquad
\hat\mu_{a \to i}(z_i) \propto \exp\{\eta_{a \to i}^\top T_i(z_i)\}.
```

Geometrically, ``\eta_{a \to i}`` is the projection of the exact log-message
onto the tangent space of the receiving family at ``q_{\lambda_i}`` in the
Fisher metric; the component orthogonal to the family is dropped. The
stationary points of the (form-constrained) Bethe free energy satisfy, on
every edge between factors ``b`` and ``c``,

```math
\lambda_i = \eta_{b \to i} + \eta_{c \to i},
```

the familiar BP "product of incoming messages", now with projected natural
parameters. This is the same fixed point as global natural-gradient
variational inference (CVI), derived edge-locally — so each edge can carry its
own family: below, the Softplus factor sends a **Gamma** message towards
``\gamma`` and a **Gaussian** message towards ``z_a``.

Compared with its relatives:

- **VMP** first tilts the factor by mean-field expectations and only then
  sends a conjugate message; the tilting changes the message content (the
  classic symptom is overconfident precision messages).
- **EP** projects a tilted marginal and divides out the cavity. NGMP has no
  cavity step: the projection point is simply the current receiving marginal,
  so an outgoing message depends on the belief it is sent *to*, and inference
  becomes a fixed-point iteration in the natural parameters.

Two practical ingredients complete the algorithm:

- **Unscented tangent projection.** The projection reduces to
  ``G_i^{-1}\operatorname{Cov}_{q_{\lambda_i}}\!\bigl[T_i(z_i),\, \ell_{a \to i}(z_i)\bigr]``,
  expectations with no closed form for Softplus.
  `TangentProjection(type = Unscented)` evaluates them with deterministic
  sigma points rather than Monte-Carlo samples.
- **Damping.** Writing one full sweep of projections as
  ``\Phi(\lambda)``, NGMP iterates ``\lambda^{(t+1)} = \Phi(\lambda^{(t)})``.
  `DampingMeta(alpha, beta, max_step)` relaxes the iteration in natural
  coordinates, ``\lambda^{(t+1)} = (1-\alpha)\lambda^{(t)} + \alpha\,\Phi(\lambda^{(t)})``,
  optionally adds momentum ``\beta``, and bounds the natural-parameter step.
  Damping changes the path, not the fixed point, and keeps the Gamma messages
  proper on this large factor graph.
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
        mse_snapshots = 12,
        mse_subsample = 240,
        mse_subsample_seed = 2_028,
        # Quick-look resolution. Raise for publication figures, or use
        # scripts/xor_softplus_ut_ngmp_animation.jl for the heavy renders.
        grid_size = 60,
        checkboard_size = (2, 2),
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

The network is a **precision-gated ensemble** of `n_neurons` linear neurons.
For an input with features ``x = [1, x_1, x_2]`` each neuron ``k`` owns two
weight vectors:

- a *mean branch* ``z^{\mathrm{mean}}_k \sim \operatorname{softdot}(x, w^{\mathrm{mean}}_k, \tau_{\mathrm{mean}})``
  — its local linear prediction, and
- a *gate branch* ``z^{a}_k \sim \operatorname{softdot}(x, w^{a}_k, \tau)``,
  pushed through the Softplus factor to yield a positive precision
  ``\gamma_k = \operatorname{softplus}(z^a_k)``.

Each neuron then attaches a factor
``\mathcal{N}(\mathrm{out} \mid z^{\mathrm{mean}}_k, \gamma_k^{-1})`` to the
shared output. Multiplying the ``K`` Gaussian factors gives

```math
p(\mathrm{out} \mid \cdot) \propto \mathcal{N}\!\left(
\mathrm{out} \,\middle|\,
\frac{\sum_k \gamma_k z^{\mathrm{mean}}_k}{\sum_k \gamma_k},\;
\Bigl(\sum_k \gamma_k\Bigr)^{-1}
\right),
```

a precision-weighted ensemble: ``\gamma_k(x)`` is an input-dependent
*responsibility*, so a neuron whose gate is large in some region of the input
plane dominates the prediction there — a mixture-of-experts built entirely
from Gaussian and Gamma beliefs.

The Softplus factor connects a **Gaussian** belief on ``z^a`` to a **Gamma**
belief on ``\gamma``. Neither exact BP message is representable in the
receiving family, so both are sent as NGMP tangent projections: the
`NGMPDependencies` objects below request the unscented projection on both
edges. The `NormalMeanPrecision` output factor gets the same treatment — its
exact message towards ``\gamma`` is likewise non-conjugate.

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
                meta = obs_damping
            }
        end
        y[observation] ~ NormalMeanPrecision(out[observation], obs_noise)
    end
end

# ╔═╡ 3c1f5a2e-8f4d-4b6a-9c3e-2d7b1e5a9f04
md"""
The variational family is structured, not fully mean-field: the joint cluster
``q(w_{\mathrm{mean}}, z_{\mathrm{mean}}, \mathrm{out}, z_a, \gamma)`` keeps
the coupling between each neuron's mean branch and its gate — exactly the
correlation that decides *which* neuron is responsible *where*. Severing it
with a full mean-field factorization would feed every neuron the same averaged
residual. `MomentForm()` stores the weight posteriors in moment
parameterization because the `softdot` rules repeatedly consume the same
weight means and covariances.
"""

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

# ╔═╡ b7e2c4d1-56a9-4f3b-8e0d-4a1c9f6b2e73
md"""
## Training

One `infer` iteration sweeps the whole graph once: every Softplus and every
`NormalMeanPrecision` output factor recomputes its unscented tangent
projection at the current edge marginals and applies the damped
natural-parameter update, then the conjugate part of the graph (the `softdot`
weight updates) responds with ordinary message passing. The Bethe free energy
is recorded per iteration as a convergence diagnostic. At the default config
(100 iterations over 640 training points and 8 neurons) this takes on the
order of a minute.
"""

# ╔═╡ 041edacb-5c47-487c-8117-0757e79d975f
ngmp_fit = run_softplus_ut_ngmp(
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

# ╔═╡ c5b2f8e3-7a1d-4c96-b04e-6f3a8d2c1e59
md"""
## Prediction is inference

There is no closed-form predictive function to evaluate: at a new input the
per-observation latents — the gate variables ``z_a`` and ``\gamma``, the mean
branches ``z_{\mathrm{mean}}`` and the ensemble output ``\mathrm{out}`` —
must themselves be inferred. Prediction therefore uses a second RxInfer graph
with the same local structure, in which

- the learned global marginals (weights, precisions, ``\beta``) are clamped
  with `FixedMarginalFormConstraint`, and
- each ``y`` receives a diffuse ``\mathcal{N}(0, 10^{12})`` factor instead of
  data, making it an effectively unobserved latent whose posterior ``q(y)``
  carries both latent model uncertainty and the learned observation noise.

Each batch runs `prediction_iterations` NGMP sweeps, because the gate messages
again require the fixed-point tangent projection. This is why evaluating the
model on many points is not free — and why this notebook tracks the learning
curve on a small test subsample and predicts over a full grid only once, at
the final training iteration. The heavy renders (high-resolution surface,
learning animation) live in `scripts/xor_softplus_ut_ngmp_animation.jl`.
"""

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

# ╔═╡ f2a7d9c4-3e61-48b2-95af-8c0d6e1b3f52
md"""
## Learning curve

Because every evaluation is an inference run, the learning curve is tracked
cheaply: test MSE is measured at `mse_snapshots` thinned training iterations
on a fixed random subsample of `mse_subsample` test points — enough to see
when learning saturates. The full test set is evaluated once, at the final
iteration, in the cell after next.
"""

# ╔═╡ ae1f0398-f75c-4539-bfd2-3fbcc98d17f8
begin
    snapshot_iterations = thinned_iterations(
        length(ngmp_fit.result.posteriors[:w_mean]),
        config.mse_snapshots,
    )
    prediction_output_mean = mean(train_data.OT)

    # A fixed random subsample keeps the per-snapshot prediction runs cheap;
    # the full test set is evaluated once at the final iteration below.
    mse_subsample_indices = let
        rng = StableRNG(config.mse_subsample_seed)
        n_subsample = min(config.mse_subsample, length(test_features))
        sort(randperm(rng, length(test_features))[1:n_subsample])
    end
    mse_features = test_features[mse_subsample_indices]
    mse_targets = test_data.OT[mse_subsample_indices]

    prediction_snapshots = map(snapshot_iterations) do training_iteration
        priors = softplus_prediction_priors(ngmp_fit.result, training_iteration)
        marginals = predict_softplus_marginals(
            priors,
            mse_features;
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
            iteration = training_iteration,
            test_mse = mean(abs2, statistics.mean .- mse_targets),
        )
    end

    constant_prediction = prediction_output_mean
    constant_mse = mean(abs2, constant_prediction .- mse_targets)
    constant_mse_full = mean(abs2, constant_prediction .- test_data.OT)
    mse_by_iteration = getproperty.(prediction_snapshots, :test_mse)
    mse_history = DataFrame(
        iteration = snapshot_iterations,
        test_mse = mse_by_iteration,
    )
end

# ╔═╡ e9d3b8a2-1c47-4f5e-a6b9-7d2f0c4e8a15
final_test_evaluation = let
    priors = softplus_prediction_priors(
        ngmp_fit.result,
        length(ngmp_fit.result.posteriors[:w_mean]),
    )
    marginals = predict_softplus_marginals(
        priors,
        test_features;
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
        mean = statistics.mean,
        variance = statistics.variance,
        mse = mean(abs2, statistics.mean .- test_data.OT),
    )
end

# ╔═╡ 87f5a99a-6251-44b1-bcd2-bf562715a35f
begin
    final_gamma = vec(ngmp_fit.result.posteriors[:γ][end])
    final_gamma_matrix = mean.(ngmp_fit.result.posteriors[:γ][end])
    final_total_gamma = vec(sum(final_gamma_matrix; dims = 1))
    run_summary = DataFrame(
        neurons = config.n_neurons,
        iterations = length(ngmp_fit.result.posteriors[:w_mean]),
        train_points = nrow(train_data),
        test_points = nrow(test_data),
        test_mse = final_test_evaluation.mse,
        constant_mse = constant_mse_full,
        minimum_predictive_variance = minimum(final_test_evaluation.variance),
        mean_predictive_variance = mean(final_test_evaluation.variance),
        maximum_predictive_variance = maximum(final_test_evaluation.variance),
        minimum_gamma_shape = minimum(shape, final_gamma),
        minimum_gamma_rate = minimum(rate, final_gamma),
        maximum_total_gamma = maximum(final_total_gamma),
        global_gamma_rate = mean(ngmp_fit.result.posteriors[:β][end]),
        gate_softdot_precision = mean(ngmp_fit.result.posteriors[:τ][end]),
        mean_softdot_precision = mean(ngmp_fit.result.posteriors[:τ_mean][end]),
        observation_precision = mean(ngmp_fit.result.posteriors[:obs_noise][end]),
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
        ylabel = "Test MSE ($(length(mse_targets))-point subsample)",
        title = "Checkerboard Softplus UT NGMP learning",
        label = "Softplus",
        xlims = (0.5, length(ngmp_fit.result.posteriors[:w_mean]) + 0.5),
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
        eachindex(ngmp_fit.result.free_energy[start_from:end]),
        ngmp_fit.result.free_energy[start_from:end];
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
    mean_weights = mean.(ngmp_fit.result.posteriors[:w_mean][end]),
    gate_weights = mean.(ngmp_fit.result.posteriors[:w_a][end]),
)

# ╔═╡ 8d66c7f6-75b7-4db6-8350-05257d80d0bb
md"""
## Learned surface

The final-iteration posterior is pushed through the prediction graph over a
`grid_size` × `grid_size` grid.
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
        ngmp_fit.result,
        length(ngmp_fit.result.posteriors[:w_mean]),
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

The animated version of the surface above — the predictive mean and variance
evolving across training iterations, next to the running test MSE — lives in
`scripts/xor_softplus_ut_ngmp_animation.jl`. The script is a self-contained
copy of this model (edit its `config` block to play with the architecture,
checkerboard size, and NGMP parameters); it writes
`viz/checkerboard_<N>x<M>_softplus_ut_ngmp_learning.gif` and a
high-resolution final surface PNG. It re-runs the prediction graph over a full
grid for every frame, so expect roughly 15–20 minutes at the default settings.
"""

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
# ╟─3c1f5a2e-8f4d-4b6a-9c3e-2d7b1e5a9f04
# ╠═70d0a30d-54c8-4515-bfb5-3d470686bc99
# ╠═7f098370-c2fc-4d0c-86d2-5487a1527765
# ╠═9e3a784a-ea10-4edc-b5c8-583bc3df77c0
# ╠═c9267a99-5dfc-4d89-b054-eab7a4c69484
# ╟─b7e2c4d1-56a9-4f3b-8e0d-4a1c9f6b2e73
# ╠═041edacb-5c47-487c-8117-0757e79d975f
# ╟─c5b2f8e3-7a1d-4c96-b04e-6f3a8d2c1e59
# ╠═db34de5d-f617-4236-aabb-3318863b8acf
# ╠═df426399-1486-4640-86f6-dfc6d5542f08
# ╠═0f9e2758-1b87-4444-944b-dd12b33d05dc
# ╠═6f70997c-dc59-4295-a168-4b9623165c9e
# ╠═f74c2668-9b51-4444-8c1e-e2027073d32d
# ╠═b59e6867-147d-43d3-9db0-ed2a07460dc7
# ╠═b3cf22f8-82e3-47a4-82a3-7dc98f9ffec8
# ╟─f2a7d9c4-3e61-48b2-95af-8c0d6e1b3f52
# ╠═ae1f0398-f75c-4539-bfd2-3fbcc98d17f8
# ╠═e9d3b8a2-1c47-4f5e-a6b9-7d2f0c4e8a15
# ╠═87f5a99a-6251-44b1-bcd2-bf562715a35f
# ╠═4758395c-51a4-43c4-b1e1-853678612c13
# ╠═726541a4-9de7-468f-9ac5-2ebcdcc18644
# ╟─a6830b67-90bd-41bd-a5be-85a6ace9eccd
# ╠═d45d778b-789a-4b66-a7c4-a2a42062007a
# ╠═8d66c7f6-75b7-4db6-8350-05257d80d0bb
# ╠═429bd3a0-8c79-4376-a6b6-eb0c63cffb77
# ╠═685ca7c0-ccea-4f7c-8067-5e948f0da331
# ╠═3395813c-c790-47c2-916e-75abb32355c4
# ╠═d47d2ef0-da37-4127-ad06-1bb602e655c4
# ╟─5bc819f5-61a4-47d3-ac36-5c022f9aca99
