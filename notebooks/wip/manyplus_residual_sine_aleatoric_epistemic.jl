### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 0e868b28-a0f4-4fdb-90f2-6652ff5ed862
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 9cf6f84c-9747-463d-870e-2b993ec0f4b7
begin
    ENV["GKSwstype"] = "100"

    using DataFrames
    using LinearAlgebra: Diagonal, diag, dot
    using Plots
    using Random
    using RxInfer
    using StableRNGs
    using Statistics
    using SurrogateModelling
    import Distributions
end

# ╔═╡ d67bc7b9-f64f-4ee6-8770-e6d7851bc35a
md"""
# Does the ManyPlus residual-sine model learn aleatoric and epistemic uncertainty?

This notebook applies the architecture from
`xor_manyplus_residual_sine_batched_ngmp.jl` to two one-dimensional synthetic
regression checks. Each task uses one training graph and one `infer` call over
all its observations; there is no posterior-as-prior batching.

**Aleatoric benchmark**

```math
x\sim\mathcal N(0,1),\qquad
y=-(x+0.5)\sin(3\pi x)+
  \mathcal N\!\left(0,\,[0.45(x+0.5)]^2\right).
```

**Epistemic benchmark**

```math
y=x^3+\mathcal N(0,9),\qquad
x\sim\tfrac12\mathcal U[-5,-3]+\tfrac12\mathcal U[3,5].
```

The missing interval ``(-3,3)`` is never shown during epistemic training.

The important part of this notebook is the decomposition. A predictive
variance alone does not identify its source. For global variables
``\theta=(w,v,b,\tau_{1:H},\tau_{c,1:H},\tau_{\rm obs})``, we evaluate

```math
\operatorname{Var}(y\mid x,\mathcal D)=
\underbrace{\operatorname{Var}_{q(\theta)}
 [\mathbb E(y\mid x,\theta)]}_{\text{epistemic}}+
\underbrace{\mathbb E_{q(\theta)}
 [\operatorname{Var}(y\mid x,\theta)]}_{\text{aleatoric}}.
```

There is no posterior Monte Carlo here. Gaussian expectations through
`ResidualSine` are analytic. The remaining one-dimensional expectation over
each independent per-neuron Gamma precision uses a deterministic
midpoint-quantile rule.
"""

# ╔═╡ 425a650d-93bb-4847-a3bc-74906ddef27d
begin
    env_int(name, default) = parse(Int, get(ENV, name, string(default)))
    env_float(name, default) =
        parse(Float64, get(ENV, name, string(default)))
    env_bool(name, default) = lowercase(strip(get(
        ENV, name, default ? "true" : "false",
    ))) in ("1", "true", "yes", "on")

    smoke_mode = env_bool("MANYPLUS_UNCERTAINTY_SMOKE", false)

    config = (
        n_aleatoric = env_int(
            "MANYPLUS_UNCERTAINTY_N_ALEATORIC", smoke_mode ? 16 : 40,
        ),
        n_epistemic = env_int(
            "MANYPLUS_UNCERTAINTY_N_EPISTEMIC", smoke_mode ? 16 : 40,
        ),
        n_neurons = env_int("MANYPLUS_UNCERTAINTY_NEURONS", 8),
        max_training_iterations = env_int(
            "MANYPLUS_UNCERTAINTY_TRAINING_ITERATIONS",
            smoke_mode ? 4 : 500,
        ),
        min_training_iterations = env_int(
            "MANYPLUS_UNCERTAINTY_MIN_TRAINING_ITERATIONS",
            smoke_mode ? 1 : 30,
        ),
        stop_atol = 0.0,
        stop_rtol = 1e-5,
        aleatoric_x_std = env_float(
            "MANYPLUS_UNCERTAINTY_X_STD", 1.0,
        ),
        data_seed = env_int("MANYPLUS_UNCERTAINTY_DATA_SEED", 2_026),
        prior_seed = env_int("MANYPLUS_UNCERTAINTY_PRIOR_SEED", 42),
        phi_rho = 0.9,
        phi_omega = 1.0,
        w_prior_scale = 2.2,
        w_prior_variance = 0.005,
        bias_prior_variance = 0.5,
        intercept_prior_variance = 100.0,
        v_prior_scale = 0.5,
        v_prior_variance = 0.01,
        tau_prior_mean = env_float(
            "MANYPLUS_UNCERTAINTY_TAU_PRIOR_MEAN", 1e3,
        ),
        tau_prior_shape = env_float(
            "MANYPLUS_UNCERTAINTY_TAU_PRIOR_SHAPE", 2.0,
        ),
        tau_c_prior_mean = env_float(
            "MANYPLUS_UNCERTAINTY_TAU_C_PRIOR_MEAN", 1e4,
        ),
        tau_c_prior_shape = env_float(
            "MANYPLUS_UNCERTAINTY_TAU_C_PRIOR_SHAPE", 2.0,
        ),
        obs_noise_prior = (100.0, 1.0),
        ngmp_alpha = 0.05,
        ngmp_beta = 0.0,
        ngmp_max_step = 1.0,
        grid_points = env_int(
            "MANYPLUS_UNCERTAINTY_GRID_POINTS", smoke_mode ? 41 : 241,
        ),
        precision_quadrature_points = env_int(
            "MANYPLUS_UNCERTAINTY_QUADRATURE", smoke_mode ? 6 : 48,
        ),
        prediction_iterations = env_int(
            "MANYPLUS_UNCERTAINTY_PREDICTION_ITERATIONS", 128,
        ),
        prediction_prior_variance = 1e12,
    )

    iseven(config.n_neurons) ||
        throw(ArgumentError("paired priors require an even neuron count"))
    minimum((config.n_aleatoric, config.n_epistemic)) >= 2 ||
        throw(ArgumentError("each data set needs at least two observations"))
    0 <= config.min_training_iterations <
        config.max_training_iterations ||
        throw(ArgumentError(
            "minimum training iterations must be below the safety cap",
        ))
    config.grid_points >= 5 ||
        throw(ArgumentError("grid_points must be at least five"))
    config.precision_quadrature_points >= 2 ||
        throw(ArgumentError("quadrature needs at least two points"))
    minimum((
        config.tau_prior_mean,
        config.tau_c_prior_mean,
    )) > 0 || throw(ArgumentError("precision prior means must be positive"))
    minimum((
        config.tau_prior_shape,
        config.tau_c_prior_shape,
    )) > 1 || throw(ArgumentError(
        "precision prior shapes must exceed one for finite E[1/tau]",
    ))
    config.prediction_iterations >= 1 ||
        throw(ArgumentError("prediction_iterations must be positive"))
    nothing
end

# ╔═╡ a4b09eba-718f-4914-927f-569f05c23917
md"""
## Fair scaling and the one-dimensional prior

Both responses are standardized before fitting because the XOR model was
configured for order-one targets, while ``x^3`` reaches magnitude 125. Reported
means and variances are transformed back to the original units.

Input scaling is fixed before seeing the generated responses:

- for the aleatoric task,
  ``x_{\rm model}=x/s_x`` with
  ``s_x=\omega\,2.2/(3\pi)``. Thus the known benchmark frequency ``3\pi`` is
  near the existing weight-prior scale 2.2 rather than making the mean test fail
  merely because of units;
- for the cubic task, ``s_x=4``, the center of the two observed intervals.

In two dimensions the XOR prior distributes ridge directions around a circle.
There are no different directions in one dimension, so the paired prior below
distributes its frequencies from 75% to 125% of the same central scale.

The deliberate ablation is in the latent precisions. Every neuron receives its
own ``\tau_i`` and ``\tau_{c,i}``. Their Gamma shape-rate priors have shapes
**$(config.tau_prior_shape)** and **$(config.tau_c_prior_shape)**, with rates
chosen to retain the old prior means **$(config.tau_prior_mean)** and
**$(config.tau_c_prior_mean)**. Shape 2 has coefficient of variation
``1/\sqrt{2}``, much broader than the old shared shapes 1000 and 10000. Weight
priors and NGMP settings remain unchanged.
"""

# ╔═╡ 044f99b5-cc45-43ef-ae57-a9ed054b7608
begin
    aleatoric_clean_mean(x) = -(x + 0.5) * sin(3π * x)
    aleatoric_true_variance(x) = abs2(0.45 * (x + 0.5))
    epistemic_clean_mean(x) = x^3

    function make_aleatoric_data(config)
        rng = StableRNG(config.data_seed)
        x = config.aleatoric_x_std .* randn(rng, config.n_aleatoric)
        clean_mean = aleatoric_clean_mean.(x)
        true_variance = aleatoric_true_variance.(x)
        y = clean_mean .+ sqrt.(true_variance) .* randn(rng, length(x))
        return DataFrame(
            x = x,
            y = y,
            clean_mean = clean_mean,
            true_variance = true_variance,
        )
    end

    function make_epistemic_data(config)
        rng = StableRNG(config.data_seed + 1)
        n_left = config.n_epistemic ÷ 2
        n_right = config.n_epistemic - n_left
        x = vcat(
            -5 .+ 2 .* rand(rng, n_left),
            3 .+ 2 .* rand(rng, n_right),
        )
        permutation = randperm(rng, length(x))
        x = x[permutation]
        clean_mean = epistemic_clean_mean.(x)
        true_variance = fill(9.0, length(x))
        y = clean_mean .+ 3 .* randn(rng, length(x))
        return DataFrame(
            x = x,
            y = y,
            clean_mean = clean_mean,
            true_variance = true_variance,
        )
    end

    aleatoric_data = make_aleatoric_data(config)
    epistemic_data = make_epistemic_data(config)
end

# ╔═╡ 9ff5377c-aa0b-4c8b-9c22-a8280fcc8f34
data_preview = let
    aleatoric_panel = scatter(
        aleatoric_data.x,
        aleatoric_data.y;
        markersize = 3,
        markeralpha = 0.55,
        markerstrokewidth = 0,
        label = "observed",
        xlabel = "x",
        ylabel = "y",
        title = "Heteroscedastic observations",
    )
    preview_x = range(-2, 2; length = 400)
    plot!(
        aleatoric_panel,
        preview_x,
        aleatoric_clean_mean.(preview_x);
        color = :black,
        linewidth = 2,
        label = "clean mean",
    )

    epistemic_panel = scatter(
        epistemic_data.x,
        epistemic_data.y;
        markersize = 3,
        markeralpha = 0.55,
        markerstrokewidth = 0,
        label = "observed",
        xlabel = "x",
        ylabel = "y",
        title = "No observations in (-3, 3)",
    )
    preview_x = range(-6, 6; length = 400)
    plot!(
        epistemic_panel,
        preview_x,
        epistemic_clean_mean.(preview_x);
        color = :black,
        linewidth = 2,
        label = "clean mean",
    )
    plot(
        aleatoric_panel,
        epistemic_panel;
        layout = (1, 2),
        size = (1_100, 390),
    )
end

# ╔═╡ c74117a8-eb60-49ac-91e7-d437dd68b351
begin
    struct RegressionScale
        x_center::Float64
        x_scale::Float64
        y_center::Float64
        y_scale::Float64
    end

    function regression_scale(data, x_scale)
        x_scale > 0 || throw(ArgumentError("x scale must be positive"))
        y_scale = std(data.y)
        isfinite(y_scale) && y_scale > 0 ||
            error("response scale must be finite and positive")
        return RegressionScale(0.0, x_scale, mean(data.y), y_scale)
    end

    model_features(x, scale) = [
        [1.0, (Float64(value) - scale.x_center) / scale.x_scale]
        for value in x
    ]
    model_targets(y, scale) =
        (Float64.(y) .- scale.y_center) ./ scale.y_scale
    response_mean_to_data(mean_value, scale) =
        scale.y_center + scale.y_scale * mean_value
    response_variance_to_data(variance_value, scale) =
        abs2(scale.y_scale) * variance_value

    aleatoric_scale = regression_scale(
        aleatoric_data,
        config.phi_omega * config.w_prior_scale / (3π),
    )
    epistemic_scale = regression_scale(epistemic_data, 4.0)

    aleatoric_features =
        model_features(aleatoric_data.x, aleatoric_scale)
    epistemic_features =
        model_features(epistemic_data.x, epistemic_scale)
    aleatoric_targets =
        model_targets(aleatoric_data.y, aleatoric_scale)
    epistemic_targets =
        model_targets(epistemic_data.y, epistemic_scale)
end

# ╔═╡ 27d4c24c-fef5-4777-b1f4-d8d8c9b8b72b
begin
    gamma_with_mean(shape, distribution_mean) =
        GammaShapeRate(shape, shape / distribution_mean)

    function make_regression_priors(config; seed = config.prior_seed)
        rng = StableRNG(seed)
        n_pairs = config.n_neurons ÷ 2
        prior_precision = Diagonal([
            inv(config.bias_prior_variance),
            inv(config.w_prior_variance),
        ])
        frequencies = n_pairs == 1 ?
                      [config.w_prior_scale] :
                      collect(range(
                          0.75 * config.w_prior_scale,
                          1.25 * config.w_prior_scale;
                          length = n_pairs,
                      ))
        w = Vector{Any}(undef, config.n_neurons)
        v = Vector{Any}(undef, config.n_neurons)

        for pair in 1:n_pairs
            frequency = frequencies[pair] * (1 + 0.03 * randn(rng))
            bias = (isodd(pair) ? 1.0 : -1.0) *
                   (π / 4) * (1 + 0.1 * randn(rng))
            for (slot, sign) in ((2pair - 1, 1.0), (2pair, -1.0))
                prior_mean = [sign * bias, frequency]
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
            :w => w,
            :v => v,
            :intercept => NormalMeanVariance(
                0.0, config.intercept_prior_variance,
            ),
            :tau => [
                gamma_with_mean(
                    config.tau_prior_shape,
                    config.tau_prior_mean,
                )
                for _ in 1:config.n_neurons
            ],
            :tau_c => [
                gamma_with_mean(
                    config.tau_c_prior_shape,
                    config.tau_c_prior_mean,
                )
                for _ in 1:config.n_neurons
            ],
            :obs_noise => GammaShapeRate(config.obs_noise_prior...),
        )
    end

    activation_meta(config) = ResidualSineMeta(
        rho = config.phi_rho,
        omega = config.phi_omega,
    )

    activation_dependencies(config) = NGMPDependencies(
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

# ╔═╡ 704093ac-73b1-42ec-91e1-2605b3b9daa5
@model function regression_manyplus_residual_sine(
    n_neurons,
    features,
    y,
    priors,
    activation,
    activation_deps,
)
    local w, v, za, h, c, out, intercept, tau, tau_c

    obs_noise ~ priors[:obs_noise]
    for neuron in 1:n_neurons
        tau[neuron] ~ priors[:tau][neuron]
        tau_c[neuron] ~ priors[:tau_c][neuron]
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
    end
    intercept ~ priors[:intercept]
    for observation in eachindex(y)
        for neuron in 1:n_neurons
            za[neuron, observation] ~ softdot(
                features[observation], w[neuron], tau[neuron],
            )
            h[neuron, observation] ~ ResidualSine(
                za[neuron, observation],
            ) where {
                dependencies = activation_deps,
                meta = activation,
            }
            c[neuron, observation] ~ softdot(
                v[neuron], h[neuron, observation], tau_c[neuron],
            )
        end
        out[observation] ~ ManyPlus(inputs = [
            c[neuron, observation] for neuron in 1:n_neurons
        ])
        y[observation] ~ NormalMeanPrecision(
            out[observation] + intercept, obs_noise,
        )
    end
end

# ╔═╡ c6f0f18b-6a74-4d30-a188-657640d24598
@constraints function regression_manyplus_constraints()
    q(w, v, za, h, c, out, intercept, tau, tau_c, obs_noise) =
        q(w, za, h, c, out, intercept)q(v)q(tau)q(tau_c)q(obs_noise)
    q(w)::MomentForm()
end

# ╔═╡ 4ad4b42a-720d-43f4-8a92-c7723d79e46a
begin
    @initialization function regression_manyplus_initialization(priors, inits)
        q(v) = deepcopy(priors[:v])
        q(za) = inits.za
        q(h) = inits.h
        q(c) = inits.c
        q(out) = inits.out
        q(tau) = deepcopy(priors[:tau])
        q(tau_c) = deepcopy(priors[:tau_c])
        q(obs_noise) = priors[:obs_noise]
        μ(w) = deepcopy(priors[:w])
        μ(intercept) = priors[:intercept]
    end

    function pushforward_inits(priors, features, config)
        activation = activation_meta(config)
        phi(value) =
            SurrogateModelling._residual_sine(value, activation)
        w_means = mean.(priors[:w])
        v_means = mean.(priors[:v])
        n = length(features)
        za = [
            NormalMeanVariance(
                dot(w_means[neuron], features[observation]), 0.5,
            )
            for neuron in 1:config.n_neurons, observation in 1:n
        ]
        h = [
            NormalMeanVariance(phi(mean(za[neuron, observation])), 1.0)
            for neuron in 1:config.n_neurons, observation in 1:n
        ]
        c = [
            NormalMeanVariance(
                v_means[neuron] * mean(h[neuron, observation]), 1.0,
            )
            for neuron in 1:config.n_neurons, observation in 1:n
        ]
        out = [
            NormalMeanVariance(
                sum(
                    mean(c[neuron, observation])
                    for neuron in 1:config.n_neurons
                ),
                1.0,
            )
            for observation in 1:n
        ]
        return (; za, h, c, out)
    end
end

# ╔═╡ c6b84a98-9ec1-44bc-95ed-a4fa2bc77b04
md"""
## One graph, one inference run

For each task, all **$(config.n_aleatoric)** or
**$(config.n_epistemic)** observations enter one factor graph and one call to
`infer`. Nothing is carried between graphs and no observation is revisited as
an epoch.

The default is 40 observations per task (20 in each observed epistemic
interval). This is a conservative size verified to keep the large structured
``q(w,z_a,h,c,\mathrm{out},b)`` block numerically stable; the counts remain
configurable through the two `MANYPLUS_UNCERTAINTY_N_*` environment variables.

Inference stops when the Bethe free energy satisfies
`StopEarlyIterationStrategy(atol=$(config.stop_atol),
rtol=$(config.stop_rtol))`, after at least
**$(config.min_training_iterations)** iterations. The
**$(config.max_training_iterations)**-iteration value is only a safety cap;
the summary below reports explicitly if that cap is reached.
"""

# ╔═╡ 470a771e-21d5-43c8-af4f-eb7f4e5c6643
begin
    function validate_distribution(distribution, label)
        distribution_mean = mean(distribution)
        means = distribution_mean isa Number ?
                [Float64(distribution_mean)] :
                Float64.(vec(distribution_mean))
        all(isfinite, means) || error("$label has a non-finite mean")
        variances = if distribution_mean isa Number
            [Float64(var(distribution))]
        else
            Float64.(diag(Matrix(cov(distribution))))
        end
        all(value -> isfinite(value) && value > 0, variances) ||
            error("$label has an invalid variance")
        return nothing
    end

    function validate_global_priors(priors)
        for (index, distribution) in enumerate(priors[:w])
            validate_distribution(distribution, "w[$index]")
        end
        for (index, distribution) in enumerate(priors[:v])
            validate_distribution(distribution, "v[$index]")
        end
        for name in (:tau, :tau_c)
            for (index, distribution) in enumerate(priors[name])
                validate_distribution(
                    distribution, "$(String(name))[$index]",
                )
            end
        end
        validate_distribution(priors[:intercept], "intercept")
        validate_distribution(priors[:obs_noise], "obs_noise")
        return nothing
    end

    function extract_global_posteriors(result)
        return Dict{Symbol, Any}(
            :w => deepcopy(collect(vec(result.posteriors[:w]))),
            :v => deepcopy(collect(vec(result.posteriors[:v]))),
            :intercept => deepcopy(result.posteriors[:intercept]),
            :tau => deepcopy(collect(vec(result.posteriors[:tau]))),
            :tau_c => deepcopy(collect(vec(result.posteriors[:tau_c]))),
            :obs_noise => deepcopy(result.posteriors[:obs_noise]),
        )
    end

    function fit_regression(targets, features, config; label)
        priors = make_regression_priors(config)
        initial_priors = deepcopy(priors)
        validate_global_priors(priors)

        stopper =
            StopEarlyIterationStrategy(config.stop_atol, config.stop_rtol)
        delayed_stopper = function (event)
            event.iteration >= config.min_training_iterations &&
                stopper(event)
            return nothing
        end
        local result
        elapsed_seconds = @elapsed result = infer(
            model = regression_manyplus_residual_sine(
                n_neurons = config.n_neurons,
                priors = priors,
                activation = activation_meta(config),
                activation_deps = activation_dependencies(config),
            ),
            data = (y = targets, features = features),
            constraints = regression_manyplus_constraints(),
            initialization = regression_manyplus_initialization(
                priors, pushforward_inits(priors, features, config),
            ),
            returnvars = (
                w = KeepLast(),
                v = KeepLast(),
                intercept = KeepLast(),
                tau = KeepLast(),
                tau_c = KeepLast(),
                obs_noise = KeepLast(),
            ),
            iterations = config.max_training_iterations,
            free_energy = true,
            callbacks = (after_iteration = delayed_stopper,),
            showprogress = false,
            options = (limit_stack_depth = 100,),
            disable_inference_error_hint = true,
        )
        all(isfinite, result.free_energy) ||
            error("training produced a non-finite free energy")
        posteriors = extract_global_posteriors(result)
        validate_global_priors(posteriors)
        iterations = length(result.free_energy)
        converged = iterations < config.max_training_iterations
        status = converged ? "converged" : "reached safety cap"
        println(
            "$label: $status after $iterations iterations, ",
            "$(round(elapsed_seconds; digits = 2)) s",
        )
        return (
            priors = posteriors,
            initial_priors = initial_priors,
            iterations = iterations,
            converged = converged,
            elapsed_seconds = elapsed_seconds,
            free_energy = Float64.(result.free_energy),
        )
    end
end

# ╔═╡ a5a56d64-c7a8-40d6-863a-eb183c97bb94
begin
    aleatoric_fit = fit_regression(
        aleatoric_targets,
        aleatoric_features,
        config;
        label = "aleatoric",
    )
    epistemic_fit = fit_regression(
        epistemic_targets,
        epistemic_features,
        config;
        label = "epistemic",
    )
    nothing
end

# ╔═╡ c7e3de3f-2f22-4306-a2b2-f1085aff94c8
training_summary = DataFrame(
    task = ["aleatoric", "epistemic"],
    observations = [
        length(aleatoric_targets),
        length(epistemic_targets),
    ],
    iterations = [aleatoric_fit.iterations, epistemic_fit.iterations],
    converged = [aleatoric_fit.converged, epistemic_fit.converged],
    training_seconds = round.(
        [
            aleatoric_fit.elapsed_seconds,
            epistemic_fit.elapsed_seconds,
        ];
        digits = 2,
    ),
    observation_precision = [
        mean(aleatoric_fit.priors[:obs_noise]),
        mean(epistemic_fit.priors[:obs_noise]),
    ],
    hidden_precision_mean = [
        mean(mean.(aleatoric_fit.priors[:tau])),
        mean(mean.(epistemic_fit.priors[:tau])),
    ],
    hidden_precision_min = [
        minimum(mean.(aleatoric_fit.priors[:tau])),
        minimum(mean.(epistemic_fit.priors[:tau])),
    ],
    hidden_precision_max = [
        maximum(mean.(aleatoric_fit.priors[:tau])),
        maximum(mean.(epistemic_fit.priors[:tau])),
    ],
    contribution_precision_mean = [
        mean(mean.(aleatoric_fit.priors[:tau_c])),
        mean(mean.(epistemic_fit.priors[:tau_c])),
    ],
    contribution_precision_min = [
        minimum(mean.(aleatoric_fit.priors[:tau_c])),
        minimum(mean.(epistemic_fit.priors[:tau_c])),
    ],
    contribution_precision_max = [
        maximum(mean.(aleatoric_fit.priors[:tau_c])),
        maximum(mean.(epistemic_fit.priors[:tau_c])),
    ],
)

# ╔═╡ 5fa4e7c0-2b2f-4cb7-b8ef-d2c4f5a6e901
per_neuron_precision_summary = let
    function precision_rows(task, fit)
        tau_mean = Float64.(mean.(fit.priors[:tau]))
        tau_c_mean = Float64.(mean.(fit.priors[:tau_c]))
        return DataFrame(
            task = fill(task, config.n_neurons),
            neuron = collect(1:config.n_neurons),
            tau_mean = tau_mean,
            tau_sd = sqrt.(Float64.(var.(fit.priors[:tau]))),
            expected_input_variance = Float64.(
                rate.(fit.priors[:tau]) ./
                (shape.(fit.priors[:tau]) .- 1),
            ),
            tau_c_mean = tau_c_mean,
            tau_c_sd = sqrt.(Float64.(var.(fit.priors[:tau_c]))),
            expected_contribution_variance = Float64.(
                rate.(fit.priors[:tau_c]) ./
                (shape.(fit.priors[:tau_c]) .- 1),
            ),
        )
    end

    vcat(
        precision_rows("aleatoric", aleatoric_fit),
        precision_rows("epistemic", epistemic_fit),
    )
end

# ╔═╡ a06c8f76-639a-4505-ab88-9f9b1db78e29
free_energy_plot = let
    panel = plot(
        xlabel = "Inference iteration",
        ylabel = "Bethe free energy",
        title = "One full-data inference run per task",
        legend = :outerright,
        size = (1_100, 400),
    )
    for (label, fit) in (
        ("aleatoric", aleatoric_fit),
        ("epistemic", epistemic_fit),
    )
        plot!(
            panel,
            eachindex(fit.free_energy),
            fit.free_energy;
            linewidth = 1.8,
            alpha = 0.8,
            label = "$label ($(fit.iterations) iterations)",
        )
    end
    panel
end

# ╔═╡ ecda9ea2-ea80-4ccc-9692-5c1e3e552f2e
md"""
## Direct `q(y)` prediction

The model's own prediction graph is the primary predictive diagnostic. The
following graph is the one-dimensional counterpart of
`xor_manyplus_prediction` in the source notebook, with its learned output
intercept included. It fixes the learned ``v`` and precision marginals, uses
the learned ``w`` and intercept posteriors as incoming prior messages, and
keeps ``q(w,z_a,h,c,\mathrm{out},b,y)`` in one structured cluster. It
terminates ``y`` with the same diffuse ``10^{12}`` pseudo-prior.

It runs for **$(config.prediction_iterations) prediction iterations**,
separately from training. We retain both `mean(q(y))` and `var(q(y))`. The
deterministic calculation below is reported as a factorized
law-of-total-variance reference, not substituted for the structured
prediction.

Each prediction location gets its own fresh graph. This keeps ``w`` joint with
that location's local chain while preventing other unobserved grid points and
their pseudo-priors from changing the marginal being requested.
"""

# ╔═╡ 94cce03d-a149-4c70-a2ab-9525976405b1
@model function regression_manyplus_prediction(
    n_neurons,
    features,
    priors,
    activation,
    activation_deps,
    y_prior_variance,
)
    local w, v, za, h, c, out, intercept, tau, tau_c, y

    obs_noise ~ priors[:obs_noise]
    for neuron in 1:n_neurons
        tau[neuron] ~ priors[:tau][neuron]
        tau_c[neuron] ~ priors[:tau_c][neuron]
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
    end
    intercept ~ priors[:intercept]
    for observation in eachindex(features)
        for neuron in 1:n_neurons
            za[neuron, observation] ~ softdot(
                features[observation], w[neuron], tau[neuron],
            )
            h[neuron, observation] ~ ResidualSine(
                za[neuron, observation],
            ) where {
                dependencies = activation_deps,
                meta = activation,
            }
            c[neuron, observation] ~ softdot(
                v[neuron], h[neuron, observation], tau_c[neuron],
            )
        end
        out[observation] ~ ManyPlus(inputs = [
            c[neuron, observation] for neuron in 1:n_neurons
        ])
        y[observation] ~ NormalMeanPrecision(
            out[observation] + intercept, obs_noise,
        )
        y[observation] ~ NormalMeanVariance(
            0.0, y_prior_variance,
        )
    end
end

# ╔═╡ b36202fd-fb34-4a43-a946-44bb412af8a5
@constraints function regression_manyplus_prediction_constraints(priors)
    q(w, v, za, h, c, out, intercept, tau, tau_c, obs_noise, y) =
        q(w, za, h, c, out, intercept, y)q(v)q(tau)q(tau_c)q(obs_noise)

    q(w)::MomentForm()
    q(obs_noise)::RxInfer.FixedMarginalFormConstraint(priors[:obs_noise])
    for (neuron, prior) in enumerate(priors[:v])
        q(v[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    for (neuron, prior) in enumerate(priors[:tau])
        q(tau[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    for (neuron, prior) in enumerate(priors[:tau_c])
        q(tau_c[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
end

# ╔═╡ e957b968-8016-4e63-b4e3-229969088292
@initialization function regression_manyplus_prediction_initialization(
    priors,
    inits,
    output_mean,
    y_prior_variance,
)
    q(v) = deepcopy(priors[:v])
    q(za) = inits.za
    q(h) = inits.h
    q(c) = inits.c
    q(out) = inits.out
    q(tau) = deepcopy(priors[:tau])
    q(tau_c) = deepcopy(priors[:tau_c])
    q(obs_noise) = priors[:obs_noise]
    q(y) = NormalMeanVariance(output_mean, y_prior_variance)

    μ(w) = deepcopy(priors[:w])
    μ(intercept) = priors[:intercept]
    μ(out) = NormalMeanVariance(output_mean, 10.0)
    μ(y) = NormalMeanVariance(output_mean, y_prior_variance)
end

# ╔═╡ 93863b38-46fa-492f-959a-46bef21ed5d3
begin
    function direct_predictive_marginal(
        priors,
        feature,
        config;
        output_mean,
    )
        features = [feature]
        result = infer(
            model = regression_manyplus_prediction(
                n_neurons = config.n_neurons,
                priors = priors,
                activation = activation_meta(config),
                activation_deps = activation_dependencies(config),
                y_prior_variance = config.prediction_prior_variance,
            ),
            data = (features = features,),
            constraints =
                regression_manyplus_prediction_constraints(priors),
            initialization =
                regression_manyplus_prediction_initialization(
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
        return only(vec(result.posteriors[:y]))
    end

    function direct_predictive_statistics(
        priors,
        x_values,
        scale,
        config,
    )
        features = model_features(x_values, scale)
        marginals = [
            direct_predictive_marginal(
                priors,
                feature,
                config;
                output_mean = 0.0,
            )
            for feature in features
        ]
        standardized_mean = Float64.(mean.(marginals))
        standardized_variance = Float64.(var.(marginals))
        all(isfinite, standardized_mean) ||
            error("direct q(y) produced a non-finite mean")
        all(
            value -> isfinite(value) && value > 0,
            standardized_variance,
        ) || error("direct q(y) produced an invalid variance")
        return (
            mean = response_mean_to_data.(
                standardized_mean, Ref(scale),
            ),
            variance = response_variance_to_data.(
                standardized_variance, Ref(scale),
            ),
            marginals = marginals,
        )
    end
end

# ╔═╡ c2152840-9e72-456a-87ef-5c609ce20c7a
md"""
## Deterministic uncertainty decomposition

For ``Z\sim\mathcal N(m,s^2)`` and
``\phi(Z)=Z+a\sin(\omega Z)``, the notebook uses

```math
\mathbb E[\sin(\omega Z)]
 =e^{-\omega^2s^2/2}\sin(\omega m),
```

```math
\mathbb E[Z\sin(\omega Z)]
 =e^{-\omega^2s^2/2}
  \left[m\sin(\omega m)+\omega s^2\cos(\omega m)\right],
```

and

```math
\mathbb E[\sin^2(\omega Z)]
 =\tfrac12\left[1-e^{-2\omega^2s^2}\cos(2\omega m)\right].
```

These give the first two moments through every sine unit exactly. At a fixed
per-neuron precision ``\tau_i``, Gaussian weight and coefficient marginals can
then be integrated analytically. Integrating their conditional-mean variance
gives epistemic uncertainty; integrating the variance caused by the stochastic
hidden and contribution nodes gives process aleatoric uncertainty.

For a Gamma shape-rate precision, ``E[1/\tau]=b/(a-1)``. Only the nonlinear
dependence of neuron ``i`` on ``\tau_i`` remains. Because the precisions are
independent, each neuron needs only its own one-dimensional deterministic
equal-probability midpoint rule—there is no ``H``-dimensional quadrature.
Results are computed at both ``Q`` and ``2Q`` nodes so the table below can
report numerical convergence.
"""

# ╔═╡ 7b9238e3-92b1-43ab-a7ce-6bba6b501a23
begin
    function normal_affine_sine_moments(
        gaussian_mean,
        gaussian_variance,
        sine_coefficient,
        omega,
    )
        variance = max(Float64(gaussian_variance), 0.0)
        mean_value = Float64(gaussian_mean)
        attenuation = exp(-0.5 * abs2(omega) * variance)
        sine_mean = attenuation * sin(omega * mean_value)
        x_sine_mean = attenuation * (
            mean_value * sin(omega * mean_value) +
            omega * variance * cos(omega * mean_value)
        )
        sine_second = 0.5 * (
            1 -
            exp(-2 * abs2(omega) * variance) *
            cos(2 * omega * mean_value)
        )
        transformed_mean =
            mean_value + sine_coefficient * sine_mean
        transformed_second =
            abs2(mean_value) + variance +
            2 * sine_coefficient * x_sine_mean +
            abs2(sine_coefficient) * sine_second
        return (
            mean = transformed_mean,
            second = transformed_second,
        )
    end

    function gamma_midpoint_nodes(distribution, n_points)
        distribution_shape = shape(distribution)
        distribution_rate = rate(distribution)
        distribution_shape > 0 && distribution_rate > 0 ||
            error("Gamma precision parameters must be positive")
        gamma_distribution = Distributions.Gamma(
            distribution_shape,
            inv(distribution_rate),
        )
        probabilities =
            ((1:n_points) .- 0.5) ./ n_points
        nodes = Distributions.quantile.(
            Ref(gamma_distribution), probabilities,
        )
        all(value -> isfinite(value) && value > 0, nodes) ||
            error("precision quadrature produced an invalid node")
        return Float64.(nodes)
    end

    function expected_inverse_precision(distribution)
        distribution_shape = shape(distribution)
        distribution_shape > 1 || throw(DomainError(
            distribution_shape,
            "E[1/precision] needs Gamma shape greater than one",
        ))
        return Float64(rate(distribution) / (distribution_shape - 1))
    end

    function decompose_at_feature(
        priors,
        feature,
        config,
        tau_nodes_by_neuron,
    )
        activation = activation_meta(config)
        sine_coefficient = activation.rho / activation.omega
        omega = activation.omega
        weight_means = mean.(priors[:w])
        weight_covariances = cov.(priors[:w])
        coefficient_means = Float64.(mean.(priors[:v]))
        coefficient_seconds =
            abs2.(coefficient_means) .+ Float64.(var.(priors[:v]))

        network_mean = 0.0
        network_epistemic_variance = 0.0
        integrated_activation_variance = 0.0

        for neuron in 1:config.n_neurons
            tau_nodes = tau_nodes_by_neuron[neuron]
            quadrature_weight = inv(length(tau_nodes))
            neuron_mean = 0.0
            neuron_second = 0.0
            neuron_activation_variance = 0.0
            projected_mean =
                dot(feature, weight_means[neuron])
            projected_variance = max(
                dot(
                    feature,
                    weight_covariances[neuron] * feature,
                ),
                0.0,
            )

            for tau in tau_nodes
                input_noise_variance = inv(tau)
                input_attenuation = exp(
                    -0.5 * abs2(omega) * input_noise_variance,
                )
                smoothed_sine_coefficient =
                    sine_coefficient * input_attenuation
                conditional_mean_moments =
                    normal_affine_sine_moments(
                        projected_mean,
                        projected_variance,
                        smoothed_sine_coefficient,
                        omega,
                    )
                unconditional_activation_moments =
                    normal_affine_sine_moments(
                        projected_mean,
                        projected_variance +
                        input_noise_variance,
                        sine_coefficient,
                        omega,
                    )

                expected_smoothed_activation =
                    conditional_mean_moments.mean
                expected_smoothed_activation_second =
                    conditional_mean_moments.second
                expected_conditional_activation_variance = max(
                    unconditional_activation_moments.second -
                    expected_smoothed_activation_second,
                    0.0,
                )

                contribution_mean =
                    coefficient_means[neuron] *
                    expected_smoothed_activation
                contribution_second =
                    coefficient_seconds[neuron] *
                    expected_smoothed_activation_second
                contribution_parameter_variance = max(
                    contribution_second -
                    abs2(contribution_mean),
                    0.0,
                )
                neuron_mean +=
                    quadrature_weight * contribution_mean
                neuron_second += quadrature_weight * (
                    contribution_parameter_variance +
                    abs2(contribution_mean)
                )
                neuron_activation_variance += quadrature_weight *
                    coefficient_seconds[neuron] *
                    expected_conditional_activation_variance
            end

            network_mean += neuron_mean
            network_epistemic_variance += max(
                neuron_second - abs2(neuron_mean),
                0.0,
            )
            integrated_activation_variance +=
                neuron_activation_variance
        end

        network_epistemic_variance =
            max(network_epistemic_variance, 0.0)
        intercept_mean = Float64(mean(priors[:intercept]))
        intercept_variance = Float64(var(priors[:intercept]))
        predictive_mean = network_mean + intercept_mean
        epistemic_variance =
            network_epistemic_variance + intercept_variance
        contribution_variance = sum(
            expected_inverse_precision.(priors[:tau_c]),
        )
        observation_variance =
            expected_inverse_precision(priors[:obs_noise])
        process_variance =
            integrated_activation_variance + contribution_variance
        aleatoric_variance =
            process_variance + observation_variance
        return (
            mean = predictive_mean,
            epistemic_variance = epistemic_variance,
            network_epistemic_variance =
                network_epistemic_variance,
            intercept_variance = intercept_variance,
            activation_variance = integrated_activation_variance,
            contribution_variance = contribution_variance,
            process_variance = process_variance,
            observation_variance = observation_variance,
            aleatoric_variance = aleatoric_variance,
            total_variance =
                epistemic_variance + aleatoric_variance,
        )
    end

    function deterministic_decomposition(
        priors,
        x_values,
        scale,
        config;
        quadrature_points = config.precision_quadrature_points,
    )
        features = model_features(x_values, scale)
        tau_nodes_by_neuron = [
            gamma_midpoint_nodes(distribution, quadrature_points)
            for distribution in priors[:tau]
        ]
        values = [
            decompose_at_feature(
                priors, feature, config, tau_nodes_by_neuron,
            )
            for feature in features
        ]
        variance_scale = abs2(scale.y_scale)
        return (
            mean = [
                response_mean_to_data(value.mean, scale)
                for value in values
            ],
            epistemic_variance = variance_scale .* [
                value.epistemic_variance for value in values
            ],
            activation_variance = variance_scale .* [
                value.activation_variance for value in values
            ],
            contribution_variance = variance_scale .* [
                value.contribution_variance for value in values
            ],
            process_variance = variance_scale .* [
                value.process_variance for value in values
            ],
            observation_variance = variance_scale .* [
                value.observation_variance for value in values
            ],
            aleatoric_variance = variance_scale .* [
                value.aleatoric_variance for value in values
            ],
            total_variance = variance_scale .* [
                value.total_variance for value in values
            ],
        )
    end
end

# ╔═╡ 9b3ef89b-f9aa-4ee2-9680-56448491cb39
begin
    aleatoric_grid =
        collect(range(-2.0, 2.0; length = config.grid_points))
    epistemic_grid =
        collect(range(-7.0, 7.0; length = config.grid_points))

    aleatoric_prediction_q = deterministic_decomposition(
        aleatoric_fit.priors,
        aleatoric_grid,
        aleatoric_scale,
        config,
    )
    aleatoric_prediction = deterministic_decomposition(
        aleatoric_fit.priors,
        aleatoric_grid,
        aleatoric_scale,
        config;
        quadrature_points = 2config.precision_quadrature_points,
    )
    aleatoric_prior_prediction = deterministic_decomposition(
        aleatoric_fit.initial_priors,
        aleatoric_grid,
        aleatoric_scale,
        config,
    )

    epistemic_prediction_q = deterministic_decomposition(
        epistemic_fit.priors,
        epistemic_grid,
        epistemic_scale,
        config,
    )
    epistemic_prediction = deterministic_decomposition(
        epistemic_fit.priors,
        epistemic_grid,
        epistemic_scale,
        config;
        quadrature_points = 2config.precision_quadrature_points,
    )
    epistemic_prior_prediction = deterministic_decomposition(
        epistemic_fit.initial_priors,
        epistemic_grid,
        epistemic_scale,
        config,
    )

    aleatoric_direct_prediction = direct_predictive_statistics(
        aleatoric_fit.priors,
        aleatoric_grid,
        aleatoric_scale,
        config,
    )
    epistemic_direct_prediction = direct_predictive_statistics(
        epistemic_fit.priors,
        epistemic_grid,
        epistemic_scale,
        config,
    )
    doubled_prediction_config = merge(
        config,
        (prediction_iterations = 2config.prediction_iterations,),
    )
    aleatoric_direct_prediction_doubled =
        direct_predictive_statistics(
            aleatoric_fit.priors,
            aleatoric_grid,
            aleatoric_scale,
            doubled_prediction_config,
        )
    epistemic_direct_prediction_doubled =
        direct_predictive_statistics(
            epistemic_fit.priors,
            epistemic_grid,
            epistemic_scale,
            doubled_prediction_config,
        )
end

# ╔═╡ 6ebc5612-45bb-423f-932d-a3f90eb59705
begin
    root_mean_square(values) = sqrt(mean(abs2, values))
    root_mean_square_error(prediction, truth) =
        root_mean_square(prediction .- truth)
    safe_ratio(numerator, denominator) =
        denominator > eps(Float64) ? numerator / denominator : Inf
    function safe_correlation(left, right)
        std(left) > 100eps(Float64) && std(right) > 100eps(Float64) ?
            cor(left, right) : 0.0
    end
    function relative_curve_change(coarse, fine)
        scale = max(maximum(abs, fine), eps(Float64))
        return maximum(abs.(coarse .- fine)) / scale
    end

    aleatoric_true_mean =
        aleatoric_clean_mean.(aleatoric_grid)
    aleatoric_true_variance_curve =
        aleatoric_true_variance.(aleatoric_grid)
    aleatoric_support = abs.(aleatoric_grid) .<= 1.5
    aleatoric_constant_mean = fill(
        mean(aleatoric_true_mean[aleatoric_support]),
        count(aleatoric_support),
    )
    aleatoric_constant_variance = fill(
        mean(aleatoric_true_variance_curve[aleatoric_support]),
        count(aleatoric_support),
    )
    aleatoric_mean_rmse = root_mean_square_error(
        aleatoric_prediction.mean[aleatoric_support],
        aleatoric_true_mean[aleatoric_support],
    )
    aleatoric_direct_mean_rmse = root_mean_square_error(
        aleatoric_direct_prediction.mean[aleatoric_support],
        aleatoric_true_mean[aleatoric_support],
    )
    aleatoric_mean_baseline_rmse = root_mean_square_error(
        aleatoric_constant_mean,
        aleatoric_true_mean[aleatoric_support],
    )
    aleatoric_variance_rmse = root_mean_square_error(
        aleatoric_prediction.aleatoric_variance[aleatoric_support],
        aleatoric_true_variance_curve[aleatoric_support],
    )
    aleatoric_constant_variance_rmse = root_mean_square_error(
        aleatoric_constant_variance,
        aleatoric_true_variance_curve[aleatoric_support],
    )
    aleatoric_variance_skill =
        1 - safe_ratio(
            aleatoric_variance_rmse,
            aleatoric_constant_variance_rmse,
        )
    aleatoric_variance_correlation = safe_correlation(
        aleatoric_prediction.aleatoric_variance[aleatoric_support],
        aleatoric_true_variance_curve[aleatoric_support],
    )
    aleatoric_direct_variance_rmse = root_mean_square_error(
        aleatoric_direct_prediction.variance[aleatoric_support],
        aleatoric_true_variance_curve[aleatoric_support],
    )
    aleatoric_direct_variance_skill =
        1 - safe_ratio(
            aleatoric_direct_variance_rmse,
            aleatoric_constant_variance_rmse,
        )
    aleatoric_direct_variance_correlation = safe_correlation(
        aleatoric_direct_prediction.variance[aleatoric_support],
        aleatoric_true_variance_curve[aleatoric_support],
    )
    aleatoric_mean_adequate =
        aleatoric_mean_rmse < 0.5aleatoric_mean_baseline_rmse
    aleatoric_shape_recovered =
        aleatoric_mean_adequate &&
        aleatoric_variance_correlation >= 0.7 &&
        aleatoric_variance_skill >= 0.2
    aleatoric_direct_mean_adequate =
        aleatoric_direct_mean_rmse <
        0.5aleatoric_mean_baseline_rmse
    aleatoric_direct_shape_recovered =
        aleatoric_direct_mean_adequate &&
        aleatoric_direct_variance_correlation >= 0.7 &&
        aleatoric_direct_variance_skill >= 0.2

    epistemic_true_mean =
        epistemic_clean_mean.(epistemic_grid)
    epistemic_in_domain =
        (abs.(epistemic_grid) .>= 3.0) .&
        (abs.(epistemic_grid) .<= 5.0)
    epistemic_gap = abs.(epistemic_grid) .<= 2.5
    epistemic_outer = abs.(epistemic_grid) .>= 5.5
    epistemic_mean_rmse = root_mean_square_error(
        epistemic_prediction.mean[epistemic_in_domain],
        epistemic_true_mean[epistemic_in_domain],
    )
    epistemic_direct_mean_rmse = root_mean_square_error(
        epistemic_direct_prediction.mean[epistemic_in_domain],
        epistemic_true_mean[epistemic_in_domain],
    )
    epistemic_constant_rmse = root_mean_square(
        epistemic_true_mean[epistemic_in_domain],
    )
    epistemic_gap_ratio = safe_ratio(
        mean(epistemic_prediction.epistemic_variance[epistemic_gap]),
        mean(epistemic_prediction.epistemic_variance[epistemic_in_domain]),
    )
    epistemic_outer_ratio = safe_ratio(
        mean(epistemic_prediction.epistemic_variance[epistemic_outer]),
        mean(epistemic_prediction.epistemic_variance[epistemic_in_domain]),
    )
    epistemic_in_domain_contraction = safe_ratio(
        mean(epistemic_prediction.epistemic_variance[epistemic_in_domain]),
        mean(epistemic_prior_prediction.epistemic_variance[epistemic_in_domain]),
    )
    epistemic_gap_contraction = safe_ratio(
        mean(epistemic_prediction.epistemic_variance[epistemic_gap]),
        mean(epistemic_prior_prediction.epistemic_variance[epistemic_gap]),
    )
    epistemic_gap_function_coverage = mean(
        abs.(
            epistemic_prediction.mean[epistemic_gap] .-
            epistemic_true_mean[epistemic_gap]
        ) .<=
        1.96 .* sqrt.(
            epistemic_prediction.epistemic_variance[epistemic_gap],
        ),
    )
    epistemic_gap_total_coverage = mean(
        abs.(
            epistemic_prediction.mean[epistemic_gap] .-
            epistemic_true_mean[epistemic_gap]
        ) .<=
        1.96 .* sqrt.(
            epistemic_prediction.total_variance[epistemic_gap],
        ),
    )
    epistemic_direct_gap_ratio = safe_ratio(
        mean(epistemic_direct_prediction.variance[epistemic_gap]),
        mean(epistemic_direct_prediction.variance[epistemic_in_domain]),
    )
    epistemic_direct_outer_ratio = safe_ratio(
        mean(epistemic_direct_prediction.variance[epistemic_outer]),
        mean(epistemic_direct_prediction.variance[epistemic_in_domain]),
    )
    epistemic_direct_gap_coverage = mean(
        abs.(
            epistemic_direct_prediction.mean[epistemic_gap] .-
            epistemic_true_mean[epistemic_gap]
        ) .<=
        1.96 .* sqrt.(
            epistemic_direct_prediction.variance[epistemic_gap],
        ),
    )
    epistemic_direct_in_domain_noise_ratio = safe_ratio(
        mean(epistemic_direct_prediction.variance[epistemic_in_domain]),
        9.0,
    )
    epistemic_mean_adequate =
        epistemic_mean_rmse < 0.5epistemic_constant_rmse
    epistemic_direct_mean_adequate =
        epistemic_direct_mean_rmse < 0.5epistemic_constant_rmse
    epistemic_gap_detected =
        epistemic_mean_adequate &&
        epistemic_gap_ratio >= 1.25 &&
        epistemic_in_domain_contraction <
        epistemic_gap_contraction
    epistemic_gap_recovered =
        epistemic_gap_detected &&
        epistemic_gap_function_coverage >= 0.8

    aleatoric_quadrature_change = maximum((
        relative_curve_change(
            aleatoric_prediction_q.epistemic_variance,
            aleatoric_prediction.epistemic_variance,
        ),
        relative_curve_change(
            aleatoric_prediction_q.aleatoric_variance,
            aleatoric_prediction.aleatoric_variance,
        ),
    ))
    epistemic_quadrature_change = maximum((
        relative_curve_change(
            epistemic_prediction_q.epistemic_variance,
            epistemic_prediction.epistemic_variance,
        ),
        relative_curve_change(
            epistemic_prediction_q.aleatoric_variance,
            epistemic_prediction.aleatoric_variance,
        ),
    ))
    aleatoric_direct_mean_change = relative_curve_change(
        aleatoric_direct_prediction.mean,
        aleatoric_prediction.mean,
    )
    aleatoric_direct_variance_change = relative_curve_change(
        aleatoric_direct_prediction.variance,
        aleatoric_prediction.total_variance,
    )
    epistemic_direct_mean_change = relative_curve_change(
        epistemic_direct_prediction.mean,
        epistemic_prediction.mean,
    )
    epistemic_direct_variance_change = relative_curve_change(
        epistemic_direct_prediction.variance,
        epistemic_prediction.total_variance,
    )
    aleatoric_prediction_iteration_change = maximum((
        relative_curve_change(
            aleatoric_direct_prediction.mean,
            aleatoric_direct_prediction_doubled.mean,
        ),
        relative_curve_change(
            aleatoric_direct_prediction.variance,
            aleatoric_direct_prediction_doubled.variance,
        ),
    ))
    epistemic_prediction_iteration_change = maximum((
        relative_curve_change(
            epistemic_direct_prediction.mean,
            epistemic_direct_prediction_doubled.mean,
        ),
        relative_curve_change(
            epistemic_direct_prediction.variance,
            epistemic_direct_prediction_doubled.variance,
        ),
    ))
    epistemic_direct_recovered =
        epistemic_direct_mean_adequate &&
        epistemic_direct_gap_ratio >= 1.25 &&
        epistemic_direct_gap_coverage >= 0.8 &&
        0.5 <= epistemic_direct_in_domain_noise_ratio <= 2.0 &&
        epistemic_prediction_iteration_change <= 0.05
end

# ╔═╡ d119e2d1-22c6-4bda-8550-6fd01c2e7380
diagnostic_summary = DataFrame(
    diagnostic = [
        "aleatoric mean RMSE / constant RMSE",
        "aleatoric direct-q(y) mean RMSE / constant RMSE",
        "aleatoric variance correlation",
        "aleatoric variance skill vs best constant",
        "direct var(q(y)) correlation with true variance",
        "direct var(q(y)) skill vs best constant",
        "epistemic mean RMSE / constant RMSE",
        "epistemic direct-q(y) mean RMSE / constant RMSE",
        "epistemic gap / observed-domain variance",
        "epistemic outer / observed-domain variance",
        "observed-domain posterior/prior contraction",
        "gap posterior/prior contraction",
        "gap coverage by 95% epistemic function band",
        "gap coverage by 95% total predictive band",
        "direct var(q(y)) gap / observed-domain ratio",
        "direct var(q(y)) outer / observed-domain ratio",
        "gap coverage by direct 95% q(y) band",
        "direct in-domain var(q(y)) / true noise variance",
        "aleatoric direct-q(y) mean relative difference",
        "aleatoric direct-q(y) variance relative difference",
        "epistemic direct-q(y) mean relative difference",
        "epistemic direct-q(y) variance relative difference",
        "aleatoric direct prediction N-to-2N iteration change",
        "epistemic direct prediction N-to-2N iteration change",
        "aleatoric Q-to-2Q relative change",
        "epistemic Q-to-2Q relative change",
    ],
    value = [
        safe_ratio(
            aleatoric_mean_rmse,
            aleatoric_mean_baseline_rmse,
        ),
        safe_ratio(
            aleatoric_direct_mean_rmse,
            aleatoric_mean_baseline_rmse,
        ),
        aleatoric_variance_correlation,
        aleatoric_variance_skill,
        aleatoric_direct_variance_correlation,
        aleatoric_direct_variance_skill,
        safe_ratio(epistemic_mean_rmse, epistemic_constant_rmse),
        safe_ratio(
            epistemic_direct_mean_rmse,
            epistemic_constant_rmse,
        ),
        epistemic_gap_ratio,
        epistemic_outer_ratio,
        epistemic_in_domain_contraction,
        epistemic_gap_contraction,
        epistemic_gap_function_coverage,
        epistemic_gap_total_coverage,
        epistemic_direct_gap_ratio,
        epistemic_direct_outer_ratio,
        epistemic_direct_gap_coverage,
        epistemic_direct_in_domain_noise_ratio,
        aleatoric_direct_mean_change,
        aleatoric_direct_variance_change,
        epistemic_direct_mean_change,
        epistemic_direct_variance_change,
        aleatoric_prediction_iteration_change,
        epistemic_prediction_iteration_change,
        aleatoric_quadrature_change,
        epistemic_quadrature_change,
    ],
)

# ╔═╡ dfd54b65-26d5-4a55-93d7-3aa820382b3d
aleatoric_uncertainty_plot = let
    interval_radius =
        1.96 .* sqrt.(aleatoric_direct_prediction.variance)
    mean_panel = plot(
        aleatoric_grid,
        aleatoric_direct_prediction.mean;
        ribbon = interval_radius,
        fillalpha = 0.16,
        color = :royalblue,
        linewidth = 2,
        label = "direct q(y) mean ± 1.96 SD",
        xlabel = "x",
        ylabel = "y",
        title = "Aleatoric benchmark: predictive fit",
    )
    plot!(
        mean_panel,
        aleatoric_grid,
        aleatoric_true_mean;
        color = :black,
        linewidth = 2,
        label = "true mean",
    )
    scatter!(
        mean_panel,
        aleatoric_data.x,
        aleatoric_data.y;
        color = :gray35,
        markersize = 2.5,
        markeralpha = 0.35,
        markerstrokewidth = 0,
        label = "training observations",
    )

    variance_panel = plot(
        aleatoric_grid,
        aleatoric_true_variance_curve;
        color = :black,
        linewidth = 2.5,
        label = "true aleatoric",
        xlabel = "x",
        ylabel = "variance",
        title = "What kind of variance was learned?",
    )
    plot!(
        variance_panel,
        aleatoric_grid,
        aleatoric_direct_prediction.variance;
        color = :seagreen,
        linewidth = 2.2,
        linestyle = :dash,
        label = "direct var(q(y))",
    )
    plot!(
        variance_panel,
        aleatoric_grid,
        aleatoric_prediction.aleatoric_variance;
        color = :firebrick,
        linewidth = 2,
        label = "model aleatoric",
    )
    plot!(
        variance_panel,
        aleatoric_grid,
        aleatoric_prediction.epistemic_variance;
        color = :darkorange,
        linewidth = 2,
        linestyle = :dash,
        label = "model epistemic",
    )
    plot!(
        variance_panel,
        aleatoric_grid,
        aleatoric_prediction.observation_variance;
        color = :steelblue,
        linewidth = 1.7,
        linestyle = :dot,
        label = "global observation component",
    )
    plot(
        mean_panel,
        variance_panel;
        layout = (1, 2),
        size = (1_200, 420),
    )
end

# ╔═╡ 01a4889e-8477-4e20-800d-46248097a4c0
epistemic_uncertainty_plot = let
    interval_radius =
        1.96 .* sqrt.(epistemic_direct_prediction.variance)
    mean_panel = plot(
        epistemic_grid,
        epistemic_direct_prediction.mean;
        ribbon = interval_radius,
        fillalpha = 0.16,
        color = :royalblue,
        linewidth = 2,
        label = "direct q(y) mean ± 1.96 SD",
        xlabel = "x",
        ylabel = "y",
        title = "Epistemic benchmark: empty central interval",
    )
    plot!(
        mean_panel,
        epistemic_grid,
        epistemic_true_mean;
        color = :black,
        linewidth = 2,
        label = "true cubic",
    )
    scatter!(
        mean_panel,
        epistemic_data.x,
        epistemic_data.y;
        color = :gray35,
        markersize = 2.5,
        markeralpha = 0.4,
        markerstrokewidth = 0,
        label = "training observations",
    )

    variance_panel = plot(
        epistemic_grid,
        epistemic_direct_prediction.variance;
        color = :seagreen,
        linewidth = 2.2,
        label = "direct var(q(y))",
        xlabel = "x",
        ylabel = "variance",
        title = "Learned variance components",
    )
    plot!(
        variance_panel,
        epistemic_grid,
        epistemic_prediction.epistemic_variance;
        color = :darkorange,
        linewidth = 1.8,
        linestyle = :dash,
        label = "decomposed epistemic",
    )
    plot!(
        variance_panel,
        epistemic_grid,
        epistemic_prediction.aleatoric_variance;
        color = :firebrick,
        linewidth = 1.8,
        label = "posterior aleatoric",
    )
    hline!(
        variance_panel,
        [9.0];
        color = :black,
        linewidth = 1.6,
        linestyle = :dot,
        label = "true aleatoric = 9",
    )
    vspan!(
        variance_panel,
        [-3, 3];
        color = :gray80,
        alpha = 0.18,
        label = "no-training-data interval",
    )

    contraction_ratio =
        epistemic_prediction.epistemic_variance ./
        max.(
            epistemic_prior_prediction.epistemic_variance,
            eps(Float64),
        )
    contraction_panel = plot(
        epistemic_grid,
        contraction_ratio;
        color = :purple,
        linewidth = 2.2,
        label = "posterior / prior epistemic",
        xlabel = "x",
        ylabel = "retained fraction",
        title = "Where did posterior uncertainty contract?",
    )
    vspan!(
        contraction_panel,
        [-3, 3];
        color = :gray80,
        alpha = 0.18,
        label = "no-training-data interval",
    )
    plot(
        mean_panel,
        variance_panel,
        contraction_panel;
        layout = (1, 3),
        size = (1_600, 420),
    )
end

# ╔═╡ a779f878-2bde-427d-8c5a-087fdfac6056
begin
    if haskey(ENV, "MANYPLUS_UNCERTAINTY_ALEATORIC_FIGURE")
        savefig(
            aleatoric_uncertainty_plot,
            ENV["MANYPLUS_UNCERTAINTY_ALEATORIC_FIGURE"],
        )
    end
    if haskey(ENV, "MANYPLUS_UNCERTAINTY_EPISTEMIC_FIGURE")
        savefig(
            epistemic_uncertainty_plot,
            ENV["MANYPLUS_UNCERTAINTY_EPISTEMIC_FIGURE"],
        )
    end
    nothing
end

# ╔═╡ f50592b4-9f17-4f3b-b4b3-215bdfdd4f6a
begin
    aleatoric_answer = if !aleatoric_direct_mean_adequate
        "inconclusive: the predictive mean did not fit the clean function well enough"
    elseif aleatoric_direct_shape_recovered
        "direct var(q(y)) passes this seeded variance-shape check"
    else
        "direct var(q(y)) does not recover the input-dependent variance in this check"
    end
    epistemic_answer = if !epistemic_direct_mean_adequate
        "inconclusive: the predictive mean did not fit the observed cubic branches well enough"
    elseif epistemic_direct_recovered
        "direct q(y) passes this seeded missing-interval and calibration check"
    elseif epistemic_direct_gap_ratio >= 1.25
        "direct q(y) detects the missing interval relatively, but fails at least one calibration check"
    else
        "direct q(y) does not produce the expected response in the missing interval"
    end
end

# ╔═╡ 7606a9a5-33bd-4eaa-872d-3cb92299bec9
md"""
## Automated interpretation

**Aleatoric result:** $aleatoric_answer.

On the well-supported interval ``[-1.5,1.5]``, the direct-prediction mean RMSE
is **$(round(aleatoric_direct_mean_rmse; sigdigits=4))**, versus
**$(round(aleatoric_mean_baseline_rmse; sigdigits=4))** for a constant mean.
Direct ``var(q(y))`` has correlation
**$(round(aleatoric_direct_variance_correlation; sigdigits=4))** with the true
variance and skill **$(round(aleatoric_direct_variance_skill; sigdigits=4))**
relative to the best constant-variance curve. Skill zero means no improvement
over that constant curve.

The likelihood in the original model has only one global `obs_noise`
precision. The stochastic `softdot` input can still create a small
input-dependent conditional variance after the nonlinear activation, so the
model is not declared homoscedastic merely by inspecting that one likelihood
line. The plotted decomposition tests whether that indirect mechanism actually
recovers the benchmark noise shape.

**Epistemic result:** $epistemic_answer.

Direct ``var(q(y))`` in the unobserved center divided by that in the two
training intervals is **$(round(epistemic_direct_gap_ratio; sigdigits=4))**.
The corresponding outside-to-training ratio is
**$(round(epistemic_direct_outer_ratio; sigdigits=4))**. In the observed
intervals, its mean variance is
**$(round(epistemic_direct_in_domain_noise_ratio; sigdigits=4))×** the true
noise variance 9.

The direct 95% ``q(y)`` band contains the known cubic at
**$(round(100epistemic_direct_gap_coverage; digits=1))%** of grid points in
the missing interval. A merely wider band is not sufficient: the pass criterion
also checks observed-domain scale and prediction-iteration convergence.

The deterministic integration changed by at most
**$(round(max(aleatoric_quadrature_change, epistemic_quadrature_change);
sigdigits=3))** relatively when increasing the precision rule from
``Q=$(config.precision_quadrature_points)`` to
``2Q=$(2config.precision_quadrature_points)`` nodes.

### Direct prediction cross-check

The plotted aleatoric band now uses the model's direct
``var(q(y))`` after **$(config.prediction_iterations)** prediction iterations.
Keeping ``w`` joint makes this materially different from a factorized
post-hoc decomposition. Increasing prediction from
**$(config.prediction_iterations)** to
**$(2config.prediction_iterations)** iterations changes the aleatoric curve by
**$(round(aleatoric_prediction_iteration_change; sigdigits=4))** and the
epistemic curve by
**$(round(epistemic_prediction_iteration_change; sigdigits=4))**.
Its maximum relative difference from the decomposed total variance is
**$(round(aleatoric_direct_variance_change; sigdigits=4))** on the aleatoric
grid and **$(round(epistemic_direct_variance_change; sigdigits=4))** on the
epistemic grid. The corresponding mean differences are
**$(round(aleatoric_direct_mean_change; sigdigits=4))** and
**$(round(epistemic_direct_mean_change; sigdigits=4))**.
"""

# ╔═╡ f0e79494-8243-4335-b7e7-faf1ec6f24de
md"""
## What this check establishes—and what it does not

- The decomposition is with respect to the learned **variational posterior**
  and the stated generative model. It does not turn the variational
  approximation into an exact Bayesian posterior.
- `process_variance` is aleatoric in the law-of-total-variance sense: it remains
  when the global parameters are known. It includes noise from `tau` and
  `tau_c`. `observation_variance` is shown separately.
- A single seeded synthetic data set is a diagnostic, not proof of frequentist
  calibration. Robust claims should repeat the notebook over several data
  seeds and report the distribution of the metrics.
- Aleatoric and epistemic uncertainty are not identifiable from observations
  without modeling assumptions. In particular, a flexible mean error can be
  absorbed by a noise model, and a restricted noise model can be mistaken for
  parameter uncertainty. That is why the notebook checks mean accuracy before
  accepting either uncertainty result.
- If the aleatoric curve fails while the mean curve succeeds, the direct remedy
  is an input-dependent positive precision/variance head. Merely plotting total
  predictive variance cannot repair or hide that structural limitation.
"""

# ╔═╡ Cell order:
# ╠═0e868b28-a0f4-4fdb-90f2-6652ff5ed862
# ╠═9cf6f84c-9747-463d-870e-2b993ec0f4b7
# ╟─d67bc7b9-f64f-4ee6-8770-e6d7851bc35a
# ╠═425a650d-93bb-4847-a3bc-74906ddef27d
# ╟─a4b09eba-718f-4914-927f-569f05c23917
# ╠═044f99b5-cc45-43ef-ae57-a9ed054b7608
# ╠═9ff5377c-aa0b-4c8b-9c22-a8280fcc8f34
# ╠═c74117a8-eb60-49ac-91e7-d437dd68b351
# ╠═27d4c24c-fef5-4777-b1f4-d8d8c9b8b72b
# ╠═704093ac-73b1-42ec-91e1-2605b3b9daa5
# ╠═c6f0f18b-6a74-4d30-a188-657640d24598
# ╠═4ad4b42a-720d-43f4-8a92-c7723d79e46a
# ╟─c6b84a98-9ec1-44bc-95ed-a4fa2bc77b04
# ╠═470a771e-21d5-43c8-af4f-eb7f4e5c6643
# ╠═a5a56d64-c7a8-40d6-863a-eb183c97bb94
# ╠═c7e3de3f-2f22-4306-a2b2-f1085aff94c8
# ╠═5fa4e7c0-2b2f-4cb7-b8ef-d2c4f5a6e901
# ╠═a06c8f76-639a-4505-ab88-9f9b1db78e29
# ╟─ecda9ea2-ea80-4ccc-9692-5c1e3e552f2e
# ╠═94cce03d-a149-4c70-a2ab-9525976405b1
# ╠═b36202fd-fb34-4a43-a946-44bb412af8a5
# ╠═e957b968-8016-4e63-b4e3-229969088292
# ╠═93863b38-46fa-492f-959a-46bef21ed5d3
# ╟─c2152840-9e72-456a-87ef-5c609ce20c7a
# ╠═7b9238e3-92b1-43ab-a7ce-6bba6b501a23
# ╠═9b3ef89b-f9aa-4ee2-9680-56448491cb39
# ╠═6ebc5612-45bb-423f-932d-a3f90eb59705
# ╠═d119e2d1-22c6-4bda-8550-6fd01c2e7380
# ╠═dfd54b65-26d5-4a55-93d7-3aa820382b3d
# ╠═01a4889e-8477-4e20-800d-46248097a4c0
# ╠═a779f878-2bde-427d-8c5a-087fdfac6056
# ╠═f50592b4-9f17-4f3b-b4b3-215bdfdd4f6a
# ╟─7606a9a5-33bd-4eaa-872d-3cb92299bec9
# ╟─f0e79494-8243-4335-b7e7-faf1ec6f24de
