### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 29b2f433-1c3a-4ff1-b46c-075338a858a8
begin
    using Pkg
    notebook_project = normpath(joinpath(
        @__DIR__,
        "..",
        "Project.toml",
    ))
    if isnothing(Base.active_project()) ||
       normpath(Base.active_project()) != notebook_project
        Pkg.activate(dirname(notebook_project))
    end
end

# ╔═╡ 6c27eb5e-1da6-4864-90f3-a82c73a23703
begin
    ENV["GKSwstype"] = "100"

    using LinearAlgebra: Diagonal, dot
    using Plots
    using Random
    using RxInfer
    using Serialization
    using StableRNGs
    using Statistics
    using SurrogateModelling
    import ProbabilisticEnsembling: Exp
end

# ╔═╡ 4544a95b-b0a5-4f56-ba08-d13162d54ee9
md"""
# Joint mean and heteroscedastic precision with shared activations and shared τ

This is the deliberately minimal first experiment:

```math
\begin{aligned}
h_i(x) &\sim \operatorname{ResidualSine}(\operatorname{softdot}(x,w_i,\tau)),\\
c_i(x) &\sim \operatorname{softdot}(v_i,h_i(x),\tau_c),\\
s_i(x) &\sim \operatorname{softdot}(g_i,h_i(x),\tau_c),\\
\mu(x) &= b_\mu + \sum_i c_i(x),\\
\eta(x) &= b_\gamma + \sum_i s_i(x),\\
\gamma(x) &= L(\eta(x)),\\
y(x) &\sim \mathcal N\!\left(\mu(x),\gamma(x)^{-1}\right).
\end{aligned}
```

The positive link `L` is selectable: the default script uses `Exp`; the
separate comparison runner uses `Squareplus`.

The mean and precision heads are learned **together** in one full-data
`infer` call, capped at 100 iterations. They share exactly the same hidden
variables `h`, one `tau`, and one `tau_c`. There is:

- no batching or posterior-as-prior training;
- no frozen or pre-trained mean;
- no global `obs_noise` fallback;
- no artificial near-deterministic copy of the precision intercept;
- no variance floor.

The only positivity relation is either
`precision[observation] ~ Exp(link_input[observation])` or
`precision[observation] ~ Squareplus(link_input[observation])`. The final
`NormalMeanPrecision(mean_output, precision)` factor uses ordinary VMP, while
the nonlinear positive link uses a damped natural-gradient update. Both neural
heads are inferred together. A weak shared-rate Gamma prior,
`precision[i] ~ GammaShapeRate(1.1, precision_rate)`, prevents a single
near-zero residual from sending its local precision to infinity;
`precision_rate` is learned in the same call and does not remove the
feature-dependent precision head.

Prediction uses the learned parameter posteriors in a second graph that keeps
`q(w, za, h, c, out, mean_output, y)` joint. Each unobserved `y` is terminated
by a Gaussian with variance `10^12`, so the reported uncertainty is the
explicit posterior `var(q(y))`. The variance plot compares that quantity
directly with the true conditional variance
`(0.45 * (x + 0.5))^2`.
"""

# ╔═╡ b898e0ec-b2a9-4afd-a726-ee8304ff4a4d
begin
    env_bool(name, default) =
        lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")
    env_int(name, default) = parse(Int, get(ENV, name, string(default)))
    env_float(name, default) = parse(Float64, get(ENV, name, string(default)))

    smoke_mode = env_bool("JOINT_HETERO_SMOKE", false)
    definitions_only = env_bool("JOINT_HETERO_DEFINITIONS_ONLY", false)
    positive_link = Symbol(lowercase(get(
        ENV,
        "JOINT_HETERO_LINK",
        "exp",
    )))

    config = (
        smoke_mode = smoke_mode,
        definitions_only = definitions_only,
        positive_link = positive_link,
        n_observations = env_int(
            "JOINT_HETERO_N", smoke_mode ? 12 : 60,
        ),
        n_neurons = env_int(
            "JOINT_HETERO_NEURONS", smoke_mode ? 2 : 8,
        ),
        max_iterations = env_int(
            "JOINT_HETERO_ITERATIONS", smoke_mode ? 2 : 100,
        ),
        min_iterations = env_int(
            "JOINT_HETERO_MIN_ITERATIONS", smoke_mode ? 0 : 30,
        ),
        stop_rtol = env_float("JOINT_HETERO_STOP_RTOL", 1e-5),
        diagnostic_every = env_int(
            "JOINT_HETERO_DIAGNOSTIC_EVERY", smoke_mode ? 0 : 10,
        ),
        grid_points = env_int(
            "JOINT_HETERO_GRID", smoke_mode ? 21 : 161,
        ),
        prediction_iterations = env_int(
            "JOINT_HETERO_PREDICTION_ITERATIONS", smoke_mode ? 2 : 100,
        ),
        prediction_prior_variance = 1e12,
        data_seed = env_int("JOINT_HETERO_DATA_SEED", 2_026),
        prior_seed = env_int("JOINT_HETERO_PRIOR_SEED", 42),
        phi_rho = 0.9,
        phi_omega = 1.0,
        w_prior_scale = 2.2,
        w_prior_variance = 0.005,
        bias_prior_variance = 0.5,
        v_prior_scale = 0.5,
        v_prior_variance = 0.01,
        g_prior_variance = env_float(
            "JOINT_HETERO_G_PRIOR_VARIANCE", 1.0,
        ),
        g_initial_scale = env_float("JOINT_HETERO_G_INITIAL_SCALE", 0.1),
        g_initial_variance = env_float(
            "JOINT_HETERO_G_INITIAL_VARIANCE", 1e-4,
        ),
        intercept_prior_variance = 100.0,
        log_precision_intercept_prior_variance = env_float(
            "JOINT_HETERO_LOG_PRECISION_INTERCEPT_VARIANCE", 4.0,
        ),
        log_precision_intercept_initial_variance = env_float(
            "JOINT_HETERO_LOG_PRECISION_INTERCEPT_INITIAL_VARIANCE", 1e-4,
        ),
        precision_kappa = env_float("JOINT_HETERO_PRECISION_KAPPA", 1.1),
        precision_rate_prior_shape = env_float(
            "JOINT_HETERO_PRECISION_RATE_SHAPE", 2.0,
        ),
        tau_prior_mean = 1e3,
        tau_prior_shape = 2.0,
        tau_c_prior_mean = 1e4,
        tau_c_prior_shape = 2.0,
        activation_alpha = env_float(
            "JOINT_HETERO_ACTIVATION_ALPHA", 0.005,
        ),
        log_alpha = env_float("JOINT_HETERO_LOG_ALPHA", 0.005),
        ngmp_max_step = env_float("JOINT_HETERO_MAX_STEP", 0.1),
        prediction_activation_alpha = env_float(
            "JOINT_HETERO_PREDICTION_ACTIVATION_ALPHA",
            0.5,
        ),
        prediction_log_alpha = env_float(
            "JOINT_HETERO_PREDICTION_LOG_ALPHA",
            0.5,
        ),
        prediction_max_step = env_float(
            "JOINT_HETERO_PREDICTION_MAX_STEP",
            1.0,
        ),
        output_path = get(
            ENV,
            "JOINT_HETERO_OUTPUT",
            positive_link === :exp ?
            "/tmp/manyplus_joint_shared_tau_qy.png" :
            "/tmp/manyplus_joint_shared_tau_qy_squareplus.png",
        ),
        posterior_path = get(
            ENV,
            "JOINT_HETERO_POSTERIORS",
            positive_link === :exp ?
            "/tmp/manyplus_joint_shared_tau_qy_posteriors.jls" :
            "/tmp/manyplus_joint_shared_tau_qy_squareplus_posteriors.jls",
        ),
    )

    iseven(config.n_neurons) ||
        throw(ArgumentError("the paired prior requires an even neuron count"))
    config.n_observations >= 4 ||
        throw(ArgumentError("at least four observations are required"))
    config.max_iterations >= 1 ||
        throw(ArgumentError("max_iterations must be positive"))
    config.precision_kappa > 1 ||
        throw(ArgumentError("precision_kappa must exceed one"))
    config.positive_link in (:exp, :squareplus) ||
        throw(ArgumentError("JOINT_HETERO_LINK must be exp or squareplus"))
    0 <= config.min_iterations < config.max_iterations ||
        throw(ArgumentError("min_iterations must be below max_iterations"))
end

# ╔═╡ d429037e-f0fc-482b-96c9-5acaf713ce7c
begin
    clean_mean(x) = -(x + 0.5) * sin(3π * x)
    true_variance(x) = abs2(0.45 * (x + 0.5))

    rng = StableRNG(config.data_seed)
    x_train = randn(rng, config.n_observations)
    true_mean_train = clean_mean.(x_train)
    true_variance_train = true_variance.(x_train)
    y_train = true_mean_train .+
              sqrt.(true_variance_train) .* randn(rng, config.n_observations)

    x_scale = config.phi_omega * config.w_prior_scale / (3π)
    y_center = mean(y_train)
    y_scale = std(y_train)

    make_features(x_values) = [
        [1.0, Float64(x) / x_scale] for x in x_values
    ]
    to_model_targets(y_values) =
        (Float64.(y_values) .- y_center) ./ y_scale
    mean_to_data(value) = y_center + y_scale * value
    variance_to_data(value) = abs2(y_scale) * value

    features_train = make_features(x_train)
    targets_train = to_model_targets(y_train)
end

# ╔═╡ 516f1cff-80e2-48de-8017-8fb92b894dfc
begin
    gamma_with_mean(shape, distribution_mean) =
        GammaShapeRate(shape, shape / distribution_mean)

    function make_priors(config)
        rng = StableRNG(config.prior_seed)
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
        g = Vector{Any}(undef, config.n_neurons)
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
                g[slot] = NormalMeanVariance(
                    0.0,
                    config.g_prior_variance,
                )
            end
        end

        return Dict{Symbol, Any}(
            :w => w,
            :v => v,
            :g => g,
            :intercept => NormalMeanVariance(
                0.0,
                config.intercept_prior_variance,
            ),
            # Targets are standardized, so precision one is a neutral start.
            :log_precision_intercept => NormalMeanVariance(
                0.0,
                config.log_precision_intercept_prior_variance,
            ),
            # E[precision_rate] = kappa gives an initial E[precision] near one.
            # The rate is then learned from every observation jointly.
            :precision_rate => gamma_with_mean(
                config.precision_rate_prior_shape,
                config.precision_kappa,
            ),
            :tau => gamma_with_mean(
                config.tau_prior_shape,
                config.tau_prior_mean,
            ),
            :tau_c => gamma_with_mean(
                config.tau_c_prior_shape,
                config.tau_c_prior_mean,
            ),
        )
    end

    activation_meta(config) = ResidualSineMeta(
        rho = config.phi_rho,
        omega = config.phi_omega,
    )

    activation_dependencies(config) = NGMPDependencies(
        out = nothing,
        in = nothing;
        projection = TangentProjection(type = ClosedForm),
        damping = DampingMeta(
            alpha = config.activation_alpha,
            beta = 0.0,
            max_step = config.ngmp_max_step,
        ),
    )

    function link_dependencies(config)
        projection =
            config.positive_link === :exp ?
            TangentProjection(type = ClosedForm) :
            TangentProjection(type = Unscented)
        return NGMPDependencies(
            out = nothing,
            in = nothing;
            projection = projection,
        )
    end

    link_damping(config) = DampingMeta(
        alpha = config.log_alpha,
        beta = 0.0,
        max_step = config.ngmp_max_step,
    )

    positive_link_value(config, value) =
        config.positive_link === :exp ?
        exp(value) :
        exp(asinh(value / 2))

end

# ╔═╡ dcebd0e1-96e6-4f96-80f7-9ffb9f27ec1a
@model function shared_hidden_joint_heteroscedastic(
    n_neurons,
    features,
    y,
    priors,
    activation,
    activation_deps,
    link_deps,
    link_meta,
    positive_link,
    precision_kappa,
)
    local w, v, g, tau, tau_c
    local za, h, c, s
    local out, mean_output, precision_score, log_precision, precision

    intercept ~ priors[:intercept]
    log_precision_intercept ~ priors[:log_precision_intercept]
    precision_rate ~ priors[:precision_rate]
    tau ~ priors[:tau]
    tau_c ~ priors[:tau_c]

    for neuron in 1:n_neurons
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
        g[neuron] ~ priors[:g][neuron]
    end

    for observation in eachindex(y)
        for neuron in 1:n_neurons
            za[neuron, observation] ~ softdot(
                features[observation],
                w[neuron],
                tau,
            )
            h[neuron, observation] ~ ResidualSine(
                za[neuron, observation],
            ) where {
                dependencies = activation_deps,
                meta = activation,
            }

            c[neuron, observation] ~ softdot(
                v[neuron],
                h[neuron, observation],
                tau_c,
            )
            s[neuron, observation] ~ softdot(
                g[neuron],
                h[neuron, observation],
                tau_c,
            )
        end

        out[observation] ~ ManyPlus(inputs = [
            c[neuron, observation] for neuron in 1:n_neurons
        ])
        mean_output[observation] :=
            out[observation] + intercept

        precision_score[observation] ~ ManyPlus(inputs = [
            s[neuron, observation] for neuron in 1:n_neurons
        ])
        log_precision[observation] :=
            precision_score[observation] + log_precision_intercept

        precision[observation] ~ GammaShapeRate(
            precision_kappa,
            precision_rate,
        )
        if positive_link === :exp
            precision[observation] ~ Exp(
                log_precision[observation],
            ) where {
                dependencies = link_deps,
                meta = link_meta,
            }
        else
            precision[observation] ~ Squareplus(
                log_precision[observation],
            ) where {
                dependencies = link_deps,
                meta = link_meta,
            }
        end
        y[observation] ~ NormalMeanPrecision(
            mean_output[observation],
            precision[observation],
        )
    end
end

# ╔═╡ c6cba3b8-4f27-49eb-b05f-90a021013a37
@constraints function shared_hidden_joint_constraints()
    q(
        w, v, g, tau, tau_c,
        za, h, c, s, out, mean_output,
        precision_score, log_precision, precision,
        intercept, log_precision_intercept, precision_rate,
    ) = q(
        w, za, h, c, out, mean_output,
    ) * q(
        s, precision_score, log_precision, precision,
    ) * q(v)q(g)q(tau)q(tau_c) *
        q(intercept)q(log_precision_intercept)q(precision_rate)
    q(w)::MomentForm()
end

# ╔═╡ 98264315-3cb0-4a23-95cc-65b707ab4e54
begin
    function training_precision_head_inits(priors, config)
        rng = StableRNG(config.prior_seed + 20_000)
        initial_means =
            config.g_initial_scale .* randn(rng, config.n_neurons)
        initial_means .-= mean(initial_means)
        return (
            g = [
                NormalMeanVariance(
                    initial_means[neuron],
                    config.g_initial_variance,
                )
                for neuron in 1:config.n_neurons
            ],
            log_precision_intercept = NormalMeanVariance(
                0.0,
                config.log_precision_intercept_initial_variance,
            ),
        )
    end

    function pushforward_inits(
        priors,
        features,
        config;
        g_distributions = priors[:g],
        log_precision_intercept_distribution =
            priors[:log_precision_intercept],
    )
        activation = activation_meta(config)
        phi(value) =
            SurrogateModelling._residual_sine(value, activation)
        w_means = mean.(priors[:w])
        v_means = mean.(priors[:v])
        g_means = mean.(g_distributions)
        intercept_mean = mean(priors[:intercept])
        log_precision_intercept_mean =
            mean(log_precision_intercept_distribution)
        n = length(features)

        za = [
            NormalMeanVariance(
                dot(w_means[neuron], features[observation]),
                0.5,
            )
            for neuron in 1:config.n_neurons, observation in 1:n
        ]
        h = [
            NormalMeanVariance(
                phi(mean(za[neuron, observation])),
                1.0,
            )
            for neuron in 1:config.n_neurons, observation in 1:n
        ]
        c = [
            NormalMeanVariance(
                v_means[neuron] * mean(h[neuron, observation]),
                1.0,
            )
            for neuron in 1:config.n_neurons, observation in 1:n
        ]
        s = [
            NormalMeanVariance(
                g_means[neuron] * mean(h[neuron, observation]),
                1.0,
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
        mean_output = [
            NormalMeanVariance(
                mean(out[observation]) + intercept_mean,
                1.0,
            )
            for observation in 1:n
        ]
        precision_score = [
            NormalMeanVariance(
                sum(
                    mean(s[neuron, observation])
                    for neuron in 1:config.n_neurons
                ),
                1.0,
            )
            for observation in 1:n
        ]
        log_precision = [
            NormalMeanVariance(
                log_precision_intercept_mean +
                mean(precision_score[observation]),
                1.0,
            )
            for observation in 1:n
        ]
        precision = [
            GammaShapeRate(
                config.precision_kappa,
                config.precision_kappa /
                positive_link_value(
                    config,
                    mean(log_precision[observation]),
                ),
            )
            for observation in 1:n
        ]
        y = [
            NormalMeanVariance(
                mean(mean_output[observation]),
                var(mean_output[observation]) +
                inv(mean(precision[observation])),
            )
            for observation in 1:n
        ]
        return (;
            za,
            h,
            c,
            s,
            out,
            mean_output,
            precision_score,
            log_precision,
            precision,
            y,
        )
    end
end

# ╔═╡ 71353e4d-bc6a-42f5-a959-cf32b25896d1
@initialization function shared_hidden_joint_initialization(
    priors,
    inits,
    precision_head_inits,
)
    q(v) = deepcopy(priors[:v])
    q(g) = deepcopy(precision_head_inits.g)
    q(tau) = deepcopy(priors[:tau])
    q(tau_c) = deepcopy(priors[:tau_c])
    q(za) = inits.za
    q(h) = inits.h
    q(c) = inits.c
    q(s) = inits.s
    q(out) = inits.out
    q(mean_output) = inits.mean_output
    q(precision_score) = inits.precision_score
    q(log_precision) = inits.log_precision
    q(precision) = inits.precision
    q(precision_rate) = priors[:precision_rate]

    μ(w) = deepcopy(priors[:w])
    q(intercept) = priors[:intercept]
    q(log_precision_intercept) =
        precision_head_inits.log_precision_intercept
end

# ╔═╡ e827fcb9-7182-46fb-8ae8-50d430e255c7
begin
    function train_joint_model(targets, features, config)
        priors = make_priors(config)
        precision_head_inits =
            training_precision_head_inits(priors, config)
        inits = pushforward_inits(
            priors,
            features,
            config;
            g_distributions = precision_head_inits.g,
            log_precision_intercept_distribution =
                precision_head_inits.log_precision_intercept,
        )
        stopper = StopEarlyIterationStrategy(0.0, config.stop_rtol)
        delayed_stopper = event -> begin
            if config.diagnostic_every > 0 &&
               event.iteration % config.diagnostic_every == 0
                println("completed joint iteration ", event.iteration)
            end
            event.iteration >= config.min_iterations && stopper(event)
            return nothing
        end

        local result
        elapsed_seconds = @elapsed result = infer(
            model = shared_hidden_joint_heteroscedastic(
                n_neurons = config.n_neurons,
                priors = priors,
                activation = activation_meta(config),
                activation_deps = activation_dependencies(config),
                link_deps = link_dependencies(config),
                link_meta = link_damping(config),
                positive_link = config.positive_link,
                precision_kappa = config.precision_kappa,
            ),
            data = (y = targets, features = features),
            constraints = shared_hidden_joint_constraints(),
            initialization = shared_hidden_joint_initialization(
                priors,
                inits,
                precision_head_inits,
            ),
            returnvars = (
                w = KeepLast(),
                v = KeepLast(),
                g = KeepLast(),
                tau = KeepLast(),
                tau_c = KeepLast(),
                intercept = KeepLast(),
                log_precision_intercept = KeepLast(),
                log_precision = KeepLast(),
                precision = KeepLast(),
                out = KeepLast(),
                mean_output = KeepLast(),
                precision_score = KeepLast(),
                precision_rate = KeepLast(),
            ),
            iterations = config.max_iterations,
            free_energy = true,
            callbacks = (after_iteration = delayed_stopper,),
            showprogress = false,
            options = (limit_stack_depth = 100,),
            disable_inference_error_hint = true,
        )

        all(isfinite, result.free_energy) ||
            error("joint training produced a non-finite free energy")
        iterations = length(result.free_energy)
        println(
            "joint fit: ",
            iterations < config.max_iterations ? "converged" : "reached cap",
            " after $iterations iterations in ",
            round(elapsed_seconds; digits = 2),
            " s",
        )
        return (; result, priors, elapsed_seconds, iterations)
    end

    joint_fit = config.definitions_only ?
                nothing :
                train_joint_model(targets_train, features_train, config)
end

# ╔═╡ 6732bf91-25bd-440c-bba5-750782461c9c
begin
    function learned_priors(fit)
        posterior = fit.result.posteriors
        return Dict{Symbol, Any}(
            :w => deepcopy(collect(vec(posterior[:w]))),
            :v => deepcopy(collect(vec(posterior[:v]))),
            :g => deepcopy(collect(vec(posterior[:g]))),
            :tau => deepcopy(posterior[:tau]),
            :tau_c => deepcopy(posterior[:tau_c]),
            :intercept => deepcopy(posterior[:intercept]),
            :log_precision_intercept =>
                deepcopy(posterior[:log_precision_intercept]),
            :precision_rate => deepcopy(posterior[:precision_rate]),
        )
    end

    fitted_priors = isnothing(joint_fit) ?
                    nothing :
                    learned_priors(joint_fit)
end

# ╔═╡ c95dd371-35b9-4ea1-8442-923f7b33918b
@model function shared_hidden_qy_prediction(
    n_neurons,
    features,
    priors,
    activation,
    activation_deps,
    link_deps,
    link_meta,
    positive_link,
    precision_kappa,
    prediction_prior_variance,
)
    local w, v, g, tau, tau_c
    local za, h, c, s
    local out, mean_output, precision_score, log_precision, precision, y

    intercept ~ priors[:intercept]
    log_precision_intercept ~ priors[:log_precision_intercept]
    precision_rate ~ priors[:precision_rate]
    tau ~ priors[:tau]
    tau_c ~ priors[:tau_c]

    for neuron in 1:n_neurons
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
        g[neuron] ~ priors[:g][neuron]
    end

    for observation in eachindex(features)
        for neuron in 1:n_neurons
            za[neuron, observation] ~ softdot(
                features[observation],
                w[neuron],
                tau,
            )
            h[neuron, observation] ~ ResidualSine(
                za[neuron, observation],
            ) where {
                dependencies = activation_deps,
                meta = activation,
            }
            c[neuron, observation] ~ softdot(
                v[neuron],
                h[neuron, observation],
                tau_c,
            )
            s[neuron, observation] ~ softdot(
                g[neuron],
                h[neuron, observation],
                tau_c,
            )
        end
        out[observation] ~ ManyPlus(inputs = [
            c[neuron, observation] for neuron in 1:n_neurons
        ])
        mean_output[observation] :=
            out[observation] + intercept

        precision_score[observation] ~ ManyPlus(inputs = [
            s[neuron, observation] for neuron in 1:n_neurons
        ])
        log_precision[observation] :=
            precision_score[observation] + log_precision_intercept
        precision[observation] ~ GammaShapeRate(
            precision_kappa,
            precision_rate,
        )
        if positive_link === :exp
            precision[observation] ~ Exp(
                log_precision[observation],
            ) where {
                dependencies = link_deps,
                meta = link_meta,
            }
        else
            precision[observation] ~ Squareplus(
                log_precision[observation],
            ) where {
                dependencies = link_deps,
                meta = link_meta,
            }
        end
        y[observation] ~ NormalMeanPrecision(
            mean_output[observation],
            precision[observation],
        )
        y[observation] ~ NormalMeanVariance(
            0.0,
            prediction_prior_variance,
        )
    end
end

# ╔═╡ f16512bc-a730-4eb6-a86b-2a2341881218
@constraints function shared_hidden_prediction_constraints()
    q(
        w, v, g, tau, tau_c,
        za, h, c, s, out, mean_output,
        precision_score, log_precision, precision, y,
        intercept, log_precision_intercept, precision_rate,
    ) = q(
        w, za, h, c, out, mean_output, y,
    ) * q(
        s, precision_score, log_precision, precision,
    ) * q(v)q(g)q(tau)q(tau_c) *
        q(intercept)q(log_precision_intercept)q(precision_rate)
    q(w)::MomentForm()
end

# ╔═╡ 52ab212c-bcd6-453c-9f03-3b302a8f8398
@initialization function shared_hidden_prediction_initialization(priors, inits)
    q(v) = deepcopy(priors[:v])
    q(g) = deepcopy(priors[:g])
    q(tau) = deepcopy(priors[:tau])
    q(tau_c) = deepcopy(priors[:tau_c])
    q(za) = inits.za
    q(h) = inits.h
    q(c) = inits.c
    q(s) = inits.s
    q(out) = inits.out
    q(mean_output) = inits.mean_output
    q(precision_score) = inits.precision_score
    q(log_precision) = inits.log_precision
    q(precision) = inits.precision
    q(y) = inits.y

    μ(w) = deepcopy(priors[:w])
    q(intercept) = priors[:intercept]
    q(log_precision_intercept) = priors[:log_precision_intercept]
    q(precision_rate) = priors[:precision_rate]
end

# ╔═╡ b07b7ba5-3b76-4267-8477-43f60b9618f0
begin
    if !config.definitions_only
        x_grid = collect(range(-2.0, 2.0; length = config.grid_points))
        features_grid = make_features(x_grid)
        prediction_config = merge(
            config,
            (
                activation_alpha =
                    config.prediction_activation_alpha,
                log_alpha = config.prediction_log_alpha,
                ngmp_max_step = config.prediction_max_step,
            ),
        )
        prediction_inits = pushforward_inits(
            fitted_priors,
            features_grid,
            prediction_config,
        )

        prediction_result = infer(
            model = shared_hidden_qy_prediction(
                n_neurons = config.n_neurons,
                priors = fitted_priors,
                activation = activation_meta(prediction_config),
                activation_deps =
                    activation_dependencies(prediction_config),
                link_deps = link_dependencies(prediction_config),
                link_meta = link_damping(prediction_config),
                positive_link = config.positive_link,
                precision_kappa = config.precision_kappa,
                prediction_prior_variance = config.prediction_prior_variance,
            ),
            data = (features = features_grid,),
            constraints = shared_hidden_prediction_constraints(),
            initialization = shared_hidden_prediction_initialization(
                fitted_priors,
                prediction_inits,
            ),
            returnvars = (
                y = KeepLast(),
            ),
            iterations = config.prediction_iterations,
            free_energy = false,
            showprogress = false,
            options = (limit_stack_depth = 100,),
            disable_inference_error_hint = true,
        )

        q_y = collect(vec(prediction_result.posteriors[:y]))
        predicted_mean = mean_to_data.(mean.(q_y))
        predictive_variance = variance_to_data.(var.(q_y))
        true_mean_grid = clean_mean.(x_grid)
        true_variance_grid = true_variance.(x_grid)
    end
end

# ╔═╡ 2d83d909-f7a4-4f35-834c-af96e10d8c8e
begin
    if !config.definitions_only
        free_energy_values = joint_fit.result.free_energy
        final_free_energy = last(free_energy_values)
        final_free_energy_relative_change =
            length(free_energy_values) > 1 ?
            abs(free_energy_values[end] - free_energy_values[end - 1]) /
            max(
                abs(free_energy_values[end]),
                abs(free_energy_values[end - 1]),
                eps(Float64),
            ) : Inf
        supported = abs.(x_grid) .<= 1.5
        qy_variance_correlation = cor(
            predictive_variance[supported],
            true_variance_grid[supported],
        )
        qy_variance_mse = mean(abs2.(
            predictive_variance[supported] .-
            true_variance_grid[supported],
        ))
        constant_variance = mean(true_variance_grid[supported])
        constant_mse = mean(abs2.(
            constant_variance .- true_variance_grid[supported],
        ))
        qy_variance_skill = 1 - qy_variance_mse / constant_mse
        mean_rmse = sqrt(mean(abs2.(
            predicted_mean[supported] .- true_mean_grid[supported],
        )))

        metrics = (
            positive_link = config.positive_link,
            prediction_iterations = config.prediction_iterations,
            prediction_activation_alpha =
                config.prediction_activation_alpha,
            prediction_log_alpha = config.prediction_log_alpha,
            prediction_max_step = config.prediction_max_step,
            observations = config.n_observations,
            neurons = config.n_neurons,
            iterations = joint_fit.iterations,
            seconds = joint_fit.elapsed_seconds,
            final_free_energy = final_free_energy,
            final_free_energy_relative_change =
                final_free_energy_relative_change,
            mean_rmse = mean_rmse,
            qy_variance_correlation = qy_variance_correlation,
            qy_variance_skill = qy_variance_skill,
            qy_variance_mean = mean(predictive_variance[supported]),
            true_variance_mean = mean(true_variance_grid[supported]),
            shared_tau = mean(fitted_priors[:tau]),
            shared_tau_c = mean(fitted_priors[:tau_c]),
            log_precision_intercept =
                mean(fitted_priors[:log_precision_intercept]),
            precision_head_norm = sqrt(sum(abs2, mean.(fitted_priors[:g]))),
            precision_rate = mean(fitted_priors[:precision_rate]),
        )
        println("JOINT_HETERO_RESULT = ", metrics)
    end
end

# ╔═╡ 71ec7d6a-cec6-44a9-8183-cde5f7b1af4a
begin
    if !config.definitions_only
        posterior_artifact = (
            config = config,
            learned_parameter_posteriors = fitted_priors,
            x_grid = x_grid,
            q_y = q_y,
            predicted_mean = predicted_mean,
            predictive_variance = predictive_variance,
            true_mean = true_mean_grid,
            true_variance = true_variance_grid,
            metrics = metrics,
        )
        serialize(config.posterior_path, posterior_artifact)
        println("saved learned posteriors to ", config.posterior_path)
    end
end

# ╔═╡ 42fbc2ca-e758-42a6-bd4e-03002540ef09
joint_heteroscedastic_plot = config.definitions_only ? nothing : let
    uncertainty = 1.96 .* sqrt.(max.(predictive_variance, 0.0))
    fit_panel = scatter(
        x_train,
        y_train;
        color = :gray60,
        markersize = 3,
        markeralpha = 0.55,
        markerstrokewidth = 0,
        label = "observations",
        xlabel = "x",
        ylabel = "y",
        title = "Joint mean + $(config.positive_link) precision head",
        legend = :bottomleft,
    )
    plot!(
        fit_panel,
        x_grid,
        predicted_mean;
        ribbon = uncertainty,
        fillalpha = 0.18,
        color = :royalblue,
        linewidth = 2,
        label = "mean(q(y*)) ± 1.96 SD(q(y*))",
    )
    plot!(
        fit_panel,
        x_grid,
        true_mean_grid;
        color = :black,
        linewidth = 2,
        label = "true mean",
    )

    variance_panel = plot(
        x_grid,
        true_variance_grid;
        color = :black,
        linewidth = 2,
        label = "true Var(y | x)",
        xlabel = "x",
        ylabel = "variance",
        title = "$(config.positive_link): true variance vs var(q(y*))",
        legend = :topleft,
    )
    plot!(
        variance_panel,
        x_grid,
        predictive_variance;
        color = :darkorange,
        linewidth = 2,
        label = "Var(q(y*))",
    )

    figure = plot(
        fit_panel,
        variance_panel;
        layout = (1, 2),
        size = (1_180, 430),
    )
    savefig(figure, config.output_path)
    figure
end

# ╔═╡ d0eec6fd-e636-440b-9d42-bc28083d4e8a
md"""
## Reading the result

The orange curve is now obtained directly from the predictive marginal:

```math
\operatorname{Var}(q(y_* \mid x)).
```

It therefore contains every uncertainty source propagated by the prediction
graph: posterior uncertainty in `w`, the shared `tau` and `tau_c`, both output
heads, and the conditional Normal observation variance. The black curve is the
true conditional variance of the synthetic generator. These are deliberately
compared directly, as requested; no manually reconstructed variance is used.
"""

# ╔═╡ Cell order:
# ╠═29b2f433-1c3a-4ff1-b46c-075338a858a8
# ╠═6c27eb5e-1da6-4864-90f3-a82c73a23703
# ╟─4544a95b-b0a5-4f56-ba08-d13162d54ee9
# ╠═b898e0ec-b2a9-4afd-a726-ee8304ff4a4d
# ╠═d429037e-f0fc-482b-96c9-5acaf713ce7c
# ╠═516f1cff-80e2-48de-8017-8fb92b894dfc
# ╠═dcebd0e1-96e6-4f96-80f7-9ffb9f27ec1a
# ╠═c6cba3b8-4f27-49eb-b05f-90a021013a37
# ╠═98264315-3cb0-4a23-95cc-65b707ab4e54
# ╠═71353e4d-bc6a-42f5-a959-cf32b25896d1
# ╠═e827fcb9-7182-46fb-8ae8-50d430e255c7
# ╠═6732bf91-25bd-440c-bba5-750782461c9c
# ╠═c95dd371-35b9-4ea1-8442-923f7b33918b
# ╠═f16512bc-a730-4eb6-a86b-2a2341881218
# ╠═52ab212c-bcd6-453c-9f03-3b302a8f8398
# ╠═b07b7ba5-3b76-4267-8477-43f60b9618f0
# ╠═2d83d909-f7a4-4f35-834c-af96e10d8c8e
# ╠═71ec7d6a-cec6-44a9-8183-cde5f7b1af4a
# ╠═42fbc2ca-e758-42a6-bd4e-03002540ef09
# ╟─d0eec6fd-e636-440b-9d42-bc28083d4e8a
