### A Pluto.jl notebook ###
# v1.0.3

# This experiment deliberately has two independent ResidualSine networks:
#
#   mean:      x -> w       -> za       -> h       -> v       -> mean
#   precision: x -> noise_w -> noise_za -> noise_h -> noise_v -> positive link
#
# The two networks have separate weights, activation variables, tau, and tau_c.
# They are nevertheless learned together in one full-data infer call because
# both meet in the same NormalMeanPrecision likelihood.
#
# The shared-activation runner is imported only for its reusable definitions.
# JOINT_HETERO_DEFINITIONS_ONLY prevents its fit or picture from being rerun.
# The default remains Squareplus so existing scripts and artifacts are
# unchanged; JOINT_HETERO_SEPARATE_LINK=exp selects Exp for a controlled test.

const SEPARATE_VARIANCE_LINK = Symbol(lowercase(get(
    ENV,
    "JOINT_HETERO_SEPARATE_LINK",
    "squareplus",
)))
SEPARATE_VARIANCE_LINK in (:exp, :squareplus) ||
    throw(ArgumentError(
        "JOINT_HETERO_SEPARATE_LINK must be exp or squareplus",
    ))

const SEPARATE_VARIANCE_OUTPUT = get(
    ENV,
    "JOINT_HETERO_SEPARATE_OUTPUT",
    "/tmp/manyplus_joint_separate_variance_squareplus_qy.png",
)
const SEPARATE_VARIANCE_POSTERIORS = get(
    ENV,
    "JOINT_HETERO_SEPARATE_POSTERIORS",
    "/tmp/manyplus_joint_separate_variance_squareplus_qy_posteriors.jls",
)

ENV["JOINT_HETERO_LINK"] = String(SEPARATE_VARIANCE_LINK)
ENV["JOINT_HETERO_DEFINITIONS_ONLY"] = "true"
ENV["JOINT_HETERO_OUTPUT"] = SEPARATE_VARIANCE_OUTPUT
ENV["JOINT_HETERO_POSTERIORS"] = SEPARATE_VARIANCE_POSTERIORS

include(joinpath(
    @__DIR__,
    "manyplus_residual_sine_joint_heteroscedastic.jl",
))

separate_output_weight_reference_neurons = env_float(
    "JOINT_HETERO_OUTPUT_WEIGHT_REFERENCE_NEURONS",
    config.n_neurons,
)
separate_output_weight_reference_neurons > 0 ||
    throw(ArgumentError(
        "JOINT_HETERO_OUTPUT_WEIGHT_REFERENCE_NEURONS must be positive",
    ))
separate_output_weight_width_scale = sqrt(
    separate_output_weight_reference_neurons / config.n_neurons,
)

separate_config = merge(config, (
    definitions_only = false,
    positive_link = SEPARATE_VARIANCE_LINK,
    output_weight_reference_neurons =
        separate_output_weight_reference_neurons,
    output_weight_width_scale =
        separate_output_weight_width_scale,
    use_precision_prior = env_bool(
        "JOINT_HETERO_USE_PRECISION_PRIOR",
        true,
    ),
    output_path = SEPARATE_VARIANCE_OUTPUT,
    posterior_path = SEPARATE_VARIANCE_POSTERIORS,
))

"""
Construct equal-strength but independent priors for the two networks.

The copied distributions describe the same prior beliefs; the model variables
created from them are different nodes and are therefore not tied.
"""
function make_separate_network_priors(config)
    priors = make_priors(config)
    width_scale = config.output_weight_width_scale
    variance_scale = abs2(width_scale)
    priors[:v] = [
        NormalMeanVariance(
            width_scale * mean(distribution),
            variance_scale * var(distribution),
        )
        for distribution in priors[:v]
    ]
    priors[:g] = [
        NormalMeanVariance(
            width_scale * mean(distribution),
            variance_scale * var(distribution),
        )
        for distribution in priors[:g]
    ]
    priors[:noise_w] = deepcopy(priors[:w])
    priors[:noise_v] = deepcopy(priors[:g])
    priors[:noise_tau] = deepcopy(priors[:tau])
    priors[:noise_tau_c] = deepcopy(priors[:tau_c])
    return priors
end

function separate_noise_head_inits(priors, config)
    shared_style_inits = training_precision_head_inits(priors, config)
    width_scale = config.output_weight_width_scale
    variance_scale = abs2(width_scale)
    return (
        noise_v = [
            NormalMeanVariance(
                width_scale * mean(distribution),
                variance_scale * var(distribution),
            )
            for distribution in shared_style_inits.g
        ],
        log_precision_intercept =
            shared_style_inits.log_precision_intercept,
    )
end

function separate_pushforward_inits(
    priors,
    features,
    config;
    noise_v_distributions = priors[:noise_v],
    log_precision_intercept_distribution =
        priors[:log_precision_intercept],
)
    # Reuse the existing initialization for the independent mean network.
    mean_inits = pushforward_inits(
        priors,
        features,
        config;
        g_distributions = noise_v_distributions,
        log_precision_intercept_distribution =
            log_precision_intercept_distribution,
    )

    activation = activation_meta(config)
    phi(value) = SurrogateModelling._residual_sine(value, activation)
    noise_w_means = mean.(priors[:noise_w])
    noise_v_means = mean.(noise_v_distributions)
    log_precision_intercept_mean =
        mean(log_precision_intercept_distribution)
    n = length(features)

    noise_za = [
        NormalMeanVariance(
            dot(noise_w_means[neuron], features[observation]),
            0.5,
        )
        for neuron in 1:config.n_neurons, observation in 1:n
    ]
    noise_h = [
        NormalMeanVariance(
            phi(mean(noise_za[neuron, observation])),
            1.0,
        )
        for neuron in 1:config.n_neurons, observation in 1:n
    ]
    s = [
        NormalMeanVariance(
            noise_v_means[neuron] *
            mean(noise_h[neuron, observation]),
            1.0,
        )
        for neuron in 1:config.n_neurons, observation in 1:n
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
            mean(mean_inits.mean_output[observation]),
            var(mean_inits.mean_output[observation]) +
            inv(mean(precision[observation])),
        )
        for observation in 1:n
    ]

    return (
        za = mean_inits.za,
        h = mean_inits.h,
        c = mean_inits.c,
        out = mean_inits.out,
        mean_output = mean_inits.mean_output,
        noise_za = noise_za,
        noise_h = noise_h,
        s = s,
        precision_score = precision_score,
        log_precision = log_precision,
        precision = precision,
        y = y,
    )
end

@model function separate_variance_joint_model(
    n_neurons,
    features,
    y,
    priors,
    mean_activation,
    mean_activation_deps,
    noise_activation,
    noise_activation_deps,
    link_deps,
    link_meta,
    positive_link,
    precision_kappa,
    use_precision_prior,
)
    local w, v, tau, tau_c
    local noise_w, noise_v, noise_tau, noise_tau_c
    local za, h, c, out, mean_output
    local noise_za, noise_h, s
    local precision_score, log_precision, precision

    intercept ~ priors[:intercept]
    log_precision_intercept ~ priors[:log_precision_intercept]
    precision_rate ~ priors[:precision_rate]
    if !use_precision_prior
        precision_rate ~ Uninformative()
    end

    tau ~ priors[:tau]
    tau_c ~ priors[:tau_c]
    noise_tau ~ priors[:noise_tau]
    noise_tau_c ~ priors[:noise_tau_c]

    for neuron in 1:n_neurons
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
        noise_w[neuron] ~ priors[:noise_w][neuron]
        noise_v[neuron] ~ priors[:noise_v][neuron]
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
                dependencies = mean_activation_deps,
                meta = mean_activation,
            }
            c[neuron, observation] ~ softdot(
                v[neuron],
                h[neuron, observation],
                tau_c,
            )

            noise_za[neuron, observation] ~ softdot(
                features[observation],
                noise_w[neuron],
                noise_tau,
            )
            noise_h[neuron, observation] ~ ResidualSine(
                noise_za[neuron, observation],
            ) where {
                dependencies = noise_activation_deps,
                meta = noise_activation,
            }
            s[neuron, observation] ~ softdot(
                noise_v[neuron],
                noise_h[neuron, observation],
                noise_tau_c,
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

        if use_precision_prior
            precision[observation] ~ GammaShapeRate(
                precision_kappa,
                precision_rate,
            )
        end
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

@constraints function separate_variance_joint_constraints()
    q(
        w, v, tau, tau_c,
        noise_w, noise_v, noise_tau, noise_tau_c,
        za, h, c, out, mean_output,
        noise_za, noise_h, s,
        precision_score, log_precision, precision,
        intercept, log_precision_intercept, precision_rate,
    ) = q(
        w, za, h, c, out, mean_output,
    ) * q(
        noise_w, noise_za, noise_h, s,
        precision_score, log_precision, precision,
    ) * q(v)q(tau)q(tau_c) *
        q(noise_v)q(noise_tau)q(noise_tau_c) *
        q(intercept)q(log_precision_intercept)q(precision_rate)
    q(w)::MomentForm()
    q(noise_w)::MomentForm()
end

@initialization function separate_variance_joint_initialization(
    priors,
    inits,
    noise_head_inits,
)
    q(v) = deepcopy(priors[:v])
    q(noise_v) = deepcopy(noise_head_inits.noise_v)
    q(tau) = deepcopy(priors[:tau])
    q(tau_c) = deepcopy(priors[:tau_c])
    q(noise_tau) = deepcopy(priors[:noise_tau])
    q(noise_tau_c) = deepcopy(priors[:noise_tau_c])

    q(za) = inits.za
    q(h) = inits.h
    q(c) = inits.c
    q(out) = inits.out
    q(mean_output) = inits.mean_output
    q(noise_za) = inits.noise_za
    q(noise_h) = inits.noise_h
    q(s) = inits.s
    q(precision_score) = inits.precision_score
    q(log_precision) = inits.log_precision
    q(precision) = inits.precision
    q(precision_rate) = priors[:precision_rate]

    μ(w) = deepcopy(priors[:w])
    μ(noise_w) = deepcopy(priors[:noise_w])
    q(intercept) = priors[:intercept]
    q(log_precision_intercept) =
        noise_head_inits.log_precision_intercept
end

function train_separate_variance_model(targets, features, config)
    priors = make_separate_network_priors(config)
    noise_head_inits = separate_noise_head_inits(priors, config)
    inits = separate_pushforward_inits(
        priors,
        features,
        config;
        noise_v_distributions = noise_head_inits.noise_v,
        log_precision_intercept_distribution =
            noise_head_inits.log_precision_intercept,
    )

    stopper = StopEarlyIterationStrategy(0.0, config.stop_rtol)
    delayed_stopper = event -> begin
        if config.diagnostic_every > 0 &&
           event.iteration % config.diagnostic_every == 0
            println(
                "completed separate-network joint iteration ",
                event.iteration,
            )
        end
        event.iteration >= config.min_iterations && stopper(event)
        return nothing
    end

    return_variables = (
        w = KeepLast(),
        v = KeepLast(),
        tau = KeepLast(),
        tau_c = KeepLast(),
        noise_w = KeepLast(),
        noise_v = KeepLast(),
        noise_tau = KeepLast(),
        noise_tau_c = KeepLast(),
        intercept = KeepLast(),
        log_precision_intercept = KeepLast(),
        log_precision = KeepLast(),
        precision = KeepLast(),
        out = KeepLast(),
        mean_output = KeepLast(),
        precision_score = KeepLast(),
    )
    if config.use_precision_prior
        return_variables = merge(
            return_variables,
            (precision_rate = KeepLast(),),
        )
    end

    local result
    elapsed_seconds = @elapsed result = infer(
        model = separate_variance_joint_model(
            n_neurons = config.n_neurons,
            priors = priors,
            mean_activation = activation_meta(config),
            mean_activation_deps = activation_dependencies(config),
            noise_activation = activation_meta(config),
            noise_activation_deps = activation_dependencies(config),
            link_deps = link_dependencies(config),
            link_meta = link_damping(config),
            positive_link = config.positive_link,
            precision_kappa = config.precision_kappa,
            use_precision_prior = config.use_precision_prior,
        ),
        data = (y = targets, features = features),
        constraints = separate_variance_joint_constraints(),
        initialization = separate_variance_joint_initialization(
            priors,
            inits,
            noise_head_inits,
        ),
        returnvars = return_variables,
        iterations = config.max_iterations,
        free_energy = true,
        callbacks = (after_iteration = delayed_stopper,),
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )

    all(isfinite, result.free_energy) ||
        error("separate-network training produced non-finite free energy")
    iterations = length(result.free_energy)
    println(
        "separate-network fit: ",
        iterations < config.max_iterations ? "converged" : "reached cap",
        " after $iterations iterations in ",
        round(elapsed_seconds; digits = 2),
        " s",
    )
    return (; result, priors, elapsed_seconds, iterations)
end

function learned_separate_priors(fit)
    posterior = fit.result.posteriors
    precision_rate = haskey(posterior, :precision_rate) ?
                     deepcopy(posterior[:precision_rate]) :
                     deepcopy(fit.priors[:precision_rate])
    return Dict{Symbol, Any}(
        :w => deepcopy(collect(vec(posterior[:w]))),
        :v => deepcopy(collect(vec(posterior[:v]))),
        :tau => deepcopy(posterior[:tau]),
        :tau_c => deepcopy(posterior[:tau_c]),
        :noise_w => deepcopy(collect(vec(posterior[:noise_w]))),
        :noise_v => deepcopy(collect(vec(posterior[:noise_v]))),
        :noise_tau => deepcopy(posterior[:noise_tau]),
        :noise_tau_c => deepcopy(posterior[:noise_tau_c]),
        :intercept => deepcopy(posterior[:intercept]),
        :log_precision_intercept =>
            deepcopy(posterior[:log_precision_intercept]),
        :precision_rate => precision_rate,
    )
end

@model function separate_variance_qy_prediction(
    n_neurons,
    features,
    priors,
    mean_activation,
    mean_activation_deps,
    noise_activation,
    noise_activation_deps,
    link_deps,
    link_meta,
    positive_link,
    precision_kappa,
    prediction_prior_variance,
    use_precision_prior,
)
    local w, v, tau, tau_c
    local noise_w, noise_v, noise_tau, noise_tau_c
    local za, h, c, out, mean_output
    local noise_za, noise_h, s
    local precision_score, log_precision, precision, y

    intercept ~ priors[:intercept]
    log_precision_intercept ~ priors[:log_precision_intercept]
    precision_rate ~ priors[:precision_rate]
    if !use_precision_prior
        precision_rate ~ Uninformative()
    end

    tau ~ priors[:tau]
    tau_c ~ priors[:tau_c]
    noise_tau ~ priors[:noise_tau]
    noise_tau_c ~ priors[:noise_tau_c]

    for neuron in 1:n_neurons
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
        noise_w[neuron] ~ priors[:noise_w][neuron]
        noise_v[neuron] ~ priors[:noise_v][neuron]
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
                dependencies = mean_activation_deps,
                meta = mean_activation,
            }
            c[neuron, observation] ~ softdot(
                v[neuron],
                h[neuron, observation],
                tau_c,
            )

            noise_za[neuron, observation] ~ softdot(
                features[observation],
                noise_w[neuron],
                noise_tau,
            )
            noise_h[neuron, observation] ~ ResidualSine(
                noise_za[neuron, observation],
            ) where {
                dependencies = noise_activation_deps,
                meta = noise_activation,
            }
            s[neuron, observation] ~ softdot(
                noise_v[neuron],
                noise_h[neuron, observation],
                noise_tau_c,
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

        if use_precision_prior
            precision[observation] ~ GammaShapeRate(
                precision_kappa,
                precision_rate,
            )
        end
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

@constraints function separate_variance_prediction_constraints()
    q(
        w, v, tau, tau_c,
        noise_w, noise_v, noise_tau, noise_tau_c,
        za, h, c, out, mean_output,
        noise_za, noise_h, s,
        precision_score, log_precision, precision, y,
        intercept, log_precision_intercept, precision_rate,
    ) = q(
        w, za, h, c, out, mean_output, y,
    ) * q(
        noise_w, noise_za, noise_h, s,
        precision_score, log_precision, precision,
    ) * q(v)q(tau)q(tau_c) *
        q(noise_v)q(noise_tau)q(noise_tau_c) *
        q(intercept)q(log_precision_intercept)q(precision_rate)
    q(w)::MomentForm()
    q(noise_w)::MomentForm()
end

@initialization function separate_variance_prediction_initialization(
    priors,
    inits,
)
    q(v) = deepcopy(priors[:v])
    q(noise_v) = deepcopy(priors[:noise_v])
    q(tau) = deepcopy(priors[:tau])
    q(tau_c) = deepcopy(priors[:tau_c])
    q(noise_tau) = deepcopy(priors[:noise_tau])
    q(noise_tau_c) = deepcopy(priors[:noise_tau_c])

    q(za) = inits.za
    q(h) = inits.h
    q(c) = inits.c
    q(out) = inits.out
    q(mean_output) = inits.mean_output
    q(noise_za) = inits.noise_za
    q(noise_h) = inits.noise_h
    q(s) = inits.s
    q(precision_score) = inits.precision_score
    q(log_precision) = inits.log_precision
    q(precision) = inits.precision
    q(y) = inits.y

    μ(w) = deepcopy(priors[:w])
    μ(noise_w) = deepcopy(priors[:noise_w])
    q(intercept) = priors[:intercept]
    q(log_precision_intercept) = priors[:log_precision_intercept]
    q(precision_rate) = priors[:precision_rate]
end

separate_definitions_only = env_bool(
    "JOINT_HETERO_SEPARATE_DEFINITIONS_ONLY",
    false,
)

if !separate_definitions_only
separate_fit = train_separate_variance_model(
    targets_train,
    features_train,
    separate_config,
)
separate_fitted_priors = learned_separate_priors(separate_fit)

separate_x_grid = collect(range(
    -2.0,
    2.0;
    length = separate_config.grid_points,
))
separate_features_grid = make_features(separate_x_grid)
separate_prediction_config = merge(
    separate_config,
    (
        activation_alpha =
            separate_config.prediction_activation_alpha,
        log_alpha = separate_config.prediction_log_alpha,
        ngmp_max_step = separate_config.prediction_max_step,
    ),
)
separate_prediction_inits = separate_pushforward_inits(
    separate_fitted_priors,
    separate_features_grid,
    separate_prediction_config,
)

separate_prediction_result = infer(
    model = separate_variance_qy_prediction(
        n_neurons = separate_config.n_neurons,
        priors = separate_fitted_priors,
        mean_activation = activation_meta(separate_prediction_config),
        mean_activation_deps =
            activation_dependencies(separate_prediction_config),
        noise_activation = activation_meta(separate_prediction_config),
        noise_activation_deps =
            activation_dependencies(separate_prediction_config),
        link_deps = link_dependencies(separate_prediction_config),
        link_meta = link_damping(separate_prediction_config),
        positive_link = separate_config.positive_link,
        precision_kappa = separate_config.precision_kappa,
        prediction_prior_variance =
            separate_config.prediction_prior_variance,
        use_precision_prior =
            separate_config.use_precision_prior,
    ),
    data = (features = separate_features_grid,),
    constraints = separate_variance_prediction_constraints(),
    initialization = separate_variance_prediction_initialization(
        separate_fitted_priors,
        separate_prediction_inits,
    ),
    returnvars = (
        y = KeepLast(),
        mean_output = KeepLast(),
        precision = KeepLast(),
        log_precision = KeepLast(),
    ),
    iterations = separate_config.prediction_iterations,
    free_energy = false,
    showprogress = false,
    options = (limit_stack_depth = 100,),
    disable_inference_error_hint = true,
)

separate_q_y = collect(vec(
    separate_prediction_result.posteriors[:y],
))
separate_q_mean_output = collect(vec(
    separate_prediction_result.posteriors[:mean_output],
))
separate_q_precision = collect(vec(
    separate_prediction_result.posteriors[:precision],
))
separate_q_log_precision = collect(vec(
    separate_prediction_result.posteriors[:log_precision],
))
separate_predicted_mean = mean_to_data.(mean.(separate_q_y))
separate_predictive_variance =
    variance_to_data.(var.(separate_q_y))
separate_mean_variance =
    variance_to_data.(var.(separate_q_mean_output))
# This is the conditional variance used by the NormalMeanPrecision VMP
# message: the inverse of the expected precision, not a reconstructed total.
separate_aleatoric_variance = variance_to_data.(
    inv.(mean.(separate_q_precision)),
)
separate_decomposition_residual =
    separate_predictive_variance .-
    separate_mean_variance .-
    separate_aleatoric_variance
separate_true_mean = clean_mean.(separate_x_grid)
separate_true_variance = true_variance.(separate_x_grid)

separate_free_energy = separate_fit.result.free_energy
separate_final_free_energy = last(separate_free_energy)
separate_final_fe_relative_change =
    length(separate_free_energy) > 1 ?
    abs(separate_free_energy[end] - separate_free_energy[end - 1]) /
    max(
        abs(separate_free_energy[end]),
        abs(separate_free_energy[end - 1]),
        eps(Float64),
    ) : Inf
separate_supported = abs.(separate_x_grid) .<= 1.5
separate_variance_correlation = cor(
    separate_predictive_variance[separate_supported],
    separate_true_variance[separate_supported],
)
separate_variance_mse = mean(abs2.(
    separate_predictive_variance[separate_supported] .-
    separate_true_variance[separate_supported],
))
separate_constant_variance =
    mean(separate_true_variance[separate_supported])
separate_constant_mse = mean(abs2.(
    separate_constant_variance .-
    separate_true_variance[separate_supported],
))
separate_variance_skill =
    1 - separate_variance_mse / separate_constant_mse
separate_aleatoric_correlation = cor(
    separate_aleatoric_variance[separate_supported],
    separate_true_variance[separate_supported],
)
separate_aleatoric_mse = mean(abs2.(
    separate_aleatoric_variance[separate_supported] .-
    separate_true_variance[separate_supported],
))
separate_aleatoric_skill =
    1 - separate_aleatoric_mse / separate_constant_mse
separate_mean_rmse = sqrt(mean(abs2.(
    separate_predicted_mean[separate_supported] .-
    separate_true_mean[separate_supported],
)))

separate_metrics = (
    architecture = :separate_residual_sine_variance_network,
    positive_link = separate_config.positive_link,
    output_weight_reference_neurons =
        separate_config.output_weight_reference_neurons,
    output_weight_width_scale =
        separate_config.output_weight_width_scale,
    use_precision_prior = separate_config.use_precision_prior,
    activation_alpha = separate_config.activation_alpha,
    log_alpha = separate_config.log_alpha,
    prediction_iterations =
        separate_config.prediction_iterations,
    prediction_activation_alpha =
        separate_config.prediction_activation_alpha,
    prediction_log_alpha =
        separate_config.prediction_log_alpha,
    prediction_max_step =
        separate_config.prediction_max_step,
    observations = separate_config.n_observations,
    neurons_per_network = separate_config.n_neurons,
    iterations = separate_fit.iterations,
    seconds = separate_fit.elapsed_seconds,
    final_free_energy = separate_final_free_energy,
    final_free_energy_relative_change =
        separate_final_fe_relative_change,
    mean_rmse = separate_mean_rmse,
    qy_variance_correlation = separate_variance_correlation,
    qy_variance_skill = separate_variance_skill,
    qy_variance_mean = mean(
        separate_predictive_variance[separate_supported],
    ),
    aleatoric_variance_mean = mean(
        separate_aleatoric_variance[separate_supported],
    ),
    aleatoric_variance_correlation =
        separate_aleatoric_correlation,
    aleatoric_variance_skill = separate_aleatoric_skill,
    mean_function_variance_mean = mean(
        separate_mean_variance[separate_supported],
    ),
    decomposition_residual_mean = mean(
        separate_decomposition_residual[separate_supported],
    ),
    true_variance_mean = mean(
        separate_true_variance[separate_supported],
    ),
    mean_tau = mean(separate_fitted_priors[:tau]),
    mean_tau_c = mean(separate_fitted_priors[:tau_c]),
    variance_tau = mean(separate_fitted_priors[:noise_tau]),
    variance_tau_c = mean(separate_fitted_priors[:noise_tau_c]),
    log_precision_intercept =
        mean(separate_fitted_priors[:log_precision_intercept]),
    precision_head_norm = sqrt(sum(
        abs2,
        mean.(separate_fitted_priors[:noise_v]),
    )),
    precision_rate = separate_config.use_precision_prior ?
                     mean(separate_fitted_priors[:precision_rate]) :
                     missing,
)
println("SEPARATE_VARIANCE_RESULT = ", separate_metrics)

separate_posterior_artifact = (
    config = separate_config,
    architecture = :separate_residual_sine_variance_network,
    learned_parameter_posteriors = separate_fitted_priors,
    x_grid = separate_x_grid,
    q_y = separate_q_y,
    q_mean_output = separate_q_mean_output,
    q_precision = separate_q_precision,
    q_log_precision = separate_q_log_precision,
    predicted_mean = separate_predicted_mean,
    predictive_variance = separate_predictive_variance,
    mean_function_variance = separate_mean_variance,
    aleatoric_variance = separate_aleatoric_variance,
    decomposition_residual = separate_decomposition_residual,
    true_mean = separate_true_mean,
    true_variance = separate_true_variance,
    metrics = separate_metrics,
)
serialize(
    separate_config.posterior_path,
    separate_posterior_artifact,
)
println(
    "saved separate-network learned posteriors to ",
    separate_config.posterior_path,
)

separate_uncertainty =
    1.96 .* sqrt.(max.(separate_predictive_variance, 0.0))
separate_fit_panel = scatter(
    x_train,
    y_train;
    color = :gray60,
    markersize = 3,
    markeralpha = 0.55,
    markerstrokewidth = 0,
    label = "observations",
    xlabel = "x",
    ylabel = "y",
    title = "Separate mean + variance networks, α=$(separate_config.log_alpha)",
    legend = :bottomleft,
)
plot!(
    separate_fit_panel,
    separate_x_grid,
    separate_predicted_mean;
    ribbon = separate_uncertainty,
    fillalpha = 0.18,
    color = :royalblue,
    linewidth = 2,
    label = "mean(q(y*)) ± 1.96 SD(q(y*))",
)
plot!(
    separate_fit_panel,
    separate_x_grid,
    separate_true_mean;
    color = :black,
    linewidth = 2,
    label = "true mean",
)

separate_variance_panel = plot(
    separate_x_grid,
    separate_true_variance;
    color = :black,
    linewidth = 2,
    label = "true Var(y | x)",
    xlabel = "x",
    ylabel = "variance",
    title = separate_config.use_precision_prior ?
            "$(separate_config.positive_link) with Gamma precision prior" :
            "$(separate_config.positive_link) without Gamma precision prior",
    legend = :topleft,
)
plot!(
    separate_variance_panel,
    separate_x_grid,
    separate_predictive_variance;
    color = :darkorange,
    linewidth = 2,
    label = "Var(q(y*))",
)
plot!(
    separate_variance_panel,
    separate_x_grid,
    separate_aleatoric_variance;
    color = :purple,
    linestyle = :dash,
    linewidth = 2,
    label = "1 / E[q(precision)]",
)
plot!(
    separate_variance_panel,
    separate_x_grid,
    separate_mean_variance;
    color = :royalblue,
    linestyle = :dot,
    linewidth = 2,
    label = "Var(q(mean))",
)

separate_variance_plot = plot(
    separate_fit_panel,
    separate_variance_panel;
    layout = (1, 2),
    size = (1_180, 430),
)
savefig(separate_variance_plot, separate_config.output_path)
println(
    "saved separate-network comparison to ",
    separate_config.output_path,
)
end
