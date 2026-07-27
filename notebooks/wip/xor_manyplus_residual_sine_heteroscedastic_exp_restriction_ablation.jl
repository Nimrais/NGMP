# Factorial ablation for the two restrictions left after replacing Squareplus
# with an exponential precision link:
#
#   1. the shared GammaShapeRate(kappa, beta) factor on every local precision;
#   2. the near-point-mass first-batch initialization of the variance head.
#
# The mean model is fitted once per task/seed and frozen.  Every ablation arm
# then starts from the same fitted mean and the same baseline precision, so the
# comparison changes only kappa and the two initial marginal variances.
#
# Screen all eight factorial arms on one seed:
#   OPENBLAS_NUM_THREADS=1 XOR_HETERO_STAGE=screen \
#     julia --project=. \
#     experiments/xor_manyplus_residual_sine_heteroscedastic_exp_restriction_ablation.jl
#
# Restrict the screen or run a selected full arm:
#   XOR_EXP_RESTRICTION_ARMS=k2_v0p1_s0p01
#   XOR_HETERO_STAGE=full

include(joinpath(
    @__DIR__,
    "xor_manyplus_residual_sine_heteroscedastic_exp_link.jl",
))

function restriction_arm_symbols(default)
    tokens = strip.(split(
        get(ENV, "XOR_EXP_RESTRICTION_ARMS", join(string.(default), ',')),
        ',',
    ))
    arms = Tuple(Symbol(lowercase(token)) for token in tokens if !isempty(token))
    isempty(arms) && throw(ArgumentError(
        "XOR_EXP_RESTRICTION_ARMS must contain at least one arm",
    ))
    return arms
end

function restriction_output_token(value)
    return replace(string(value), '.' => 'p', '-' => 'm')
end

function restriction_arm(
    name,
    kappa,
    coefficient_initial_variance,
    score_initial_variance,
    ;
    direct = false,
    warm_precision = NaN,
)
    return (
        name = Symbol(name),
        kappa = Float64(kappa),
        coefficient_initial_variance = Float64(
            coefficient_initial_variance,
        ),
        score_initial_variance = Float64(score_initial_variance),
        direct = Bool(direct),
        warm_precision = Float64(warm_precision),
    )
end

function available_restriction_arms()
    point = 1e-10
    wide_coefficient = 0.1
    wide_score = 0.01
    return (
        restriction_arm(:k50_point, 50.0, point, point),
        restriction_arm(:k50_v0p1, 50.0, wide_coefficient, point),
        restriction_arm(:k50_s0p01, 50.0, point, wide_score),
        restriction_arm(
            :k50_v0p1_s0p01, 50.0, wide_coefficient, wide_score,
        ),
        restriction_arm(:k2_point, 2.0, point, point),
        restriction_arm(:k2_v0p1, 2.0, wide_coefficient, point),
        restriction_arm(:k2_s0p01, 2.0, point, wide_score),
        restriction_arm(
            :k2_v0p1_s0p01, 2.0, wide_coefficient, wide_score,
        ),
        restriction_arm(:k10_point, 10.0, point, point),
        restriction_arm(:k5_point, 5.0, point, point),
        restriction_arm(:k1p5_point, 1.5, point, point),
        restriction_arm(:k1p25_point, 1.25, point, point),
        restriction_arm(:k1p1_point, 1.1, point, point),
        restriction_arm(:k2_v1em6, 2.0, 1e-6, point),
        restriction_arm(:k2_v1em4, 2.0, 1e-4, point),
        restriction_arm(:k2_v0p01, 2.0, 0.01, point),
        restriction_arm(
            :unpooled_k10_point,
            10.0,
            point,
            point;
            direct = true,
        ),
        restriction_arm(
            :unpooled_k2_point,
            2.0,
            point,
            point;
            direct = true,
        ),
        restriction_arm(
            :unpooled_k1p1_point,
            1.1,
            point,
            point;
            direct = true,
        ),
        restriction_arm(
            :unpooled_k1p1_v1em4,
            1.1,
            1e-4,
            point;
            direct = true,
        ),
        restriction_arm(
            :k1p1_warm20,
            1.1,
            1e-4,
            1e-4;
            warm_precision = 20.0,
        ),
        restriction_arm(
            :k1p1_warm5,
            1.1,
            1e-4,
            1e-4;
            warm_precision = 5.0,
        ),
        restriction_arm(
            :k1p1_warm10,
            1.1,
            1e-4,
            1e-4;
            warm_precision = 10.0,
        ),
        restriction_arm(
            :k1p1_warm50,
            1.1,
            1e-4,
            1e-4;
            warm_precision = 50.0,
        ),
        restriction_arm(
            :unpooled_k1p1_warm20,
            1.1,
            1e-4,
            1e-4;
            direct = true,
            warm_precision = 20.0,
        ),
        restriction_arm(
            :unpooled_k1p1_warm50,
            1.1,
            1e-4,
            1e-4;
            direct = true,
            warm_precision = 50.0,
        ),
        restriction_arm(
            :unpooled_k1p1_warm100,
            1.1,
            1e-4,
            1e-4;
            direct = true,
            warm_precision = 100.0,
        ),
    )
end

function default_restriction_arm_names()
    return (
        :k50_point,
        :k50_v0p1,
        :k50_s0p01,
        :k50_v0p1_s0p01,
        :k2_point,
        :k2_v0p1,
        :k2_s0p01,
        :k2_v0p1_s0p01,
    )
end

function selected_restriction_arms()
    available = available_restriction_arms()
    names = restriction_arm_symbols(default_restriction_arm_names())
    unknown = setdiff(names, getfield.(available, :name))
    isempty(unknown) || throw(ArgumentError(
        "unknown restriction arms: $(join(string.(unknown), ','))",
    ))
    return Tuple(only(filter(arm -> arm.name === name, available)) for name in names)
end

function exp_restriction_config()
    base = heteroscedastic_config()
    base.stage in (:smoke, :screen, :full) || throw(ArgumentError(
        "restriction ablation supports XOR_HETERO_STAGE=smoke, screen, or full",
    ))
    output_dir = get(
        ENV,
        "OUTPUT_DIR",
        joinpath(
            @__DIR__,
            "xor_manyplus_residual_sine_heteroscedastic_exp_restriction_output",
        ),
    )
    return validate_config(merge(base, (
        output_dir = output_dir,
        save_plots = false,
    )))
end

function arm_config(config, arm)
    return validate_config(merge(config, (
        full_kappa = arm.kappa,
        noise_v_initial_variance = arm.coefficient_initial_variance,
        noise_intercept_initial_variance = arm.score_initial_variance,
        save_outputs = false,
        save_plots = false,
    )))
end

function variance_warm_start(
    priors,
    features,
    observations,
    fixed_means,
    config,
    coefficient_precision,
)
    coefficient_precision > 0 || throw(ArgumentError(
        "warm-start coefficient precision must be positive",
    ))
    activation = activation_meta(config)
    weight_means = mean.(priors[:noise_w])
    hidden_columns = [
        [
            SurrogateModelling._residual_sine(
                dot(weight, feature),
                activation,
            )
            for feature in features
        ]
        for weight in weight_means
    ]
    design = hcat(hidden_columns..., ones(length(features)))
    residual_squared = max.(
        abs2.(observations .- fixed_means),
        eps(Float64),
    )
    coefficients = zeros(size(design, 2))
    coefficients[end] = log(priors[:baseline_precision])
    prior_center = copy(coefficients)
    prior_precision = Diagonal(vcat(
        fill(coefficient_precision, length(weight_means)),
        inv(config.noise_intercept_prior_variance),
    ))
    objective(candidate) = begin
        score = clamp.(design * candidate, -20.0, 20.0)
        likelihood = 0.5sum(-score .+ residual_squared .* exp.(score))
        displacement = candidate - prior_center
        likelihood + 0.5dot(
            displacement,
            prior_precision * displacement,
        )
    end
    iterations = 0
    for iteration in 1:30
        score = clamp.(design * coefficients, -20.0, 20.0)
        weights = residual_squared .* exp.(score)
        design_transpose = transpose(design)
        gradient = 0.5design_transpose * (weights .- 1) +
                   prior_precision * (coefficients - prior_center)
        hessian = 0.5design_transpose * (design .* weights) +
                  prior_precision
        step = hessian \ gradient
        previous_objective = objective(coefficients)
        scale = 1.0
        while scale > 1e-6 &&
              objective(coefficients - scale * step) > previous_objective
            scale *= 0.5
        end
        coefficients -= scale * step
        iterations = iteration
        norm(scale * step) < 1e-8 && break
    end
    all(isfinite, coefficients) ||
        error("variance warm start produced non-finite coefficients")
    return (
        coefficients = Float64.(coefficients[1:end-1]),
        intercept = Float64(last(coefficients)),
        coefficient_precision = Float64(coefficient_precision),
        iterations,
        objective = Float64(objective(coefficients)),
    )
end

function make_exp_noise_priors(
    config::NamedTuple,
    n_noise_neurons::Integer,
    kappa::Real,
    baseline_precision::Real,
)
    priors = invoke(
        make_exp_noise_priors,
        Tuple{Any, Any, Any, Any},
        config,
        n_noise_neurons,
        kappa,
        baseline_precision,
    )
    if hasproperty(config, :variance_warm_start) &&
       !isnothing(config.variance_warm_start)
        priors[:variance_warm_start] = deepcopy(
            config.variance_warm_start,
        )
    end
    return priors
end

function exp_noise_pushforward_inits(
    priors::AbstractDict,
    features,
    n_noise_neurons::Integer,
    config::NamedTuple;
    cold_start,
)
    if !cold_start || !haskey(priors, :variance_warm_start)
        return invoke(
            exp_noise_pushforward_inits,
            Tuple{Any, Any, Any, Any},
            priors,
            features,
            n_noise_neurons,
            config;
            cold_start,
        )
    end

    warm = priors[:variance_warm_start]
    length(warm.coefficients) == n_noise_neurons || error(
        "warm-start coefficient width does not match the noise head",
    )
    activation = activation_meta(config)
    phi(value) = SurrogateModelling._residual_sine(value, activation)
    weight_means = mean.(priors[:noise_w])
    n_observations = length(features)
    noise_v = [
        NormalMeanVariance(
            warm.coefficients[neuron],
            config.noise_v_initial_variance,
        )
        for neuron in 1:n_noise_neurons
    ]
    za = [
        NormalMeanVariance(
            dot(weight_means[neuron], features[observation]),
            0.5,
        )
        for neuron in 1:n_noise_neurons,
            observation in 1:n_observations
    ]
    h = [
        NormalMeanVariance(phi(mean(za[neuron, observation])), 1.0)
        for neuron in 1:n_noise_neurons,
            observation in 1:n_observations
    ]
    c = Matrix{Any}(undef, n_noise_neurons + 1, n_observations)
    for observation in 1:n_observations, neuron in 1:n_noise_neurons
        c[neuron, observation] = NormalMeanVariance(
            warm.coefficients[neuron] * mean(h[neuron, observation]),
            1e-4,
        )
    end
    for observation in 1:n_observations
        c[n_noise_neurons + 1, observation] = NormalMeanVariance(
            warm.intercept,
            1e-4,
        )
    end
    score_means = [
        sum(
            mean(c[input, observation])
            for input in 1:(n_noise_neurons + 1)
        )
        for observation in 1:n_observations
    ]
    score_variance = max(
        config.noise_intercept_initial_variance,
        1e-4,
    )
    score = [
        NormalMeanVariance(value, score_variance)
        for value in score_means
    ]
    gamma = [
        GammaShapeRate(
            priors[:kappa],
            priors[:kappa] / exp(value),
        )
        for value in score_means
    ]
    intercept = NormalMeanVariance(warm.intercept, score_variance)
    return (; noise_v, za, h, c, score, gamma, intercept)
end

@model function hetero_fixed_mean_direct_exp_model(
    n_noise_neurons,
    features,
    fixed_mean,
    y,
    priors,
    activation,
    noise_activation_deps,
    exp_deps,
)
    local noise_w, noise_v, noise_za, noise_h, noise_c, noise_score, gamma

    noise_tau ~ priors[:noise_tau]
    noise_tau_c ~ priors[:noise_tau_c]
    noise_intercept ~ priors[:noise_intercept]
    for neuron in 1:n_noise_neurons
        noise_w[neuron] ~ priors[:noise_w][neuron]
        noise_v[neuron] ~ priors[:noise_v][neuron]
    end
    for observation in eachindex(y)
        for neuron in 1:n_noise_neurons
            noise_za[neuron, observation] ~ softdot(
                features[observation],
                noise_w[neuron],
                noise_tau,
            )
            noise_h[neuron, observation] ~ ResidualSine(
                noise_za[neuron, observation],
            ) where {
                dependencies = noise_activation_deps,
                meta = activation,
            }
            noise_c[neuron, observation] ~ softdot(
                noise_v[neuron],
                noise_h[neuron, observation],
                noise_tau_c,
            )
        end
        noise_c[n_noise_neurons + 1, observation] ~ NormalMeanPrecision(
            noise_intercept,
            1e12,
        )
        noise_score[observation] ~ ManyPlus(inputs = [
            noise_c[input, observation]
            for input in 1:(n_noise_neurons + 1)
        ])
        gamma[observation] ~ priors[:direct_gamma]
        gamma[observation] ~ Exp(noise_score[observation]) where {
            dependencies = exp_deps,
        }
        y[observation] ~ NormalMeanPrecision(
            fixed_mean[observation],
            gamma[observation],
        )
    end
end

@constraints function direct_exp_training_constraints()
    q(
        noise_w,
        noise_v,
        noise_za,
        noise_h,
        noise_c,
        noise_score,
        gamma,
        noise_tau,
        noise_tau_c,
        noise_intercept,
    ) = q(noise_w, noise_za, noise_h, noise_c, noise_score, gamma) *
        q(noise_v)q(noise_tau)q(noise_tau_c)q(noise_intercept)
    q(noise_w)::MomentForm()
end

@initialization function direct_exp_training_initialization(priors, inits)
    q(noise_v) = inits.noise_v
    q(noise_za) = inits.za
    q(noise_h) = inits.h
    q(noise_c) = inits.c
    q(noise_score) = inits.score
    q(gamma) = inits.gamma
    q(noise_tau) = priors[:noise_tau]
    q(noise_tau_c) = priors[:noise_tau_c]
    q(noise_intercept) = inits.intercept
    μ(noise_w) = deepcopy(priors[:noise_w])
end

function direct_exp_posterior_priors(result, previous)
    carried = Dict{Symbol, Any}()
    for name in (
        :w,
        :v,
        :mean_intercept,
        :tau,
        :tau_c,
        :obs_noise,
    )
        haskey(previous, name) &&
            (carried[name] = deepcopy(previous[name]))
    end
    carried[:noise_w] = deepcopy(collect(vec(
        result.posteriors[:noise_w],
    )))
    carried[:noise_v] = deepcopy(collect(vec(
        result.posteriors[:noise_v],
    )))
    carried[:noise_tau] = deepcopy(result.posteriors[:noise_tau])
    carried[:noise_tau_c] = deepcopy(result.posteriors[:noise_tau_c])
    carried[:noise_intercept] = deepcopy(
        result.posteriors[:noise_intercept],
    )
    for name in (
        :noise_beta,
        :direct_gamma,
        :baseline_obs_noise,
        :baseline_precision,
        :score_intercept,
        :kappa,
    )
        carried[name] = deepcopy(previous[name])
    end
    return carried
end

function run_direct_exp_batch(
    priors,
    observations,
    fixed_mean,
    features,
    config,
    spec;
    cold_start,
)
    activation_deps = activation_dependencies(
        config;
        alpha = spec.link_alpha,
        max_step = spec.link_max_step,
    )
    exp_deps = exp_link_dependencies(
        config,
        spec.link_alpha,
        spec.link_max_step,
    )
    inits = exp_noise_pushforward_inits(
        priors,
        features,
        spec.n_noise_neurons,
        config;
        cold_start,
    )
    local result
    elapsed = @elapsed result = infer(
        model = hetero_fixed_mean_direct_exp_model(
            n_noise_neurons = spec.n_noise_neurons,
            priors = priors,
            activation = activation_meta(config),
            noise_activation_deps = activation_deps,
            exp_deps = exp_deps,
        ),
        data = (
            y = observations,
            fixed_mean = fixed_mean,
            features = features,
        ),
        constraints = direct_exp_training_constraints(),
        initialization = direct_exp_training_initialization(priors, inits),
        returnvars = (
            noise_w = KeepLast(),
            noise_v = KeepLast(),
            noise_tau = KeepLast(),
            noise_tau_c = KeepLast(),
            noise_intercept = KeepLast(),
        ),
        iterations = config.max_batch_iterations,
        free_energy = true,
        callbacks = (after_iteration = make_delayed_stopper(config),),
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    all(isfinite, result.free_energy) ||
        error("direct Exp batch has non-finite free energy")
    updated = direct_exp_posterior_priors(result, priors)
    validate_noise_priors(updated)
    expected_states = 2length(observations)
    length(exp_deps.states) == expected_states || error(
        "direct Exp state count $(length(exp_deps.states)) != $expected_states",
    )
    return (
        priors = updated,
        iterations = length(result.free_energy),
        elapsed_seconds = elapsed,
        free_energy = Float64.(result.free_energy),
        link_states = length(exp_deps.states),
    )
end

function run_direct_exp_training(
    train,
    features,
    fixed_means,
    batches,
    initial_priors,
    config,
    spec,
)
    carried = deepcopy(initial_priors)
    reports = NamedTuple[]
    for (batch, indices) in enumerate(batches)
        fitted = run_direct_exp_batch(
            carried,
            train.y[indices],
            fixed_means[indices],
            features[indices],
            config,
            spec;
            cold_start = batch == 1,
        )
        carried = fitted.priors
        push!(reports, (
            batch = batch,
            observations = length(indices),
            iterations = fitted.iterations,
            elapsed_seconds = fitted.elapsed_seconds,
            free_energy = fitted.free_energy,
            final_free_energy = last(fitted.free_energy),
            link_states = fitted.link_states,
        ))
    end
    return (; priors = carried, reports)
end

@model function hetero_noise_direct_exp_prediction_model(
    n_noise_neurons,
    features,
    priors,
    activation,
    activation_deps,
    exp_deps,
)
    local noise_w, noise_v, noise_za, noise_h, noise_c, noise_score, gamma

    noise_tau ~ priors[:noise_tau]
    noise_tau_c ~ priors[:noise_tau_c]
    noise_intercept ~ priors[:noise_intercept]
    for neuron in 1:n_noise_neurons
        noise_w[neuron] ~ priors[:noise_w][neuron]
        noise_v[neuron] ~ priors[:noise_v][neuron]
    end
    for observation in eachindex(features)
        for neuron in 1:n_noise_neurons
            noise_za[neuron, observation] ~ softdot(
                features[observation],
                noise_w[neuron],
                noise_tau,
            )
            noise_h[neuron, observation] ~ ResidualSine(
                noise_za[neuron, observation],
            ) where {
                dependencies = activation_deps,
                meta = activation,
            }
            noise_c[neuron, observation] ~ softdot(
                noise_v[neuron],
                noise_h[neuron, observation],
                noise_tau_c,
            )
        end
        noise_c[n_noise_neurons + 1, observation] ~ NormalMeanPrecision(
            noise_intercept,
            1e12,
        )
        noise_score[observation] ~ ManyPlus(inputs = [
            noise_c[input, observation]
            for input in 1:(n_noise_neurons + 1)
        ])
        gamma[observation] ~ priors[:direct_gamma]
        gamma[observation] ~ Exp(noise_score[observation]) where {
            dependencies = exp_deps,
        }
    end
end

@constraints function direct_exp_prediction_constraints(priors)
    q(
        noise_w,
        noise_v,
        noise_za,
        noise_h,
        noise_c,
        noise_score,
        gamma,
        noise_tau,
        noise_tau_c,
        noise_intercept,
    ) = q(noise_w)q(noise_v)q(noise_tau)q(noise_tau_c) *
        q(noise_intercept)q(noise_za, noise_h, noise_c, noise_score, gamma)
    q(noise_tau)::RxInfer.FixedMarginalFormConstraint(priors[:noise_tau])
    q(noise_tau_c)::RxInfer.FixedMarginalFormConstraint(priors[:noise_tau_c])
    q(noise_intercept)::RxInfer.FixedMarginalFormConstraint(
        priors[:noise_intercept],
    )
    for (neuron, prior) in enumerate(priors[:noise_w])
        q(noise_w[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    for (neuron, prior) in enumerate(priors[:noise_v])
        q(noise_v[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
end

@initialization function direct_exp_prediction_initialization(priors, inits)
    q(noise_w) = deepcopy(priors[:noise_w])
    q(noise_v) = deepcopy(priors[:noise_v])
    q(noise_tau) = priors[:noise_tau]
    q(noise_tau_c) = priors[:noise_tau_c]
    q(noise_intercept) = priors[:noise_intercept]
    q(noise_za) = inits.za
    q(noise_h) = inits.h
    q(noise_c) = inits.c
    q(noise_score) = inits.score
    q(gamma) = inits.gamma
end

function predict_direct_exp_noise_batch(priors, features, config, spec)
    isempty(features) && return Any[]
    exp_deps = exp_link_dependencies(
        config,
        spec.link_alpha,
        spec.link_max_step,
    )
    activation_deps = activation_dependencies(
        config;
        alpha = spec.link_alpha,
        max_step = spec.link_max_step,
    )
    inits = exp_noise_pushforward_inits(
        priors,
        features,
        spec.n_noise_neurons,
        config;
        cold_start = false,
    )
    result = infer(
        model = hetero_noise_direct_exp_prediction_model(
            n_noise_neurons = spec.n_noise_neurons,
            priors = priors,
            activation = activation_meta(config),
            activation_deps = activation_deps,
            exp_deps = exp_deps,
        ),
        data = (features = features,),
        constraints = direct_exp_prediction_constraints(priors),
        initialization = direct_exp_prediction_initialization(priors, inits),
        returnvars = (gamma = KeepLast(),),
        iterations = max(1, config.prediction_iterations),
        free_energy = false,
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    length(exp_deps.states) == 2length(features) ||
        error("direct Exp prediction state count is wrong")
    return collect(vec(result.posteriors[:gamma]))
end

function predict_direct_exp_noise(priors, features, config, spec)
    marginals = Vector{Any}(undef, length(features))
    for first_index in 1:config.prediction_batch_size:length(features)
        indices = first_index:min(
            first_index + config.prediction_batch_size - 1,
            length(features),
        )
        marginals[indices] = predict_direct_exp_noise_batch(
            priors,
            features[indices],
            config,
            spec,
        )
    end
    precision_mean = Float64.(mean.(marginals))
    aleatoric_variance = gamma_inverse_mean.(marginals)
    all(value -> isfinite(value) && value > 0, precision_mean) ||
        error("direct Exp prediction contains invalid precision means")
    return (; precision_mean, aleatoric_variance, marginals)
end

function direct_exp_prediction(priors, features, config, spec)
    mean_prediction = predict_mean(priors, features, config)
    noise_prediction = predict_direct_exp_noise(
        priors,
        features,
        config,
        spec,
    )
    return (
        mean = mean_prediction.mean,
        epistemic_variance = mean_prediction.epistemic_variance,
        aleatoric_variance = noise_prediction.aleatoric_variance,
        total_variance = mean_prediction.epistemic_variance .+
                         noise_prediction.aleatoric_variance,
        precision_mean = noise_prediction.precision_mean,
    )
end

function fit_direct_exp!(
    rows,
    task,
    seed,
    split,
    baseline,
    train,
    train_features,
    fixed_means,
    evaluation,
    evaluation_features,
    batches,
    config,
)
    baseline_obs_noise = baseline.fitted.priors[:obs_noise]
    baseline_precision = mean(baseline_obs_noise)
    for (attempt, spec) in enumerate(exp_retry_specs(config))
        noise_priors = make_exp_noise_priors(
            config,
            spec.n_noise_neurons,
            spec.kappa,
            baseline_precision,
        )
        noise_priors[:direct_gamma] = GammaShapeRate(
            spec.kappa,
            spec.kappa / baseline_precision,
        )
        initial_priors = merge_hierarchy_priors(
            baseline.fitted.priors,
            noise_priors,
            baseline_obs_noise,
        )
        try
            fitted = run_direct_exp_training(
                train,
                train_features,
                fixed_means,
                batches,
                initial_priors,
                config,
                spec,
            )
            prediction = direct_exp_prediction(
                fitted.priors,
                evaluation_features,
                config,
                spec,
            )
            metrics = evaluate_prediction(prediction, evaluation, config)
            summary = batch_summary(fitted.reports)
            row = exp_result_row(
                config;
                task = String(task),
                seed,
                model = "fixed_mean_exp_direct",
                attempt,
                status = "success",
                split_metadata(split)...,
                summary...,
                metrics...,
                initial_precision = baseline_precision,
                initial_log_precision = log(baseline_precision),
                final_precision_mean = mean(prediction.precision_mean),
                mean_alpha = config.mean_ngmp_alpha,
                link_alpha = spec.link_alpha,
                link_max_step = spec.link_max_step,
                mean_coefficients = compact_means(fitted.priors[:v]),
                mean_intercept_mean = mean(
                    fitted.priors[:mean_intercept],
                ),
                noise_coefficients = compact_means(
                    fitted.priors[:noise_v],
                ),
                noise_intercept_mean = mean(
                    fitted.priors[:noise_intercept],
                ),
            )
            record_exp_row!(rows, row, config)
            println(
                "fixed_mean/direct-exp task=$task seed=$seed ",
                "attempt=$attempt ",
                "MSE=$(round(metrics.mean_mse; digits=5)) ",
                "NLL=$(round(metrics.nll; digits=5)) ",
                "noise-corr=$(round(metrics.aleatoric_correlation; digits=4))",
            )
            return (; fitted, prediction, spec, row)
        catch exception
            message = concise_error(exception)
            record_exp_row!(rows, exp_result_row(
                config;
                task = String(task),
                seed,
                model = "fixed_mean_exp_direct",
                attempt,
                status = "failure",
                error = message,
                split_metadata(split)...,
                initial_precision = baseline_precision,
                initial_log_precision = log(baseline_precision),
                mean_alpha = config.mean_ngmp_alpha,
                link_alpha = spec.link_alpha,
                link_max_step = spec.link_max_step,
            ), config)
            println(
                "fixed_mean/direct-exp retry $attempt failed for ",
                "$task/$seed: $message",
            )
        end
    end
    return nothing
end

function restriction_grid_diagnostics(
    fitted,
    task,
    config,
    local_config,
    arm,
)
    axes, grid = grid_data(task, config)
    prediction = arm.direct ?
                 direct_exp_prediction(
        fitted.fitted.priors,
        build_features(grid),
        local_config,
        fitted.spec,
    ) : exp_hierarchy_prediction(
        fitted.fitted.priors,
        build_features(grid),
        local_config,
        fitted.spec,
    )
    learned = prediction.aleatoric_variance
    truth = grid.true_variance
    diagnostics = (
        grid_variance_minimum = minimum(learned),
        grid_variance_maximum = maximum(learned),
        grid_variance_ratio = maximum(learned) / minimum(learned),
        grid_variance_cv = std(learned) / mean(learned),
        grid_variance_correlation = std(truth) > 0 ? cor(learned, truth) : 0.0,
        true_grid_variance_ratio = maximum(truth) / minimum(truth),
    )
    return (; axes, grid, prediction, diagnostics)
end

function persist_restriction_rows(rows, config)
    config.save_outputs || return nothing
    ensure_output_directory(config)
    CSV.write(joinpath(config.output_dir, "attempts.csv"), DataFrame(rows))
    return nothing
end

function append_restriction_rows!(
    rows,
    local_rows,
    config,
    arm;
    diagnostics = nothing,
)
    grid_fields = isnothing(diagnostics) ? (
        grid_variance_minimum = NaN,
        grid_variance_maximum = NaN,
        grid_variance_ratio = NaN,
        grid_variance_cv = NaN,
        grid_variance_correlation = NaN,
        true_grid_variance_ratio = NaN,
    ) : diagnostics
    for row in local_rows
        successful_arm = row.status == "success" &&
                         row.model in (
            "fixed_mean_exp",
            "fixed_mean_exp_direct",
        )
        push!(rows, merge(row, (
            arm = String(arm.name),
            hierarchy = arm.direct ? "independent_gamma" : "shared_gamma",
            kappa = arm.kappa,
            coefficient_initial_variance =
                arm.coefficient_initial_variance,
            score_initial_variance = arm.score_initial_variance,
            warm_precision = arm.warm_precision,
        ), successful_arm ? grid_fields : (
            grid_variance_minimum = NaN,
            grid_variance_maximum = NaN,
            grid_variance_ratio = NaN,
            grid_variance_cv = NaN,
            grid_variance_correlation = NaN,
            true_grid_variance_ratio = NaN,
        )))
    end
    persist_restriction_rows(rows, config)
    return nothing
end

function append_restriction_baseline_rows!(rows, local_rows, config)
    baseline_arm = restriction_arm(:baseline, NaN, NaN, NaN)
    append_restriction_rows!(rows, local_rows, config, baseline_arm)
    return nothing
end

function save_restriction_grid(
    diagnostic,
    task,
    seed,
    arm,
    config,
)
    config.save_outputs || return nothing
    ensure_output_directory(config)
    prefix = join((
        String(config.stage),
        String(task),
        string(seed),
        String(arm.name),
    ), "_")
    CSV.write(joinpath(config.output_dir, prefix * "_grid.csv"), DataFrame(
        x1 = diagnostic.grid.x1,
        x2 = diagnostic.grid.x2,
        true_mean = diagnostic.grid.clean_mean,
        true_aleatoric_variance = diagnostic.grid.true_variance,
        predictive_mean = diagnostic.prediction.mean,
        epistemic_variance = diagnostic.prediction.epistemic_variance,
        aleatoric_variance = diagnostic.prediction.aleatoric_variance,
        total_variance = diagnostic.prediction.total_variance,
    ))
    return nothing
end

function write_restriction_note(config, arms)
    config.save_outputs || return nothing
    ensure_output_directory(config)
    arm_list = join(string.(getfield.(arms, :name)), ", ")
    open(joinpath(config.output_dir, "README.md"), "w") do io
        println(io, "# Exp hierarchy and initialization restriction ablation")
        println(io)
        println(io, "The constant-noise mean model is fitted once per task and seed, then frozen for every arm. The Exp link, residual-sine head, priors, data split, batching, and inference settings are otherwise unchanged.")
        println(io)
        println(io, "The mean model includes a learned Normal intercept with prior mean $(config.mean_intercept_prior_mean) and variance $(config.mean_intercept_prior_variance). For each observation, its identity output is appended directly to ManyPlus without an activation.")
        println(io)
        println(io, "The factorial arms vary κ ∈ {50,2}, the initial variance of q(noise_v) ∈ {1e-10,0.1}, and the initial variance of q(noise_intercept) and q(noise_score) ∈ {1e-10,0.01}.")
        println(io)
        println(io, "Follow-up `unpooled_*` arms replace GammaShapeRate(κ, shared β) with a fixed independent Gamma anchor of the same initial mean for every observation. The Exp NGMP rules require a Gamma-family message, so this is the weakest valid no-shared-β graph rather than a literally prior-free output.")
        println(io)
        println(io, "Follow-up `warm*` arms initialize q(noise_v) and q(noise_intercept) at the regularized Newton solution of the frozen-mean Gaussian log-likelihood on the training residuals. This changes only the optimizer start; the factor-graph priors and one-pass likelihood updates are unchanged.")
        println(io)
        println(io, "Arms in this run: $arm_list.")
    end
    return nothing
end

function run_exp_restriction_ablation(config = exp_restriction_config())
    arms = selected_restriction_arms()
    write_restriction_note(config, arms)
    rows = NamedTuple[]
    evaluation_split = config.stage === :full ? :test : :validation
    for seed in config.seeds
        paired = make_paired_datasets(config, seed)
        split = deterministic_split(
            config.n_samples,
            config,
            config.split_seed + seed,
        )
        for task in config.tasks
            println("\n=== Restriction ablation: task=$task seed=$seed ===")
            data = paired[task]
            train = subset_data(data, split.train)
            evaluation = subset_data(
                data,
                getproperty(split, evaluation_split),
            )
            train_features = build_features(train)
            evaluation_features = build_features(evaluation)
            batches = deterministic_batch_ranges(
                length(train.y),
                config.n_training_batches,
            )

            baseline_rows = NamedTuple[]
            local_base_config = merge(config, (
                save_outputs = false,
                save_plots = false,
            ))
            baseline = fit_exp_baseline!(
                baseline_rows,
                task,
                seed,
                split,
                train,
                train_features,
                evaluation,
                evaluation_features,
                batches,
                local_base_config,
            )
            append_restriction_baseline_rows!(
                rows,
                baseline_rows,
                config,
            )
            isnothing(baseline) && continue
            fixed_means = try
                predict_mean(
                    baseline.fitted.priors,
                    train_features,
                    config,
                ).mean
            catch exception
                println(
                    "training-point prediction failed: ",
                    concise_error(exception),
                )
                continue
            end

            for arm in arms
                println(
                    "arm=$(arm.name) κ=$(arm.kappa) ",
                    "q(v) init var=$(arm.coefficient_initial_variance) ",
                    "q(score) init var=$(arm.score_initial_variance) ",
                    "warm precision=$(arm.warm_precision)",
                )
                local_config = arm_config(config, arm)
                if isfinite(arm.warm_precision)
                    baseline_precision = mean(
                        baseline.fitted.priors[:obs_noise],
                    )
                    warm_priors = make_exp_noise_priors(
                        local_config,
                        local_config.full_noise_neurons,
                        arm.kappa,
                        baseline_precision,
                    )
                    warm = variance_warm_start(
                        warm_priors,
                        train_features,
                        train.y,
                        fixed_means,
                        local_config,
                        arm.warm_precision,
                    )
                    local_config = merge(
                        local_config,
                        (variance_warm_start = warm,),
                    )
                    println(
                        "warm coefficients=",
                        join(round.(warm.coefficients; digits = 4), ','),
                        " intercept=",
                        round(warm.intercept; digits = 4),
                    )
                end
                local_rows = NamedTuple[]
                fitted = arm.direct ?
                         fit_direct_exp!(
                    local_rows,
                    task,
                    seed,
                    split,
                    baseline,
                    train,
                    train_features,
                    fixed_means,
                    evaluation,
                    evaluation_features,
                    batches,
                    local_config,
                ) : fit_exp_hierarchy!(
                    local_rows,
                    task,
                    seed,
                    split,
                    baseline,
                    train,
                    train_features,
                    fixed_means,
                    evaluation,
                    evaluation_features,
                    batches,
                    local_config,
                )
                diagnostic = if isnothing(fitted)
                    nothing
                else
                    try
                        restriction_grid_diagnostics(
                            fitted,
                            task,
                            config,
                            local_config,
                            arm,
                        )
                    catch exception
                        println(
                            "grid diagnostic failed for $(arm.name): ",
                            concise_error(exception),
                        )
                        nothing
                    end
                end
                append_restriction_rows!(
                    rows,
                    local_rows,
                    config,
                    arm;
                    diagnostics = isnothing(diagnostic) ?
                                  nothing : diagnostic.diagnostics,
                )
                isnothing(diagnostic) || save_restriction_grid(
                    diagnostic,
                    task,
                    seed,
                    arm,
                    config,
                )
            end
        end
    end
    persist_restriction_rows(rows, config)
    successes = count(
        row -> row.status == "success" &&
               row.model in (
            "fixed_mean_exp",
            "fixed_mean_exp_direct",
        ),
        rows,
    )
    failures = count(row -> row.status == "failure", rows)
    println(
        "\nRestriction ablation completed: $successes successful arms, ",
        "$failures failed attempts",
    )
    successes > 0 || error("no restriction arm completed successfully")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_exp_restriction_ablation()
end
