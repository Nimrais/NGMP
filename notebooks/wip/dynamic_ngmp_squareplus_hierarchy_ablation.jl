# Staged ETTh2 ablation for two separate Squareplus hypotheses:
#
#   1. Does retaining q(z, gamma, out) make the shared consensus output useful?
#   2. Does Squareplus let the nonlinear low-rank CT projector learn routing?
#
# The default `all` workflow audits Unscented against Quadrature(64), evaluates
# native-out and CT-direct separately, and runs CT+out only when one component
# passes a predeclared validation criterion.  Kappa is fixed at one throughout.

include(joinpath(@__DIR__, "dynamic_ngmp_native_softplus.jl"))

function hierarchy_env_symbols(name, default)
    values = unique(Symbol.(lowercase.(strip.(split(
        get(ENV, name, join(string.(default), ',')),
        ',',
    )))))
    isempty(values) && throw(ArgumentError("$name must not be empty"))
    return values
end

function squareplus_hierarchy_config()
    base = native_softplus_config()
    stage = Symbol(lowercase(get(
        ENV,
        "SQUAREPLUS_HIERARCHY_STAGE",
        "smoke",
    )))
    stage in (:smoke, :audit, :screen, :full, :all) || throw(ArgumentError(
        "SQUAREPLUS_HIERARCHY_STAGE must be smoke, audit, screen, full, or all",
    ))
    projection_mode = Symbol(lowercase(get(
        ENV,
        "SQUAREPLUS_HIERARCHY_PROJECTION",
        "auto",
    )))
    projection_mode in (:auto, :unscented, :quadrature64) || throw(
        ArgumentError(
            "SQUAREPLUS_HIERARCHY_PROJECTION must be auto, unscented, or " *
            "quadrature64",
        ),
    )
    seeds = projector_env_widths(
        "SQUAREPLUS_HIERARCHY_SEEDS",
        (2_026, 2_027, 2_028),
    )
    selected_arm_token = lowercase(strip(get(
        ENV,
        "SQUAREPLUS_HIERARCHY_SELECTED_ARM",
        "",
    )))
    selected_arm = isempty(selected_arm_token) ? nothing : Symbol(selected_arm_token)
    isnothing(selected_arm) || selected_arm in (
        :native_square_out,
        :ct_square_direct,
        :ct_square_out,
    ) || throw(ArgumentError(
        "SQUAREPLUS_HIERARCHY_SELECTED_ARM must be native_square_out, " *
        "ct_square_direct, or ct_square_out",
    ))
    base.kappa == 1.0 || throw(ArgumentError(
        "Squareplus hierarchy ablation fixes PROJECTOR_BETA_FLOOR_WEIGHT at 1",
    ))
    dataset_token = lowercase(base.dataset)
    default_output_name =
        "dynamic_ngmp_squareplus_hierarchy_$(dataset_token)_h$(base.horizon)"
    return merge(base, (;
        stage,
        projection_mode,
        selected_arm,
        seeds,
        hidden_width = projector_env_int("SQUAREPLUS_HIERARCHY_WIDTH", 8),
        rank_multiplier = projector_env_int(
            "SQUAREPLUS_HIERARCHY_RANK_MULTIPLIER",
            2,
        ),
        prior_variance = projector_env_float(
            "SQUAREPLUS_HIERARCHY_CT_VARIANCE",
            0.25,
        ),
        head_prior_mean_scale = projector_env_float(
            "SQUAREPLUS_HIERARCHY_HEAD_MEAN_SCALE",
            4.0,
        ),
        head_prior_precision = projector_env_float(
            "SQUAREPLUS_HIERARCHY_HEAD_PRECISION",
            0.01,
        ),
        ct_precision_mean = projector_env_float(
            "SQUAREPLUS_HIERARCHY_CT_PRECISION_MEAN",
            10.0,
        ),
        ct_dof_multiplier = projector_env_float(
            "SQUAREPLUS_HIERARCHY_CT_DOF_MULTIPLIER",
            1.0,
        ),
        obs_projection = Symbol(lowercase(get(
            ENV,
            "SQUAREPLUS_HIERARCHY_OBS_PROJECTION",
            "delta",
        ))),
        obs_alpha = projector_env_float(
            "SQUAREPLUS_HIERARCHY_OBS_ALPHA",
            0.2,
        ),
        obs_beta = projector_env_float(
            "SQUAREPLUS_HIERARCHY_OBS_BETA",
            0.0,
        ),
        obs_max_step = projector_env_float(
            "SQUAREPLUS_HIERARCHY_OBS_MAX_STEP",
            1.0,
        ),
        obs_precision_shape = projector_env_float(
            "SQUAREPLUS_HIERARCHY_OBS_PRECISION_SHAPE",
            10.0,
        ),
        obs_precision_mean = projector_env_float(
            "SQUAREPLUS_HIERARCHY_OBS_PRECISION_MEAN",
            10.0,
        ),
        smoke_observations = projector_env_int(
            "SQUAREPLUS_HIERARCHY_SMOKE_OBSERVATIONS",
            24,
        ),
        smoke_iterations = projector_env_int(
            "SQUAREPLUS_HIERARCHY_SMOKE_ITERATIONS",
            2,
        ),
        screen_train = projector_env_int(
            "SQUAREPLUS_HIERARCHY_SCREEN_TRAIN",
            2_048,
        ),
        screen_validation = projector_env_int(
            "SQUAREPLUS_HIERARCHY_SCREEN_VALIDATION",
            1_024,
        ),
        screen_iterations = projector_env_int(
            "SQUAREPLUS_HIERARCHY_SCREEN_ITERATIONS",
            20,
        ),
        full_iterations = projector_env_int(
            "SQUAREPLUS_HIERARCHY_FULL_ITERATIONS",
            20,
        ),
        output_prefix = get(
            ENV,
            "OUTPUT_PREFIX",
            joinpath(PROJECTOR_ROOT, "viz", default_output_name),
        ),
    ))
end

function hierarchy_observation_projection(name)
    name === :delta && return TangentProjection(type = DeltaApproximation)
    name === :unscented && return TangentProjection(type = Unscented)
    name === :quadrature16 && return TangentProjection(type = Quadrature(16))
    throw(ArgumentError(
        "SQUAREPLUS_HIERARCHY_OBS_PROJECTION must be delta, unscented, or " *
        "quadrature16",
    ))
end

@model function dynamic_native_squareplus_out(
    n_forecasters,
    n_obs,
    y,
    features,
    predictions,
    priors,
    link_dependencies,
    link_damping,
    observation_dependencies,
    observation_damping,
)
    local w, z, gamma, tau, beta, out, obs_noise

    obs_noise ~ priors[:obs_noise]
    for i in 1:n_forecasters
        w[i] ~ priors[:w][i]
        tau[i] ~ priors[:tau][i]
        beta[i] ~ priors[:beta][i]
    end
    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], tau[i]) where {
                meta = LowRankMeta(),
            }
            gamma[i, j] ~ GammaShapeRate(1.0, beta[i])
            gamma[i, j] ~ Squareplus(z[i, j]) where {
                dependencies = link_dependencies,
                meta = link_damping,
            }
            out[j] ~ NormalMeanPrecision(
                predictions[i, j],
                gamma[i, j],
            ) where {
                dependencies = observation_dependencies,
                meta = observation_damping,
            }
        end
        y[j] ~ NormalMeanPrecision(out[j], obs_noise)
    end
end

@model function dynamic_ct_squareplus_direct(
    n_forecasters,
    n_obs,
    y,
    features,
    predictions,
    priors,
    map_meta,
    hidden_meta,
    map_dependencies,
    hidden_dependencies,
    activation,
    activation_dependencies,
    link_dependencies,
    link_damping,
)
    local h1, s, h2, w, z, gamma, tau, beta
    local a_map, a_hidden, P_map, P_hidden

    a_map ~ priors[:a_map]
    a_hidden ~ priors[:a_hidden]
    P_map ~ priors[:P_map]
    P_hidden ~ priors[:P_hidden]
    for i in 1:n_forecasters
        w[i] ~ priors[:w][i]
        tau[i] ~ priors[:tau][i]
        beta[i] ~ priors[:beta][i]
    end
    for j in 1:n_obs
        h1[j] ~ ContinuousTransition(features[j], a_map, P_map) where {
            dependencies = map_dependencies,
            meta = map_meta,
        }
        s[j] ~ MvResidualSine(h1[j]) where {
            dependencies = activation_dependencies,
            meta = activation,
        }
        h2[j] ~ ContinuousTransition(s[j], a_hidden, P_hidden) where {
            dependencies = hidden_dependencies,
            meta = hidden_meta,
        }
        for i in 1:n_forecasters
            z[i, j] ~ softdot(w[i], h2[j], tau[i])
            gamma[i, j] ~ GammaShapeRate(1.0, beta[i])
            gamma[i, j] ~ Squareplus(z[i, j]) where {
                dependencies = link_dependencies,
                meta = link_damping,
            }
            y[j] ~ NormalMeanPrecision(predictions[i, j], gamma[i, j])
        end
    end
end

@model function dynamic_ct_log_direct(
    n_forecasters,
    n_obs,
    y,
    features,
    predictions,
    priors,
    map_meta,
    hidden_meta,
    map_dependencies,
    hidden_dependencies,
    activation,
    activation_dependencies,
    link_dependencies,
    link_damping,
)
    local h1, s, h2, w, z, gamma, tau, beta
    local a_map, a_hidden, P_map, P_hidden

    a_map ~ priors[:a_map]
    a_hidden ~ priors[:a_hidden]
    P_map ~ priors[:P_map]
    P_hidden ~ priors[:P_hidden]
    for i in 1:n_forecasters
        w[i] ~ priors[:w][i]
        tau[i] ~ priors[:tau][i]
        beta[i] ~ priors[:beta][i]
    end
    for j in 1:n_obs
        h1[j] ~ ContinuousTransition(features[j], a_map, P_map) where {
            dependencies = map_dependencies,
            meta = map_meta,
        }
        s[j] ~ MvResidualSine(h1[j]) where {
            dependencies = activation_dependencies,
            meta = activation,
        }
        h2[j] ~ ContinuousTransition(s[j], a_hidden, P_hidden) where {
            dependencies = hidden_dependencies,
            meta = hidden_meta,
        }
        for i in 1:n_forecasters
            z[i, j] ~ softdot(w[i], h2[j], tau[i])
            gamma[i, j] ~ GammaShapeRate(1.0, beta[i])
            z[i, j] ~ Log(gamma[i, j]) where {
                dependencies = link_dependencies,
                meta = link_damping,
            }
            y[j] ~ NormalMeanPrecision(predictions[i, j], gamma[i, j])
        end
    end
end

@model function dynamic_ct_squareplus_out(
    n_forecasters,
    n_obs,
    y,
    features,
    predictions,
    priors,
    map_meta,
    hidden_meta,
    map_dependencies,
    hidden_dependencies,
    activation,
    activation_dependencies,
    link_dependencies,
    link_damping,
    observation_dependencies,
    observation_damping,
)
    local h1, s, h2, w, z, gamma, tau, beta, out, obs_noise
    local a_map, a_hidden, P_map, P_hidden

    a_map ~ priors[:a_map]
    a_hidden ~ priors[:a_hidden]
    P_map ~ priors[:P_map]
    P_hidden ~ priors[:P_hidden]
    obs_noise ~ priors[:obs_noise]
    for i in 1:n_forecasters
        w[i] ~ priors[:w][i]
        tau[i] ~ priors[:tau][i]
        beta[i] ~ priors[:beta][i]
    end
    for j in 1:n_obs
        h1[j] ~ ContinuousTransition(features[j], a_map, P_map) where {
            dependencies = map_dependencies,
            meta = map_meta,
        }
        s[j] ~ MvResidualSine(h1[j]) where {
            dependencies = activation_dependencies,
            meta = activation,
        }
        h2[j] ~ ContinuousTransition(s[j], a_hidden, P_hidden) where {
            dependencies = hidden_dependencies,
            meta = hidden_meta,
        }
        for i in 1:n_forecasters
            z[i, j] ~ softdot(w[i], h2[j], tau[i])
            gamma[i, j] ~ GammaShapeRate(1.0, beta[i])
            gamma[i, j] ~ Squareplus(z[i, j]) where {
                dependencies = link_dependencies,
                meta = link_damping,
            }
            out[j] ~ NormalMeanPrecision(
                predictions[i, j],
                gamma[i, j],
            ) where {
                dependencies = observation_dependencies,
                meta = observation_damping,
            }
        end
        y[j] ~ NormalMeanPrecision(out[j], obs_noise)
    end
end

@constraints function native_squareplus_out_constraints()
    q(w, z, gamma, out, tau, beta, obs_noise) =
        q(w)q(z, gamma, out)q(tau)q(beta)q(obs_noise)
    q(w)::MomentForm()
end

@constraints function ct_hierarchy_direct_constraints()
    q(
        h1,
        s,
        h2,
        z,
        gamma,
        a_map,
        a_hidden,
        P_map,
        P_hidden,
        w,
        tau,
        beta,
    ) = q(h1)q(s, h2, z, gamma)q(a_map)q(a_hidden)q(P_map)q(P_hidden)q(w)q(tau)q(beta)
    q(a_map)::MomentForm()
    q(a_hidden)::MomentForm()
    q(w)::MomentForm()
end

@constraints function ct_hierarchy_out_constraints()
    q(
        h1,
        s,
        h2,
        z,
        gamma,
        out,
        a_map,
        a_hidden,
        P_map,
        P_hidden,
        w,
        tau,
        beta,
        obs_noise,
    ) = q(h1)q(s, h2, z, gamma, out)q(a_map)q(a_hidden)q(P_map)q(P_hidden)q(w)q(tau)q(beta)q(obs_noise)
    q(a_map)::MomentForm()
    q(a_hidden)::MomentForm()
    q(w)::MomentForm()
end

function hierarchy_output_initial_moments(targets)
    return mean(targets), max(var(targets), 1e-3)
end

function native_squareplus_out_initialization(priors, targets)
    output_mean, output_variance = hierarchy_output_initial_moments(targets)
    return @initialization begin
        q(w) = deepcopy(priors[:w])
        q(z) = NormalMeanVariance(0.0, 1.0)
        q(gamma) = GammaShapeScale(1.0, 1.0)
        q(out) = NormalMeanVariance(output_mean, output_variance)
        q(tau) = deepcopy(priors[:tau])
        q(beta) = deepcopy(priors[:beta])
        q(obs_noise) = priors[:obs_noise]
        μ(gamma) = GammaShapeScale(1.0, 1.0)
        μ(out) = NormalMeanVariance(output_mean, output_variance)
    end
end

function ct_squareplus_out_initialization(
    priors,
    activation,
    hidden_width,
    targets,
)
    s_mean, s_covariance = SurrogateModelling._mv_residual_sine_mean_cov(
        zeros(hidden_width),
        Matrix(Diagonal(ones(hidden_width))),
        activation,
    )
    output_mean, output_variance = hierarchy_output_initial_moments(targets)
    return @initialization begin
        q(a_map) = priors[:a_map]
        q(a_hidden) = priors[:a_hidden]
        q(P_map) = priors[:P_map]
        q(P_hidden) = priors[:P_hidden]
        q(w) = deepcopy(priors[:w])
        q(tau) = deepcopy(priors[:tau])
        q(beta) = deepcopy(priors[:beta])
        q(obs_noise) = priors[:obs_noise]
        q(h1) = MvNormalMeanCovariance(
            zeros(hidden_width),
            Diagonal(ones(hidden_width)),
        )
        q(s) = MvNormalMeanCovariance(s_mean, s_covariance)
        q(h2) = MvNormalMeanCovariance(
            zeros(hidden_width),
            Diagonal(ones(hidden_width)),
        )
        q(z) = NormalMeanVariance(0.0, 1.0)
        q(gamma) = GammaShapeScale(1.0, 1.0)
        q(out) = NormalMeanVariance(output_mean, output_variance)
        μ(gamma) = GammaShapeScale(1.0, 1.0)
        μ(out) = NormalMeanVariance(output_mean, output_variance)
    end
end

hierarchy_has_out(::Val{:native_square_out}) = true
hierarchy_has_out(::Val{:ct_square_out}) = true
hierarchy_has_out(::Val) = false

hierarchy_is_ct(::Val{:ct_log_direct}) = true
hierarchy_is_ct(::Val{:ct_square_direct}) = true
hierarchy_is_ct(::Val{:ct_square_out}) = true
hierarchy_is_ct(::Val) = false

hierarchy_link(::Val{:native_log_direct}) = Val(:log)
hierarchy_link(::Val{:ct_log_direct}) = Val(:log)
hierarchy_link(::Val) = Val(:squareplus)

hierarchy_architecture_name(::Val{architecture}) where {architecture} = architecture

function hierarchy_link_dependencies(::Val{:log}, config)
    return NGMPDependencies(out = nothing, in = nothing)
end

function hierarchy_link_dependencies(::Val{:squareplus}, config)
    return NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = native_squareplus_projection(config.squareplus_projection),
    )
end

function hierarchy_observation_dependencies(config)
    return NGMPDependencies(
        out = nothing,
        τ = nothing,
        projection = hierarchy_observation_projection(config.obs_projection),
    )
end

function hierarchy_observation_damping(config)
    return DampingMeta(
        alpha = config.obs_alpha,
        beta = config.obs_beta,
        max_step = config.obs_max_step,
    )
end

function hierarchy_obs_noise_prior(config)
    return GammaShapeRate(
        config.obs_precision_shape,
        config.obs_precision_shape / config.obs_precision_mean,
    )
end

function hierarchy_ct_dependencies(architecture, config)
    ct_damping = DampingMeta(
        alpha = config.ct_alpha,
        beta = config.ct_beta,
        max_step = config.ct_max_step,
    )
    dependencies = (;
        map = NGMPDependencies(a = nothing, damping = deepcopy(ct_damping)),
        hidden = NGMPDependencies(a = nothing, damping = deepcopy(ct_damping)),
        activation = NGMPDependencies(
            out = nothing,
            in = nothing,
            projection = TangentProjection(type = ClosedForm),
            damping = DampingMeta(
                alpha = config.activation_alpha,
                beta = config.activation_beta,
                max_step = config.activation_max_step,
            ),
        ),
        link = hierarchy_link_dependencies(hierarchy_link(architecture), config),
    )
    return hierarchy_has_out(architecture) ? merge(dependencies, (;
        observation = hierarchy_observation_dependencies(config),
    )) : dependencies
end

function check_hierarchy_states(name, dependencies, expected, iterations)
    length(dependencies.states) == expected || error(
        "$name state count $(length(dependencies.states)) != $expected",
    )
    firing_counts = getfield.(dependencies.states, :nfired)
    all(>=(iterations), firing_counts) || error(
        "$name firing counts $(extrema(firing_counts)) do not cover " *
        "$iterations iterations",
    )
    return extrema(firing_counts)
end

function run_native_squareplus_out_fit(
    training_data,
    base_priors,
    config;
    iterations,
)
    n_forecasters = size(training_data.predictions, 1)
    priors = normalized_native_priors(base_priors)
    priors[:obs_noise] = hierarchy_obs_noise_prior(config)
    link_dependencies = hierarchy_link_dependencies(Val(:squareplus), config)
    observation_dependencies = hierarchy_observation_dependencies(config)
    measured = @timed infer(
        model = dynamic_native_squareplus_out(
            n_forecasters = n_forecasters,
            n_obs = length(training_data.y),
            priors = priors,
            link_dependencies = link_dependencies,
            link_damping = DampingMeta(alpha = config.alpha, beta = config.beta),
            observation_dependencies = observation_dependencies,
            observation_damping = hierarchy_observation_damping(config),
        ),
        data = (
            y = training_data.y,
            features = training_data.features,
            predictions = training_data.predictions,
        ),
        constraints = native_squareplus_out_constraints(),
        initialization = native_squareplus_out_initialization(
            priors,
            training_data.y,
        ),
        iterations = iterations,
        free_energy = config.free_energy,
        showprogress = config.show_progress,
        options = (limit_stack_depth = 500,),
        disable_inference_error_hint = true,
        returnvars = (
            w = KeepLast(),
            tau = KeepLast(),
            beta = KeepLast(),
            obs_noise = KeepLast(),
        ),
    )
    n_local = n_forecasters * length(training_data.y)
    check_hierarchy_states(:native_square_link, link_dependencies, 2n_local, iterations)
    check_hierarchy_states(
        :native_square_observation,
        observation_dependencies,
        2n_local,
        iterations,
    )
    result = measured.value
    return (;
        result,
        posterior = (;
            w = result.posteriors[:w],
            tau = result.posteriors[:tau],
            beta = result.posteriors[:beta],
            obs_noise = result.posteriors[:obs_noise],
        ),
        training_seconds = measured.time,
        training_bytes = measured.bytes,
        dependencies = (;
            link = link_dependencies,
            observation = observation_dependencies,
        ),
        parameter_count =
            n_forecasters * length(first(training_data.features)) + 1,
    )
end

function hierarchy_ct_model(
    ::Val{:ct_square_direct},
    n_forecasters,
    n_obs,
    setup,
    activation,
    dependencies,
    config,
)
    return dynamic_ct_squareplus_direct(
        n_forecasters = n_forecasters,
        n_obs = n_obs,
        priors = setup.priors,
        map_meta = setup.map_meta,
        hidden_meta = setup.hidden_meta,
        map_dependencies = dependencies.map,
        hidden_dependencies = dependencies.hidden,
        activation = activation,
        activation_dependencies = dependencies.activation,
        link_dependencies = dependencies.link,
        link_damping = DampingMeta(alpha = config.alpha, beta = config.beta),
    )
end

function hierarchy_ct_model(
    ::Val{:ct_log_direct},
    n_forecasters,
    n_obs,
    setup,
    activation,
    dependencies,
    config,
)
    return dynamic_ct_log_direct(
        n_forecasters = n_forecasters,
        n_obs = n_obs,
        priors = setup.priors,
        map_meta = setup.map_meta,
        hidden_meta = setup.hidden_meta,
        map_dependencies = dependencies.map,
        hidden_dependencies = dependencies.hidden,
        activation = activation,
        activation_dependencies = dependencies.activation,
        link_dependencies = dependencies.link,
        link_damping = DampingMeta(alpha = config.alpha, beta = config.beta),
    )
end

function hierarchy_ct_model(
    ::Val{:ct_square_out},
    n_forecasters,
    n_obs,
    setup,
    activation,
    dependencies,
    config,
)
    return dynamic_ct_squareplus_out(
        n_forecasters = n_forecasters,
        n_obs = n_obs,
        priors = setup.priors,
        map_meta = setup.map_meta,
        hidden_meta = setup.hidden_meta,
        map_dependencies = dependencies.map,
        hidden_dependencies = dependencies.hidden,
        activation = activation,
        activation_dependencies = dependencies.activation,
        link_dependencies = dependencies.link,
        link_damping = DampingMeta(alpha = config.alpha, beta = config.beta),
        observation_dependencies = dependencies.observation,
        observation_damping = hierarchy_observation_damping(config),
    )
end

hierarchy_ct_constraints(::Val{:ct_square_out}) = ct_hierarchy_out_constraints()
hierarchy_ct_constraints(::Val) = ct_hierarchy_direct_constraints()

function hierarchy_ct_initialization(
    architecture,
    setup,
    activation,
    hidden_width,
    targets,
)
    if hierarchy_has_out(architecture)
        return ct_squareplus_out_initialization(
            setup.priors,
            activation,
            hidden_width,
            targets,
        )
    end
    return nonlinear_projector_initialization(
        setup.priors,
        activation,
        hidden_width,
    )
end

function run_hierarchy_ct_fit(
    architecture,
    training_data,
    base_priors,
    config;
    iterations,
)
    n_forecasters = size(training_data.predictions, 1)
    setup = make_dynamic_projector_setup(
        training_data.features,
        base_priors,
        config.hidden_width,
        n_forecasters,
        config,
    )
    if hierarchy_has_out(architecture)
        priors = copy(setup.priors)
        priors[:obs_noise] = hierarchy_obs_noise_prior(config)
        setup = merge(setup, (;
            priors,
            parameter_count = setup.parameter_count + 1,
        ))
    end
    dependencies = hierarchy_ct_dependencies(architecture, config)
    activation = ResidualSineMeta(rho = config.rho, omega = config.omega)
    measured = @timed infer(
        model = hierarchy_ct_model(
            architecture,
            n_forecasters,
            length(training_data.y),
            setup,
            activation,
            dependencies,
            config,
        ),
        data = (
            y = training_data.y,
            features = training_data.features,
            predictions = training_data.predictions,
        ),
        constraints = hierarchy_ct_constraints(architecture),
        initialization = hierarchy_ct_initialization(
            architecture,
            setup,
            activation,
            config.hidden_width,
            training_data.y,
        ),
        iterations = iterations,
        free_energy = config.free_energy,
        showprogress = config.show_progress,
        options = (limit_stack_depth = 500,),
        disable_inference_error_hint = true,
        returnvars = hierarchy_has_out(architecture) ? (
            a_map = KeepLast(),
            a_hidden = KeepLast(),
            P_map = KeepLast(),
            P_hidden = KeepLast(),
            w = KeepLast(),
            tau = KeepLast(),
            beta = KeepLast(),
            obs_noise = KeepLast(),
        ) : (
            a_map = KeepLast(),
            a_hidden = KeepLast(),
            P_map = KeepLast(),
            P_hidden = KeepLast(),
            w = KeepLast(),
            tau = KeepLast(),
            beta = KeepLast(),
        ),
    )
    n_observations = length(training_data.y)
    n_local = n_forecasters * n_observations
    check_hierarchy_states(:ct_map, dependencies.map, n_observations, iterations)
    check_hierarchy_states(
        :ct_hidden,
        dependencies.hidden,
        n_observations,
        iterations,
    )
    check_hierarchy_states(
        :ct_activation,
        dependencies.activation,
        2n_observations,
        iterations,
    )
    check_hierarchy_states(:ct_link, dependencies.link, 2n_local, iterations)
    if hierarchy_has_out(architecture)
        check_hierarchy_states(
            :ct_observation,
            dependencies.observation,
            2n_local,
            iterations,
        )
    end
    result = measured.value
    posterior = (;
        a_map = result.posteriors[:a_map],
        a_hidden = result.posteriors[:a_hidden],
        P_map = result.posteriors[:P_map],
        P_hidden = result.posteriors[:P_hidden],
        w = result.posteriors[:w],
        tau = result.posteriors[:tau],
        beta = result.posteriors[:beta],
    )
    if hierarchy_has_out(architecture)
        posterior = merge(posterior, (;
            obs_noise = result.posteriors[:obs_noise],
        ))
    end
    return (;
        result,
        posterior,
        setup,
        activation,
        predictive_activation = ResidualSineProjectorActivation(activation),
        mode = :nonlinear,
        structure = hierarchy_has_out(architecture) ? :out : :direct,
        training_seconds = measured.time,
        training_bytes = measured.bytes,
        dependencies,
        parameter_count = setup.parameter_count,
    )
end

run_hierarchy_fit(
    ::Val{:native_log_direct},
    training_data,
    base_priors,
    config;
    iterations,
) = run_native_projector_fit(training_data, base_priors, config; iterations)

run_hierarchy_fit(
    ::Val{:native_square_direct},
    training_data,
    base_priors,
    config;
    iterations,
) = run_native_positive_fit(
    Val(:squareplus),
    training_data,
    base_priors,
    config;
    iterations,
)

run_hierarchy_fit(
    ::Val{:native_square_out},
    training_data,
    base_priors,
    config;
    iterations,
) = run_native_squareplus_out_fit(
    training_data,
    base_priors,
    config;
    iterations,
)

run_hierarchy_fit(
    architecture::Union{
        Val{:ct_log_direct},
        Val{:ct_square_direct},
        Val{:ct_square_out},
    },
    training_data,
    base_priors,
    config;
    iterations,
) = run_hierarchy_ct_fit(
    architecture,
    training_data,
    base_priors,
    config;
    iterations,
)

function hierarchy_latent_moments(architecture, fit, features)
    return hierarchy_is_ct(architecture) ?
           projector_log_precision_moments(fit, features) :
           native_log_precision_moments(fit.posterior, features)
end

function hierarchy_base_prediction(
    ::Val{:log},
    latent_mean,
    latent_variance,
    predictions,
    beta,
    config,
)
    return ensemble_predictive_statistics(
        latent_mean,
        latent_variance,
        predictions,
        beta;
        kappa = config.kappa,
    )
end

function hierarchy_base_prediction(
    ::Val{:squareplus},
    latent_mean,
    latent_variance,
    predictions,
    beta,
    config,
)
    return positive_ensemble_predictive_statistics(
        Val(:squareplus),
        latent_mean,
        latent_variance,
        predictions,
        beta;
        kappa = config.kappa,
        quadrature_order = config.prediction_order,
    )
end

function hierarchy_observation_variance(fit)
    posterior_shape = shape(fit.posterior.obs_noise)
    posterior_shape > 1 || throw(DomainError(
        posterior_shape,
        "observation-precision posterior shape must exceed one",
    ))
    return rate(fit.posterior.obs_noise) / (posterior_shape - 1)
end

function hierarchy_prediction(architecture, fit, evaluation_data, config)
    latent_mean, latent_variance = hierarchy_latent_moments(
        architecture,
        fit,
        evaluation_data.features,
    )
    base = hierarchy_base_prediction(
        hierarchy_link(architecture),
        latent_mean,
        latent_variance,
        evaluation_data.predictions,
        fit.posterior.beta,
        config,
    )
    prediction = merge(base, (; latent_mean, latent_variance))
    if hierarchy_has_out(architecture)
        observation_variance = hierarchy_observation_variance(fit)
        prediction = merge(prediction, (;
            std = sqrt.(abs2.(base.std) .+ observation_variance),
            consensus_variance = abs2.(base.std),
            observation_variance,
        ))
    end
    return prediction
end

function hierarchy_routing_diagnostics(prediction, targets, predictions)
    oracle_expert = [
        argmin(abs2.(view(predictions, :, j) .- targets[j]))
        for j in eachindex(targets)
    ]
    selected_by_weight = [
        findmax(@view prediction.weights[:, j])[2]
        for j in eachindex(oracle_expert)
    ]
    selected_by_latent_mean = [
        findmax(@view prediction.latent_mean[:, j])[2]
        for j in eachindex(oracle_expert)
    ]
    oracle_weight = [
        prediction.weights[oracle_expert[j], j]
        for j in eachindex(oracle_expert)
    ]
    temporal_std = mean([
        std(@view prediction.weights[i, :])
        for i in axes(prediction.weights, 1)
    ])
    return (;
        routing_accuracy = mean(selected_by_weight .== oracle_expert),
        mean_routing_accuracy =
            mean(selected_by_latent_mean .== oracle_expert),
        oracle_weight_mean = mean(oracle_weight),
        weight_temporal_std_mean = temporal_std,
    )
end

function hierarchy_latent_diagnostics(prediction)
    latent_mean = prediction.latent_mean
    n_forecasters, n_observations = size(latent_mean)
    between_expert_variance = n_forecasters > 1 ? mean([
        var(@view latent_mean[:, j]) for j in 1:n_observations
    ]) : 0.0
    temporal_std = n_observations > 1 ? mean([
        std(@view latent_mean[i, :]) for i in 1:n_forecasters
    ]) : 0.0
    inverse_precision_mean = if hasproperty(prediction, :inverse_precision)
        mean(prediction.inverse_precision)
    else
        mean(exp.(clamp.(
            .-prediction.latent_mean .+ prediction.latent_variance ./ 2,
            -50.0,
            50.0,
        )))
    end
    return (;
        latent_variance_mean = mean(prediction.latent_variance),
        latent_between_expert_variance = between_expert_variance,
        latent_temporal_std_mean = temporal_std,
        inverse_precision_mean,
    )
end

function hierarchy_ct_diagnostics(architecture, fit, features)
    if !hierarchy_is_ct(architecture)
        return (;
            ct_hidden_variance_fraction = missing,
            ct_head_variance_fraction = missing,
            ct_interaction_variance_fraction = missing,
            ct_noise_variance_fraction = missing,
            ct_head_mean_norm = missing,
            ct_map_precision_mean = missing,
            ct_second_precision_mean = missing,
        )
    end
    diagnostics = summarize_projector_uncertainty(fit, features)
    return (;
        ct_hidden_variance_fraction = diagnostics.component_fractions.hidden,
        ct_head_variance_fraction = diagnostics.component_fractions.head,
        ct_interaction_variance_fraction =
            diagnostics.component_fractions.interaction,
        ct_noise_variance_fraction = diagnostics.component_fractions.noise,
        ct_head_mean_norm = diagnostics.head_mean_norm_mean,
        ct_map_precision_mean = diagnostics.map_precision_mean,
        ct_second_precision_mean = diagnostics.second_precision_mean,
    )
end

function hierarchy_observation_diagnostics(architecture, fit)
    if !hierarchy_has_out(architecture)
        return (;
            observation_precision_mean = missing,
            observation_precision_variance = missing,
            observation_variance = missing,
        )
    end
    return (;
        observation_precision_mean = mean(fit.posterior.obs_noise),
        observation_precision_variance = var(fit.posterior.obs_noise),
        observation_variance = hierarchy_observation_variance(fit),
    )
end

function evaluate_hierarchy_fit(
    architecture,
    fit,
    evaluation_data,
    config;
    seed = config.projector_seed,
)
    measured = @timed hierarchy_prediction(
        architecture,
        fit,
        evaluation_data,
        config,
    )
    prediction = measured.value
    metrics = dynamic_predictive_metrics(
        prediction,
        evaluation_data.y,
        evaluation_data.predictions,
    )
    routing = hierarchy_routing_diagnostics(
        prediction,
        evaluation_data.y,
        evaluation_data.predictions,
    )
    latent = hierarchy_latent_diagnostics(prediction)
    ct = hierarchy_ct_diagnostics(
        architecture,
        fit,
        evaluation_data.features,
    )
    observation = hierarchy_observation_diagnostics(architecture, fit)
    finite = all(isfinite, prediction.mean) &&
             all(isfinite, prediction.std) &&
             all(isfinite, prediction.weights) &&
             all(isfinite, metrics.weight_error_correlations) &&
             all(isfinite, (
                 metrics.mae,
                 metrics.rmse,
                 metrics.mean_log_likelihood,
                 metrics.pinball,
                 metrics.effective_experts_mean,
                 routing.routing_accuracy,
                 routing.oracle_weight_mean,
                 routing.weight_temporal_std_mean,
                 latent.latent_variance_mean,
                 latent.latent_between_expert_variance,
                 latent.latent_temporal_std_mean,
                 latent.inverse_precision_mean,
             ))
    return (;
        architecture = string(hierarchy_architecture_name(architecture)),
        seed,
        projection = hierarchy_link(architecture) isa Val{:log} ?
                     "closed_form" : string(config.squareplus_projection),
        has_out = hierarchy_has_out(architecture),
        is_ct = hierarchy_is_ct(architecture),
        finite,
        fit.training_seconds,
        fit.training_bytes,
        prediction_seconds = measured.time,
        prediction_bytes = measured.bytes,
        fit.parameter_count,
        metrics...,
        routing...,
        latent...,
        ct...,
        observation...,
        prediction,
    )
end

function hierarchy_metrics_row(stage, split, evaluation)
    return (;
        stage = string(stage),
        split = string(split),
        architecture = evaluation.architecture,
        seed = evaluation.seed,
        projection = evaluation.projection,
        has_out = evaluation.has_out,
        is_ct = evaluation.is_ct,
        finite = evaluation.finite,
        parameter_count = evaluation.parameter_count,
        training_seconds = evaluation.training_seconds,
        training_gib = evaluation.training_bytes / 2.0^30,
        prediction_seconds = evaluation.prediction_seconds,
        mae = evaluation.mae,
        rmse = evaluation.rmse,
        mean_log_likelihood = evaluation.mean_log_likelihood,
        log_likelihood_std = evaluation.log_likelihood_std,
        coverage95 = evaluation.coverage95,
        pinball = evaluation.pinball,
        effective_experts_mean = evaluation.effective_experts_mean,
        effective_experts_min = evaluation.effective_experts_min,
        effective_experts_max = evaluation.effective_experts_max,
        weight_error_correlation_mean =
            evaluation.weight_error_correlation_mean,
        routing_accuracy = evaluation.routing_accuracy,
        oracle_weight_mean = evaluation.oracle_weight_mean,
        weight_temporal_std_mean = evaluation.weight_temporal_std_mean,
        latent_variance_mean = evaluation.latent_variance_mean,
        latent_between_expert_variance =
            evaluation.latent_between_expert_variance,
        latent_temporal_std_mean = evaluation.latent_temporal_std_mean,
        inverse_precision_mean = evaluation.inverse_precision_mean,
        ct_hidden_variance_fraction = evaluation.ct_hidden_variance_fraction,
        ct_head_variance_fraction = evaluation.ct_head_variance_fraction,
        ct_interaction_variance_fraction =
            evaluation.ct_interaction_variance_fraction,
        ct_noise_variance_fraction = evaluation.ct_noise_variance_fraction,
        ct_head_mean_norm = evaluation.ct_head_mean_norm,
        ct_map_precision_mean = evaluation.ct_map_precision_mean,
        ct_second_precision_mean = evaluation.ct_second_precision_mean,
        observation_precision_mean = evaluation.observation_precision_mean,
        observation_precision_variance =
            evaluation.observation_precision_variance,
        observation_variance = evaluation.observation_variance,
    )
end

function print_hierarchy_table(rows)
    println()
    println(
        rpad("architecture", 25),
        lpad("seed", 7),
        lpad("RMSE", 11),
        lpad("mean LL", 12),
        lpad("eff. K", 10),
        lpad("weight sd", 12),
        lpad("route", 9),
        lpad("train s", 11),
    )
    for row in rows
        println(
            rpad(row.architecture, 25),
            lpad(string(row.seed), 7),
            lpad(string(round(row.rmse; digits = 6)), 11),
            lpad(string(round(row.mean_log_likelihood; digits = 6)), 12),
            lpad(string(round(row.effective_experts_mean; digits = 4)), 10),
            lpad(string(round(row.weight_temporal_std_mean; digits = 6)), 12),
            lpad(string(round(row.routing_accuracy; digits = 4)), 9),
            lpad(string(round(row.training_seconds; digits = 3)), 11),
        )
    end
end

function save_hierarchy_rows(rows, config, suffix)
    config.save_outputs || return nothing
    filename = config.output_prefix * "_$(suffix).csv"
    mkpath(dirname(filename))
    CSV.write(filename, DataFrame(rows))
    return filename
end

function hierarchy_run_arm(
    architecture,
    training_data,
    evaluation_data,
    base_priors,
    config;
    iterations,
    seed = config.projector_seed,
)
    run_config = merge(config, (; projector_seed = seed))
    fit = run_hierarchy_fit(
        architecture,
        training_data,
        base_priors,
        run_config;
        iterations,
    )
    evaluation = evaluate_hierarchy_fit(
        architecture,
        fit,
        evaluation_data,
        run_config;
        seed,
    )
    return (; fit, evaluation, config = run_config)
end

function hierarchy_warm_paths(
    architectures,
    training_data,
    base_priors,
    config,
)
    config.benchmark_warmup || return nothing
    n = min(4, length(training_data.y))
    tiny = (;
        y = training_data.y[1:n],
        features = training_data.features[1:n],
        predictions = training_data.predictions[:, 1:n],
    )
    warm_config = merge(config, (; free_energy = false, show_progress = false))
    println("Warming hierarchy paths ...")
    for architecture in architectures
        hierarchy_run_arm(
            architecture,
            tiny,
            tiny,
            base_priors,
            warm_config;
            iterations = 1,
        )
    end
    GC.gc()
    return nothing
end

function hierarchy_split(data, config; smoke = false)
    n_train = smoke ?
        min(config.smoke_observations, length(data.y_validation) - 1) :
        min(config.screen_train, length(data.y_validation) - 1)
    n_validation = smoke ? n_train : config.screen_validation
    validation_stop = min(n_train + n_validation, length(data.y_validation))
    return (;
        training = dynamic_projector_slice(data, 1:n_train),
        validation = dynamic_projector_slice(
            data,
            (n_train + 1):validation_stop,
        ),
    )
end

function run_hierarchy_smoke(data, config)
    split = hierarchy_split(data, config; smoke = true)
    architectures = (
        Val(:native_log_direct),
        Val(:native_square_direct),
        Val(:native_square_out),
        Val(:ct_log_direct),
        Val(:ct_square_direct),
        Val(:ct_square_out),
    )
    rows = NamedTuple[]
    runs = Dict{Symbol, Any}()
    smoke_config = merge(config, (;
        squareplus_projection = :unscented,
        benchmark_warmup = false,
    ))
    for architecture in architectures
        run = hierarchy_run_arm(
            architecture,
            split.training,
            split.validation,
            data.base_priors,
            smoke_config;
            iterations = config.smoke_iterations,
        )
        name = hierarchy_architecture_name(architecture)
        runs[name] = run
        push!(rows, hierarchy_metrics_row(
            :smoke,
            :validation,
            run.evaluation,
        ))
        GC.gc()
    end
    print_hierarchy_table(rows)
    path = save_hierarchy_rows(rows, config, "smoke_metrics")
    return (; rows, runs, path, split)
end

function run_hierarchy_projection_audit(data, config)
    split = hierarchy_split(data, config)
    rows = NamedTuple[]
    runs = Dict{Symbol, Any}()
    for projection in (:unscented, :quadrature64)
        run_config = merge(config, (; squareplus_projection = projection))
        hierarchy_warm_paths(
            (Val(:native_square_direct),),
            split.training,
            data.base_priors,
            run_config,
        )
        run = hierarchy_run_arm(
            Val(:native_square_direct),
            split.training,
            split.validation,
            data.base_priors,
            run_config;
            iterations = config.screen_iterations,
        )
        runs[projection] = run
        push!(rows, hierarchy_metrics_row(
            :projection_audit,
            :validation,
            run.evaluation,
        ))
        GC.gc()
    end
    unscented = runs[:unscented].evaluation
    quadrature = runs[:quadrature64].evaluation
    agrees = abs(unscented.rmse - quadrature.rmse) < 0.005 &&
             abs(
                 unscented.effective_experts_mean -
                 quadrature.effective_experts_mean,
             ) < 0.05
    selected_projection = agrees ? :unscented : :quadrature64
    println(
        "Projection audit selected $selected_projection: ",
        "ΔRMSE=$(abs(unscented.rmse - quadrature.rmse)), ",
        "Δeffective-K=$(abs(unscented.effective_experts_mean - quadrature.effective_experts_mean))",
    )
    print_hierarchy_table(rows)
    path = save_hierarchy_rows(rows, config, "projection_audit_metrics")
    return (; rows, runs, path, split, agrees, selected_projection)
end

hierarchy_rmse_success(candidate, baseline) =
    baseline.rmse - candidate.rmse >= 0.005

hierarchy_probabilistic_success(candidate, baseline) =
    candidate.mean_log_likelihood - baseline.mean_log_likelihood >= 0.02 &&
    candidate.rmse - baseline.rmse <= 0.01

hierarchy_routing_success(candidate, baseline) =
    baseline.effective_experts_mean - candidate.effective_experts_mean >= 0.1

hierarchy_general_success(candidate, baseline) =
    hierarchy_rmse_success(candidate, baseline) ||
    hierarchy_probabilistic_success(candidate, baseline) ||
    hierarchy_routing_success(candidate, baseline)

function run_hierarchy_screen(data, config, selected_projection)
    split = hierarchy_split(data, config)
    run_config = merge(config, (; squareplus_projection = selected_projection))
    base_architectures = (
        Val(:native_square_direct),
        Val(:native_square_out),
        Val(:ct_log_direct),
        Val(:ct_square_direct),
    )
    hierarchy_warm_paths(
        base_architectures,
        split.training,
        data.base_priors,
        run_config,
    )
    runs = Dict{Symbol, Any}()
    rows = NamedTuple[]
    for architecture in base_architectures
        run = hierarchy_run_arm(
            architecture,
            split.training,
            split.validation,
            data.base_priors,
            run_config;
            iterations = config.screen_iterations,
        )
        name = hierarchy_architecture_name(architecture)
        runs[name] = run
        push!(rows, hierarchy_metrics_row(
            :screen,
            :validation,
            run.evaluation,
        ))
        GC.gc()
    end

    native_out_useful = hierarchy_rmse_success(
        runs[:native_square_out].evaluation,
        runs[:native_square_direct].evaluation,
    ) || hierarchy_probabilistic_success(
        runs[:native_square_out].evaluation,
        runs[:native_square_direct].evaluation,
    )
    ct_routing_useful = hierarchy_routing_success(
        runs[:ct_square_direct].evaluation,
        runs[:ct_log_direct].evaluation,
    )
    run_combined = native_out_useful || ct_routing_useful
    if run_combined
        architecture = Val(:ct_square_out)
        run = hierarchy_run_arm(
            architecture,
            split.training,
            split.validation,
            data.base_priors,
            run_config;
            iterations = config.screen_iterations,
        )
        runs[:ct_square_out] = run
        push!(rows, hierarchy_metrics_row(
            :screen,
            :validation,
            run.evaluation,
        ))
    else
        println(
            "Skipping CT+out: neither native out nor CT routing passed its " *
            "predeclared validation criterion",
        )
    end

    ct_direct_useful = hierarchy_general_success(
        runs[:ct_square_direct].evaluation,
        runs[:ct_log_direct].evaluation,
    )
    ct_out_useful = haskey(runs, :ct_square_out) && hierarchy_general_success(
        runs[:ct_square_out].evaluation,
        runs[:ct_square_direct].evaluation,
    )
    for seed in Iterators.drop(config.seeds, 1)
        for architecture in (
            ct_direct_useful ? (Val(:ct_square_direct),) : (),
            ct_out_useful ? (Val(:ct_square_out),) : (),
        )
            for actual_architecture in architecture
                run = hierarchy_run_arm(
                    actual_architecture,
                    split.training,
                    split.validation,
                    data.base_priors,
                    run_config;
                    iterations = config.screen_iterations,
                    seed,
                )
                name = hierarchy_architecture_name(actual_architecture)
                runs[Symbol(name, :_seed_, seed)] = run
                push!(rows, hierarchy_metrics_row(
                    :robustness,
                    :validation,
                    run.evaluation,
                ))
                GC.gc()
            end
        end
    end

    qualified = Symbol[]
    native_out_useful && push!(qualified, :native_square_out)
    ct_direct_useful && push!(qualified, :ct_square_direct)
    ct_out_useful && push!(qualified, :ct_square_out)
    selected_arm = isempty(qualified) ? nothing : first(sort(
        qualified;
        by = name -> (
            runs[name].evaluation.rmse,
            -runs[name].evaluation.mean_log_likelihood,
            string(name),
        ),
    ))
    println(
        "Validation decisions: native-out=$native_out_useful, " *
        "CT-routing=$ct_routing_useful, CT-direct=$ct_direct_useful, " *
        "CT+out=$ct_out_useful, selected=$(something(selected_arm, :none))",
    )
    print_hierarchy_table(rows)
    path = save_hierarchy_rows(rows, config, "screen_metrics")
    return (;
        rows,
        runs,
        path,
        split,
        selected_projection,
        native_out_useful,
        ct_routing_useful,
        ct_direct_useful,
        ct_out_useful,
        selected_arm,
    )
end

function run_hierarchy_full(data, config, screen)
    isnothing(screen.selected_arm) && return (;
        skipped = true,
        reason = "no structural arm passed validation criteria",
    )
    training_data = dynamic_projector_slice(
        data,
        eachindex(data.y_validation),
    )
    test_data = dynamic_projector_slice(
        data,
        eachindex(data.y_test);
        test = true,
    )
    run_config = merge(config, (;
        squareplus_projection = screen.selected_projection,
    ))
    selected = Val(screen.selected_arm)
    architectures = (
        Val(:native_log_direct),
        Val(:native_square_direct),
        selected,
    )
    hierarchy_warm_paths(
        unique(architectures),
        training_data,
        data.base_priors,
        run_config,
    )
    rows = NamedTuple[]
    runs = Dict{Symbol, Any}()
    for architecture in unique(architectures)
        run = hierarchy_run_arm(
            architecture,
            training_data,
            test_data,
            data.base_priors,
            run_config;
            iterations = config.full_iterations,
        )
        name = hierarchy_architecture_name(architecture)
        runs[name] = run
        push!(rows, hierarchy_metrics_row(:full, :test, run.evaluation))
        GC.gc()
    end
    print_hierarchy_table(rows)
    path = save_hierarchy_rows(rows, config, "test_metrics")
    if config.save_outputs
        payload = Dict{Symbol, Any}(:target => test_data.y)
        for (name, run) in runs
            payload[Symbol(name, :_mean)] = run.evaluation.prediction.mean
            payload[Symbol(name, :_std)] = run.evaluation.prediction.std
            payload[Symbol(name, :_weights)] = run.evaluation.prediction.weights
        end
        jldsave(config.output_prefix * "_test_predictions.jld2"; payload...)
    end
    return (; skipped = false, rows, runs, path)
end

function run_squareplus_hierarchy_study(
    config = squareplus_hierarchy_config(),
)
    config.dataset == "ETTh2" || throw(ArgumentError(
        "Squareplus hierarchy ablation is fixed to ETTh2",
    ))
    data = load_dynamic_projector_data(config)
    println(
        "Squareplus hierarchy ablation: dataset=$(config.dataset), " *
        "horizon=$(config.horizon), stage=$(config.stage), " *
        "projection=$(config.projection_mode), kappa=$(config.kappa)",
    )
    config.stage === :smoke && return run_hierarchy_smoke(data, config)
    if config.stage === :audit
        return run_hierarchy_projection_audit(data, config)
    end
    if config.stage === :screen
        selected_projection = if config.projection_mode === :auto
            run_hierarchy_projection_audit(data, config).selected_projection
        else
            config.projection_mode
        end
        return run_hierarchy_screen(data, config, selected_projection)
    end
    if config.stage === :full
        isnothing(config.selected_arm) && throw(ArgumentError(
            "SQUAREPLUS_HIERARCHY_SELECTED_ARM is required for the full stage",
        ))
        config.projection_mode === :auto && throw(ArgumentError(
            "SQUAREPLUS_HIERARCHY_PROJECTION must be explicit for the full stage",
        ))
        screen = (;
            selected_arm = config.selected_arm,
            selected_projection = config.projection_mode,
        )
        return run_hierarchy_full(data, config, screen)
    end
    smoke = run_hierarchy_smoke(data, config)
    audit = config.projection_mode === :auto ?
        run_hierarchy_projection_audit(data, config) :
        (; selected_projection = config.projection_mode)
    screen = run_hierarchy_screen(
        data,
        config,
        audit.selected_projection,
    )
    full = run_hierarchy_full(data, config, screen)
    return (; smoke, audit, screen, full)
end

if abspath(PROGRAM_FILE) == @__FILE__
    squareplus_hierarchy_study = run_squareplus_hierarchy_study()
end
