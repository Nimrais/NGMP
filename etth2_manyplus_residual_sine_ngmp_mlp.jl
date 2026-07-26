include(joinpath(@__DIR__, "etth2_softplus_ut_ngmp_mlp.jl"))

const ETTH2_MANYPLUS_OPTIMIZERS = (
    damped=(optimizer=:damped, alpha=0.5, beta=0.0),
    vector_transport_05=(optimizer=:vector_transport, alpha=0.2, beta=0.5),
    vector_transport_08=(optimizer=:vector_transport, alpha=0.2, beta=0.8),
    projected_nesterov_09=(optimizer=:projected_nesterov, alpha=0.2, beta=0.9),
)

const ETTH2_MANYPLUS_CONFIG = merge(
    ETTH2_MLP_CONFIG,
    (
        arms=keys(ETTH2_MANYPLUS_OPTIMIZERS),
        optimizers=ETTH2_MANYPLUS_OPTIMIZERS,
        # ManyPlus creates a broad reactive sum graph. Yield message updates
        # more frequently than the Softplus model to avoid native stack
        # overflow on the 32-unit, 250-observation minibatches.
        limit_stack_depth=100,
        # The primary research comparison uses the repository's exact RxInfer
        # rules. `:direct_approximate_messages` remains an explicitly labelled
        # approximation and must not be treated as the same inference method.
        inference_backend=:direct_approximate_messages,
        # More than one iteration with α=0.5 can accumulate locally improper
        # ResidualSine backward sites until the structured weight cavity is no
        # longer positive definite. One iteration matches the original safe
        # posterior-as-prior minibatch schedule.
        message_passing_iterations=1,
        # Local nonlinear sites are rebuilt for every minibatch. Keep their
        # update identical across comparison arms; optimizer history is carried
        # only by the persistent global Gaussian parameters below.
        local_ngmp_alpha=0.05,
        local_ngmp_max_step=1.0,
        global_transport_max_backtracks=40,
        direct_inner_iterations=5,
        direct_weight_covariance=:diagonal, # :diagonal (fast) or :full
        direct_site_tolerance=1e-5,
        direct_max_site_precision=1e4,
        direct_max_abs_weight=1.0,
        direct_max_abs_coefficient=0.25,
        direct_max_abs_intercept=5.0,
        direct_residual_clip=20.0,
        direct_min_precision=1e-8,
        direct_max_precision=1e8,
        direct_max_backtracks=12,
        direct_loss_tolerance=1e-10,
    ),
)

function make_etth2_manyplus_direct_priors(input_count, hidden_count, settings)
    rng = MersenneTwister(settings.seed)
    weight_mean = randn(rng, hidden_count, input_count)
    for unit in 1:hidden_count
        direction_norm = norm(@view weight_mean[unit, :])
        weight_mean[unit, :] .*= sqrt(2 / input_count) /
                                max(direction_norm, eps())
    end
    return (
        weight_mean,
        weight_precision=cat(
            (
                Matrix(Diagonal(fill(
                    inv(settings.residual_sine_weight_variance),
                    input_count,
                ))) for _ in 1:hidden_count
            )...;
            dims=3,
        ),
        coefficient_mean=[
            (isodd(unit) ? 1.0 : -1.0) / hidden_count
            for unit in 1:hidden_count
        ],
        coefficient_precision=fill(
            inv(settings.residual_sine_coefficient_variance),
            hidden_count,
        ),
        intercept_mean=0.0,
        intercept_precision=1.0,
        hidden_precision=1e3,
        contribution_precision=1e4,
        observation_shape=2.0,
        observation_rate=2.0,
    )
end

function direct_manyplus_zero_sites(hidden_count, input_count)
    return (
        weight_xi=zeros(hidden_count, input_count),
        weight_Λ=zeros(input_count, input_count, hidden_count),
        coefficient_xi=zeros(hidden_count),
        coefficient_Λ=zeros(hidden_count),
        intercept_xi=0.0,
        intercept_Λ=0.0,
    )
end

function direct_preactivation_variance(input_matrix, covariance, mode)
    if mode === :diagonal
        covariance_diagonal = hcat([
            diag(@view covariance[:, :, unit])
            for unit in axes(covariance, 3)
        ]...)'
        return abs2.(input_matrix) * transpose(covariance_diagonal)
    elseif mode === :full
        return hcat([
            [
                dot(
                    @view(input_matrix[observation, :]),
                    @view(covariance[:, :, unit]) *
                    @view(input_matrix[observation, :]),
                )
                for observation in axes(input_matrix, 1)
            ] for unit in axes(covariance, 3)
        ]...)
    end
    throw(ArgumentError("direct_weight_covariance must be :diagonal or :full"))
end

function direct_manyplus_posterior(priors, site, settings)
    sanitize_precision(value) = isfinite(value) ?
        clamp(
            value,
            settings.direct_min_precision,
            settings.direct_max_precision,
        ) :
        settings.direct_max_precision
    coefficient_precision =
        sanitize_precision.(priors.coefficient_precision .+ site.coefficient_Λ)
    intercept_precision =
        sanitize_precision(priors.intercept_precision + site.intercept_Λ)
    hidden_count, input_count = size(priors.weight_mean)
    weight_mean = similar(priors.weight_mean)
    weight_covariance = Array{Float64}(undef, input_count, input_count, hidden_count)
    for unit in 1:hidden_count
        precision = Matrix(Hermitian(
            @view(priors.weight_precision[:, :, unit]) .+
            @view(site.weight_Λ[:, :, unit]),
        ))
        precision[diagind(precision)] .= sanitize_precision.(
            precision[diagind(precision)],
        )
        natural =
            @view(priors.weight_precision[:, :, unit]) *
            @view(priors.weight_mean[unit, :]) +
            @view(site.weight_xi[unit, :])
        if settings.direct_weight_covariance === :diagonal
            precision_diagonal = clamp.(
                diag(precision),
                settings.direct_min_precision,
                settings.direct_max_precision,
            )
            weight_mean[unit, :] = natural ./ precision_diagonal
            weight_covariance[:, :, unit] =
                Diagonal(inv.(precision_diagonal))
        else
            factor = cholesky(Hermitian(precision); check=false)
            if !issuccess(factor)
                decomposition = eigen(Hermitian(precision))
                precision = decomposition.vectors * Diagonal(
                    clamp.(
                        decomposition.values,
                        settings.direct_min_precision,
                        settings.direct_max_precision,
                    ),
                ) * decomposition.vectors'
                factor = cholesky(Hermitian(precision))
            end
            weight_mean[unit, :] = factor \ natural
            weight_covariance[:, :, unit] = inv(factor)
        end
    end
    coefficient_natural =
        priors.coefficient_precision .* priors.coefficient_mean .+
        site.coefficient_xi
    intercept_natural =
        priors.intercept_precision * priors.intercept_mean + site.intercept_xi
    sanitize_mean(value, bound) = isfinite(value) ?
        clamp(value, -bound, bound) :
        0.0
    return (
        weight_mean=sanitize_mean.(
            weight_mean,
            settings.direct_max_abs_weight,
        ),
        weight_covariance,
        coefficient_mean=sanitize_mean.(
            coefficient_natural ./ coefficient_precision,
            settings.direct_max_abs_coefficient,
        ),
        coefficient_variance=inv.(coefficient_precision),
        intercept_mean=sanitize_mean(
            intercept_natural / intercept_precision,
            settings.direct_max_abs_intercept,
        ),
        intercept_variance=inv(intercept_precision),
    )
end

function direct_manyplus_forward(marginals, inputs, settings)
    input_matrix = reduce(vcat, permutedims.(inputs))
    preactivation = input_matrix * transpose(marginals.weight_mean)
    preactivation_variance = direct_preactivation_variance(
        input_matrix,
        marginals.weight_covariance,
        settings.direct_weight_covariance,
    )
    activation = preactivation .+
        (settings.residual_sine_rho / settings.residual_sine_omega) .*
        exp.(
            -0.5 * settings.residual_sine_omega^2 .*
            preactivation_variance,
        ) .*
        sin.(settings.residual_sine_omega .* preactivation)
    activation_derivative = 1 .+
        settings.residual_sine_rho .*
        cos.(settings.residual_sine_omega .* preactivation)
    predicted_mean = marginals.intercept_mean .+
                     activation * marginals.coefficient_mean
    return (;
        input_matrix,
        preactivation,
        activation,
        activation_derivative,
        predicted_mean,
    )
end

function direct_manyplus_batch_loss(priors, site, inputs, targets, settings)
    marginals = direct_manyplus_posterior(priors, site, settings)
    prediction = direct_manyplus_forward(
        marginals,
        inputs,
        settings,
    ).predicted_mean
    all(isfinite, prediction) || return Inf
    return mean(abs2, targets .- prediction)
end

function direct_manyplus_backtrack(
    priors,
    old_site,
    proposed_site,
    inputs,
    targets,
    settings,
)
    old_loss = direct_manyplus_batch_loss(
        priors,
        old_site,
        inputs,
        targets,
        settings,
    )
    step = direct_site_direction(old_site, proposed_site)
    for backtrack in 0:settings.direct_max_backtracks
        scale = exp2(-backtrack)
        candidate = direct_sanitize_site(
            direct_add_site(old_site, step, scale),
            settings,
        )
        candidate_loss = direct_manyplus_batch_loss(
            priors,
            candidate,
            inputs,
            targets,
            settings,
        )
        if isfinite(candidate_loss) &&
           candidate_loss <= old_loss + settings.direct_loss_tolerance
            return candidate, candidate_loss, backtrack
        end
    end
    return old_site, old_loss, settings.direct_max_backtracks + 1
end

function direct_manyplus_target_sites(
    priors,
    site,
    inputs,
    targets,
    settings,
)
    marginals = direct_manyplus_posterior(priors, site, settings)
    input_matrix = reduce(vcat, permutedims.(inputs))
    hidden_count, input_count = size(marginals.weight_mean)
    observation_count = length(targets)
    preactivation_mean = input_matrix * transpose(marginals.weight_mean)
    preactivation_variance = direct_preactivation_variance(
        input_matrix,
        marginals.weight_covariance,
        settings.direct_weight_covariance,
    )
    preactivation_variance .+= inv(priors.hidden_precision)
    hidden_mean = similar(preactivation_mean)
    hidden_variance = similar(preactivation_variance)
    rho = settings.residual_sine_rho
    omega = settings.residual_sine_omega
    activation_meta = ResidualSineMeta(rho=rho, omega=omega)
    for index in eachindex(preactivation_mean)
        hidden_mean[index], hidden_variance[index] =
            SurrogateModelling._residual_sine_mean_var_1d(
                preactivation_mean[index],
                preactivation_variance[index],
                activation_meta,
            )
    end
    contribution_mean =
        hidden_mean .* transpose(marginals.coefficient_mean)
    contribution_variance =
        abs2.(hidden_mean) .* transpose(marginals.coefficient_variance) .+
        hidden_variance .* transpose(
            abs2.(marginals.coefficient_mean) .+
            marginals.coefficient_variance,
        )
    contribution_variance .+= inv(priors.contribution_precision)
    output_mean = vec(sum(contribution_mean; dims=2))
    output_variance = vec(sum(contribution_variance; dims=2))
    observation_variance =
        priors.observation_rate / max(priors.observation_shape - 1, 1.0)

    target_site = direct_manyplus_zero_sites(hidden_count, input_count)
    max_precision = settings.direct_max_site_precision

    for unit in 1:hidden_count
        for observation in 1:observation_count
            other_mean =
                output_mean[observation] -
                contribution_mean[observation, unit]
            other_variance =
                output_variance[observation] -
                contribution_variance[observation, unit]
            contribution_message_mean =
                targets[observation] - marginals.intercept_mean - other_mean
            contribution_message_variance =
                observation_variance +
                marginals.intercept_variance +
                max(other_variance, 0.0)
            product_variance =
                contribution_message_variance +
                inv(priors.contribution_precision)

            second_hidden_moment =
                abs2(hidden_mean[observation, unit]) +
                hidden_variance[observation, unit]
            coefficient_precision =
                second_hidden_moment / product_variance
            target_site.coefficient_Λ[unit] += coefficient_precision
            target_site.coefficient_xi[unit] +=
                hidden_mean[observation, unit] *
                contribution_message_mean / product_variance

            second_coefficient_moment =
                abs2(marginals.coefficient_mean[unit]) +
                marginals.coefficient_variance[unit]
            hidden_message_precision =
                second_coefficient_moment / product_variance
            hidden_message_precision <= eps() && continue
            hidden_message_weighted_mean =
                marginals.coefficient_mean[unit] *
                contribution_message_mean / product_variance
            backward_projection =
                SurrogateModelling._project_residual_sine_backward_1d(
                    NormalMeanVariance(
                        preactivation_mean[observation, unit],
                        preactivation_variance[observation, unit],
                    ),
                    SurrogateModelling.ResidualSineGaussianBackwardMessage(
                        hidden_message_weighted_mean,
                        hidden_message_precision,
                        activation_meta,
                    ),
                )
            backward_natural =
                ExponentialFamily.getnaturalparameters(backward_projection)
            preactivation_message_precision =
                max(-2 * backward_natural[2], settings.direct_min_precision)
            preactivation_message_mean =
                backward_natural[1] / preactivation_message_precision
            softdot_variance =
                inv(preactivation_message_precision) +
                inv(priors.hidden_precision)
            if !isfinite(preactivation_message_mean) ||
               !isfinite(softdot_variance) ||
               softdot_variance <= 0
                continue
            end
            feature = @view input_matrix[observation, :]
            if settings.direct_weight_covariance === :diagonal
                diagonal_view =
                    @view target_site.weight_Λ[:, :, unit]
                diagonal_view[diagind(diagonal_view)] .+=
                    abs2.(feature) ./ softdot_variance
            else
                target_site.weight_Λ[:, :, unit] .+=
                    (feature * feature') ./ softdot_variance
            end
            target_site.weight_xi[unit, :] .+=
                feature .* (preactivation_message_mean / softdot_variance)
        end
    end
    intercept_denominator = observation_variance .+ output_variance
    intercept_precision = sum(inv, intercept_denominator)
    target_site = merge(
        target_site,
        (
            intercept_xi=sum(
                (targets .- output_mean) ./ intercept_denominator,
            ),
            intercept_Λ=min(intercept_precision, max_precision),
        ),
    )
    return direct_sanitize_site(target_site, settings)
end

function direct_site_map(operation, sites...)
    names = keys(first(sites))
    values = map(names) do name
        operation((getfield(site, name) for site in sites)...)
    end
    return NamedTuple{names}(values)
end

direct_site_direction(site, target) =
    direct_site_map((current, proposed) -> proposed .- current, site, target)
direct_zero_like_site(site) = direct_site_map(value -> zero.(value), site)
direct_add_site(site, direction, scale) =
    direct_site_map((value, step) -> value .+ scale .* step, site, direction)

function direct_clamp_site(site)
    names = keys(site)
    values = map(names) do name
        value = getfield(site, name)
        occursin("Λ", String(name)) ? max.(value, 0.0) : value
    end
    return NamedTuple{names}(values)
end

function direct_sanitize_site(site, settings)
    names = keys(site)
    values = map(names) do name
        value = getfield(site, name)
        if name === :weight_Λ
            sanitized = similar(value)
            for unit in axes(value, 3)
                matrix = map(
                    item -> isfinite(item) ? item : 0.0,
                    @view(value[:, :, unit]),
                )
                sanitized[:, :, unit] = (matrix .+ matrix') ./ 2
            end
            sanitized
        elseif occursin("Λ", String(name))
            (item -> isfinite(item) ?
                clamp(item, 0.0, settings.direct_max_site_precision) :
                0.0).(value)
        else
            mean_bound = startswith(String(name), "weight_") ?
                         settings.direct_max_abs_weight :
                         startswith(String(name), "coefficient_") ?
                         settings.direct_max_abs_coefficient :
                         settings.direct_max_abs_intercept
            max_natural =
                settings.direct_max_site_precision * mean_bound
            (item -> isfinite(item) ?
                clamp(item, -max_natural, max_natural) :
                0.0).(value)
        end
    end
    return NamedTuple{names}(values)
end

function direct_site_dot(left, right)
    return sum(
        sum(getfield(left, name) .* getfield(right, name))
        for name in keys(left)
    )
end

function direct_site_delta(left, right)
    return maximum(
        maximum(abs.(getfield(left, name) .- getfield(right, name)))
        for name in keys(left)
    )
end

function direct_site_metric(site, damping)
    names = keys(site)
    values = map(names) do name
        base = if name === :weight_xi
            precision = getfield(site, :weight_Λ)
            hcat([
                diag(@view precision[:, :, unit])
                for unit in axes(precision, 3)
            ]...)'
        else
            precision_name = endswith(String(name), "_xi") ?
                             Symbol(String(name)[1:(end - 3)] * "_Λ") :
                             name
            haskey(site, precision_name) ?
                getfield(site, precision_name) :
                getfield(site, name)
        end
        max.(abs.(base), damping)
    end
    return NamedTuple{names}(values)
end

function direct_manyplus_optimizer_step(
    site,
    direction,
    previous_direction,
    previous_update,
    previous_metric,
    has_previous,
    priors,
    inputs,
    targets,
    settings,
)
    if settings.optimizer === :projected_nesterov
        denominator = direct_site_dot(previous_direction, previous_direction)
        coefficient = has_previous && denominator > settings.nesterov_eps ?
                      direct_site_dot(previous_direction, direction) /
                      denominator :
                      0.0
        lookahead = direct_sanitize_site(
            direct_add_site(
                site,
                direction,
                settings.alpha * settings.beta * coefficient,
            ),
            settings,
        )
        lookahead_target = direct_manyplus_target_sites(
            priors,
            lookahead,
            inputs,
            targets,
            settings,
        )
        lookahead_direction = direct_site_direction(
            lookahead,
            lookahead_target,
        )
        next_site = direct_sanitize_site(
            direct_add_site(site, lookahead_direction, settings.alpha),
            settings,
        )
        return next_site, direction, previous_update,
               direct_site_metric(
                   next_site,
                   settings.vector_transport_damping,
               ), true
    elseif settings.optimizer === :vector_transport
        current_metric = direct_site_metric(
            site,
            settings.vector_transport_damping,
        )
        transported = direct_site_map(
            (update, old_metric, new_metric) ->
                update .* sqrt.(old_metric ./ new_metric),
            previous_update,
            previous_metric,
            current_metric,
        )
        update = direct_site_map(
            (momentum, gradient) ->
                settings.beta .* momentum .+ settings.alpha .* gradient,
            transported,
            direction,
        )
        next_site = direct_sanitize_site(
            direct_add_site(site, update, 1.0),
            settings,
        )
        return next_site, previous_direction, update,
               direct_site_metric(
                   next_site,
                   settings.vector_transport_damping,
               ), has_previous
    elseif settings.optimizer === :damped
        next_site = direct_sanitize_site(
            direct_add_site(site, direction, settings.alpha),
            settings,
        )
        return next_site, previous_direction, previous_update,
               previous_metric, has_previous
    end
    throw(ArgumentError("unsupported optimizer $(settings.optimizer)"))
end

function infer_etth2_manyplus_direct(priors, inputs, targets, settings)
    hidden_count, input_count = size(priors.weight_mean)
    site = direct_manyplus_zero_sites(hidden_count, input_count)
    previous_direction = direct_zero_like_site(site)
    previous_update = direct_zero_like_site(site)
    previous_metric = direct_site_metric(
        site,
        settings.vector_transport_damping,
    )
    has_previous = false
    delta = Inf
    for _ in 1:settings.direct_inner_iterations
        target_site = direct_manyplus_target_sites(
            priors,
            site,
            inputs,
            targets,
            settings,
        )
        direction = direct_site_direction(site, target_site)
        old_site = site
        proposed_site, proposed_previous_direction,
        proposed_previous_update, proposed_previous_metric,
        proposed_has_previous = direct_manyplus_optimizer_step(
            site,
            direction,
            previous_direction,
            previous_update,
            previous_metric,
            has_previous,
            priors,
            inputs,
            targets,
            settings,
        )
        site, _, _ = direct_manyplus_backtrack(
            priors,
            old_site,
            proposed_site,
            inputs,
            targets,
            settings,
        )
        accepted_update = direct_site_direction(old_site, site)
        previous_direction = proposed_previous_direction
        previous_update = settings.optimizer === :vector_transport ?
                          accepted_update :
                          proposed_previous_update
        previous_metric = direct_site_metric(
            site,
            settings.vector_transport_damping,
        )
        has_previous = proposed_has_previous
        delta = direct_site_delta(site, old_site)
        delta < settings.direct_site_tolerance && break
    end
    posterior = direct_manyplus_posterior(priors, site, settings)
    forward = direct_manyplus_forward(posterior, inputs, settings)
    residual = clamp.(
        targets .- forward.predicted_mean,
        -settings.direct_residual_clip,
        settings.direct_residual_clip,
    )
    hidden_count, input_count = size(posterior.weight_mean)
    weight_precision = Array{Float64}(
        undef,
        input_count,
        input_count,
        hidden_count,
    )
    for unit in 1:hidden_count
        covariance = @view posterior.weight_covariance[:, :, unit]
        weight_precision[:, :, unit] =
            settings.direct_weight_covariance === :diagonal ?
            Diagonal(inv.(diag(covariance))) :
            inv(Hermitian(covariance))
    end
    return (
        weight_mean=posterior.weight_mean,
        weight_precision,
        coefficient_mean=posterior.coefficient_mean,
        coefficient_precision=inv.(posterior.coefficient_variance),
        intercept_mean=posterior.intercept_mean,
        intercept_precision=inv(posterior.intercept_variance),
        hidden_precision=priors.hidden_precision,
        contribution_precision=priors.contribution_precision,
        observation_shape=priors.observation_shape + length(targets) / 2,
        observation_rate=priors.observation_rate + sum(abs2, residual) / 2,
    ), (; delta)
end

function train_etth2_manyplus_direct(priors, inputs, targets, settings)
    observation_count = length(targets)
    batch_size = isnothing(settings.training_batch_size) ?
                 observation_count :
                 min(settings.training_batch_size, observation_count)
    rng = MersenneTwister(settings.seed)
    learned = priors
    final_stats = nothing
    progress = Progress(
        settings.inference_iterations * cld(observation_count, batch_size);
        desc="direct manyplus $(settings.optimizer) ",
        enabled=settings.show_progress,
    )
    for _ in 1:settings.inference_iterations
        order = randperm(rng, observation_count)
        for first_index in 1:batch_size:observation_count
            indices = order[
                first_index:min(first_index + batch_size - 1, observation_count)
            ]
            learned, final_stats = infer_etth2_manyplus_direct(
                learned,
                inputs[indices],
                targets[indices],
                settings,
            )
            ProgressMeter.next!(progress)
        end
    end
    ProgressMeter.finish!(progress)
    return learned, final_stats
end

function predict_etth2_manyplus_direct(priors, inputs, settings)
    marginals = (
        weight_mean=priors.weight_mean,
        weight_covariance=cat((
            settings.direct_weight_covariance === :diagonal ?
            Diagonal(inv.(diag(@view priors.weight_precision[:, :, unit]))) :
            inv(Hermitian(@view priors.weight_precision[:, :, unit]))
            for unit in axes(priors.weight_precision, 3)
        )...; dims=3),
        coefficient_mean=priors.coefficient_mean,
        coefficient_variance=inv.(priors.coefficient_precision),
        intercept_mean=priors.intercept_mean,
        intercept_variance=inv(priors.intercept_precision),
    )
    input_matrix = reduce(vcat, permutedims.(inputs))
    preactivation_mean = input_matrix * transpose(marginals.weight_mean)
    preactivation_variance = direct_preactivation_variance(
        input_matrix,
        marginals.weight_covariance,
        settings.direct_weight_covariance,
    )
    preactivation_variance .+= inv(priors.hidden_precision)
    hidden_mean = similar(preactivation_mean)
    hidden_variance = similar(preactivation_variance)
    activation = ResidualSineMeta(
        rho=settings.residual_sine_rho,
        omega=settings.residual_sine_omega,
    )
    for index in eachindex(preactivation_mean)
        hidden_mean[index], hidden_variance[index] =
            SurrogateModelling._residual_sine_mean_var_1d(
                preactivation_mean[index],
                preactivation_variance[index],
                activation,
            )
    end
    predicted_mean =
        marginals.intercept_mean .+
        hidden_mean * marginals.coefficient_mean
    predicted_variance = fill(
        marginals.intercept_variance +
        priors.observation_rate / max(priors.observation_shape - 1, eps()),
        length(inputs),
    )
    for unit in axes(marginals.weight_mean, 1)
        predicted_variance .+=
            abs2.(@view hidden_mean[:, unit]) .*
            marginals.coefficient_variance[unit] .+
            (@view hidden_variance[:, unit]) .*
            (
                abs2(marginals.coefficient_mean[unit]) +
                marginals.coefficient_variance[unit]
            )
    end
    predicted_std = sqrt.(max.(predicted_variance, eps()))
    all(isfinite, predicted_mean) ||
        error("direct ManyPlus prediction produced a non-finite mean")
    all(value -> isfinite(value) && value > 0, predicted_std) ||
        error("direct ManyPlus prediction produced a non-finite standard deviation")
    return predicted_mean, predicted_std
end

function seeded_global_ngmp_state(prior, damping)
    state = NaturalGradientMP.NGMPEdgeState(nothing; damping)
    family, natural = NaturalGradientMP.natural_parameters(prior)
    state.message = deepcopy(prior)
    state.η = copy(natural)
    state.momentum = zero(natural)
    state.previous_direction = zero(natural)
    state.metric = NaturalGradientMP.diagonal_fisher_metric(
        family,
        state.η,
        damping.metric_damping,
    )
    # Mark the prior as the optimizer's base point without pretending that an
    # update has already occurred. A zero previous direction makes the first
    # Nesterov correction vanish, as it should.
    state.nfired = 1
    return state
end

function valid_global_gaussian(message::UnivariateNormalDistributionsFamily)
    message_precision = precision(message)
    return isfinite(weightedmean(message)) &&
           isfinite(message_precision) &&
           message_precision > 0
end

function valid_global_gaussian(message::MultivariateNormalDistributionsFamily)
    weighted_mean, message_precision = weightedmean_precision(message)
    return all(isfinite, weighted_mean) &&
           all(isfinite, message_precision) &&
           isposdef(Hermitian((message_precision .+ message_precision') ./ 2))
end

function apply_global_ngmp_with_backtracking!(
    state,
    target;
    max_backtracks,
)
    previous_natural = copy(state.η)
    previous_message = state.message
    proposed = NaturalGradientMP.apply_damping!(state, target)
    valid_global_gaussian(proposed) && return proposed

    proposed_natural = copy(state.η)
    for backtrack in 1:max_backtracks
        scale = exp2(-backtrack)
        candidate_natural = previous_natural .+
                            scale .* (proposed_natural .- previous_natural)
        candidate = NaturalGradientMP.from_natural(
            first(NaturalGradientMP.natural_parameters(target)),
            candidate_natural,
        )
        if valid_global_gaussian(candidate)
            state.η = candidate_natural
            state.momentum = candidate_natural .- previous_natural
            state.message = candidate
            state.metric = NaturalGradientMP.diagonal_fisher_metric(
                first(NaturalGradientMP.natural_parameters(candidate)),
                state.η,
                NaturalGradientMP.optimizer_metric_damping(state),
            )
            return candidate
        end
    end

    state.η = previous_natural
    state.momentum .= 0
    state.message = previous_message
    state.metric = NaturalGradientMP.diagonal_fisher_metric(
        first(NaturalGradientMP.natural_parameters(previous_message)),
        state.η,
        NaturalGradientMP.optimizer_metric_damping(state),
    )
    return previous_message
end

function make_manyplus_global_optimizer(priors, settings)
    damping = DampingMeta(
        alpha=settings.alpha,
        beta=settings.beta,
        max_step=settings.max_step,
        method=settings.optimizer,
        eps=settings.nesterov_eps,
        metric_damping=settings.vector_transport_damping,
    )
    return (
        weight=[
            seeded_global_ngmp_state(prior, damping)
            for prior in priors.weight
        ],
        coefficient=[
            seeded_global_ngmp_state(prior, damping)
            for prior in priors.coefficient
        ],
        intercept=seeded_global_ngmp_state(priors.intercept, damping),
    )
end

function transport_manyplus_global_priors!(
    optimizer,
    target,
    settings,
)
    weight = [
        apply_global_ngmp_with_backtracking!(
            optimizer.weight[unit],
            target.weight[unit];
            max_backtracks=settings.global_transport_max_backtracks,
        )
        for unit in eachindex(target.weight)
    ]
    coefficient = [
        apply_global_ngmp_with_backtracking!(
            optimizer.coefficient[unit],
            target.coefficient[unit];
            max_backtracks=settings.global_transport_max_backtracks,
        )
        for unit in eachindex(target.coefficient)
    ]
    intercept = apply_global_ngmp_with_backtracking!(
        optimizer.intercept,
        target.intercept;
        max_backtracks=settings.global_transport_max_backtracks,
    )
    return (
        weight,
        coefficient,
        intercept,
        # Precision hyperparameters remain conjugate RxInfer posteriors. They
        # are deliberately not subjected to Gaussian transport geometry.
        hidden_precision=target.hidden_precision,
        contribution_precision=target.contribution_precision,
        observation_precision=target.observation_precision,
    )
end

function train_etth2_manyplus_global_transport(
    priors,
    inputs,
    targets,
    settings,
)
    observation_count = length(targets)
    batch_size = isnothing(settings.training_batch_size) ?
                 observation_count :
                 min(settings.training_batch_size, observation_count)
    rng = MersenneTwister(settings.seed)
    learned = priors
    optimizer = make_manyplus_global_optimizer(priors, settings)
    final_result = nothing
    progress = Progress(
        settings.inference_iterations * cld(observation_count, batch_size);
        desc="rxinfer manyplus global $(settings.optimizer) ",
        enabled=settings.show_progress,
    )
    for _ in 1:settings.inference_iterations
        order = randperm(rng, observation_count)
        for first_index in 1:batch_size:observation_count
            indices = order[
                first_index:min(first_index + batch_size - 1, observation_count)
            ]
            target_priors, final_result = infer_etth2_manyplus(
                learned,
                inputs[indices],
                targets[indices],
                settings,
            )
            learned = transport_manyplus_global_priors!(
                optimizer,
                target_priors,
                settings,
            )
            ProgressMeter.next!(progress)
        end
    end
    ProgressMeter.finish!(progress)
    return learned, final_result
end

function write_etth2_manyplus_report(path, config, metrics)
    labels = Dict(
        :damped => "Damped (α=0.5)",
        :vector_transport_05 => "Vector transport (β=0.5)",
        :vector_transport_08 => "Vector transport (β=0.8)",
        :projected_nesterov_09 => "Projected Nesterov (β=0.9)",
    )
    open(path, "w") do io
        println(io, "# ManyPlus residual-sine NGMP MLP on ETTh2")
        println(io)
        println(io, "Horizon: **$(config.horizon)**  ")
        println(io, "Hidden units: **$(config.hidden_count)**  ")
        println(io, "Input features: **$(config.input_count)**  ")
        println(io, "Training observations: **$(config.training_observations)**  ")
        batch_label = isnothing(config.training_batch_size) ?
                      "full graph" :
                      config.training_batch_size
        println(io, "Training batch size: **$batch_label**  ")
        println(io, "Complete data passes: **$(config.inference_iterations)**  ")
        println(io, "Inference backend: **$(config.inference_backend)**  ")
        if config.inference_backend === :rxinfer
            println(io, "RxInfer message-passing iterations per minibatch: **$(config.message_passing_iterations)**  ")
            println(io, "Persistent optimizer state: **global Gaussian posterior transport**  ")
            println(io, "Local ResidualSine damping α: **$(config.local_ngmp_alpha)**  ")
        end
        if config.inference_backend === :direct_approximate_messages
            println(io, "Direct inner site updates: **$(config.direct_inner_iterations)**  ")
            println(io, "Direct weight covariance: **$(config.direct_weight_covariance)**  ")
            println(io)
            println(io, "> **Research warning:** this graph-free backend is an approximation, not an algebraically equivalent execution of the RxInfer model. It uses the repository's exact closed-form ResidualSine forward moments and backward Fisher projection, but retains $(config.direct_weight_covariance) weight covariance, simplified cavity scheduling, moment-matched product messages, and additional trust-region stabilization.")
        end
        println(io, "Prediction batch/iterations: **$(config.prediction_batch_size)/$(config.prediction_iterations)**  ")
        println(io, "Inputs include expert forecasts: **$(config.include_expert_predictions)**")
        println(io)
        println(io, "| Optimizer | MAE | MSE ± 95% CI | NLL ± 95% CI | Coverage 95% | Pinball |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        for arm in config.arms
            value = metrics[arm]
            println(
                io,
                "| $(labels[arm]) | $(@sprintf("%.4f", value.mae)) | " *
                "$(@sprintf("%.4f", value.mse)) ± $(@sprintf("%.4f", value.mse_ci95)) | " *
                "$(@sprintf("%.4f", value.negative_log_likelihood)) ± $(@sprintf("%.4f", value.negative_log_likelihood_ci95)) | " *
                "$(@sprintf("%.4f", value.coverage95)) | $(@sprintf("%.4f", value.pinball)) |",
            )
        end
    end
end

function run_etth2_manyplus_optimizer_comparison(;
    session,
    prepare_only=false,
    rebuild_cache=false,
)
    session_path = isabspath(session) ?
                   normpath(session) :
                   normpath(joinpath(ROOT, session))
    _, raw, cache, cache_path = prepare_data(session_path; rebuild_cache)
    println("prepared_cache=$cache_path")
    prepare_only && return (; cache_path)

    settings = ETTH2_MANYPLUS_CONFIG
    settings.hidden_count > 0 ||
        throw(ArgumentError("hidden_count must be positive"))
    horizon = Int(raw["params"]["horizon"])
    y_val = cache["y_val"]
    y_test = cache["y_test"]
    training_observations = settings.training_observations == 0 ?
                            length(y_val) :
                            min(settings.training_observations, length(y_val))
    training_batch_size = isnothing(settings.training_batch_size) ?
                          nothing :
                          min(
                              settings.training_batch_size,
                              training_observations,
                          )
    train_inputs = etth2_mlp_inputs(
        cache["features_val"][1:training_observations],
        cache["predictions_val"][:, 1:training_observations];
        include_expert_predictions=settings.include_expert_predictions,
    )
    test_inputs = etth2_mlp_inputs(
        cache["features_test"],
        cache["predictions_test"];
        include_expert_predictions=settings.include_expert_predictions,
    )
    initial_priors = if settings.inference_backend === :direct_approximate_messages
        make_etth2_manyplus_direct_priors(
            length(first(train_inputs)),
            settings.hidden_count,
            settings,
        )
    elseif settings.inference_backend === :rxinfer
        make_etth2_manyplus_priors(
            length(first(train_inputs)),
            settings.hidden_count,
            settings,
        )
    else
        throw(ArgumentError(
            "inference_backend must be :direct_approximate_messages or :rxinfer",
        ))
    end

    metrics = Dict{Symbol,Any}()
    predictions = Dict{Symbol,Any}()
    posteriors = Dict{Symbol,Any}()
    for arm in settings.arms
        optimizer_settings = settings.optimizers[arm]
        arm_settings = merge(
            settings,
            (
                optimizer=optimizer_settings.optimizer,
                alpha=optimizer_settings.alpha,
                beta=optimizer_settings.beta,
                training_batch_size=training_batch_size,
            ),
        )
        batch_label = isnothing(training_batch_size) ?
                      "full" :
                      training_batch_size
        println(
            "running manyplus arm=$arm " *
            "optimizer=$(arm_settings.optimizer) " *
            "alpha=$(arm_settings.alpha) " *
            "beta=$(arm_settings.beta) horizon=$horizon " *
            "hidden=$(settings.hidden_count) " *
            "observations=$training_observations batch=$batch_label",
        )
        learned_priors, predicted_mean, predicted_std =
            if settings.inference_backend === :direct_approximate_messages
                learned, _ = train_etth2_manyplus_direct(
                    deepcopy(initial_priors),
                    train_inputs,
                    y_val[1:training_observations],
                    arm_settings,
                )
                prediction_mean, prediction_std =
                    predict_etth2_manyplus_direct(
                        learned,
                        test_inputs,
                        arm_settings,
                    )
                learned, prediction_mean, prediction_std
            else
                learned, _ = train_etth2_manyplus_global_transport(
                    deepcopy(initial_priors),
                    train_inputs,
                    y_val[1:training_observations],
                    arm_settings,
                )
                prediction_mean, prediction_std = predict_etth2_manyplus(
                    learned,
                    test_inputs,
                    arm_settings,
                )
                learned, prediction_mean, prediction_std
            end
        metrics[arm] = predictive_metrics(
            predicted_mean,
            predicted_std,
            y_test,
        )
        predictions[arm] = (mean=predicted_mean, std=predicted_std)
        posteriors[arm] = learned_priors
        println("arm=$arm metrics=$(metrics[arm])")
    end

    config = (;
        horizon,
        hidden_count=settings.hidden_count,
        input_count=length(first(train_inputs)),
        include_expert_predictions=settings.include_expert_predictions,
        training_observations,
        training_batch_size,
        inference_iterations=settings.inference_iterations,
        prediction_batch_size=min(
            settings.prediction_batch_size,
            length(test_inputs),
        ),
        prediction_iterations=settings.prediction_iterations,
        arms=collect(settings.arms),
        optimizers=settings.optimizers,
        inference_backend=settings.inference_backend,
        message_passing_iterations=settings.message_passing_iterations,
        local_ngmp_alpha=settings.local_ngmp_alpha,
        direct_inner_iterations=settings.direct_inner_iterations,
        direct_weight_covariance=settings.direct_weight_covariance,
        seed=settings.seed,
    )
    mkpath(joinpath(ROOT, "results"))
    batch_token = isnothing(training_batch_size) ?
                  "full" :
                  training_batch_size
    backend_token = settings.inference_backend === :direct_approximate_messages ?
                    "direct_approximate_messages" :
                    "rxinfer"
    stem = "etth2_manyplus_residual_sine_$(backend_token)_optimizer_comparison_h$(horizon)_h$(settings.hidden_count)_b$(batch_token)_passes$(settings.inference_iterations)"
    jld2_path = joinpath(ROOT, "results", "$stem.jld2")
    markdown_path = joinpath(ROOT, "results", "$stem.md")
    jldsave(
        jld2_path;
        config,
        metrics,
        predictions,
        posteriors,
        y_test,
    )
    write_etth2_manyplus_report(markdown_path, config, metrics)
    println("results_jld2=$jld2_path")
    println("results_markdown=$markdown_path")
    return (;
        config,
        metrics,
        predictions,
        posteriors,
        jld2_path,
        markdown_path,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    cli = parse_cli(ARGS)
    for session in cli.sessions
        run_etth2_manyplus_optimizer_comparison(;
            session,
            prepare_only=cli.prepare_only,
            rebuild_cache=cli.rebuild_cache,
        )
    end
end
