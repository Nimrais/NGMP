include(joinpath(@__DIR__, "dynamic_vmp_vs_ngmp_etth2.jl"))

using LinearAlgebra: Diagonal, Hermitian, I, cholesky, diag, diagind, eigen, isposdef, issuccess, norm
using ProgressMeter
using Random

const ETTH2_MLP_ARMS = (:ngmp, :relaxed)
const ETTH2_MLP_CONFIG = (
    hidden_count=32,
    include_expert_predictions=true,
    training_observations=0, # 0 uses the complete validation split
    training_batch_size=250, # nothing builds one full training graph
    inference_iterations=20, # complete passes over all training observations
    prediction_batch_size=250,
    prediction_iterations=3,
    alpha=0.2,
    beta=0.0, # use e.g. 0.9 for :projected_nesterov, 0.5 for :vector_transport
    max_step=1.0,
    optimizer=:damped, # :damped, :projected_nesterov, or :vector_transport
    nesterov_eps=1e-8,
    vector_transport_damping=1e-6,
    residual_sine_rho=0.9,
    residual_sine_omega=1.0,
    residual_sine_weight_variance=0.1,
    residual_sine_coefficient_variance=0.1,
    arms=ETTH2_MLP_ARMS,
    limit_stack_depth=500,
    show_progress=true,
    seed=42,
)

@model function etth2_softplus_ngmp_mlp(
    inputs,
    targets,
    n_obs,
    hidden_count,
    priors,
    softplus_dependencies,
    damping,
    output_dependencies,
    output_damping,
)
    local hidden_weight, gate_weight, hidden_mean, gate_score, gate_precision, output

    hidden_precision ~ priors.hidden_precision
    gate_score_precision ~ priors.gate_score_precision
    observation_precision ~ priors.observation_precision
    gate_rate ~ priors.gate_rate

    for hidden in 1:hidden_count
        hidden_weight[hidden] ~ priors.hidden_weight[hidden]
        gate_weight[hidden] ~ priors.gate_weight[hidden]
    end

    for observation in 1:n_obs
        for hidden in 1:hidden_count
            gate_score[hidden, observation] ~ softdot(
                inputs[observation],
                gate_weight[hidden],
                gate_score_precision,
            ) where { meta=LowRankMeta() }
            gate_precision[hidden, observation] ~ GammaShapeRate(1.0, gate_rate)
            gate_precision[hidden, observation] ~ Softplus(
                gate_score[hidden, observation],
            ) where {
                dependencies=softplus_dependencies,
                meta=damping,
            }
            hidden_mean[hidden, observation] ~ softdot(
                inputs[observation],
                hidden_weight[hidden],
                hidden_precision,
            ) where { meta=LowRankMeta() }
            output[observation] ~ NormalMeanPrecision(
                hidden_mean[hidden, observation],
                gate_precision[hidden, observation],
            ) where {
                dependencies=output_dependencies,
                meta=output_damping,
            }
        end
        targets[observation] ~ NormalMeanPrecision(
            output[observation],
            observation_precision,
        )
    end
end

@model function etth2_softplus_ngmp_mlp_relaxed(
    inputs,
    targets,
    n_obs,
    hidden_count,
    priors,
    softplus_dependencies,
    damping,
    output_dependencies,
    output_damping,
)
    local hidden_weight, gate_weight, hidden_mean, gate_score, gate_precision, output

    hidden_precision ~ priors.hidden_precision
    gate_score_precision ~ priors.gate_score_precision
    observation_precision ~ priors.observation_precision
    gate_rate ~ priors.gate_rate

    for hidden in 1:hidden_count
        hidden_weight[hidden] ~ priors.hidden_weight[hidden]
        gate_weight[hidden] ~ priors.gate_weight[hidden]
    end

    for observation in 1:n_obs
        for hidden in 1:hidden_count
            gate_score[hidden, observation] ~ softdot(
                inputs[observation],
                gate_weight[hidden],
                gate_score_precision,
            )
            gate_precision[hidden, observation] ~ GammaShapeRate(1.0, gate_rate)
            gate_precision[hidden, observation] ~ Softplus(
                gate_score[hidden, observation],
            ) where {
                dependencies=softplus_dependencies,
                meta=damping,
            }
            hidden_mean[hidden, observation] ~ softdot(
                inputs[observation],
                hidden_weight[hidden],
                hidden_precision,
            )
            output[observation] ~ NormalMeanPrecision(
                hidden_mean[hidden, observation],
                gate_precision[hidden, observation],
            ) where {
                dependencies=output_dependencies,
                meta=output_damping,
            }
        end
        targets[observation] ~ NormalMeanPrecision(
            output[observation],
            observation_precision,
        )
    end
end

@constraints function etth2_mlp_ngmp_constraints()
    q(
        hidden_weight,
        gate_weight,
        hidden_mean,
        gate_score,
        gate_precision,
        output,
        hidden_precision,
        gate_score_precision,
        observation_precision,
        gate_rate,
    ) = q(hidden_weight)q(hidden_mean, output)q(gate_score, gate_precision)q(gate_weight)q(hidden_precision)q(gate_score_precision)q(observation_precision)q(gate_rate)
    q(hidden_weight)::MomentForm()
    q(gate_weight)::MomentForm()
end

@constraints function etth2_mlp_relaxed_constraints()
    q(
        hidden_weight,
        gate_weight,
        hidden_mean,
        gate_score,
        gate_precision,
        output,
        hidden_precision,
        gate_score_precision,
        observation_precision,
        gate_rate,
    ) = q(hidden_weight, hidden_mean, output)q(gate_weight, gate_score, gate_precision)q(hidden_precision)q(gate_score_precision)q(observation_precision)q(gate_rate)
    q(hidden_weight)::MomentForm()
    q(gate_weight)::MomentForm()
end

@initialization function etth2_mlp_initialization(priors, hidden_count, observation_count)
    q(hidden_weight) = deepcopy(priors.hidden_weight)
    q(gate_weight) = deepcopy(priors.gate_weight)
    q(hidden_mean) = fill(NormalMeanVariance(0.0, 1.0), hidden_count, observation_count)
    q(gate_score) = fill(NormalMeanVariance(0.0, 1.0), hidden_count, observation_count)
    q(gate_precision) = fill(GammaShapeScale(2.0, 1.0), hidden_count, observation_count)
    q(output) = fill(NormalMeanVariance(0.0, 1.0), observation_count)
    q(hidden_precision) = priors.hidden_precision
    q(gate_score_precision) = priors.gate_score_precision
    q(observation_precision) = priors.observation_precision
    q(gate_rate) = priors.gate_rate
    μ(hidden_weight) = deepcopy(priors.hidden_weight)
end

function make_etth2_mlp_priors(input_count, hidden_count; seed)
    rng = MersenneTwister(seed)
    hidden_precision_matrix = Diagonal(fill(1e-2, input_count))
    gate_precision_matrix = Diagonal(fill(1.0, input_count))
    hidden_weight = [
        MvNormalWeightedMeanPrecision(
            hidden_precision_matrix * (0.01 .* randn(rng, input_count)),
            hidden_precision_matrix,
        ) for _ in 1:hidden_count
    ]
    gate_weight = [
        MvNormalWeightedMeanPrecision(
            gate_precision_matrix * (0.01 .* randn(rng, input_count)),
            gate_precision_matrix,
        ) for _ in 1:hidden_count
    ]
    return (
        hidden_weight,
        gate_weight,
        hidden_precision=GammaShapeRate(1e4, 1.0),
        gate_score_precision=GammaShapeRate(1e3, 1.0),
        # ETTh2 targets are standardized with order-one variance. The MNIST
        # prior Gamma(1000, 1) incorrectly forced predictive σ near 0.032.
        observation_precision=GammaShapeRate(2.0, 2.0),
        gate_rate=GammaShapeRate(10.0, 10.0),
    )
end

function etth2_mlp_inputs(features, predictions; include_expert_predictions)
    if include_expert_predictions
        return [vcat(features[j], predictions[:, j]) for j in eachindex(features)]
    end
    return [collect(feature) for feature in features]
end

function information_form_prior(posterior, previous; max_backtracks=40)
    posterior_weighted_mean, posterior_precision = weightedmean_precision(posterior)
    previous_weighted_mean, previous_precision = weightedmean_precision(previous)
    posterior_weighted_mean = collect(posterior_weighted_mean)
    previous_weighted_mean = collect(previous_weighted_mean)
    posterior_precision = Matrix(posterior_precision)
    previous_precision = Matrix(previous_precision)
    for backtrack in 0:max_backtracks
        step = exp2(-backtrack)
        weighted_mean = previous_weighted_mean .+ step .* (
            posterior_weighted_mean .- previous_weighted_mean
        )
        precision_matrix = previous_precision .+ step .* (
            posterior_precision .- previous_precision
        )
        precision_matrix = Matrix(Hermitian((precision_matrix .+ precision_matrix') ./ 2))
        if all(isfinite, weighted_mean) && all(isfinite, precision_matrix) &&
           isposdef(Hermitian(precision_matrix))
            return MvNormalWeightedMeanPrecision(weighted_mean, precision_matrix)
        end
    end
    error("Could not transport a batch weight posterior to a positive-definite prior")
end

function learned_etth2_mlp_priors(result, old_priors)
    posteriors = result.posteriors
    hidden_weight = map(
        information_form_prior,
        last(posteriors[:hidden_weight]),
        old_priors.hidden_weight,
    )
    gate_weight = map(
        information_form_prior,
        last(posteriors[:gate_weight]),
        old_priors.gate_weight,
    )
    return (
        hidden_weight,
        gate_weight,
        hidden_precision=deepcopy(last(posteriors[:hidden_precision])),
        gate_score_precision=deepcopy(last(posteriors[:gate_score_precision])),
        observation_precision=deepcopy(last(posteriors[:observation_precision])),
        gate_rate=deepcopy(last(posteriors[:gate_rate])),
    )
end

function infer_etth2_mlp(
    arm,
    priors,
    inputs,
    targets;
    hidden_count,
    iterations,
    training_batch_size,
    repeat_batch,
    alpha,
    beta,
    max_step,
    optimizer,
    nesterov_eps,
    vector_transport_damping,
    limit_stack_depth,
    showprogress,
)
    observation_count = length(targets)
    observation_count > 0 || throw(ArgumentError("training data must not be empty"))
    repeat_batch > 0 || throw(ArgumentError("repeat_batch must be positive"))
    n_obs, data = if isnothing(training_batch_size)
        observation_count, (inputs=inputs, targets=targets)
    else
        training_batch_size > 0 || throw(ArgumentError(
            "training_batch_size must be positive or nothing",
        ))
        batch_size = min(training_batch_size, observation_count)
        batch_size, (
            inputs=ProbabilisticEnsembling.SubsampledData(inputs, batch_size, repeat_batch),
            targets=ProbabilisticEnsembling.SubsampledData(targets, batch_size, repeat_batch),
        )
    end
    softplus_dependencies = NGMPDependencies(
        out=nothing,
        in=nothing,
        projection=TangentProjection(type=Unscented),
    )
    output_dependencies = NGMPDependencies(
        τ=nothing,
        projection=TangentProjection(type=Unscented),
    )
    damping = DampingMeta(;
        alpha,
        beta,
        max_step,
        method=optimizer,
        eps=nesterov_eps,
        metric_damping=vector_transport_damping,
    )
    output_damping = DampingMeta(;
        alpha,
        beta,
        max_step,
        method=optimizer,
        eps=nesterov_eps,
        metric_damping=vector_transport_damping,
    )
    model = arm === :ngmp ? etth2_softplus_ngmp_mlp :
            arm === :relaxed ? etth2_softplus_ngmp_mlp_relaxed :
            throw(ArgumentError("unsupported arm $arm"))
    constraints = arm === :ngmp ? etth2_mlp_ngmp_constraints() :
                  etth2_mlp_relaxed_constraints()
    result = infer(
        model=model(;
            n_obs,
            hidden_count,
            priors,
            softplus_dependencies,
            damping,
            output_dependencies,
            output_damping,
        ),
        data=data,
        constraints=constraints,
        initialization=etth2_mlp_initialization(priors, hidden_count, n_obs),
        iterations=iterations,
        options=(limit_stack_depth=limit_stack_depth,),
        showprogress=showprogress,
        disable_inference_error_hint=true,
    )
    isempty(softplus_dependencies.states) && error("Softplus NGMP edges were not activated")
    isempty(output_dependencies.states) && error("output NGMP edges were not activated")
    return learned_etth2_mlp_priors(result, priors), result
end

function train_etth2_mlp(
    arm,
    priors,
    inputs,
    targets;
    hidden_count,
    iterations,
    training_batch_size,
    alpha,
    beta,
    max_step,
    optimizer,
    nesterov_eps,
    vector_transport_damping,
    limit_stack_depth,
    showprogress,
    seed,
)
    iterations > 0 || throw(ArgumentError("inference_iterations must be positive"))
    observation_count = length(targets)
    if isnothing(training_batch_size)
        return infer_etth2_mlp(
            arm,
            priors,
            inputs,
            targets;
            hidden_count,
            iterations,
            training_batch_size=nothing,
            repeat_batch=1,
            alpha,
            beta,
            max_step,
            optimizer,
            nesterov_eps,
            vector_transport_damping,
            limit_stack_depth,
            showprogress,
        )
    end

    training_batch_size > 0 || throw(ArgumentError("training_batch_size must be positive"))
    rng = MersenneTwister(seed)
    learned = priors
    final_result = nothing
    batches_per_epoch = cld(observation_count, training_batch_size)
    progress = Progress(
        iterations * batches_per_epoch;
        desc="mlp $arm minibatches ",
        enabled=showprogress,
    )
    for _ in 1:iterations
        order = randperm(rng, observation_count)
        for first_index in 1:training_batch_size:observation_count
            last_index = min(first_index + training_batch_size - 1, observation_count)
            indices = order[first_index:last_index]
            # Match the MNIST schedule: fit one concrete minibatch and carry
            # its learned global marginals forward to the next minibatch.
            learned, final_result = infer_etth2_mlp(
                arm,
                learned,
                inputs[indices],
                targets[indices];
                hidden_count,
                iterations=1,
                training_batch_size=nothing,
                repeat_batch=1,
                alpha,
                beta,
                max_step,
                optimizer,
                nesterov_eps,
                vector_transport_damping,
                limit_stack_depth,
                showprogress=false,
            )
            ProgressMeter.next!(progress)
        end
    end
    ProgressMeter.finish!(progress)
    return learned, final_result
end

@constraints function etth2_mlp_prediction_constraints(priors)
    q(
        hidden_weight,
        gate_weight,
        hidden_mean,
        gate_score,
        gate_precision,
        output,
        hidden_precision,
        gate_score_precision,
        observation_precision,
        gate_rate,
    ) = q(hidden_weight)q(hidden_mean, output)q(gate_score, gate_precision)q(gate_weight)q(hidden_precision)q(gate_score_precision)q(observation_precision)q(gate_rate)
    for (index, prior) in enumerate(deepcopy(priors.hidden_weight))
        q(hidden_weight[index])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    for (index, prior) in enumerate(deepcopy(priors.gate_weight))
        q(gate_weight[index])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    q(hidden_precision)::RxInfer.FixedMarginalFormConstraint(priors.hidden_precision)
    q(gate_score_precision)::RxInfer.FixedMarginalFormConstraint(priors.gate_score_precision)
    q(observation_precision)::RxInfer.FixedMarginalFormConstraint(priors.observation_precision)
    q(gate_rate)::RxInfer.FixedMarginalFormConstraint(priors.gate_rate)
end

function predict_etth2_mlp_batch(
    priors,
    inputs;
    hidden_count,
    iterations,
    alpha,
    beta,
    max_step,
    optimizer,
    nesterov_eps,
    vector_transport_damping,
    limit_stack_depth,
)
    observation_count = length(inputs)
    softplus_dependencies = NGMPDependencies(
        out=nothing,
        in=nothing,
        projection=TangentProjection(type=Unscented),
    )
    output_dependencies = NGMPDependencies(
        τ=nothing,
        projection=TangentProjection(type=Unscented),
    )
    damping = DampingMeta(;
        alpha,
        beta,
        max_step,
        method=optimizer,
        eps=nesterov_eps,
        metric_damping=vector_transport_damping,
    )
    result = infer(
        model=etth2_softplus_ngmp_mlp(;
            n_obs=observation_count,
            hidden_count,
            priors,
            softplus_dependencies,
            damping,
            output_dependencies,
            output_damping=damping,
        ),
        data=(inputs=inputs, targets=fill(missing, observation_count)),
        constraints=etth2_mlp_prediction_constraints(priors),
        initialization=etth2_mlp_initialization(priors, hidden_count, observation_count),
        iterations=iterations,
        free_energy=false,
        options=(limit_stack_depth=limit_stack_depth,),
        disable_inference_error_hint=true,
    )
    # `predictions[:targets]` is only the observation-factor message here and
    # omits q(output) uncertainty. Convolve the latent output posterior with
    # the Gaussian observation noise for the actual posterior predictive.
    output = last(result.posteriors[:output])
    shape, rate = params(priors.observation_precision)
    noise_variance = shape > 1 ? rate / (shape - 1) : inv(mean(priors.observation_precision))
    return mean.(output), sqrt.(var.(output) .+ noise_variance)
end

function predict_etth2_mlp(priors, inputs; batch_size, kwargs...)
    n_obs = length(inputs)
    n_obs > 0 || throw(ArgumentError("prediction data must not be empty"))
    batch_size > 0 || throw(ArgumentError("prediction batch size must be positive"))
    predicted_mean = Vector{Float64}(undef, n_obs)
    predicted_std = Vector{Float64}(undef, n_obs)
    for first_index in 1:batch_size:n_obs
        last_index = min(first_index + batch_size - 1, n_obs)
        indices = first_index:last_index
        batch_mean, batch_std = predict_etth2_mlp_batch(priors, inputs[indices]; kwargs...)
        predicted_mean[indices] = batch_mean
        predicted_std[indices] = batch_std
    end
    return predicted_mean, predicted_std
end

@model function etth2_manyplus_residual_sine(
    inputs,
    targets,
    n_obs,
    hidden_count,
    priors,
    activation,
    activation_dependencies,
)
    local weight, coefficient, preactivation, hidden, contribution, output

    hidden_precision ~ priors.hidden_precision
    contribution_precision ~ priors.contribution_precision
    observation_precision ~ priors.observation_precision
    intercept ~ priors.intercept
    for unit in 1:hidden_count
        weight[unit] ~ priors.weight[unit]
        coefficient[unit] ~ priors.coefficient[unit]
    end
    for observation in 1:n_obs
        for unit in 1:hidden_count
            preactivation[unit, observation] ~ softdot(
                inputs[observation],
                weight[unit],
                hidden_precision,
            )
            hidden[unit, observation] ~ ResidualSine(
                preactivation[unit, observation],
            ) where {
                dependencies=activation_dependencies,
                meta=activation,
            }
            contribution[unit, observation] ~ softdot(
                coefficient[unit],
                hidden[unit, observation],
                contribution_precision,
            )
        end
        output[observation] ~ ManyPlus(
            inputs=[
                contribution[unit, observation] for unit in 1:hidden_count
            ],
        )
        targets[observation] ~ NormalMeanPrecision(
            output[observation] + intercept,
            observation_precision,
        )
    end
end

@constraints function etth2_manyplus_constraints()
    q(
        weight,
        coefficient,
        preactivation,
        hidden,
        contribution,
        output,
        intercept,
        hidden_precision,
        contribution_precision,
        observation_precision,
    ) = q(weight, preactivation, hidden, contribution, output, intercept)q(coefficient)q(hidden_precision)q(contribution_precision)q(observation_precision)
    q(weight)::MomentForm()
end

@initialization function etth2_manyplus_initialization(priors, initial)
    q(coefficient) = deepcopy(priors.coefficient)
    q(preactivation) = initial.preactivation
    q(hidden) = initial.hidden
    q(contribution) = initial.contribution
    q(output) = initial.output
    q(hidden_precision) = priors.hidden_precision
    q(contribution_precision) = priors.contribution_precision
    q(observation_precision) = priors.observation_precision
    μ(weight) = deepcopy(priors.weight)
    μ(intercept) = priors.intercept
end

function make_etth2_manyplus_priors(input_count, hidden_count, settings)
    rng = MersenneTwister(settings.seed)
    weight_precision = Diagonal(fill(
        inv(settings.residual_sine_weight_variance),
        input_count,
    ))
    weight = [
        begin
            direction = randn(rng, input_count)
            direction ./= max(norm(direction), eps())
            direction .*= sqrt(2 / input_count)
            MvNormalWeightedMeanPrecision(
                weight_precision * direction,
                weight_precision,
            )
        end for _ in 1:hidden_count
    ]
    coefficient = [
        NormalMeanVariance(
            (isodd(unit) ? 1.0 : -1.0) / hidden_count,
            settings.residual_sine_coefficient_variance,
        ) for unit in 1:hidden_count
    ]
    return (
        weight,
        coefficient,
        intercept=NormalMeanVariance(0.0, 1.0),
        hidden_precision=GammaShapeRate(1e3, 1.0),
        contribution_precision=GammaShapeRate(1e4, 1.0),
        observation_precision=GammaShapeRate(2.0, 2.0),
    )
end

function etth2_manyplus_initial_values(priors, inputs, hidden_count, activation)
    weight_mean = mean.(priors.weight)
    coefficient_mean = mean.(priors.coefficient)
    observation_count = length(inputs)
    phi(value) = SurrogateModelling._residual_sine(value, activation)
    preactivation = [
        NormalMeanVariance(dot(weight_mean[unit], inputs[observation]), 0.5)
        for unit in 1:hidden_count, observation in 1:observation_count
    ]
    hidden = [
        NormalMeanVariance(phi(mean(preactivation[unit, observation])), 1.0)
        for unit in 1:hidden_count, observation in 1:observation_count
    ]
    contribution = [
        NormalMeanVariance(
            coefficient_mean[unit] * mean(hidden[unit, observation]),
            1.0,
        )
        for unit in 1:hidden_count, observation in 1:observation_count
    ]
    output = [
        NormalMeanVariance(
            sum(mean(contribution[unit, observation]) for unit in 1:hidden_count),
            1.0,
        ) for observation in 1:observation_count
    ]
    return (; preactivation, hidden, contribution, output)
end

function learned_etth2_manyplus_priors(result)
    posteriors = result.posteriors
    return (
        weight=deepcopy(collect(vec(posteriors[:weight]))),
        coefficient=deepcopy(collect(vec(posteriors[:coefficient]))),
        intercept=deepcopy(posteriors[:intercept]),
        hidden_precision=deepcopy(posteriors[:hidden_precision]),
        contribution_precision=deepcopy(posteriors[:contribution_precision]),
        observation_precision=deepcopy(posteriors[:observation_precision]),
    )
end

function infer_etth2_manyplus(priors, inputs, targets, settings)
    observation_count = length(targets)
    activation = ResidualSineMeta(
        rho=settings.residual_sine_rho,
        omega=settings.residual_sine_omega,
    )
    dependencies = NGMPDependencies(
        out=nothing,
        in=nothing,
        projection=TangentProjection(type=ClosedForm),
        damping=DampingMeta(
            alpha=hasproperty(settings, :local_ngmp_alpha) ?
                  settings.local_ngmp_alpha :
                  settings.alpha,
            beta=0.0,
            max_step=hasproperty(settings, :local_ngmp_max_step) ?
                     settings.local_ngmp_max_step :
                     settings.max_step,
            method=:damped,
            eps=settings.nesterov_eps,
            metric_damping=settings.vector_transport_damping,
        ),
    )
    result = infer(
        model=etth2_manyplus_residual_sine(;
            n_obs=observation_count,
            hidden_count=settings.hidden_count,
            priors,
            activation,
            activation_dependencies=dependencies,
        ),
        data=(inputs=inputs, targets=targets),
        constraints=etth2_manyplus_constraints(),
        initialization=etth2_manyplus_initialization(
            priors,
            etth2_manyplus_initial_values(
                priors,
                inputs,
                settings.hidden_count,
                activation,
            ),
        ),
        iterations=hasproperty(settings, :message_passing_iterations) ?
                   settings.message_passing_iterations :
                   1,
        returnvars=(
            weight=KeepLast(),
            coefficient=KeepLast(),
            intercept=KeepLast(),
            hidden_precision=KeepLast(),
            contribution_precision=KeepLast(),
            observation_precision=KeepLast(),
        ),
        free_energy=false,
        showprogress=false,
        options=(limit_stack_depth=settings.limit_stack_depth,),
        disable_inference_error_hint=true,
    )
    isempty(dependencies.states) && error("ResidualSine NGMP edges were not activated")
    return learned_etth2_manyplus_priors(result), result
end

function train_etth2_manyplus(priors, inputs, targets, settings)
    observation_count = length(targets)
    batch_size = isnothing(settings.training_batch_size) ?
                 observation_count :
                 min(settings.training_batch_size, observation_count)
    rng = MersenneTwister(settings.seed)
    learned = priors
    final_result = nothing
    progress = Progress(
        settings.inference_iterations * cld(observation_count, batch_size);
        desc="mlp manyplus_residual_sine minibatches ",
        enabled=settings.show_progress,
    )
    for _ in 1:settings.inference_iterations
        order = randperm(rng, observation_count)
        for first_index in 1:batch_size:observation_count
            indices = order[
                first_index:min(first_index + batch_size - 1, observation_count)
            ]
            learned, final_result = infer_etth2_manyplus(
                learned,
                inputs[indices],
                targets[indices],
                settings,
            )
            ProgressMeter.next!(progress)
        end
    end
    ProgressMeter.finish!(progress)
    return learned, final_result
end

@constraints function etth2_manyplus_prediction_constraints(priors)
    q(
        weight,
        coefficient,
        preactivation,
        hidden,
        contribution,
        output,
        intercept,
        hidden_precision,
        contribution_precision,
        observation_precision,
    ) = q(weight)q(coefficient)q(hidden_precision)q(contribution_precision)q(observation_precision)q(intercept)q(preactivation, hidden, contribution, output)
    for (unit, prior) in enumerate(deepcopy(priors.weight))
        q(weight[unit])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    for (unit, prior) in enumerate(deepcopy(priors.coefficient))
        q(coefficient[unit])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    q(intercept)::RxInfer.FixedMarginalFormConstraint(priors.intercept)
    q(hidden_precision)::RxInfer.FixedMarginalFormConstraint(priors.hidden_precision)
    q(contribution_precision)::RxInfer.FixedMarginalFormConstraint(priors.contribution_precision)
    q(observation_precision)::RxInfer.FixedMarginalFormConstraint(priors.observation_precision)
end

function predict_etth2_manyplus_batch(priors, inputs, settings)
    observation_count = length(inputs)
    activation = ResidualSineMeta(
        rho=settings.residual_sine_rho,
        omega=settings.residual_sine_omega,
    )
    dependencies = NGMPDependencies(
        out=nothing,
        in=nothing,
        projection=TangentProjection(type=ClosedForm),
        damping=DampingMeta(
            alpha=hasproperty(settings, :local_ngmp_alpha) ?
                  settings.local_ngmp_alpha :
                  settings.alpha,
            beta=0.0,
            max_step=hasproperty(settings, :local_ngmp_max_step) ?
                     settings.local_ngmp_max_step :
                     settings.max_step,
            method=:damped,
            eps=settings.nesterov_eps,
            metric_damping=settings.vector_transport_damping,
        ),
    )
    result = infer(
        model=etth2_manyplus_residual_sine(;
            n_obs=observation_count,
            hidden_count=settings.hidden_count,
            priors,
            activation,
            activation_dependencies=dependencies,
        ),
        data=(inputs=inputs, targets=fill(missing, observation_count)),
        constraints=etth2_manyplus_prediction_constraints(priors),
        initialization=etth2_manyplus_initialization(
            priors,
            etth2_manyplus_initial_values(
                priors,
                inputs,
                settings.hidden_count,
                activation,
            ),
        ),
        iterations=settings.prediction_iterations,
        free_energy=false,
        options=(limit_stack_depth=settings.limit_stack_depth,),
        disable_inference_error_hint=true,
    )
    output = last(result.posteriors[:output])
    output_mean = mean.(output) .+ mean(priors.intercept)
    shape, rate = params(priors.observation_precision)
    noise_variance = shape > 1 ? rate / (shape - 1) :
                     inv(mean(priors.observation_precision))
    output_std = sqrt.(var.(output) .+ var(priors.intercept) .+ noise_variance)
    return output_mean, output_std
end

function predict_etth2_manyplus(priors, inputs, settings)
    predicted_mean = Vector{Float64}(undef, length(inputs))
    predicted_std = Vector{Float64}(undef, length(inputs))
    batch_size = min(settings.prediction_batch_size, length(inputs))
    for first_index in 1:batch_size:length(inputs)
        indices = first_index:min(first_index + batch_size - 1, length(inputs))
        batch_mean, batch_std = predict_etth2_manyplus_batch(
            priors,
            inputs[indices],
            settings,
        )
        predicted_mean[indices] = batch_mean
        predicted_std[indices] = batch_std
    end
    return predicted_mean, predicted_std
end

function write_etth2_mlp_report(path, config, metrics)
    open(path, "w") do io
        println(io, "# One-hidden-layer Softplus NGMP MLP on ETTh2")
        println(io)
        println(io, "Horizon: **$(config.horizon)**  ")
        println(io, "Hidden units: **$(config.hidden_count)**  ")
        println(io, "Input features: **$(config.input_count)**  ")
        println(io, "Training observations: **$(config.training_observations)**  ")
        batch_label = isnothing(config.training_batch_size) ? "full graph" : config.training_batch_size
        println(io, "Training batch size: **$batch_label**  ")
        println(io, "Inference iterations (complete data passes): **$(config.inference_iterations)**  ")
        println(io, "Prediction batch/iterations: **$(config.prediction_batch_size)/$(config.prediction_iterations)**  ")
        println(io, "NGMP optimizer: **$(config.optimizer)**  ")
        println(io, "Inputs include expert forecasts: **$(config.include_expert_predictions)**")
        println(io)
        println(io, "| Method | MAE | MSE ± 95% CI | NLL ± 95% CI | Coverage 95% | Pinball |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        labels = Dict(
            :ngmp => "Softplus UT-NGMP",
            :relaxed => "Softplus UT-NGMP relaxed",
            :manyplus_residual_sine => "ManyPlus residual-sine NGMP",
        )
        for arm in config.arms
            value = metrics[arm]
            println(io,
                "| $(labels[arm]) | $(@sprintf("%.4f", value.mae)) | " *
                "$(@sprintf("%.4f", value.mse)) ± $(@sprintf("%.4f", value.mse_ci95)) | " *
                "$(@sprintf("%.4f", value.negative_log_likelihood)) ± $(@sprintf("%.4f", value.negative_log_likelihood_ci95)) | " *
                "$(@sprintf("%.4f", value.coverage95)) | $(@sprintf("%.4f", value.pinball)) |",
            )
        end
    end
end

function run_etth2_mlp(; session, prepare_only=false, rebuild_cache=false)
    session_path = isabspath(session) ? normpath(session) : normpath(joinpath(ROOT, session))
    _, raw, cache, cache_path = prepare_data(session_path; rebuild_cache)
    println("prepared_cache=$cache_path")
    prepare_only && return (; cache_path)

    settings = ETTH2_MLP_CONFIG
    settings.hidden_count > 0 || throw(ArgumentError("hidden_count must be positive"))
    all(arm -> arm in ETTH2_MLP_ARMS, settings.arms) || throw(ArgumentError(
        "arms must be selected from $(collect(ETTH2_MLP_ARMS))",
    ))
    horizon = Int(raw["params"]["horizon"])
    y_val = cache["y_val"]
    y_test = cache["y_test"]
    training_observations = settings.training_observations == 0 ? length(y_val) :
                            min(settings.training_observations, length(y_val))
    training_batch_size = isnothing(settings.training_batch_size) ? nothing :
                          min(settings.training_batch_size, training_observations)
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
    priors = make_etth2_mlp_priors(length(first(train_inputs)), settings.hidden_count; seed=settings.seed)
    manyplus_priors = make_etth2_manyplus_priors(
        length(first(train_inputs)),
        settings.hidden_count,
        settings,
    )
    metrics = Dict{Symbol,Any}()
    predictions = Dict{Symbol,Any}()
    posteriors = Dict{Symbol,Any}()
    for arm in settings.arms
        batch_label = isnothing(training_batch_size) ? "full" : training_batch_size
        println("running mlp arm=$arm optimizer=$(settings.optimizer) horizon=$horizon hidden=$(settings.hidden_count) observations=$training_observations batch=$batch_label")
        learned_priors, predicted_mean, predicted_std = if arm === :manyplus_residual_sine
            learned, _ = train_etth2_manyplus(
                deepcopy(manyplus_priors),
                train_inputs,
                y_val[1:training_observations],
                settings,
            )
            prediction_mean, prediction_std = predict_etth2_manyplus(
                learned,
                test_inputs,
                settings,
            )
            learned, prediction_mean, prediction_std
        else
            learned, _ = train_etth2_mlp(
                arm,
                deepcopy(priors),
                train_inputs,
                y_val[1:training_observations];
                hidden_count=settings.hidden_count,
                iterations=settings.inference_iterations,
                training_batch_size,
                alpha=settings.alpha,
                beta=settings.beta,
                max_step=settings.max_step,
                optimizer=settings.optimizer,
                nesterov_eps=settings.nesterov_eps,
                vector_transport_damping=settings.vector_transport_damping,
                limit_stack_depth=settings.limit_stack_depth,
                showprogress=settings.show_progress,
                seed=settings.seed,
            )
            prediction_mean, prediction_std = predict_etth2_mlp(
                learned,
                test_inputs;
                batch_size=min(settings.prediction_batch_size, length(test_inputs)),
                hidden_count=settings.hidden_count,
                iterations=settings.prediction_iterations,
                alpha=settings.alpha,
                beta=settings.beta,
                max_step=settings.max_step,
                optimizer=settings.optimizer,
                nesterov_eps=settings.nesterov_eps,
                vector_transport_damping=settings.vector_transport_damping,
                limit_stack_depth=settings.limit_stack_depth,
            )
            learned, prediction_mean, prediction_std
        end
        metrics[arm] = predictive_metrics(predicted_mean, predicted_std, y_test)
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
        prediction_batch_size=min(settings.prediction_batch_size, length(test_inputs)),
        prediction_iterations=settings.prediction_iterations,
        optimizer=settings.optimizer,
        arms=collect(settings.arms),
    )
    mkpath(joinpath(ROOT, "results"))
    batch_token = isnothing(training_batch_size) ? "full" : training_batch_size
    stem = "etth2_softplus_ut_ngmp_mlp_$(settings.optimizer)_h$(horizon)_h$(settings.hidden_count)_b$(batch_token)_passes$(settings.inference_iterations)"
    jld2_path = joinpath(ROOT, "results", "$stem.jld2")
    markdown_path = joinpath(ROOT, "results", "$stem.md")
    jldsave(jld2_path; config, metrics, predictions, posteriors, y_test)
    write_etth2_mlp_report(markdown_path, config, metrics)
    println("results_jld2=$jld2_path")
    println("results_markdown=$markdown_path")
    return (; config, metrics, predictions, posteriors, jld2_path, markdown_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    cli = parse_cli(ARGS)
    for session in cli.sessions
        run_etth2_mlp(;
            session,
            prepare_only=cli.prepare_only,
            rebuild_cache=cli.rebuild_cache,
        )
    end
end
