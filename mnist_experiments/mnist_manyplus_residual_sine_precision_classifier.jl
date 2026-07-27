using LinearAlgebra
using ProgressMeter
using Random
using RxInfer
using Statistics
using SurrogateModelling
import ProbabilisticEnsembling: Exp

include(joinpath(@__DIR__, "image_classification_utils.jl"))

# ============================================================================
# Heteroscedastic multiclass extension of the ManyPlus ResidualSine model.
#
# For image i, hidden unit k, and class c:
#
#   z[k,i]       = w[k]' x[i]
#   h[k,i]       = ResidualSine(z[k,i])
#   mean[c,i]    = sum_k v[c,k] h[k,i]
#   logprec[i]   = sum_k g[k] h[k,i]
#   precision[i] = exp(logprec[i])
#   onehot[c,i] ~ Normal(mean[c,i], precision[i])
#
# The final line is deliberately a Gaussian one-hot classification surrogate.
# It supplies a well-defined learned per-image variance head, but is not an
# exact Categorical likelihood and must not be reported as one.
#
# STATUS: research scaffold. RxInfer currently constructs this graph but cannot
# derive a complete update schedule for both learned heads. It is deliberately
# excluded from run_residual_sine_mnist_sweep.sh until that is resolved.
# ============================================================================

const PRECISION_CLASSIFIER_ARMS = (
    :damped,
    :vector_transport_05,
    :vector_transport_08,
    :projected_nesterov_09,
)

function precision_classifier_optimizer(arm)
    arm === :damped &&
        return (method=:damped, alpha=0.5, beta=0.0)
    arm === :vector_transport_05 &&
        return (method=:vector_transport, alpha=0.2, beta=0.5)
    arm === :vector_transport_08 &&
        return (method=:vector_transport, alpha=0.2, beta=0.8)
    arm === :projected_nesterov_09 &&
        return (method=:projected_nesterov, alpha=0.2, beta=0.9)
    throw(ArgumentError(
        "unknown optimizer arm $arm; use one of $PRECISION_CLASSIFIER_ARMS"))
end

function precision_classifier_dependencies(
    arm;
    max_step=1.0,
    eps=1e-8,
    metric_damping=1e-6,
)
    optimizer = precision_classifier_optimizer(arm)
    return NGMPDependencies(
        out=nothing,
        in=nothing,
        projection=TangentProjection(type=ClosedForm),
        damping=DampingMeta(
            alpha=optimizer.alpha,
            beta=optimizer.beta,
            max_step=max_step,
            method=optimizer.method,
            eps=eps,
            metric_damping=metric_damping,
        ),
    )
end

@model function manyplus_residual_sine_precision_classifier(
    features,
    targets,
    observation_count,
    input_count,
    hidden_count,
    class_count,
    priors,
    activation,
    activation_dependencies,
    precision_dependencies,
)
    local weight
    local mean_coefficient
    local precision_coefficient
    local preactivation
    local hidden
    local mean_contribution
    local precision_contribution
    local mean_output
    local precision_score
    local log_precision
    local precision

    hidden_precision ~ priors.hidden_precision
    contribution_precision ~ priors.contribution_precision
    log_precision_precision ~ priors.log_precision_precision

    for unit in 1:hidden_count
        weight[unit] ~ priors.weight[unit]
        precision_coefficient[unit] ~ priors.precision_coefficient[unit]
    end
    for cls in 1:class_count
        for unit in 1:hidden_count
            mean_coefficient[cls, unit] ~
                priors.mean_coefficient[cls, unit]
        end
    end

    for observation in 1:observation_count
        for unit in 1:hidden_count
            preactivation[unit, observation] ~ softdot(
                features[observation],
                weight[unit],
                hidden_precision,
            )
            hidden[unit, observation] ~ ResidualSine(
                preactivation[unit, observation],
            ) where {
                dependencies=activation_dependencies,
                meta=activation,
            }
            precision_contribution[unit, observation] ~ softdot(
                precision_coefficient[unit],
                hidden[unit, observation],
                contribution_precision,
            )
            for cls in 1:class_count
                mean_contribution[cls, unit, observation] ~ softdot(
                    mean_coefficient[cls, unit],
                    hidden[unit, observation],
                    contribution_precision,
                )
            end
        end

        for cls in 1:class_count
            mean_output[cls, observation] ~ ManyPlus(
                inputs=[
                    mean_contribution[cls, unit, observation]
                    for unit in 1:hidden_count
                ],
            )
        end
        precision_score[observation] ~ ManyPlus(
            inputs=[
                precision_contribution[unit, observation]
                for unit in 1:hidden_count
            ],
        )
        log_precision[observation] ~ NormalMeanPrecision(
            precision_score[observation],
            log_precision_precision,
        )
        precision[observation] ~ Exp(
            log_precision[observation],
        ) where {
            dependencies=precision_dependencies,
        }
        for cls in 1:class_count
            targets[cls, observation] ~ NormalMeanPrecision(
                mean_output[cls, observation],
                precision[observation],
            )
        end
    end
end

@constraints function precision_classifier_constraints()
    q(
        weight,
        mean_coefficient,
        precision_coefficient,
        preactivation,
        hidden,
        mean_contribution,
        precision_contribution,
        mean_output,
        precision_score,
        log_precision,
        precision,
        hidden_precision,
        contribution_precision,
        log_precision_precision,
    ) = q(
        weight,
        preactivation,
        hidden,
        mean_contribution,
        precision_contribution,
        mean_output,
        precision_score,
        log_precision,
    )q(mean_coefficient)q(precision_coefficient)q(precision)q(hidden_precision)q(contribution_precision)q(log_precision_precision)
    q(weight)::MomentForm()
end

@initialization function precision_classifier_initialization(priors, initial)
    q(mean_coefficient) = deepcopy(priors.mean_coefficient)
    q(precision_coefficient) = deepcopy(priors.precision_coefficient)
    q(preactivation) = initial.preactivation
    q(hidden) = initial.hidden
    q(mean_contribution) = initial.mean_contribution
    q(precision_contribution) = initial.precision_contribution
    q(mean_output) = initial.mean_output
    q(precision_score) = initial.precision_score
    q(log_precision) = initial.log_precision
    q(precision) = initial.precision
    q(hidden_precision) = priors.hidden_precision
    q(contribution_precision) = priors.contribution_precision
    q(log_precision_precision) = priors.log_precision_precision
    μ(weight) = deepcopy(priors.weight)
end

function make_precision_classifier_priors(
    input_count,
    hidden_count,
    classes;
    seed=1,
    weight_variance=0.05,
    coefficient_variance=0.1,
)
    hidden_count >= 2 ||
        throw(ArgumentError("ManyPlus requires hidden_count >= 2"))
    rng = MersenneTwister(seed)
    class_count = length(classes)
    weight_precision =
        Diagonal(fill(inv(weight_variance), input_count))
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
    mean_coefficient = [
        NormalMeanVariance(
            0.05 * randn(rng),
            coefficient_variance,
        )
        for _ in 1:class_count, _ in 1:hidden_count
    ]
    precision_coefficient = [
        NormalMeanVariance(0.0, coefficient_variance)
        for _ in 1:hidden_count
    ]
    return (
        weight,
        mean_coefficient,
        precision_coefficient,
        hidden_precision=GammaShapeRate(1e3, 1.0),
        contribution_precision=GammaShapeRate(1e4, 1.0),
        log_precision_precision=GammaShapeRate(1e3, 1.0),
        classes=collect(classes),
    )
end

function precision_classifier_initial_values(
    priors,
    features,
    hidden_count,
    class_count,
    activation,
)
    weight_mean = mean.(priors.weight)
    mean_coefficient_mean = mean.(priors.mean_coefficient)
    precision_coefficient_mean = mean.(priors.precision_coefficient)
    observation_count = length(features)
    phi(value) = SurrogateModelling._residual_sine(value, activation)

    preactivation = [
        NormalMeanVariance(
            dot(weight_mean[unit], features[observation]),
            0.5,
        )
        for unit in 1:hidden_count, observation in 1:observation_count
    ]
    hidden = [
        NormalMeanVariance(phi(mean(preactivation[unit, observation])), 1.0)
        for unit in 1:hidden_count, observation in 1:observation_count
    ]
    mean_contribution = [
        NormalMeanVariance(
            mean_coefficient_mean[cls, unit] *
                mean(hidden[unit, observation]),
            1.0,
        )
        for cls in 1:class_count, unit in 1:hidden_count,
            observation in 1:observation_count
    ]
    precision_contribution = [
        NormalMeanVariance(
            precision_coefficient_mean[unit] *
                mean(hidden[unit, observation]),
            1.0,
        )
        for unit in 1:hidden_count, observation in 1:observation_count
    ]
    mean_output = [
        NormalMeanVariance(
            sum(
                mean(mean_contribution[cls, unit, observation])
                for unit in 1:hidden_count
            ),
            1.0,
        )
        for cls in 1:class_count, observation in 1:observation_count
    ]
    precision_score = [
        NormalMeanVariance(
            sum(
                mean(precision_contribution[unit, observation])
                for unit in 1:hidden_count
            ),
            1.0,
        )
        for observation in 1:observation_count
    ]
    log_precision = [
        NormalMeanVariance(
            mean(precision_score[observation]),
            1.0,
        )
        for observation in 1:observation_count
    ]
    precision = [
        begin
            precision_mean =
                exp(clamp(mean(log_precision[observation]), -10.0, 10.0))
            GammaShapeRate(100.0, 100.0 / precision_mean)
        end
        for observation in 1:observation_count
    ]
    return (;
        preactivation,
        hidden,
        mean_contribution,
        precision_contribution,
        mean_output,
        precision_score,
        log_precision,
        precision,
    )
end

function onehot_targets(labels, classes)
    class_to_index =
        Dict(label => index for (index, label) in enumerate(classes))
    targets = zeros(Float64, length(classes), length(labels))
    for (observation, label) in enumerate(labels)
        targets[class_to_index[label], observation] = 1.0
    end
    return targets
end

function precision_classifier_softmax(logits)
    shifted = logits .- maximum(logits)
    weights = exp.(shifted)
    return weights ./ sum(weights)
end

function learned_precision_classifier_priors(result, classes)
    posteriors = result.posteriors
    return (
        weight=deepcopy(collect(vec(posteriors[:weight]))),
        mean_coefficient=deepcopy(collect(posteriors[:mean_coefficient])),
        precision_coefficient=deepcopy(
            collect(vec(posteriors[:precision_coefficient]))),
        hidden_precision=deepcopy(posteriors[:hidden_precision]),
        contribution_precision=deepcopy(
            posteriors[:contribution_precision]),
        log_precision_precision=deepcopy(
            posteriors[:log_precision_precision]),
        classes=copy(classes),
    )
end

function infer_precision_classifier_batch(
    priors,
    features,
    labels;
    hidden_count,
    optimizer=:damped,
    residual_sine_rho=0.9,
    residual_sine_omega=1.0,
    iterations=5,
    limit_stack_depth=1000,
)
    class_count = length(priors.classes)
    activation = ResidualSineMeta(
        rho=residual_sine_rho,
        omega=residual_sine_omega,
    )
    activation_dependencies =
        precision_classifier_dependencies(optimizer)
    precision_dependencies =
        precision_classifier_dependencies(optimizer)
    initial = precision_classifier_initial_values(
        priors,
        features,
        hidden_count,
        class_count,
        activation,
    )
    result = infer(
        model=manyplus_residual_sine_precision_classifier(;
            observation_count=length(labels),
            input_count=length(first(features)),
            hidden_count,
            class_count,
            priors,
            activation,
            activation_dependencies,
            precision_dependencies,
        ),
        data=(
            features=features,
            targets=onehot_targets(labels, priors.classes),
        ),
        constraints=precision_classifier_constraints(),
        initialization=precision_classifier_initialization(priors, initial),
        iterations=iterations,
        returnvars=(
            weight=KeepLast(),
            mean_coefficient=KeepLast(),
            precision_coefficient=KeepLast(),
            hidden_precision=KeepLast(),
            contribution_precision=KeepLast(),
            log_precision_precision=KeepLast(),
        ),
        free_energy=false,
        showprogress=false,
        options=(limit_stack_depth=limit_stack_depth,),
        disable_inference_error_hint=true,
    )
    isempty(activation_dependencies.states) &&
        error("ResidualSine NGMP edges were not activated")
    isempty(precision_dependencies.states) &&
        error("Exp precision NGMP edges were not activated")
    return learned_precision_classifier_priors(result, priors.classes), result
end

function predict_precision_classifier(
    priors,
    input;
    residual_sine_rho=0.9,
    residual_sine_omega=1.0,
)
    activation = ResidualSineMeta(
        rho=residual_sine_rho,
        omega=residual_sine_omega,
    )
    weight_mean = mean.(priors.weight)
    mean_coefficient_mean = mean.(priors.mean_coefficient)
    precision_coefficient_mean = mean.(priors.precision_coefficient)
    hidden = [
        SurrogateModelling._residual_sine(
            dot(weight_mean[unit], input),
            activation,
        )
        for unit in eachindex(weight_mean)
    ]
    logits = mean_coefficient_mean * hidden
    log_precision = dot(precision_coefficient_mean, hidden)
    predicted_precision = exp(clamp(log_precision, -10.0, 10.0))
    probabilities = precision_classifier_softmax(logits)
    return (
        label=priors.classes[argmax(probabilities)],
        probabilities,
        logits,
        precision=predicted_precision,
        std=inv(sqrt(predicted_precision)),
    )
end

function evaluate_precision_classifier(
    priors,
    inputs,
    labels;
    max_images=size(inputs, 2),
    residual_sine_rho=0.9,
    residual_sine_omega=1.0,
)
    count_images = min(max_images, length(labels))
    correct = 0
    nll = 0.0
    precision_sum = 0.0
    class_to_index =
        Dict(label => index for (index, label) in enumerate(priors.classes))
    for observation in 1:count_images
        prediction = predict_precision_classifier(
            priors,
            @view(inputs[:, observation]);
            residual_sine_rho,
            residual_sine_omega,
        )
        correct += prediction.label == labels[observation]
        class_index = class_to_index[labels[observation]]
        nll -= log(max(prediction.probabilities[class_index], 1e-12))
        precision_sum += prediction.precision
    end
    return (
        accuracy=correct / count_images,
        nll=nll / count_images,
        mean_precision=precision_sum / count_images,
    )
end

function train_precision_classifier(
    ;
    dataset=:mnist,
    ntrain=1000,
    nval=100,
    ntest=100,
    split_sampling=:balanced,
    image_size=nothing,
    classes=nothing,
    hidden_count=8,
    batch_size=10,
    passes=1,
    seed=1,
    optimizer=:damped,
    residual_sine_rho=0.9,
    residual_sine_omega=1.0,
    inference_iterations=5,
    eval_max_images=1000,
    limit_stack_depth=1000,
)
    passes == 1 || throw(ArgumentError(
        "passes must equal 1: posterior-as-prior replay would double-count " *
        "the same observations in this RxInfer model"))
    data = load_flattened_image_dataset(
        dataset;
        ntrain,
        nval,
        ntest,
        seed,
        image_size,
        split_sampling,
        classes,
    )
    priors = make_precision_classifier_priors(
        size(data.train_x, 1),
        hidden_count,
        data.classes;
        seed=seed + 20,
    )
    rng = MersenneTwister(seed + 30)
    order = shuffle(rng, collect(eachindex(data.train_y)))
    starts = collect(1:batch_size:length(order))
    progress = Progress(
        length(starts);
        desc="precision-head $optimizer ",
    )
    batch_seconds = Float64[]

    println(
        "RxInfer ManyPlus ResidualSine precision classifier ",
        "dataset=$(data.name) optimizer=$optimizer hidden=$hidden_count",
    )
    println(
        "train=$(length(data.train_y)) val=$(length(data.val_y)) ",
        "test=$(length(data.test_y)) split_sampling=$split_sampling ",
        "batch=$batch_size passes=$passes inference_iterations=$inference_iterations",
    )
    println(
        "likelihood=Gaussian-onehot-surrogate ",
        "precision_head=shared-scalar-per-image",
    )

    for start_index in starts
        indices =
            order[start_index:min(start_index + batch_size - 1, end)]
        features = [
            collect(@view(data.train_x[:, index]))
            for index in indices
        ]
        elapsed = @elapsed begin
            priors, _ = infer_precision_classifier_batch(
                priors,
                features,
                data.train_y[indices];
                hidden_count,
                optimizer,
                residual_sine_rho,
                residual_sine_omega,
                iterations=inference_iterations,
                limit_stack_depth,
            )
        end
        push!(batch_seconds, elapsed)
        ProgressMeter.next!(progress)
    end

    train_metrics = evaluate_precision_classifier(
        priors,
        data.train_x,
        data.train_y;
        max_images=min(eval_max_images, length(data.train_y)),
        residual_sine_rho,
        residual_sine_omega,
    )
    val_metrics = evaluate_precision_classifier(
        priors,
        data.val_x,
        data.val_y;
        max_images=min(eval_max_images, length(data.val_y)),
        residual_sine_rho,
        residual_sine_omega,
    )
    test_metrics = evaluate_precision_classifier(
        priors,
        data.test_x,
        data.test_y;
        max_images=min(eval_max_images, length(data.test_y)),
        residual_sine_rho,
        residual_sine_omega,
    )
    println(
        "train_acc=$(round(train_metrics.accuracy, digits=6)) ",
        "val_acc=$(round(val_metrics.accuracy, digits=6)) ",
        "test_acc=$(round(test_metrics.accuracy, digits=6)) ",
        "test_nll=$(round(test_metrics.nll, digits=6)) ",
        "test_mean_precision=$(round(test_metrics.mean_precision, digits=6)) ",
        "elapsed_seconds=$(round(sum(batch_seconds), digits=3))",
    )
    return (;
        priors,
        data,
        train_metrics,
        val_metrics,
        test_metrics,
        batch_seconds,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    error(
        "experimental precision-head graph: RxInfer does not yet produce a " *
        "complete update schedule; use run_residual_sine_mnist_sweep.sh for " *
        "the validated ResidualSine classifiers",
    )
end
