using RxInfer
using ExponentialFamily
using LinearAlgebra
using MLDatasets
using ProbabilisticEnsembling: LowRankMeta
using ProgressMeter
using Random
using Statistics
using SurrogateModelling

# Native NGMP version of the flattened MNIST experiment. Unlike
# mnist_softplus_rxinfer_mlp_flattened.jl, this model puts Softplus directly in
# the RxInfer graph and lets NGMPDependencies refresh its messages.

const DEFAULT_CLASSES = collect(0:9)

stable_softplus(x::Real) = max(x, zero(x)) + log1p(exp(-abs(x)))

function optimizer_settings(optimizer::Symbol; alpha, beta, max_step)
    optimizer === :damped && return (alpha=alpha, beta=0.0, max_step=max_step)
    optimizer === :momentum && return (alpha=alpha, beta=beta, max_step=Inf)
    optimizer === :bounded_momentum && return (alpha=alpha, beta=beta, max_step=max_step)
    throw(ArgumentError(
        "unknown optimizer $optimizer; use :damped, :momentum, or :bounded_momentum",
    ))
end

function projection_strategy(projection::Symbol; quadrature_points::Int=32)
    projection === :unscented && return TangentProjection(type=Unscented)
    projection === :delta && return TangentProjection(type=DeltaApproximation)
    projection === :quadrature && return TangentProjection(type=Quadrature(quadrature_points))
    throw(ArgumentError(
        "unknown projection $projection; use :unscented, :delta, or :quadrature",
    ))
end

@model function mnist_softplus_ut_ngmp_model(
    features,
    targets,
    hidden_count,
    class_count,
    priors,
    softplus_dependencies,
    damping,
    observation_dependencies,
    observation_damping,
)
    local hidden_weight, gate_weight, hidden_mean, gate_score, gate_precision, output

    hidden_precision ~ priors.hidden_precision
    gate_score_precision ~ priors.gate_score_precision
    observation_precision ~ priors.observation_precision
    gate_rate ~ priors.gate_rate

    for cls in 1:class_count, hidden in 1:hidden_count
        hidden_weight[cls, hidden] ~ priors.hidden_weight[cls, hidden]
    end
    for cls in 1:class_count, hidden in 1:hidden_count
        gate_weight[cls, hidden] ~ priors.gate_weight[cls, hidden]
    end

    for observation in eachindex(features)
        for cls in 1:class_count, hidden in 1:hidden_count
            gate_score[cls, hidden, observation] ~ softdot(
                features[observation],
                gate_weight[cls, hidden],
                gate_score_precision,
            ) where { meta=LowRankMeta() }
            gate_precision[cls, hidden, observation] ~ GammaShapeRate(1.0, gate_rate)
            gate_precision[cls, hidden, observation] ~ Softplus(
                gate_score[cls, hidden, observation],
            ) where {
                dependencies=softplus_dependencies,
                meta=damping,
            }

            hidden_mean[cls, hidden, observation] ~ softdot(
                features[observation],
                hidden_weight[cls, hidden],
                hidden_precision,
            ) where { meta=LowRankMeta() }
            output[cls, observation] ~ NormalMeanPrecision(
                hidden_mean[cls, hidden, observation],
                gate_precision[cls, hidden, observation],
            ) where {
                dependencies=observation_dependencies,
                meta=observation_damping,
            }
        end

        for cls in 1:class_count
            targets[cls, observation] ~ NormalMeanPrecision(
                output[cls, observation],
                observation_precision,
            )
        end
    end
end

@constraints function mnist_softplus_ut_ngmp_constraints()
    # Standard (non-relaxed) NGMP factorization, analogous to ETTh2's
    # q(w)q(z, gamma): global weights remain separate from per-observation
    # nonlinear latent clusters. A relaxed form would instead join each weight
    # bank with hidden_mean/output or gate_score/gate_precision.
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

@initialization function mnist_softplus_ut_ngmp_initialization(
    priors,
    hidden_count,
    class_count,
    observation_count,
)
    q(gate_weight) = deepcopy(priors.gate_weight)
    q(hidden_weight) = deepcopy(priors.hidden_weight)
    q(hidden_mean) = fill(
        NormalMeanVariance(0.0, 1.0),
        class_count,
        hidden_count,
        observation_count,
    )
    q(output) = fill(NormalMeanVariance(0.0, 1.0), class_count, observation_count)
    q(gate_score) = fill(
        NormalMeanVariance(0.0, 1.0),
        class_count,
        hidden_count,
        observation_count,
    )
    q(gate_precision) = fill(
        GammaShapeScale(2.0, 1.0),
        class_count,
        hidden_count,
        observation_count,
    )
    q(hidden_precision) = priors.hidden_precision
    q(gate_score_precision) = priors.gate_score_precision
    q(observation_precision) = priors.observation_precision
    q(gate_rate) = priors.gate_rate
    μ(hidden_weight) = deepcopy(priors.hidden_weight)
end

function make_ngmp_mlp_priors(
    input_count,
    hidden_count,
    class_count;
    seed=42,
    hidden_weight_precision=1e-2,
    gate_weight_precision=1.0,
)
    rng = MersenneTwister(seed)
    hidden_lambda = Diagonal(fill(hidden_weight_precision, input_count))
    gate_lambda = Diagonal(fill(gate_weight_precision, input_count))

    hidden_weight = [
        MvNormalWeightedMeanPrecision(
            hidden_lambda * (0.01 .* randn(rng, input_count)),
            hidden_lambda,
        ) for cls in 1:class_count, hidden in 1:hidden_count
    ]
    gate_weight = [
        MvNormalWeightedMeanPrecision(
            gate_lambda * (0.01 .* randn(rng, input_count)),
            gate_lambda,
        ) for cls in 1:class_count, hidden in 1:hidden_count
    ]

    return (
        hidden_weight=hidden_weight,
        gate_weight=gate_weight,
        hidden_precision=GammaShapeRate(1e4, 1.0),
        gate_score_precision=GammaShapeRate(1e3, 1.0),
        observation_precision=GammaShapeRate(1e3, 1.0),
        gate_rate=GammaShapeRate(10.0, 10.0),
    )
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

    error("Could not construct a positive-definite prior from the batch posterior")
end

function posterior_priors(result, old_priors)
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
        hidden_weight=reshape(
            hidden_weight,
            size(old_priors.hidden_weight),
        ),
        gate_weight=reshape(
            gate_weight,
            size(old_priors.gate_weight),
        ),
        hidden_precision=deepcopy(last(posteriors[:hidden_precision])),
        gate_score_precision=deepcopy(last(posteriors[:gate_score_precision])),
        observation_precision=deepcopy(last(posteriors[:observation_precision])),
        gate_rate=deepcopy(last(posteriors[:gate_rate])),
    )
end

function infer_ngmp_mlp_batch(
    priors,
    features,
    targets;
    hidden_count,
    class_count,
    iterations=3,
    optimizer=:damped,
    projection=:unscented,
    alpha=0.2,
    beta=0.0,
    max_step=1.0,
    quadrature_points=32,
)
    settings = optimizer_settings(optimizer; alpha, beta, max_step)
    dependencies = NGMPDependencies(
        out=nothing,
        in=nothing,
        projection=projection_strategy(projection; quadrature_points),
    )
    damping = DampingMeta(; settings...)
    observation_dependencies = NGMPDependencies(
        τ=nothing,
        projection=projection_strategy(projection; quadrature_points),
    )
    observation_damping = DampingMeta(; settings...)
    observation_count = length(features)

    result = infer(
        model=mnist_softplus_ut_ngmp_model(
            hidden_count=hidden_count,
            class_count=class_count,
            priors=priors,
            softplus_dependencies=dependencies,
            damping=damping,
            observation_dependencies=observation_dependencies,
            observation_damping=observation_damping,
        ),
        data=(features=features, targets=targets),
        constraints=mnist_softplus_ut_ngmp_constraints(),
        initialization=mnist_softplus_ut_ngmp_initialization(
            priors,
            hidden_count,
            class_count,
            observation_count,
        ),
        iterations=iterations,
        options=(limit_stack_depth=500,),
        disable_inference_error_hint=true,
    )

    @assert !isempty(observation_dependencies.states)
    @assert all(
        state -> state.usermeta isa DampingMeta,
        observation_dependencies.states,
    ) "NormalMeanPrecision NGMP edges were activated without DampingMeta"

    return posterior_priors(result, priors), result
end

function downsample14(image)
    output = zeros(Float64, 14, 14)
    for row in 1:14, col in 1:14
        output[row, col] = mean(@view image[(2row-1):(2row), (2col-1):(2col)])
    end
    return output
end

function balanced_indices(labels, count, classes, rng; excluded=Set{Int}())
    per_class = div(count, length(classes))
    remainder = count - per_class * length(classes)
    selected = Int[]
    for (position, cls) in enumerate(classes)
        available = [index for index in eachindex(labels) if labels[index] == cls && index ∉ excluded]
        shuffle!(rng, available)
        take = per_class + (position <= remainder ? 1 : 0)
        append!(selected, available[1:min(take, length(available))])
    end
    shuffle!(rng, selected)
    return selected
end

function materialize_split(images, labels, indices, classes)
    features = Vector{Vector{Float64}}(undef, length(indices))
    targets = zeros(Float64, length(classes), length(indices))
    class_to_index = Dict(cls => index for (index, cls) in enumerate(classes))
    output_labels = Vector{Int}(undef, length(indices))

    for (position, index) in enumerate(indices)
        pixels = vec(downsample14(@view images[:, :, index]))
        features[position] = vcat(1.0, pixels)
        output_labels[position] = labels[index]
        targets[class_to_index[labels[index]], position] = 1.0
    end
    return (features=features, targets=targets, labels=output_labels)
end

function load_mnist_splits(; ntrain=1000, nval=100, ntest=100, classes=DEFAULT_CLASSES, seed=1)
    rng = MersenneTwister(seed)
    train_data = MNIST(split=:train)
    test_data = MNIST(split=:test)
    train_images = Float64.(train_data.features)
    train_labels = Int.(train_data.targets)
    test_images = Float64.(test_data.features)
    test_labels = Int.(test_data.targets)

    train_indices = balanced_indices(train_labels, ntrain, classes, rng)
    val_indices = balanced_indices(
        train_labels,
        nval,
        classes,
        rng;
        excluded=Set(train_indices),
    )
    test_indices = balanced_indices(test_labels, ntest, classes, rng)

    return (
        train=materialize_split(train_images, train_labels, train_indices, classes),
        val=materialize_split(train_images, train_labels, val_indices, classes),
        test=materialize_split(test_images, test_labels, test_indices, classes),
    )
end

function weight_means(priors)
    hidden = map(mean, priors.hidden_weight)
    gates = map(mean, priors.gate_weight)
    return hidden, gates
end

function predict_ngmp_mlp(priors, feature, classes)
    hidden_weights, gate_weights = weight_means(priors)
    hidden_count = size(gate_weights, 2)
    class_count = length(classes)
    scores = zeros(class_count)
    for cls in 1:class_count
        gates = [
            stable_softplus(dot(gate_weights[cls, hidden], feature)) for
            hidden in 1:hidden_count
        ]
        denominator = max(sum(gates), eps(Float64))
        scores[cls] = sum(
            gates[hidden] * dot(hidden_weights[cls, hidden], feature) for
            hidden in 1:hidden_count
        ) / denominator
    end
    return classes[argmax(scores)], scores
end

function evaluate_ngmp_mlp(priors, split, classes)
    correct = count(eachindex(split.labels)) do index
        prediction, _ = predict_ngmp_mlp(priors, split.features[index], classes)
        prediction == split.labels[index]
    end
    return correct / length(split.labels)
end

function train_mnist_softplus_ut_ngmp_mlp(
    ;
    ntrain=1000,
    nval=100,
    ntest=100,
    hidden_count=32,
    batch_size=32,
    epochs=10,
    inference_iterations=3,
    classes=DEFAULT_CLASSES,
    seed=1,
    optimizer=:damped,
    projection=:unscented,
    alpha=0.2,
    beta=0.0,
    max_step=1.0,
    quadrature_points=32,
)
    data = load_mnist_splits(; ntrain, nval, ntest, classes, seed)
    input_count = length(first(data.train.features))
    class_count = length(classes)
    priors = make_ngmp_mlp_priors(
        input_count,
        hidden_count,
        class_count;
        seed=seed + 10,
    )
    best_priors = deepcopy(priors)
    best_val_acc = -Inf
    best_epoch = 0
    history = NamedTuple[]
    rng = MersenneTwister(seed + 20)

    println("Native RxInfer Softplus UT NGMP MLP-like MNIST classes=$classes factorization=standard_ngmp")
    println("train=$ntrain val=$nval test=$ntest input=$input_count hidden=$hidden_count batch=$batch_size epochs=$epochs iterations=$inference_iterations optimizer=$optimizer projection=$projection alpha=$alpha beta=$beta max_step=$max_step")

    for epoch in 1:epochs
        order = shuffle(rng, collect(eachindex(data.train.labels)))
        starts = 1:batch_size:length(order)
        progress = Progress(length(starts); desc="ngmp mlp epoch $epoch/$epochs ")

        for first_index in starts
            indices = order[first_index:min(first_index + batch_size - 1, end)]
            features = data.train.features[indices]
            targets = data.train.targets[:, indices]
            priors, _ = infer_ngmp_mlp_batch(
                priors,
                features,
                targets;
                hidden_count,
                class_count,
                iterations=inference_iterations,
                optimizer,
                projection,
                alpha,
                beta,
                max_step,
                quadrature_points,
            )
            ProgressMeter.next!(progress)
        end

        train_acc = evaluate_ngmp_mlp(priors, data.train, classes)
        val_acc = evaluate_ngmp_mlp(priors, data.val, classes)
        push!(history, (epoch=epoch, train_acc=train_acc, val_acc=val_acc))
        println("epoch=$epoch train_acc=$(round(train_acc, digits=3)) val_acc=$(round(val_acc, digits=3))")

        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_epoch = epoch
            best_priors = deepcopy(priors)
        end
    end

    test_acc = evaluate_ngmp_mlp(best_priors, data.test, classes)
    println("best_val_epoch=$best_epoch best_val_acc=$(round(best_val_acc, digits=3)) test_acc=$(round(test_acc, digits=3))")
    return (priors=best_priors, history=history, data=data, test_acc=test_acc)
end

function smoke_test()
    features = [[1.0, -0.5], [1.0, 0.5]]
    targets = [1.0 0.0; 0.0 1.0]
    priors = make_ngmp_mlp_priors(2, 2, 2; seed=7)
    updated, result = infer_ngmp_mlp_batch(
        priors,
        features,
        targets;
        hidden_count=2,
        class_count=2,
        iterations=2,
        optimizer=:damped,
    )
    @assert size(updated.hidden_weight) == (2, 2)
    @assert size(updated.gate_weight) == (2, 2)
    @assert haskey(result.posteriors, :gate_precision)
    println("smoke_test=passed")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--smoke-test" in ARGS
        smoke_test()
    else
        train_mnist_softplus_ut_ngmp_mlp()
    end
end
