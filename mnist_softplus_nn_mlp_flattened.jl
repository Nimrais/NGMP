using LinearAlgebra
using Random
using Statistics
using ProgressMeter

include("mnist_softplus_rxinfer_mlp_flattened.jl")

# ============================================================================
# Neural-network baseline for the flattened RxInfer MLP.
#
# Architecture matches mnist_softplus_rxinfer_mlp_flattened.jl:
#
#   input[n,k]  = vec(downsample14(image))[k]
#   x[0]        = input
#   a[l]        = W[l] * x[l - 1]
#   x[l]        = softplus(a[l])
#   logits      = c + U * x[end]
#   label[n]    = softmax(logits[n, :])
#
# This file uses the same flattened MNIST loader and same parameter shapes as
# the RxInfer MLP, but trains the means directly with cross-entropy backprop and
# Adam. It is dependency-free beyond the packages already used by the demos.
# ============================================================================

struct NNMLP
    hidden_weights::Vector{Matrix{Float64}}
    u_m::Matrix{Float64}
    c_m::Vector{Float64}
    classes::Vector{Int}
end

mutable struct MLPAdamState
    m_w::Vector{Matrix{Float64}}
    v_w::Vector{Matrix{Float64}}
    m_u::Matrix{Float64}
    v_u::Matrix{Float64}
    m_c::Vector{Float64}
    v_c::Vector{Float64}
    t::Int
end

function MLPAdamState(model)
    return MLPAdamState(
        [zeros(size(w)) for w in model.hidden_weights],
        [zeros(size(w)) for w in model.hidden_weights],
        zeros(size(model.u_m)), zeros(size(model.u_m)),
        zeros(size(model.c_m)), zeros(size(model.c_m)),
        0,
    )
end

function init_nn_mlp(input_count, hidden_count, classes;
    hidden_layers=1,
    w_init_scale=0.01, u_init_scale=0.1, seed=1)
    hidden_layers >= 1 || throw(ArgumentError("hidden_layers must be at least 1"))
    hidden_count >= 1 || throw(ArgumentError("hidden_count must be at least 1"))
    rng = MersenneTwister(seed)
    weights = Matrix{Float64}[]
    push!(weights, w_init_scale .* randn(rng, hidden_count, input_count))
    for _ in 2:hidden_layers
        push!(weights, w_init_scale .* randn(rng, hidden_count, hidden_count))
    end
    return NNMLP(
        weights,
        u_init_scale .* randn(rng, length(classes), hidden_count),
        zeros(length(classes)),
        collect(classes),
    )
end

copy_nn_mlp(model) = NNMLP(
    copy.(model.hidden_weights), copy(model.u_m), copy(model.c_m), copy(model.classes))

function forward_nn_mlp(model, batch_x)
    input_count, n_count = size(batch_x)
    @assert input_count == size(first(model.hidden_weights), 2)
    class_count = length(model.classes)

    preactivations = Matrix{Float64}[]
    hidden_activations = Matrix{Float64}[]
    activation = batch_x
    for weights in model.hidden_weights
        preactivation = weights * activation
        activation = softplus.(preactivation)
        push!(preactivations, preactivation)
        push!(hidden_activations, activation)
    end

    logits = model.c_m .+ model.u_m * activation
    probs = zeros(Float64, class_count, n_count)
    for n in 1:n_count
        probs[:, n] .= softmax_probs(@view logits[:, n])
    end

    return (preactivations=preactivations, hidden_activations=hidden_activations,
        preactivation=last(preactivations), hidden=last(hidden_activations),
        logits=logits, probs=probs)
end

function nn_mlp_loss_and_grads(model, batch_x, batch_y)
    cache = forward_nn_mlp(model, batch_x)
    _, n_count = size(batch_x)
    class_to_index = Dict(label => i for (i, label) in enumerate(model.classes))

    dlogits = copy(cache.probs)
    loss = 0.0
    correct = 0
    for n in 1:n_count
        cls = class_to_index[batch_y[n]]
        loss -= log(max(cache.probs[cls, n], 1e-12))
        dlogits[cls, n] -= 1.0
        correct += model.classes[argmax(@view cache.probs[:, n])] == batch_y[n]
    end
    loss /= n_count
    dlogits ./= n_count

    grad_u = dlogits * transpose(last(cache.hidden_activations))
    grad_c = vec(sum(dlogits; dims=2))
    dhidden = transpose(model.u_m) * dlogits
    grad_w = Vector{Matrix{Float64}}(undef, length(model.hidden_weights))
    for layer in length(model.hidden_weights):-1:1
        dpreactivation = dhidden .* sigmoid.(cache.preactivations[layer])
        previous = layer == 1 ? batch_x : cache.hidden_activations[layer - 1]
        grad_w[layer] = dpreactivation * transpose(previous)
        if layer > 1
            dhidden = transpose(model.hidden_weights[layer]) * dpreactivation
        end
    end

    return loss, (w=grad_w, u=grad_u, c=grad_c), correct / n_count
end

function nn_mlp_adam_update!(param, grad, m, v, t;
    lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
    if weight_decay > 0
        grad = grad .+ weight_decay .* param
    end
    m .= beta1 .* m .+ (1 - beta1) .* grad
    v .= beta2 .* v .+ (1 - beta2) .* (grad .^ 2)
    mhat = m ./ (1 - beta1^t)
    vhat = v ./ (1 - beta2^t)
    param .-= lr .* mhat ./ (sqrt.(vhat) .+ eps)
    return nothing
end

function train_nn_mlp_batch!(model, opt, batch_x, batch_y;
    lr=1e-3, weight_decay=0.0)
    loss, grads, acc = nn_mlp_loss_and_grads(model, batch_x, batch_y)
    opt.t += 1
    for layer in eachindex(model.hidden_weights)
        nn_mlp_adam_update!(model.hidden_weights[layer], grads.w[layer],
            opt.m_w[layer], opt.v_w[layer], opt.t; lr, weight_decay)
    end
    nn_mlp_adam_update!(model.u_m, grads.u, opt.m_u, opt.v_u, opt.t; lr, weight_decay)
    nn_mlp_adam_update!(model.c_m, grads.c, opt.m_c, opt.v_c, opt.t; lr, weight_decay)
    return (loss=loss, acc=acc)
end

function predict_batch_nn_mlp(model, batch_x)
    cache = forward_nn_mlp(model, batch_x)
    n_count = size(batch_x, 2)
    labels = zeros(Int, n_count)
    for n in 1:n_count
        labels[n] = model.classes[argmax(@view cache.probs[:, n])]
    end
    return labels, cache.probs
end

function evaluate_nn_mlp(model, x, y; max_images=size(x, 2), batch_size=256)
    n = min(max_images, size(x, 2))
    n == 0 && return 0.0
    correct = 0
    seen = 0
    for start_idx in 1:batch_size:n
        stop_idx = min(start_idx + batch_size - 1, n)
        labels, _ = predict_batch_nn_mlp(model, x[:, start_idx:stop_idx])
        for (j, idx) in enumerate(start_idx:stop_idx)
            correct += labels[j] == y[idx]
            seen += 1
        end
    end
    return correct / seen
end

function train_nn_mlp_demo(; ntrain=10000, nval=1000, ntest=1000,
    hidden_count=32,
    hidden_layers=7,
    batch_size=32,
    epochs=50,
    seed=1,
    lr=1e-3,
    weight_decay=0.0,
    w_init_scale=0.01,
    u_init_scale=0.1,
    classes=collect(0:9),
    eval_max_images=1000)
    data = select_flattened_mnist(; ntrain, nval, ntest, seed, digits=classes)
    input_count = size(data.train_x, 1)
    model = init_nn_mlp(input_count, hidden_count, classes;
        hidden_layers,
        w_init_scale, u_init_scale, seed=seed + 20)
    opt = MLPAdamState(model)
    rng = MersenneTwister(seed + 30)

    println("Neural flattened softplus MLP MNIST classes=$(classes)")
    println("train=$(length(data.train_y)) val=$(length(data.val_y)) test=$(length(data.test_y)) input=$input_count hidden_layers=$hidden_layers hidden_count=$hidden_count batch=$batch_size epochs=$epochs lr=$lr weight_decay=$weight_decay w_init_scale=$w_init_scale u_init_scale=$u_init_scale")

    history = NamedTuple[]
    best_val_acc = -Inf
    best_model = copy_nn_mlp(model)
    best_epoch = 0

    for epoch in 1:epochs
        order = shuffle(rng, collect(1:length(data.train_y)))
        starts = collect(1:batch_size:length(order))
        progress = Progress(length(starts); desc="nn mlp epoch $epoch/$epochs ")
        losses = Float64[]
        batch_accs = Float64[]
        for start_idx in starts
            inds = order[start_idx:min(start_idx + batch_size - 1, end)]
            stats = train_nn_mlp_batch!(model, opt, data.train_x[:, inds], data.train_y[inds];
                lr, weight_decay)
            push!(losses, stats.loss)
            push!(batch_accs, stats.acc)
            ProgressMeter.next!(progress)
        end

        train_acc = evaluate_nn_mlp(model, data.train_x, data.train_y;
            max_images=min(eval_max_images, length(data.train_y)))
        val_acc = evaluate_nn_mlp(model, data.val_x, data.val_y;
            max_images=min(eval_max_images, length(data.val_y)))
        push!(history, (epoch=epoch, loss=mean(losses),
            batch_acc=mean(batch_accs), train_acc=train_acc, val_acc=val_acc))
        println("epoch=$epoch loss=$(round(mean(losses), digits=4)) batch_acc=$(round(mean(batch_accs), digits=3)) train_acc=$(round(train_acc, digits=3)) val_acc=$(round(val_acc, digits=3))")

        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_model = copy_nn_mlp(model)
            best_epoch = epoch
        end
    end

    test_acc = evaluate_nn_mlp(best_model, data.test_x, data.test_y;
        max_images=min(eval_max_images, length(data.test_y)))
    println("best_val_epoch=$best_epoch best_val_acc=$(round(best_val_acc, digits=3)) test_acc=$(round(test_acc, digits=3))")
    return (model=best_model, history=history, data=data, test_acc=test_acc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_nn_mlp_demo()
end
