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
#   a[n,h]      = dot(W[h, :], input[n, :])
#   x[n,h]      = softplus(a[n,h])
#   logits[n,c] = c[c] + dot(U[c, :], x[n, :])
#   label[n]    = softmax(logits[n, :])
#
# This file uses the same flattened MNIST loader and same parameter shapes as
# the RxInfer MLP, but trains the means directly with cross-entropy backprop and
# Adam. It is dependency-free beyond the packages already used by the demos.
# ============================================================================

mutable struct MLPAdamState
    m_w::Matrix{Float64}
    v_w::Matrix{Float64}
    m_u::Matrix{Float64}
    v_u::Matrix{Float64}
    m_c::Vector{Float64}
    v_c::Vector{Float64}
    t::Int
end

function MLPAdamState(model)
    return MLPAdamState(
        zeros(size(model.w_m)), zeros(size(model.w_m)),
        zeros(size(model.u_m)), zeros(size(model.u_m)),
        zeros(size(model.c_m)), zeros(size(model.c_m)),
        0,
    )
end

function init_nn_mlp(input_count, hidden_count, classes;
    w_init_scale=0.01, u_init_scale=0.1, seed=1)
    return init_mlp_priors(input_count, hidden_count, classes;
        w_init_scale, u_init_scale, seed)
end

copy_nn_mlp(model) = copy_mlp_priors(model)

function forward_nn_mlp(model, batch_x)
    input_count, n_count = size(batch_x)
    @assert input_count == size(model.w_m, 2)
    hidden_count = size(model.w_m, 1)
    class_count = length(model.classes)

    preactivation = zeros(Float64, hidden_count, n_count)
    hidden = zeros(Float64, hidden_count, n_count)
    logits = zeros(Float64, class_count, n_count)
    probs = zeros(Float64, class_count, n_count)

    preactivation .= model.w_m * batch_x
    hidden .= softplus.(preactivation)
    logits .= model.c_m .+ model.u_m * hidden
    for n in 1:n_count
        probs[:, n] .= softmax_probs(@view logits[:, n])
    end

    return (preactivation=preactivation, hidden=hidden,
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

    grad_u = dlogits * transpose(cache.hidden)
    grad_c = vec(sum(dlogits; dims=2))
    dhidden = transpose(model.u_m) * dlogits
    dpreactivation = dhidden .* sigmoid.(cache.preactivation)
    grad_w = dpreactivation * transpose(batch_x)

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
    nn_mlp_adam_update!(model.w_m, grads.w, opt.m_w, opt.v_w, opt.t; lr, weight_decay)
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

function train_nn_mlp_demo(; ntrain=1000, nval=100, ntest=100,
    hidden_count=32,
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
        w_init_scale, u_init_scale, seed=seed + 20)
    opt = MLPAdamState(model)
    rng = MersenneTwister(seed + 30)

    println("Neural flattened softplus MLP MNIST classes=$(classes)")
    println("train=$(length(data.train_y)) val=$(length(data.val_y)) test=$(length(data.test_y)) input=$input_count hidden=$hidden_count batch=$batch_size epochs=$epochs lr=$lr weight_decay=$weight_decay w_init_scale=$w_init_scale u_init_scale=$u_init_scale")

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
