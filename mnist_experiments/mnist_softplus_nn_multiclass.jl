using LinearAlgebra
using Random
using Statistics
using ProgressMeter

include(joinpath(@__DIR__, "mnist_softplus_rxinfer_surrogate_multiclass.jl"))

# ============================================================================
# Neural-network baseline with the same global parameter shapes as the
# RxInfer surrogate MNIST model:
#
#   local_kernel[feature, patch_pixel]
#   classifier_weight[class, pooled_feature]
#   classifier_bias[class]
#
# Forward pass:
#
#   feature_strength[n,p,f] = softplus(dot(local_kernel[f, :], patch[n,p,:]))
#   pooled[n,k]            = mean over spatial bin/feature
#   label[n]               = softmax(classifier_bias + classifier_weight * pooled[n])
#
# This is deliberately dependency-free, so it can run in the current project
# without adding Flux/Lux. Gradients are computed explicitly and optimized with
# Adam.
# ============================================================================

mutable struct AdamState
    m_b::Matrix{Float64}
    v_b::Matrix{Float64}
    m_u::Matrix{Float64}
    v_u::Matrix{Float64}
    m_c::Vector{Float64}
    v_c::Vector{Float64}
    t::Int
end

function AdamState(model)
    return AdamState(
        zeros(size(model.b_m)), zeros(size(model.b_m)),
        zeros(size(model.u_m)), zeros(size(model.u_m)),
        zeros(size(model.c_m)), zeros(size(model.c_m)),
        0,
    )
end

function init_nn_model(num_features, num_spatial_bins;
    r_count=4, classes=collect(0:9), seed=1)
    return init_priors(num_features, num_spatial_bins;
        r_count, classes, seed)
end

copy_nn_model(model) = copy_priors(model)

function label_index_map(classes)
    return Dict(label => i for (i, label) in enumerate(classes))
end

function forward_nn(model, batch_x; num_spatial_bins=49)
    p_count, r_count, n_count = size(batch_x)
    f_count = size(model.b_m, 1)
    class_count = length(model.classes)
    @assert r_count == size(model.b_m, 2)

    bin_ids = patch_bin_ids(p_count; num_bins=num_spatial_bins)
    bin_counts = [count(==(q), bin_ids) for q in 1:num_spatial_bins]

    preactivation = zeros(Float64, n_count, p_count, f_count)
    strength = zeros(Float64, n_count, p_count, f_count)
    pooled = zeros(Float64, n_count, f_count * num_spatial_bins)
    logits = zeros(Float64, n_count, class_count)
    probs = zeros(Float64, n_count, class_count)

    for n in 1:n_count, p in 1:p_count, f in 1:f_count
        a = 0.0
        for r in 1:r_count
            a += model.b_m[f, r] * batch_x[p, r, n]
        end
        preactivation[n, p, f] = a
        strength[n, p, f] = softplus(a)
    end

    for n in 1:n_count, p in 1:p_count, f in 1:f_count
        q = bin_ids[p]
        k = (f - 1) * num_spatial_bins + q
        pooled[n, k] += strength[n, p, f] / bin_counts[q]
    end

    for n in 1:n_count
        logits[n, :] .= model.c_m .+ model.u_m * @view pooled[n, :]
        probs[n, :] .= softmax_probs(@view logits[n, :])
    end

    return (preactivation=preactivation, strength=strength,
        pooled=pooled, logits=logits, probs=probs,
        bin_ids=bin_ids, bin_counts=bin_counts)
end

function nn_loss_and_grads(model, batch_x, batch_y; num_spatial_bins=49)
    cache = forward_nn(model, batch_x; num_spatial_bins)
    p_count, r_count, n_count = size(batch_x)
    f_count = size(model.b_m, 1)
    class_to_index = label_index_map(model.classes)

    dlogits = copy(cache.probs)
    loss = 0.0
    correct = 0
    for n in 1:n_count
        cls = class_to_index[batch_y[n]]
        loss -= log(max(cache.probs[n, cls], 1e-12))
        dlogits[n, cls] -= 1.0
        correct += model.classes[argmax(@view cache.probs[n, :])] == batch_y[n]
    end
    loss /= n_count
    dlogits ./= n_count

    grad_u = zeros(size(model.u_m))
    grad_c = vec(sum(dlogits; dims=1))
    dpooled = zeros(size(cache.pooled))
    for n in 1:n_count
        grad_u .+= dlogits[n, :] * transpose(@view cache.pooled[n, :])
        dpooled[n, :] .= transpose(model.u_m) * @view dlogits[n, :]
    end

    dstrength = zeros(Float64, n_count, p_count, f_count)
    for n in 1:n_count, p in 1:p_count, f in 1:f_count
        q = cache.bin_ids[p]
        k = (f - 1) * num_spatial_bins + q
        dstrength[n, p, f] += dpooled[n, k] / cache.bin_counts[q]
    end

    grad_b = zeros(size(model.b_m))
    for n in 1:n_count, p in 1:p_count, f in 1:f_count
        da = dstrength[n, p, f] * sigmoid(cache.preactivation[n, p, f])
        for r in 1:r_count
            grad_b[f, r] += da * batch_x[p, r, n]
        end
    end

    return loss, (b=grad_b, u=grad_u, c=grad_c), correct / n_count
end

function adam_update!(param, grad, m, v, t;
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

function train_nn_batch!(model, opt, batch_x, batch_y;
    num_spatial_bins=49, lr=1e-3, weight_decay=0.0)
    loss, grads, acc = nn_loss_and_grads(model, batch_x, batch_y; num_spatial_bins)
    opt.t += 1
    adam_update!(model.b_m, grads.b, opt.m_b, opt.v_b, opt.t; lr, weight_decay)
    adam_update!(model.u_m, grads.u, opt.m_u, opt.v_u, opt.t; lr, weight_decay)
    adam_update!(model.c_m, grads.c, opt.m_c, opt.v_c, opt.t; lr, weight_decay)
    return (loss=loss, acc=acc)
end

function predict_batch_nn(model, batch_x; num_spatial_bins=49)
    cache = forward_nn(model, batch_x; num_spatial_bins)
    n_count = size(batch_x, 3)
    labels = zeros(Int, n_count)
    for n in 1:n_count
        labels[n] = model.classes[argmax(@view cache.probs[n, :])]
    end
    return labels, cache.probs
end

function evaluate_nn(model, x, y; max_images=size(x, 3), num_spatial_bins=49,
    batch_size=256)
    n = min(max_images, size(x, 3))
    n == 0 && return 0.0
    correct = 0
    seen = 0
    for start_idx in 1:batch_size:n
        stop_idx = min(start_idx + batch_size - 1, n)
        labels, _ = predict_batch_nn(model, x[:, :, start_idx:stop_idx]; num_spatial_bins)
        for (j, idx) in enumerate(start_idx:stop_idx)
            correct += labels[j] == y[idx]
            seen += 1
        end
    end
    return correct / seen
end

function train_nn_demo(; ntrain=10000, nval=1000, ntest=1000,
    num_features=4, num_spatial_bins=49,
    batch_size=32, epochs=50,
    seed=1, lr=1e-3, weight_decay=0.0,
    patch_side=2,
    classes=collect(0:9),
    eval_max_images=1000)
    data = select_mnist(; ntrain, nval, ntest, seed, patch_side, digits=classes)
    model = init_nn_model(num_features, num_spatial_bins;
        r_count=size(data.train_x, 2), classes, seed=seed + 10)
    opt = AdamState(model)
    rng = MersenneTwister(seed + 20)

    println("Neural softplus MNIST categorical classes=$(classes)")
    println("train=$(length(data.train_y)) val=$(length(data.val_y)) test=$(length(data.test_y)) F=$num_features bins=$num_spatial_bins patch_side=$patch_side batch=$batch_size epochs=$epochs lr=$lr weight_decay=$weight_decay")

    history = NamedTuple[]
    best_val_acc = -Inf
    best_model = copy_nn_model(model)
    best_epoch = 0
    for epoch in 1:epochs
        order = shuffle(rng, collect(1:length(data.train_y)))
        batch_starts = collect(1:batch_size:length(order))
        progress = Progress(length(batch_starts); desc="nn epoch $epoch/$epochs ")
        epoch_losses = Float64[]
        epoch_accs = Float64[]
        for start_idx in batch_starts
            inds = order[start_idx:min(start_idx + batch_size - 1, end)]
            stats = train_nn_batch!(model, opt, data.train_x[:, :, inds], data.train_y[inds];
                num_spatial_bins, lr, weight_decay)
            push!(epoch_losses, stats.loss)
            push!(epoch_accs, stats.acc)
            ProgressMeter.next!(progress)
        end
        train_acc = evaluate_nn(model, data.train_x, data.train_y;
            max_images=min(eval_max_images, length(data.train_y)), num_spatial_bins)
        val_acc = evaluate_nn(model, data.val_x, data.val_y;
            max_images=min(eval_max_images, length(data.val_y)), num_spatial_bins)
        push!(history, (epoch=epoch, loss=mean(epoch_losses),
            batch_acc=mean(epoch_accs), train_acc=train_acc, val_acc=val_acc))
        println("epoch=$epoch loss=$(round(mean(epoch_losses), digits=4)) batch_acc=$(round(mean(epoch_accs), digits=3)) train_acc=$(round(train_acc, digits=3)) val_acc=$(round(val_acc, digits=3))")
        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_model = copy_nn_model(model)
            best_epoch = epoch
        end
    end

    test_acc = evaluate_nn(best_model, data.test_x, data.test_y;
        max_images=min(eval_max_images, length(data.test_y)), num_spatial_bins)
    println("best_val_epoch=$best_epoch best_val_acc=$(round(best_val_acc, digits=3)) test_acc=$(round(test_acc, digits=3))")
    return (model=best_model, history=history, data=data, test_acc=test_acc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_nn_demo()
end
