using LinearAlgebra
using Random
using Statistics
using ProgressMeter

include("mnist_softplus_rxinfer_surrogate_multiclass.jl")

# ============================================================================
# Experimental deeper MNIST classifier on top of the existing softplus image
# feature model.
#
# Base script:
#   a1[n,p,f] ~ Normal(0, sigma_a1^2)
#   x1[n,p,f] ~= softplus(a1[n,p,f])
#   image[n,p,r] ~ Normal(dot(b[:,r], x1[n,p,:]), sigma_y2)
#   label[n] ~ Categorical(softmax(c + U * pooled_x1[n]))
#
# This script keeps the image feature inference from the base script and
# replaces the direct classifier with:
#
#   g[n,k]        = pooled_x1[n,k]
#   a2[n,h]       ~= Normal(dot(W[h,:], g[n,:]), sigma_hidden2)
#   x2[n,h]       ~= softplus(a2[n,h])
#   label[n]      ~ Categorical(softmax(c + U * x2[n,:]))
#
# The hidden classifier is trained with the same Gaussian-site idea: nonlinear
# and product factors are refreshed as local quadratic/Gaussian sites, then the
# global Gaussian priors over W, U, c are updated batch-by-batch.
# ============================================================================

function init_deep_classifier_priors(input_count, hidden_count, classes;
    w_var=1.0, u_var=1.0, c_var=4.0,
    seed=1)
    rng = MersenneTwister(seed)
    class_count = length(classes)
    w_m = 0.01 .* randn(rng, hidden_count, input_count)
    w_Λ = fill(inv(w_var), hidden_count, input_count)
    u_m = 0.01 .* randn(rng, class_count, hidden_count)
    u_Λ = fill(inv(u_var), class_count, hidden_count)
    return (w_m=w_m, w_Λ=w_Λ,
        u_m=u_m, u_Λ=u_Λ,
        c_m=zeros(class_count), c_Λ=fill(inv(c_var), class_count),
        classes=collect(classes))
end

copy_deep_priors(priors) = (
    w_m=copy(priors.w_m), w_Λ=copy(priors.w_Λ),
    u_m=copy(priors.u_m), u_Λ=copy(priors.u_Λ),
    c_m=copy(priors.c_m), c_Λ=copy(priors.c_Λ),
    classes=copy(priors.classes),
)

function pool_marginals(mx, vx, num_spatial_bins)
    n_count, p_count, f_count = size(mx)
    bin_ids = patch_bin_ids(p_count; num_bins=num_spatial_bins)
    bin_counts = [count(==(q), bin_ids) for q in 1:num_spatial_bins]
    input_count = f_count * num_spatial_bins
    gmean = zeros(n_count, input_count)
    gvar = zeros(n_count, input_count)
    for n in 1:n_count, f in 1:f_count, q in 1:num_spatial_bins
        idx = (f - 1) * num_spatial_bins + q
        count_q = bin_counts[q]
        for p in 1:p_count
            if bin_ids[p] == q
                gmean[n, idx] += mx[n, p, f] / count_q
                gvar[n, idx] += vx[n, p, f] / count_q^2
            end
        end
    end
    return gmean, gvar
end

function deep_predict_probs(priors, gmean)
    hidden_logits = priors.w_m * gmean
    hidden = softplus.(hidden_logits)
    logits = priors.c_m .+ priors.u_m * hidden
    return softmax_probs(logits), hidden
end

function deep_predict_label(priors, gmean)
    probs, _ = deep_predict_probs(priors, gmean)
    return priors.classes[argmax(probs)], probs
end

site_metric_array(site; damping=1e-6) = max.(abs.(site), damping)

function update_deep_site!(site, target, previous_update, previous_metric;
    alpha, vector_transport=false, momentum=0.8,
    damping=1e-6, precision=false)
    if vector_transport
        current_metric = site_metric_array(site; damping)
        transported_update = previous_update .* sqrt.(previous_metric ./ current_metric)
        next_update = momentum .* transported_update .+ alpha .* (target .- site)
        site .+= next_update
        previous_update .= next_update
        previous_metric .= site_metric_array(site; damping)
    else
        site .= (1 - alpha) .* site .+ alpha .* target
    end
    precision && (site .= max.(site, 0.0))
    return nothing
end

function infer_batch_deep_classifier(priors, gmean, gvar, labels;
    sigma_hidden2=0.05^2,
    alpha=0.08,
    global_site_scale=1.0,
    vector_transport=false,
    vector_transport_momentum=0.8,
    vector_transport_damping=1e-6,
    max_inner=12,
    tol=1e-4)
    n_count, input_count = size(gmean)
    hidden_count = size(priors.w_m, 1)
    class_count = length(priors.classes)

    w_xi_base = priors.w_m .* priors.w_Λ
    w_Λ_base = copy(priors.w_Λ)
    u_xi_base = priors.u_m .* priors.u_Λ
    u_Λ_base = copy(priors.u_Λ)
    c_xi_base = priors.c_m .* priors.c_Λ
    c_Λ_base = copy(priors.c_Λ)

    sp_a_xi = zeros(n_count, hidden_count)
    sp_a_Λ = zeros(n_count, hidden_count)
    sp_x_xi = zeros(n_count, hidden_count)
    sp_x_Λ = zeros(n_count, hidden_count)
    link_a_xi = zeros(n_count, hidden_count)
    link_a_Λ = zeros(n_count, hidden_count)
    link_w_xi = zeros(n_count, hidden_count, input_count)
    link_w_Λ = zeros(n_count, hidden_count, input_count)
    cls_x_xi = zeros(n_count, hidden_count)
    cls_x_Λ = zeros(n_count, hidden_count)
    cls_u_xi = zeros(n_count, class_count, hidden_count)
    cls_u_Λ = zeros(n_count, class_count, hidden_count)
    cls_c_xi = zeros(n_count, class_count)
    cls_c_Λ = zeros(n_count, class_count)

    prev_sp_a_xi = zero(sp_a_xi)
    metric_sp_a_xi = site_metric_array(sp_a_xi; damping=vector_transport_damping)
    prev_sp_a_Λ = zero(sp_a_Λ)
    metric_sp_a_Λ = site_metric_array(sp_a_Λ; damping=vector_transport_damping)
    prev_sp_x_xi = zero(sp_x_xi)
    metric_sp_x_xi = site_metric_array(sp_x_xi; damping=vector_transport_damping)
    prev_sp_x_Λ = zero(sp_x_Λ)
    metric_sp_x_Λ = site_metric_array(sp_x_Λ; damping=vector_transport_damping)
    prev_link_a_xi = zero(link_a_xi)
    metric_link_a_xi = site_metric_array(link_a_xi; damping=vector_transport_damping)
    prev_link_a_Λ = zero(link_a_Λ)
    metric_link_a_Λ = site_metric_array(link_a_Λ; damping=vector_transport_damping)
    prev_link_w_xi = zero(link_w_xi)
    metric_link_w_xi = site_metric_array(link_w_xi; damping=vector_transport_damping)
    prev_link_w_Λ = zero(link_w_Λ)
    metric_link_w_Λ = site_metric_array(link_w_Λ; damping=vector_transport_damping)
    prev_cls_x_xi = zero(cls_x_xi)
    metric_cls_x_xi = site_metric_array(cls_x_xi; damping=vector_transport_damping)
    prev_cls_x_Λ = zero(cls_x_Λ)
    metric_cls_x_Λ = site_metric_array(cls_x_Λ; damping=vector_transport_damping)
    prev_cls_u_xi = zero(cls_u_xi)
    metric_cls_u_xi = site_metric_array(cls_u_xi; damping=vector_transport_damping)
    prev_cls_u_Λ = zero(cls_u_Λ)
    metric_cls_u_Λ = site_metric_array(cls_u_Λ; damping=vector_transport_damping)
    prev_cls_c_xi = zero(cls_c_xi)
    metric_cls_c_xi = site_metric_array(cls_c_xi; damping=vector_transport_damping)
    prev_cls_c_Λ = zero(cls_c_Λ)
    metric_cls_c_Λ = site_metric_array(cls_c_Λ; damping=vector_transport_damping)

    ma2 = zeros(n_count, hidden_count)
    va2 = fill(1.0, n_count, hidden_count)
    mx2 = fill(softplus(0.0), n_count, hidden_count)
    vx2 = fill(0.25, n_count, hidden_count)
    last_delta = Inf

    local w_m, w_v, u_m, u_v, c_m, c_v
    for _ in 1:max_inner
        old_mx2 = copy(mx2)

        w_xi = w_xi_base .+ global_site_scale .* dropdims(sum(link_w_xi; dims=1), dims=1)
        w_Λ = max.(w_Λ_base .+ global_site_scale .* dropdims(sum(link_w_Λ; dims=1), dims=1), 1e-6)
        u_xi = u_xi_base .+ global_site_scale .* dropdims(sum(cls_u_xi; dims=1), dims=1)
        u_Λ = max.(u_Λ_base .+ global_site_scale .* dropdims(sum(cls_u_Λ; dims=1), dims=1), 1e-6)
        c_xi = c_xi_base .+ global_site_scale .* vec(sum(cls_c_xi; dims=1))
        c_Λ = max.(c_Λ_base .+ global_site_scale .* vec(sum(cls_c_Λ; dims=1)), 1e-6)

        w_m = w_xi ./ w_Λ
        w_v = inv.(w_Λ)
        u_m = u_xi ./ u_Λ
        u_v = inv.(u_Λ)
        c_m = c_xi ./ c_Λ
        c_v = inv.(c_Λ)

        for n in 1:n_count, h in 1:hidden_count
            ma2[n, h], va2[n, h] = gaussian_from_nat(sp_a_xi[n, h] + link_a_xi[n, h],
                sp_a_Λ[n, h] + link_a_Λ[n, h])
            mx2[n, h], vx2[n, h] = gaussian_from_nat(sp_x_xi[n, h] + cls_x_xi[n, h],
                sp_x_Λ[n, h] + cls_x_Λ[n, h])
        end

        sp_a_xi_star = similar(sp_a_xi)
        sp_a_Λ_star = similar(sp_a_Λ)
        sp_x_xi_star = similar(sp_x_xi)
        sp_x_Λ_star = similar(sp_x_Λ)
        for n in 1:n_count, h in 1:hidden_count
            sp_a_xi_star[n, h], sp_a_Λ_star[n, h] =
                positive_softplus_to_a_site(ma2[n, h], mx2[n, h], vx2[n, h], sigma_hidden2)
            sp_x_xi_star[n, h], sp_x_Λ_star[n, h] =
                softplus_to_x_site(0.0, 1.0, sigma_hidden2)
        end
        update_deep_site!(sp_a_xi, sp_a_xi_star, prev_sp_a_xi, metric_sp_a_xi;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping)
        update_deep_site!(sp_a_Λ, sp_a_Λ_star, prev_sp_a_Λ, metric_sp_a_Λ;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping, precision=true)
        update_deep_site!(sp_x_xi, sp_x_xi_star, prev_sp_x_xi, metric_sp_x_xi;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping)
        update_deep_site!(sp_x_Λ, sp_x_Λ_star, prev_sp_x_Λ, metric_sp_x_Λ;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping, precision=true)

        link_a_xi_star = zero(link_a_xi)
        link_a_Λ_star = zero(link_a_Λ)
        link_w_xi_star = zero(link_w_xi)
        link_w_Λ_star = zero(link_w_Λ)
        for n in 1:n_count, h in 1:hidden_count
            prod_mean = zeros(input_count)
            prod_var = zeros(input_count)
            total_mean = 0.0
            total_var = sigma_hidden2
            for k in 1:input_count
                prod_mean[k], prod_var[k] = mean_var_product(gmean[n, k], gvar[n, k], w_m[h, k], w_v[h, k])
                total_mean += prod_mean[k]
                total_var += prod_var[k]
            end
            for k in 1:input_count
                residual = ma2[n, h] - (total_mean - prod_mean[k])
                noise = max(total_var - prod_var[k] + va2[n, h], sigma_hidden2)

                gg, hh = product_log_derivatives(gmean[n, k], residual, w_m[h, k], w_v[h, k], noise)
                xi_g, λ_g = positive_site_from_grad_hess(gmean[n, k], gg, hh; max_precision=1e3)
                # pooled inputs are local deterministic summaries, so push their
                # evidence only through W and a2 in this first deep experiment.
                _ = xi_g + λ_g

                gw, hw = product_log_derivatives(w_m[h, k], residual, gmean[n, k], gvar[n, k], noise)
                link_w_xi_star[n, h, k], link_w_Λ_star[n, h, k] =
                    positive_site_from_grad_hess(w_m[h, k], gw, hw; max_precision=1e3)
            end
            noise_a = max(total_var, sigma_hidden2)
            link_a_xi_star[n, h] = total_mean / noise_a
            link_a_Λ_star[n, h] = inv(noise_a)
        end
        update_deep_site!(link_a_xi, link_a_xi_star, prev_link_a_xi, metric_link_a_xi;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping)
        update_deep_site!(link_a_Λ, link_a_Λ_star, prev_link_a_Λ, metric_link_a_Λ;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping, precision=true)
        update_deep_site!(link_w_xi, link_w_xi_star, prev_link_w_xi, metric_link_w_xi;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping)
        update_deep_site!(link_w_Λ, link_w_Λ_star, prev_link_w_Λ, metric_link_w_Λ;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping, precision=true)

        cls_x_xi_star = zero(cls_x_xi)
        cls_x_Λ_star = zero(cls_x_Λ)
        cls_u_xi_star = zero(cls_u_xi)
        cls_u_Λ_star = zero(cls_u_Λ)
        cls_c_xi_star = zero(cls_c_xi)
        cls_c_Λ_star = zero(cls_c_Λ)
        for n in 1:n_count
            prod_mean = zeros(class_count, hidden_count)
            prod_var = zeros(class_count, hidden_count)
            logits = copy(c_m)
            for cls in 1:class_count, h in 1:hidden_count
                prod_mean[cls, h], prod_var[cls, h] = mean_var_product(mx2[n, h], vx2[n, h], u_m[cls, h], u_v[cls, h])
                logits[cls] += prod_mean[cls, h]
            end
            xi_t, lambda_t, _ = categorical_logit_sites(logits, labels[n], priors.classes)
            pseudo_y = xi_t ./ lambda_t

            for cls in 1:class_count
                pseudo_noise = inv(lambda_t[cls])
                for h in 1:hidden_count
                    residual = pseudo_y[cls] - (c_m[cls] + sum(@view prod_mean[cls, :]) - prod_mean[cls, h])
                    noise = max(pseudo_noise + c_v[cls] + sum(@view prod_var[cls, :]) - prod_var[cls, h],
                        pseudo_noise)
                    gx, hx = product_log_derivatives(mx2[n, h], residual, u_m[cls, h], u_v[cls, h], noise)
                    xi_x, λ_x = positive_site_from_grad_hess(mx2[n, h], gx, hx; max_precision=1e3)
                    cls_x_xi_star[n, h] += xi_x
                    cls_x_Λ_star[n, h] += λ_x

                    gu, hu = product_log_derivatives(u_m[cls, h], residual, mx2[n, h], vx2[n, h], noise)
                    cls_u_xi_star[n, cls, h], cls_u_Λ_star[n, cls, h] =
                        positive_site_from_grad_hess(u_m[cls, h], gu, hu; max_precision=1e3)
                end
                residual_c = pseudo_y[cls] - sum(@view prod_mean[cls, :])
                noise_c = max(pseudo_noise + sum(@view prod_var[cls, :]), pseudo_noise)
                cls_c_xi_star[n, cls] = residual_c / noise_c
                cls_c_Λ_star[n, cls] = inv(noise_c)
            end
        end
        update_deep_site!(cls_x_xi, cls_x_xi_star, prev_cls_x_xi, metric_cls_x_xi;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping)
        update_deep_site!(cls_x_Λ, cls_x_Λ_star, prev_cls_x_Λ, metric_cls_x_Λ;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping, precision=true)
        update_deep_site!(cls_u_xi, cls_u_xi_star, prev_cls_u_xi, metric_cls_u_xi;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping)
        update_deep_site!(cls_u_Λ, cls_u_Λ_star, prev_cls_u_Λ, metric_cls_u_Λ;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping, precision=true)
        update_deep_site!(cls_c_xi, cls_c_xi_star, prev_cls_c_xi, metric_cls_c_xi;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping)
        update_deep_site!(cls_c_Λ, cls_c_Λ_star, prev_cls_c_Λ, metric_cls_c_Λ;
            alpha, vector_transport, momentum=vector_transport_momentum,
            damping=vector_transport_damping, precision=true)

        last_delta = maximum(abs.(mx2 .- old_mx2))
        last_delta < tol && break
    end

    return (w_m=w_m, w_Λ=inv.(w_v), u_m=u_m, u_Λ=inv.(u_v),
        c_m=c_m, c_Λ=inv.(c_v), classes=priors.classes),
    (delta=last_delta, hidden_mean=mx2, hidden_var=vx2)
end

function train_deep_softplus_demo(; ntrain=1000, nval=100, ntest=100,
    num_features=4, num_spatial_bins=49,
    hidden_count=32, batch_size=32,
    epochs=3, seed=1, alpha=0.08,
    classes=collect(0:9),
    vector_transport=false,
    vector_transport_momentum=0.8,
    vector_transport_damping=1e-6,
    max_image_inner=10,
    max_classifier_inner=12)
    data = select_mnist(; ntrain, nval, ntest, seed, digits=classes)
    image_priors = init_priors(num_features, num_spatial_bins;
        r_count=size(data.train_x, 2),
        classes, seed=seed + 10)
    input_count = num_features * num_spatial_bins
    deep_priors = init_deep_classifier_priors(input_count, hidden_count, classes;
        seed=seed + 20)
    rng = MersenneTwister(seed + 30)

    println("Deep softplus MNIST classifier")
    println("train=$(length(data.train_y)) val=$(length(data.val_y)) test=$(length(data.test_y)) F=$num_features bins=$num_spatial_bins hidden=$hidden_count batch=$batch_size epochs=$epochs vector_transport=$vector_transport")

    for epoch in 1:epochs
        order = shuffle(rng, collect(1:length(data.train_y)))
        starts = collect(1:batch_size:length(order))
        progress = Progress(length(starts); desc="epoch $epoch/$epochs ")
        for start_idx in starts
            inds = order[start_idx:min(start_idx + batch_size - 1, end)]

            # First infer/update the image feature model exactly as in the base
            # script, but ignore its direct classifier for prediction.
            image_priors, image_stats = infer_batch_rxinfer_streaming(
                image_priors, data.train_x[:, :, inds], data.train_y[inds];
                num_features, num_spatial_bins, alpha,
                global_site_scale=inv(epochs),
                max_inner=max_image_inner,
            )

            marginals = image_stats.marginals
            gmean, gvar = pool_marginals(marginals.mx, marginals.vx, num_spatial_bins)
            deep_priors, _ = infer_batch_deep_classifier(
                deep_priors, gmean, gvar, data.train_y[inds];
                alpha, global_site_scale=inv(epochs),
                vector_transport,
                vector_transport_momentum,
                vector_transport_damping,
                max_inner=max_classifier_inner,
            )
            ProgressMeter.next!(progress)
        end

        train_acc = evaluate_deep_classifier(image_priors, deep_priors, data.train_x, data.train_y;
            num_spatial_bins, max_images=min(100, length(data.train_y)))
        val_acc = evaluate_deep_classifier(image_priors, deep_priors, data.val_x, data.val_y;
            num_spatial_bins, max_images=min(100, length(data.val_y)))
        println("epoch=$epoch train_acc=$(round(train_acc, digits = 3)) val_acc=$(round(val_acc, digits = 3))")
    end

    test_acc = evaluate_deep_classifier(image_priors, deep_priors, data.test_x, data.test_y;
        num_spatial_bins, max_images=min(100, length(data.test_y)))
    println("holdout_test_acc=$(round(test_acc, digits = 3))")
    return (image_priors=image_priors, deep_priors=deep_priors,
        data=data, test_acc=test_acc)
end

function evaluate_deep_classifier(image_priors, deep_priors, x, y;
    num_spatial_bins=49,
    max_images=size(x, 3))
    n = min(max_images, size(x, 3))
    correct = 0
    for i in 1:n
        m = infer_image_features_rx(image_priors, @view x[:, :, i])
        gmean, _ = pool_marginals(m.mx, m.vx, num_spatial_bins)
        pred, _ = deep_predict_label(deep_priors, vec(gmean))
        correct += pred == y[i]
    end
    return correct / n
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_deep_softplus_demo()
end
