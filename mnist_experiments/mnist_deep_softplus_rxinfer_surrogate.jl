using RxInfer
using Rocket
using LinearAlgebra
using Random
using Statistics
using ProgressMeter

include(joinpath(@__DIR__, "mnist_softplus_rxinfer_surrogate_multiclass.jl"))

# ============================================================================
# Real deeper surrogate model:
#
#   a1[n,p,f] ~ Normal(0, sigma_a1^2)
#   x1[n,p,f] ~= softplus(a1[n,p,f])
#   y[n,p,r]  ~ Normal(dot(B[:,r], x1[n,p,:]), sigma_y2)
#
#   g[n,k]    = pooled(x1[n,:,:])
#   a2[n,h]   ~ Normal(dot(W[h,:], g[n,:]), sigma_hidden2)
#   x2[n,h]   ~= softplus(a2[n,h])
#   label[n]  ~ Categorical(softmax(c + U * x2[n,:]))
#
# RxInfer sees only Gaussian priors and Gaussian surrogate leaves. The
# non-conjugate softplus, bilinear dot products, pooling, and categorical sites
# are refreshed outside the graph, then one exact Gaussian BP sweep is run.
# ============================================================================

@model function deep_softplus_mnist_surrogate(
    a1_site_y, a1_site_Λ,
    x1_sp_y, x1_sp_Λ,
    x1_img_y, x1_img_Λ,
    b_img_y, b_img_Λ,
    x1_hid_y, x1_hid_Λ,
    a2_link_y, a2_link_Λ,
    a2_site_y, a2_site_Λ,
    x2_sp_y, x2_sp_Λ,
    x2_cls_y, x2_cls_Λ,
    w_hid_y, w_hid_Λ,
    u_cls_y, u_cls_Λ,
    c_cls_y, c_cls_Λ,
    b_prior_m, b_prior_Λ,
    w_prior_m, w_prior_Λ,
    u_prior_m, u_prior_Λ,
    c_prior_m, c_prior_Λ,
    sigma_a12,
    x_weak_precision,
    n_count,
    p_count,
    f_count,
    r_count,
    input_count,
    hidden_count,
    class_count,
)
    local local_kernel
    local hidden_weight
    local classifier_weight
    local classifier_bias
    local a1
    local x1
    local a2
    local x2

    for f in 1:f_count, r in 1:r_count
        local_kernel[f, r] ~ Normal(mean=b_prior_m[f, r], precision=b_prior_Λ[f, r])
    end
    for h in 1:hidden_count, k in 1:input_count
        hidden_weight[h, k] ~ Normal(mean=w_prior_m[h, k], precision=w_prior_Λ[h, k])
    end
    for cls in 1:class_count, h in 1:hidden_count
        classifier_weight[cls, h] ~ Normal(mean=u_prior_m[cls, h], precision=u_prior_Λ[cls, h])
    end
    for cls in 1:class_count
        classifier_bias[cls] ~ Normal(mean=c_prior_m[cls], precision=c_prior_Λ[cls])
    end

    for n in 1:n_count, p in 1:p_count, f in 1:f_count
        a1[n, p, f] ~ Normal(mean=0.0, precision=inv(sigma_a12))
        a1_site_y[n, p, f] ~ Normal(mean=a1[n, p, f], precision=a1_site_Λ[n, p, f])

        x1[n, p, f] ~ Normal(mean=0.0, precision=x_weak_precision)
        x1_sp_y[n, p, f] ~ Normal(mean=x1[n, p, f], precision=x1_sp_Λ[n, p, f])
        x1_hid_y[n, p, f] ~ Normal(mean=x1[n, p, f], precision=x1_hid_Λ[n, p, f])

        for r in 1:r_count
            x1_img_y[n, p, f, r] ~ Normal(mean=x1[n, p, f], precision=x1_img_Λ[n, p, f, r])
            b_img_y[n, p, f, r] ~ Normal(mean=local_kernel[f, r], precision=b_img_Λ[n, p, f, r])
        end
    end

    for n in 1:n_count, h in 1:hidden_count
        a2[n, h] ~ Normal(mean=0.0, precision=x_weak_precision)
        a2_link_y[n, h] ~ Normal(mean=a2[n, h], precision=a2_link_Λ[n, h])
        a2_site_y[n, h] ~ Normal(mean=a2[n, h], precision=a2_site_Λ[n, h])

        x2[n, h] ~ Normal(mean=0.0, precision=x_weak_precision)
        x2_sp_y[n, h] ~ Normal(mean=x2[n, h], precision=x2_sp_Λ[n, h])
        x2_cls_y[n, h] ~ Normal(mean=x2[n, h], precision=x2_cls_Λ[n, h])
    end

    for n in 1:n_count, h in 1:hidden_count, k in 1:input_count
        w_hid_y[n, h, k] ~ Normal(mean=hidden_weight[h, k], precision=w_hid_Λ[n, h, k])
    end
    for n in 1:n_count, cls in 1:class_count, h in 1:hidden_count
        u_cls_y[n, cls, h] ~ Normal(mean=classifier_weight[cls, h], precision=u_cls_Λ[n, cls, h])
    end
    for n in 1:n_count, cls in 1:class_count
        c_cls_y[n, cls] ~ Normal(mean=classifier_bias[cls], precision=c_cls_Λ[n, cls])
    end
end

function deep_zero_sites(n_count, p_count, f_count, r_count, input_count, hidden_count, class_count)
    return (
        a1_xi=zeros(n_count, p_count, f_count),
        a1_Λ=zeros(n_count, p_count, f_count),
        x1_sp_xi=zeros(n_count, p_count, f_count),
        x1_sp_Λ=zeros(n_count, p_count, f_count),
        x1_img_xi=zeros(n_count, p_count, f_count, r_count),
        x1_img_Λ=zeros(n_count, p_count, f_count, r_count),
        b_img_xi=zeros(n_count, p_count, f_count, r_count),
        b_img_Λ=zeros(n_count, p_count, f_count, r_count),
        x1_hid_xi=zeros(n_count, p_count, f_count),
        x1_hid_Λ=zeros(n_count, p_count, f_count),
        a2_link_xi=zeros(n_count, hidden_count),
        a2_link_Λ=zeros(n_count, hidden_count),
        a2_xi=zeros(n_count, hidden_count),
        a2_Λ=zeros(n_count, hidden_count),
        x2_sp_xi=zeros(n_count, hidden_count),
        x2_sp_Λ=zeros(n_count, hidden_count),
        x2_cls_xi=zeros(n_count, hidden_count),
        x2_cls_Λ=zeros(n_count, hidden_count),
        w_hid_xi=zeros(n_count, hidden_count, input_count),
        w_hid_Λ=zeros(n_count, hidden_count, input_count),
        u_cls_xi=zeros(n_count, class_count, hidden_count),
        u_cls_Λ=zeros(n_count, class_count, hidden_count),
        c_cls_xi=zeros(n_count, class_count),
        c_cls_Λ=zeros(n_count, class_count),
    )
end

function deep_observation(site, priors; global_site_scale=1.0)
    a1_y, a1_Λ = nat_to_observations(site.a1_xi, site.a1_Λ)
    x1_sp_y, x1_sp_Λ = nat_to_observations(site.x1_sp_xi, site.x1_sp_Λ)
    x1_img_y, x1_img_Λ = nat_to_observations(site.x1_img_xi, site.x1_img_Λ)
    b_img_y, b_img_Λ = nat_to_observations(site.b_img_xi, site.b_img_Λ; scale=global_site_scale)
    x1_hid_y, x1_hid_Λ = nat_to_observations(site.x1_hid_xi, site.x1_hid_Λ)
    a2_link_y, a2_link_Λ = nat_to_observations(site.a2_link_xi, site.a2_link_Λ)
    a2_site_y, a2_site_Λ = nat_to_observations(site.a2_xi, site.a2_Λ)
    x2_sp_y, x2_sp_Λ = nat_to_observations(site.x2_sp_xi, site.x2_sp_Λ)
    x2_cls_y, x2_cls_Λ = nat_to_observations(site.x2_cls_xi, site.x2_cls_Λ)
    w_hid_y, w_hid_Λ = nat_to_observations(site.w_hid_xi, site.w_hid_Λ; scale=global_site_scale)
    u_cls_y, u_cls_Λ = nat_to_observations(site.u_cls_xi, site.u_cls_Λ; scale=global_site_scale)
    c_cls_y, c_cls_Λ = nat_to_observations(site.c_cls_xi, site.c_cls_Λ; scale=global_site_scale)

    return (
        a1_site_y=a1_y, a1_site_Λ=a1_Λ,
        x1_sp_y=x1_sp_y, x1_sp_Λ=x1_sp_Λ,
        x1_img_y=x1_img_y, x1_img_Λ=x1_img_Λ,
        b_img_y=b_img_y, b_img_Λ=b_img_Λ,
        x1_hid_y=x1_hid_y, x1_hid_Λ=x1_hid_Λ,
        a2_link_y=a2_link_y, a2_link_Λ=a2_link_Λ,
        a2_site_y=a2_site_y, a2_site_Λ=a2_site_Λ,
        x2_sp_y=x2_sp_y, x2_sp_Λ=x2_sp_Λ,
        x2_cls_y=x2_cls_y, x2_cls_Λ=x2_cls_Λ,
        w_hid_y=w_hid_y, w_hid_Λ=w_hid_Λ,
        u_cls_y=u_cls_y, u_cls_Λ=u_cls_Λ,
        c_cls_y=c_cls_y, c_cls_Λ=c_cls_Λ,
        b_prior_m=priors.b_m, b_prior_Λ=priors.b_Λ,
        w_prior_m=priors.w_m, w_prior_Λ=priors.w_Λ,
        u_prior_m=priors.u_m, u_prior_Λ=priors.u_Λ,
        c_prior_m=priors.c_m, c_prior_Λ=priors.c_Λ,
    )
end

function deep_dims(site, priors)
    n_count, p_count, f_count = size(site.x1_sp_xi)
    r_count = size(priors.b_m, 2)
    hidden_count, input_count = size(priors.w_m)
    class_count = length(priors.classes)
    return (; n_count, p_count, f_count, r_count, input_count, hidden_count, class_count)
end

function run_deep_bp(site, priors; sigma_a12=1.0, global_site_scale=1.0)
    dims = deep_dims(site, priors)
    return infer(
        model=deep_softplus_mnist_surrogate(; sigma_a12, x_weak_precision=1e-6, dims...),
        data=deep_observation(site, priors; global_site_scale),
        returnvars=Dict(
            :a1 => KeepLast(),
            :x1 => KeepLast(),
            :local_kernel => KeepLast(),
            :hidden_weight => KeepLast(),
            :a2 => KeepLast(),
            :x2 => KeepLast(),
            :classifier_weight => KeepLast(),
            :classifier_bias => KeepLast(),
        ),
        options=(limit_stack_depth=500,),
    )
end

function deep_array_mean_var(qs)
    return mean.(qs), var.(qs)
end

function deep_extract_marginals(result)
    p = result.posteriors
    ma1, va1 = deep_array_mean_var(p[:a1])
    mx1, vx1 = deep_array_mean_var(p[:x1])
    mb, vb = deep_array_mean_var(p[:local_kernel])
    mw, vw = deep_array_mean_var(p[:hidden_weight])
    ma2, va2 = deep_array_mean_var(p[:a2])
    mx2, vx2 = deep_array_mean_var(p[:x2])
    mu, vu = deep_array_mean_var(p[:classifier_weight])
    mc, vc = deep_array_mean_var(p[:classifier_bias])
    return (; ma1, va1, mx1, vx1, mb, vb, mw, vw, ma2, va2, mx2, vx2, mu, vu, mc, vc)
end

function init_deep_priors(num_features, num_spatial_bins, hidden_count, classes;
    r_count=4, seed=1,
    b_var=1.0, w_var=1.0, u_var=1.0, c_var=4.0)
    rng = MersenneTwister(seed)
    input_count = num_features * num_spatial_bins
    class_count = length(classes)
    b_m = 0.05 .+ 0.01 .* randn(rng, num_features, r_count)
    b_Λ = fill(inv(b_var), num_features, r_count)
    # The hidden layer receives positive softplus features. A tiny zero-mean
    # initialization keeps a2 near 0, so x2 stays at softplus(0) for every image.
    # Xavier-scale weights give the second layer enough image-dependent signal.
    w_scale = sqrt(2 / input_count)
    w_m = w_scale .* randn(rng, hidden_count, input_count)
    w_Λ = fill(inv(w_var), hidden_count, input_count)
    u_scale = sqrt(1 / hidden_count)
    u_m = u_scale .* randn(rng, class_count, hidden_count)
    u_Λ = fill(inv(u_var), class_count, hidden_count)
    return (b_m=b_m, b_Λ=b_Λ, w_m=w_m, w_Λ=w_Λ,
        u_m=u_m, u_Λ=u_Λ, c_m=zeros(class_count),
        c_Λ=fill(inv(c_var), class_count), classes=collect(classes))
end

function deep_posterior_as_priors(m, old_priors)
    return (b_m=m.mb, b_Λ=inv.(m.vb),
        w_m=m.mw, w_Λ=inv.(m.vw),
        u_m=m.mu, u_Λ=inv.(m.vu),
        c_m=m.mc, c_Λ=inv.(m.vc),
        classes=copy(old_priors.classes))
end

function deep_pool(mx, vx, num_spatial_bins)
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
    return gmean, gvar, bin_ids, bin_counts
end

function refresh_deep_sites(batch_x, batch_y, m, old_site;
    num_spatial_bins=49,
    classes=collect(0:(size(m.mu, 1)-1)),
    sigma_a12=1.0,
    sigma_sp1_2=0.05^2,
    sigma_hidden2=0.10^2,
    sigma_sp2_2=0.05^2,
    sigma_y2=0.18^2,
    label_strength=3.0,
    include_labels=true)
    p_count, r_count, n_count = size(batch_x)
    f_count = size(m.mx1, 3)
    hidden_count, input_count = size(m.mw)
    class_count = length(classes)
    target = deep_zero_sites(n_count, p_count, f_count, r_count, input_count, hidden_count, class_count)
    gmean, gvar, bin_ids, bin_counts = deep_pool(m.mx1, m.vx1, num_spatial_bins)

    for n in 1:n_count, p in 1:p_count, f in 1:f_count
        x_cav_xi = m.mx1[n, p, f] / m.vx1[n, p, f] - old_site.x1_sp_xi[n, p, f]
        x_cav_Λ = inv(m.vx1[n, p, f]) - old_site.x1_sp_Λ[n, p, f]
        mx_cav, vx_cav = gaussian_from_nat(x_cav_xi, x_cav_Λ)
        target.a1_xi[n, p, f], target.a1_Λ[n, p, f] =
            positive_softplus_to_a_site(m.ma1[n, p, f], mx_cav, vx_cav, sigma_sp1_2)
        target.x1_sp_xi[n, p, f], target.x1_sp_Λ[n, p, f] =
            softplus_to_x_site(m.ma1[n, p, f], m.va1[n, p, f], sigma_sp1_2)
    end

    for n in 1:n_count, p in 1:p_count, r in 1:r_count
        y = batch_x[p, r, n]
        prod_mean = zeros(f_count)
        prod_var = zeros(f_count)
        total_mean = 0.0
        total_var = sigma_y2
        for f in 1:f_count
            prod_mean[f], prod_var[f] = mean_var_product(m.mx1[n, p, f], m.vx1[n, p, f], m.mb[f, r], m.vb[f, r])
            total_mean += prod_mean[f]
            total_var += prod_var[f]
        end
        for f in 1:f_count
            residual = y - (total_mean - prod_mean[f])
            noise = max(total_var - prod_var[f], sigma_y2)
            gx, hx = product_log_derivatives(m.mx1[n, p, f], residual, m.mb[f, r], m.vb[f, r], noise)
            target.x1_img_xi[n, p, f, r], target.x1_img_Λ[n, p, f, r] =
                positive_site_from_grad_hess(m.mx1[n, p, f], gx, hx; max_precision=1e3)
            gb, hb = product_log_derivatives(m.mb[f, r], residual, m.mx1[n, p, f], m.vx1[n, p, f], noise)
            target.b_img_xi[n, p, f, r], target.b_img_Λ[n, p, f, r] =
                positive_site_from_grad_hess(m.mb[f, r], gb, hb; max_precision=1e3)
        end
    end

    for n in 1:n_count, h in 1:hidden_count
        prod_mean = zeros(input_count)
        prod_var = zeros(input_count)
        total_mean = 0.0
        total_var = sigma_hidden2
        for k in 1:input_count
            prod_mean[k], prod_var[k] = mean_var_product(gmean[n, k], gvar[n, k], m.mw[h, k], m.vw[h, k])
            total_mean += prod_mean[k]
            total_var += prod_var[k]
        end
        for k in 1:input_count
            residual = m.ma2[n, h] - (total_mean - prod_mean[k])
            noise = max(total_var - prod_var[k] + m.va2[n, h], sigma_hidden2)
            gg, hg = product_log_derivatives(gmean[n, k], residual, m.mw[h, k], m.vw[h, k], noise)
            xi_g, λ_g = positive_site_from_grad_hess(gmean[n, k], gg, hg; max_precision=1e3)
            f = cld(k, num_spatial_bins)
            q = k - (f - 1) * num_spatial_bins
            count_q = bin_counts[q]
            for p in 1:p_count
                if bin_ids[p] == q
                    target.x1_hid_xi[n, p, f] += xi_g / count_q
                    target.x1_hid_Λ[n, p, f] += λ_g / count_q^2
                end
            end

            gw, hw = product_log_derivatives(m.mw[h, k], residual, gmean[n, k], gvar[n, k], noise)
            target.w_hid_xi[n, h, k], target.w_hid_Λ[n, h, k] =
                positive_site_from_grad_hess(m.mw[h, k], gw, hw; max_precision=1e3)
        end
        noise_a = max(total_var, sigma_hidden2)
        target.a2_link_xi[n, h] = total_mean / noise_a
        target.a2_link_Λ[n, h] = inv(noise_a)
    end

    for n in 1:n_count, h in 1:hidden_count
        x_cav_xi = m.mx2[n, h] / m.vx2[n, h] - old_site.x2_sp_xi[n, h]
        x_cav_Λ = inv(m.vx2[n, h]) - old_site.x2_sp_Λ[n, h]
        mx_cav, vx_cav = gaussian_from_nat(x_cav_xi, x_cav_Λ)
        target.a2_xi[n, h], target.a2_Λ[n, h] =
            positive_softplus_to_a_site(m.ma2[n, h], mx_cav, vx_cav, sigma_sp2_2)
        target.x2_sp_xi[n, h], target.x2_sp_Λ[n, h] =
            softplus_to_x_site(m.ma2[n, h], m.va2[n, h], sigma_sp2_2)
    end

    if include_labels
        for n in 1:n_count
            g = vec(gmean[n, :])
            a_hidden = m.mw * g
            x_hidden = softplus.(a_hidden)
            slope_hidden = sigmoid.(a_hidden)
            logits = m.mc .+ m.mu * x_hidden
            probs = clamp.(softmax_probs(logits), 1e-6, 1 - 1e-6)
            target_class = Float64.(classes .== batch_y[n])
            grad_logits = label_strength .* (target_class .- probs)
            curv_logits = label_strength .* probs .* (1 .- probs)

            for cls in 1:class_count
                xi_c, λ_c = positive_site_from_grad_hess(
                    m.mc[cls],
                    grad_logits[cls],
                    -curv_logits[cls];
                    max_precision=1e3,
                )
                target.c_cls_xi[n, cls] += xi_c
                target.c_cls_Λ[n, cls] += λ_c

                for h in 1:hidden_count
                    grad_u = grad_logits[cls] * x_hidden[h]
                    hess_u = -curv_logits[cls] * x_hidden[h]^2
                    xi_u, λ_u = positive_site_from_grad_hess(
                        m.mu[cls, h],
                        grad_u,
                        hess_u;
                        max_precision=1e3,
                    )
                    target.u_cls_xi[n, cls, h] += xi_u
                    target.u_cls_Λ[n, cls, h] += λ_u
                end
            end

            grad_x2 = vec(m.mu' * grad_logits)
            curv_x2 = vec((m.mu .^ 2)' * curv_logits)
            for h in 1:hidden_count
                xi_x, λ_x = positive_site_from_grad_hess(
                    m.mx2[n, h],
                    grad_x2[h],
                    -curv_x2[h];
                    max_precision=1e3,
                )
                target.x2_cls_xi[n, h] += xi_x
                target.x2_cls_Λ[n, h] += λ_x

                grad_a = grad_x2[h] * slope_hidden[h]
                curv_a = curv_x2[h] * slope_hidden[h]^2
                for k in 1:input_count
                    xi_w, λ_w = positive_site_from_grad_hess(
                        m.mw[h, k],
                        grad_a * g[k],
                        -curv_a * (g[k]^2 + gvar[n, k]);
                        max_precision=1e3,
                    )
                    target.w_hid_xi[n, h, k] += xi_w
                    target.w_hid_Λ[n, h, k] += λ_w

                    xi_g, λ_g = positive_site_from_grad_hess(
                        g[k],
                        grad_a * m.mw[h, k],
                        -curv_a * (m.mw[h, k]^2 + m.vw[h, k]);
                        max_precision=1e3,
                    )
                    f = cld(k, num_spatial_bins)
                    q = k - (f - 1) * num_spatial_bins
                    count_q = bin_counts[q]
                    for p in 1:p_count
                        if bin_ids[p] == q
                            target.x1_hid_xi[n, p, f] += xi_g / count_q
                            target.x1_hid_Λ[n, p, f] += λ_g / count_q^2
                        end
                    end
                end
            end
        end
    end

    return target
end

function infer_deep_batch(priors, batch_x, batch_y;
    num_features=size(priors.b_m, 1),
    num_spatial_bins=49,
    hidden_count=size(priors.w_m, 1),
    alpha=0.08,
    global_site_scale=1.0,
    label_strength=3.0,
    vector_transport=false,
    vector_transport_momentum=0.8,
    vector_transport_damping=1e-6,
    max_inner=8,
    tol=1e-4,
    include_labels=true)
    p_count, r_count, n_count = size(batch_x)
    input_count = num_features * num_spatial_bins
    class_count = length(priors.classes)
    site = deep_zero_sites(n_count, p_count, num_features, r_count, input_count, hidden_count, class_count)
    previous_update = zero_like_sites(site)
    previous_metric = diagonal_site_metric(site; damping=vector_transport_damping)
    last_delta = Inf
    local m
    for _ in 1:max_inner
        result = run_deep_bp(site, priors; global_site_scale)
        m = deep_extract_marginals(result)
        target = refresh_deep_sites(batch_x, batch_y, m, site;
            num_spatial_bins, classes=priors.classes,
            label_strength,
            include_labels)
        if vector_transport
            direction = site_direction(site, target)
            new_site, previous_update, previous_metric = vector_transport_site_step(
                site,
                direction,
                previous_update,
                previous_metric;
                alpha,
                momentum=vector_transport_momentum,
                damping=vector_transport_damping,
            )
        else
            new_site = damp_sites(site, target, alpha)
        end
        last_delta = site_update_norm(new_site, site)
        site = new_site
        last_delta < tol && break
    end
    result = run_deep_bp(site, priors; global_site_scale)
    m = deep_extract_marginals(result)
    return deep_posterior_as_priors(m, priors), (delta=last_delta, marginals=m)
end

function predict_deep(priors, image_patches; num_spatial_bins=49, max_inner=8)
    batch_x = reshape(Array(image_patches), size(image_patches, 1), size(image_patches, 2), 1)
    _, stats = infer_deep_batch(priors, batch_x, [0];
        num_spatial_bins,
        global_site_scale=0.0,
        max_inner,
        include_labels=false)
    m = stats.marginals
    gmean, _, _, _ = deep_pool(m.mx1, m.vx1, num_spatial_bins)
    hidden_mean = softplus.(priors.w_m * vec(gmean[1, :]))
    logits = priors.c_m .+ priors.u_m * hidden_mean
    probs = softmax_probs(logits)
    return priors.classes[argmax(probs)], probs
end

function evaluate_deep(priors, x, y; num_spatial_bins=49, max_images=size(x, 3), max_inner=8)
    n = min(max_images, size(x, 3))
    correct = 0
    for i in 1:n
        pred, _ = predict_deep(priors, @view x[:, :, i]; num_spatial_bins, max_inner)
        correct += pred == y[i]
    end
    return correct / n
end

function discriminative_deep_mean_step(priors, marginals, batch_y;
    num_spatial_bins=49,
    learning_rate=0.02)
    gmean, _, _, _ = deep_pool(marginals.mx1, marginals.vx1, num_spatial_bins)
    batch_count = length(batch_y)
    w_m = copy(priors.w_m)
    u_m = copy(priors.u_m)
    c_m = copy(priors.c_m)
    gw = zeros(size(w_m))
    gu = zeros(size(u_m))
    gc = zeros(size(c_m))

    for n in 1:batch_count
        g = vec(gmean[n, :])
        a_hidden = w_m * g
        x_hidden = softplus.(a_hidden)
        probs = softmax_probs(c_m .+ u_m * x_hidden)
        target_class = Float64.(priors.classes .== batch_y[n])
        grad_logits = target_class .- probs
        gu .+= grad_logits * x_hidden'
        gc .+= grad_logits
        grad_hidden = vec(u_m' * grad_logits) .* sigmoid.(a_hidden)
        gw .+= grad_hidden * g'
    end

    scale = learning_rate / batch_count
    return (b_m=priors.b_m, b_Λ=priors.b_Λ,
        w_m=w_m .+ scale .* gw, w_Λ=priors.w_Λ,
        u_m=u_m .+ scale .* gu, u_Λ=priors.u_Λ,
        c_m=c_m .+ scale .* gc, c_Λ=priors.c_Λ,
        classes=priors.classes)
end

function train_deep_rxinfer_demo(; ntrain=1000, nval=100, ntest=100,
    num_features=4, num_spatial_bins=49,
    hidden_count=32, batch_size=32,
    epochs=3, seed=1, alpha=0.08,
    classes=collect(0:9),
    label_strength=3.0,
    vector_transport=false,
    vector_transport_momentum=0.8,
    vector_transport_damping=1e-6,
    discriminative_mean_step=true,
    discriminative_lr=0.02,
    max_inner=6)
    data = select_mnist(; ntrain, nval, ntest, seed, digits=classes)
    priors = init_deep_priors(num_features, num_spatial_bins, hidden_count, classes;
        r_count=size(data.train_x, 2), seed=seed + 10)
    rng = MersenneTwister(seed + 20)
    site_scale = inv(epochs)

    println("Deep RxInfer surrogate MNIST")
    println("train=$(length(data.train_y)) val=$(length(data.val_y)) test=$(length(data.test_y)) F=$num_features bins=$num_spatial_bins hidden=$hidden_count batch=$batch_size epochs=$epochs")

    history = NamedTuple[]
    for epoch in 1:epochs
        order = shuffle(rng, collect(1:length(data.train_y)))
        starts = collect(1:batch_size:length(order))
        progress = Progress(length(starts); desc="epoch $epoch/$epochs ")
        for start_idx in starts
            inds = order[start_idx:min(start_idx + batch_size - 1, end)]
            priors, stats = infer_deep_batch(priors, data.train_x[:, :, inds], data.train_y[inds];
                num_features, num_spatial_bins, hidden_count,
                alpha, global_site_scale=site_scale,
                label_strength,
                vector_transport,
                vector_transport_momentum,
                vector_transport_damping,
                max_inner)
            if discriminative_mean_step
                priors = discriminative_deep_mean_step(priors, stats.marginals, data.train_y[inds];
                    num_spatial_bins,
                    learning_rate=discriminative_lr)
            end
            ProgressMeter.next!(progress)
        end
        train_acc = evaluate_deep(priors, data.train_x, data.train_y;
            num_spatial_bins, max_images=min(64, length(data.train_y)),
            max_inner)
        val_acc = evaluate_deep(priors, data.val_x, data.val_y;
            num_spatial_bins, max_images=min(64, length(data.val_y)),
            max_inner)
        push!(history, (epoch=epoch, train_acc=train_acc, val_acc=val_acc))
        println("epoch=$epoch train_acc=$(round(train_acc, digits = 3)) val_acc=$(round(val_acc, digits = 3))")
    end
    test_acc = evaluate_deep(priors, data.test_x, data.test_y;
        num_spatial_bins, max_images=min(64, length(data.test_y)),
        max_inner)
    println("holdout_test_acc=$(round(test_acc, digits = 3))")
    return (priors=priors, history=history, data=data, test_acc=test_acc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_deep_rxinfer_demo()
end
