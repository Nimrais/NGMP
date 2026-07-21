using RxInfer
using Rocket
using LinearAlgebra
using Random
using Statistics
using ProgressMeter

include(joinpath(@__DIR__, "mnist_softplus_rxinfer_surrogate_multiclass.jl"))
include(joinpath(@__DIR__, "image_classification_utils.jl"))

# ============================================================================
# RxInfer surrogate MLP on flattened MNIST pixels.
#
# This is the non-convolutional analogue of
# mnist_softplus_rxinfer_surrogate_multiclass.jl. Images are downsampled to
# 14x14 and flattened to 196 deterministic inputs. The probabilistic model is:
#
#   hidden_weight[hidden, input] ~ Normal(...)
#   classifier_weight[class, hidden] ~ Normal(...)
#   classifier_bias[class] ~ Normal(...)
#
#   hidden_preactivation[image, hidden] ~ dot(hidden_weight[hidden, :], input[image, :])
#   hidden_strength[image, hidden] := softplus(hidden_preactivation[image, hidden])
#   digit_label[image] ~ Categorical(softmax(classifier_bias +
#                                           classifier_weight * hidden_strength))
#
# The RxInfer graph below contains only Gaussian priors and Gaussian surrogate
# leaves. Non-conjugate softplus, linear-weight/input links, and categorical
# softmax factors are refreshed outside the graph, exactly like the mini-CNN
# surrogate script.
# ============================================================================

@model function softplus_mlp_flattened_surrogate(
    a_link_y, a_link_Λ,
    a_sp_y, a_sp_Λ,
    x_sp_y, x_sp_Λ,
    w_link_y, w_link_Λ,
    x_cls_y, x_cls_Λ,
    u_cls_y, u_cls_Λ,
    c_cls_y, c_cls_Λ,
    w_prior_m, w_prior_Λ,
    u_prior_m, u_prior_Λ,
    c_prior_m, c_prior_Λ,
    x_weak_precision,
    n_count,
    input_count,
    hidden_count,
    class_count,
)
    local hidden_weight
    local classifier_weight
    local classifier_bias
    local hidden_preactivation
    local hidden_strength

    for h in 1:hidden_count, k in 1:input_count
        hidden_weight[h, k] ~ Normal(mean=w_prior_m[h, k], precision=w_prior_Λ[h, k])
    end

    for cls in 1:class_count, h in 1:hidden_count
        classifier_weight[cls, h] ~ Normal(mean=u_prior_m[cls, h], precision=u_prior_Λ[cls, h])
    end

    for cls in 1:class_count
        classifier_bias[cls] ~ Normal(mean=c_prior_m[cls], precision=c_prior_Λ[cls])
    end

    for n in 1:n_count, h in 1:hidden_count
        hidden_preactivation[n, h] ~ Normal(mean=0.0, precision=x_weak_precision)
        a_link_y[n, h] ~ Normal(mean=hidden_preactivation[n, h], precision=a_link_Λ[n, h])
        a_sp_y[n, h] ~ Normal(mean=hidden_preactivation[n, h], precision=a_sp_Λ[n, h])

        hidden_strength[n, h] ~ Normal(mean=0.0, precision=x_weak_precision)
        x_sp_y[n, h] ~ Normal(mean=hidden_strength[n, h], precision=x_sp_Λ[n, h])
        x_cls_y[n, h] ~ Normal(mean=hidden_strength[n, h], precision=x_cls_Λ[n, h])
    end

    for n in 1:n_count, h in 1:hidden_count, k in 1:input_count
        w_link_y[n, h, k] ~ Normal(mean=hidden_weight[h, k], precision=w_link_Λ[n, h, k])
    end

    for n in 1:n_count, cls in 1:class_count, h in 1:hidden_count
        u_cls_y[n, cls, h] ~ Normal(mean=classifier_weight[cls, h], precision=u_cls_Λ[n, cls, h])
    end

    for n in 1:n_count, cls in 1:class_count
        c_cls_y[n, cls] ~ Normal(mean=classifier_bias[cls], precision=c_cls_Λ[n, cls])
    end
end

function mlp_zero_sites(n_count, input_count, hidden_count, class_count)
    return (
        a_link_xi=zeros(n_count, hidden_count),
        a_link_Λ=zeros(n_count, hidden_count),
        a_sp_xi=zeros(n_count, hidden_count),
        a_sp_Λ=zeros(n_count, hidden_count),
        x_sp_xi=zeros(n_count, hidden_count),
        x_sp_Λ=zeros(n_count, hidden_count),
        w_link_xi=zeros(n_count, hidden_count, input_count),
        w_link_Λ=zeros(n_count, hidden_count, input_count),
        x_cls_xi=zeros(n_count, hidden_count),
        x_cls_Λ=zeros(n_count, hidden_count),
        u_cls_xi=zeros(n_count, class_count, hidden_count),
        u_cls_Λ=zeros(n_count, class_count, hidden_count),
        c_cls_xi=zeros(n_count, class_count),
        c_cls_Λ=zeros(n_count, class_count),
    )
end

function init_mlp_priors(input_count, hidden_count, classes;
    w_var=1.0, u_var=1.0, c_var=4.0,
    w_init_scale=0.01, u_init_scale=0.1,
    seed=1)
    rng = MersenneTwister(seed)
    class_count = length(classes)
    return (
        w_m=w_init_scale .* randn(rng, hidden_count, input_count),
        w_Λ=fill(inv(w_var), hidden_count, input_count),
        u_m=u_init_scale .* randn(rng, class_count, hidden_count),
        u_Λ=fill(inv(u_var), class_count, hidden_count),
        c_m=zeros(class_count),
        c_Λ=fill(inv(c_var), class_count),
        classes=collect(classes),
    )
end

copy_mlp_priors(priors) = (
    w_m=copy(priors.w_m), w_Λ=copy(priors.w_Λ),
    u_m=copy(priors.u_m), u_Λ=copy(priors.u_Λ),
    c_m=copy(priors.c_m), c_Λ=copy(priors.c_Λ),
    classes=copy(priors.classes),
)

function mlp_surrogate_observation(site, priors; global_site_scale=1.0)
    a_link_y, a_link_Λ = nat_to_observations(site.a_link_xi, site.a_link_Λ)
    a_sp_y, a_sp_Λ = nat_to_observations(site.a_sp_xi, site.a_sp_Λ)
    x_sp_y, x_sp_Λ = nat_to_observations(site.x_sp_xi, site.x_sp_Λ)
    w_link_y, w_link_Λ = nat_to_observations(site.w_link_xi, site.w_link_Λ; scale=global_site_scale)
    x_cls_y, x_cls_Λ = nat_to_observations(site.x_cls_xi, site.x_cls_Λ)
    u_cls_y, u_cls_Λ = nat_to_observations(site.u_cls_xi, site.u_cls_Λ; scale=global_site_scale)
    c_cls_y, c_cls_Λ = nat_to_observations(site.c_cls_xi, site.c_cls_Λ; scale=global_site_scale)

    return (
        a_link_y=a_link_y, a_link_Λ=a_link_Λ,
        a_sp_y=a_sp_y, a_sp_Λ=a_sp_Λ,
        x_sp_y=x_sp_y, x_sp_Λ=x_sp_Λ,
        w_link_y=w_link_y, w_link_Λ=w_link_Λ,
        x_cls_y=x_cls_y, x_cls_Λ=x_cls_Λ,
        u_cls_y=u_cls_y, u_cls_Λ=u_cls_Λ,
        c_cls_y=c_cls_y, c_cls_Λ=c_cls_Λ,
        w_prior_m=priors.w_m, w_prior_Λ=priors.w_Λ,
        u_prior_m=priors.u_m, u_prior_Λ=priors.u_Λ,
        c_prior_m=priors.c_m, c_prior_Λ=priors.c_Λ,
    )
end

function mlp_surrogate_dims(site, priors)
    n_count, hidden_count = size(site.a_sp_xi)
    input_count = size(priors.w_m, 2)
    class_count = length(priors.classes)
    return (n_count=n_count, input_count=input_count,
        hidden_count=hidden_count, class_count=class_count)
end

function run_mlp_surrogate_bp(site, priors; global_site_scale=1.0)
    dims = mlp_surrogate_dims(site, priors)
    return infer(
        model=softplus_mlp_flattened_surrogate(; x_weak_precision=1e-6, dims...),
        data=mlp_surrogate_observation(site, priors; global_site_scale),
        options=(limit_stack_depth=500,),
    )
end

function extract_mlp_marginals_from_posteriors(posteriors)
    ma, va = array_mean_var(posteriors[:hidden_preactivation])
    mx, vx = array_mean_var(posteriors[:hidden_strength])
    mw, vw = array_mean_var(posteriors[:hidden_weight])
    mu, vu = array_mean_var(posteriors[:classifier_weight])
    mc, vc = array_mean_var(posteriors[:classifier_bias])
    return (ma=ma, va=va, mx=mx, vx=vx,
        mw=mw, vw=vw, mu=mu, vu=vu, mc=mc, vc=vc)
end

extract_mlp_marginals(result) = extract_mlp_marginals_from_posteriors(result.posteriors)

function effective_observation_natural(xi, lambda; scale=1.0)
    scaled_precision = scale * max(lambda, 0.0)
    scaled_precision <= SITE_EPS && return 0.0, SITE_EPS
    precision = max(scaled_precision, SITE_EPS)
    observation_xi = xi / max(lambda, SITE_EPS) * precision
    return observation_xi, precision
end

function local_gaussian_marginals(xi_sites, lambda_sites; weak_precision=1e-6)
    posterior_xi = zeros(size(first(xi_sites)))
    posterior_precision = fill(weak_precision, size(first(lambda_sites)))
    for (xi, lambda) in zip(xi_sites, lambda_sites)
        for i in eachindex(xi, lambda)
            observation_xi, observation_precision =
                effective_observation_natural(xi[i], lambda[i])
            posterior_xi[i] += observation_xi
            posterior_precision[i] += observation_precision
        end
    end
    variance = inv.(posterior_precision)
    return posterior_xi .* variance, variance
end

function global_gaussian_marginals(
    prior_m, prior_precision, site_xi, site_precision;
    global_site_scale=1.0,
)
    posterior_xi = prior_m .* prior_precision
    posterior_precision = copy(prior_precision)
    for global_index in CartesianIndices(prior_m)
        site_tail = Tuple(global_index)
        for n in axes(site_xi, 1)
            index = (n, site_tail...)
            observation_xi, observation_precision = effective_observation_natural(
                site_xi[index...], site_precision[index...];
                scale=global_site_scale)
            posterior_xi[global_index] += observation_xi
            posterior_precision[global_index] += observation_precision
        end
    end
    variance = inv.(posterior_precision)
    return posterior_xi .* variance, variance
end

function direct_mlp_marginals(site, priors; global_site_scale=1.0)
    ma, va = local_gaussian_marginals(
        (site.a_link_xi, site.a_sp_xi),
        (site.a_link_Λ, site.a_sp_Λ))
    mx, vx = local_gaussian_marginals(
        (site.x_sp_xi, site.x_cls_xi),
        (site.x_sp_Λ, site.x_cls_Λ))
    mw, vw = global_gaussian_marginals(
        priors.w_m, priors.w_Λ, site.w_link_xi, site.w_link_Λ;
        global_site_scale)
    mu, vu = global_gaussian_marginals(
        priors.u_m, priors.u_Λ, site.u_cls_xi, site.u_cls_Λ;
        global_site_scale)
    mc, vc = global_gaussian_marginals(
        priors.c_m, priors.c_Λ, site.c_cls_xi, site.c_cls_Λ;
        global_site_scale)
    return (; ma, va, mx, vx, mw, vw, mu, vu, mc, vc)
end

function compute_mlp_marginals(site, priors;
    global_site_scale=1.0, inference_backend=:direct)
    if inference_backend === :direct
        return direct_mlp_marginals(site, priors; global_site_scale)
    elseif inference_backend === :rxinfer
        return extract_mlp_marginals(
            run_mlp_surrogate_bp(site, priors; global_site_scale))
    end
    throw(ArgumentError("inference_backend must be :direct or :rxinfer"))
end

function mlp_posterior_as_priors(m, old_priors)
    return (
        w_m=m.mw, w_Λ=inv.(m.vw),
        u_m=m.mu, u_Λ=inv.(m.vu),
        c_m=m.mc, c_Λ=inv.(m.vc),
        classes=copy(old_priors.classes),
    )
end

function refresh_mlp_sites(batch_x, batch_y, marginals, old_site;
    sigma_sp2=0.05^2,
    sigma_hidden2=0.05^2,
    direct_weight_site_scale=0.5,
    discriminative_site_scale=2.0,
    product_classifier_site_scale=0.0,
    classes=collect(0:(size(marginals.mu, 1)-1)))
    n_count, input_count = size(batch_x)
    hidden_count = size(marginals.mx, 2)
    class_count = size(marginals.mu, 1)
    target = mlp_zero_sites(n_count, input_count, hidden_count, class_count)

    ma, va = marginals.ma, marginals.va
    mx, vx = marginals.mx, marginals.vx
    mw, vw = marginals.mw, marginals.vw
    mu, vu = marginals.mu, marginals.vu
    mc, vc = marginals.mc, marginals.vc

    batch_x_squared = abs2.(batch_x)
    linear_mean = batch_x * transpose(mw)
    linear_var = batch_x_squared * transpose(vw)
    linear_var .+= sigma_hidden2

    for n in 1:n_count, h in 1:hidden_count
        x_cav_xi = mx[n, h] / vx[n, h] - old_site.x_sp_xi[n, h]
        x_cav_Λ = inv(vx[n, h]) - old_site.x_sp_Λ[n, h]
        mx_cav, vx_cav = gaussian_from_nat(x_cav_xi, x_cav_Λ)
        target.a_sp_xi[n, h], target.a_sp_Λ[n, h] =
            positive_softplus_to_a_site(ma[n, h], mx_cav, vx_cav, sigma_sp2)

        a_cav_xi = ma[n, h] / va[n, h] - old_site.a_sp_xi[n, h]
        a_cav_Λ = inv(va[n, h]) - old_site.a_sp_Λ[n, h]
        ma_cav, va_cav = gaussian_from_nat(a_cav_xi, a_cav_Λ)
        target.x_sp_xi[n, h], target.x_sp_Λ[n, h] =
            softplus_to_x_site(ma_cav, va_cav, sigma_sp2)
    end

    # Deterministic-input linear hidden layer:
    #   a[n,h] ~= sum_k hidden_weight[h,k] * batch_x[n,k]
    for n in 1:n_count, h in 1:hidden_count
        total_mean = linear_mean[n, h]
        total_var = linear_var[n, h]

        noise_a = max(total_var, sigma_hidden2)
        target.a_link_xi[n, h] = total_mean / noise_a
        target.a_link_Λ[n, h] = inv(noise_a)

        if target.a_sp_Λ[n, h] > SITE_EPS
            a_pseudo = target.a_sp_xi[n, h] / target.a_sp_Λ[n, h]
            a_noise = inv(target.a_sp_Λ[n, h])
            for k in 1:input_count
                xnk = batch_x[n, k]
                if abs(xnk) > 1e-12
                    product_mean = xnk * mw[h, k]
                    product_var = batch_x_squared[n, k] * vw[h, k]
                    residual = a_pseudo - (total_mean - product_mean)
                    noise = max(total_var - product_var + a_noise, sigma_hidden2)
                    target.w_link_Λ[n, h, k] = min(xnk^2 / noise, 1e3)
                    target.w_link_xi[n, h, k] = xnk * residual / noise
                end
            end
        end
    end

    # Direct discriminative site for the actual MLP forward map. The CNN model
    # learns its first layer from image reconstruction; this flattened MLP has
    # no reconstruction term, so W, U, and c need a direct supervised site.
    if discriminative_site_scale > 0
        x_det = softplus.(linear_mean)
        s_det = sigmoid.(linear_mean)
        logits_det = x_det * transpose(mu)
        logits_det .+= transpose(mc)
        for n in 1:n_count
            probs = softmax_probs(@view logits_det[n, :])
            grad_logits = .-probs
            target_index = findfirst(==(batch_y[n]), classes)
            isnothing(target_index) && error("Unknown class label $(batch_y[n])")
            grad_logits[target_index] += 1.0
            lambda_logits = probs .* (1 .- probs)

            for cls in 1:class_count
                grad_c = discriminative_site_scale * grad_logits[cls]
                lambda_c = discriminative_site_scale * lambda_logits[cls]
                target.c_cls_xi[n, cls] += grad_c + lambda_c * mc[cls]
                target.c_cls_Λ[n, cls] += lambda_c

                for h in 1:hidden_count
                    grad_u = discriminative_site_scale * grad_logits[cls] * x_det[n, h]
                    lambda_u = min(discriminative_site_scale * lambda_logits[cls] * x_det[n, h]^2, 1e3)
                    target.u_cls_xi[n, cls, h] += grad_u + lambda_u * mu[cls, h]
                    target.u_cls_Λ[n, cls, h] += lambda_u
                end
            end

            for h in 1:hidden_count
                grad_a = s_det[n, h] * sum(grad_logits[cls] * mu[cls, h] for cls in 1:class_count)
                lambda_a = s_det[n, h]^2 * sum(lambda_logits[cls] * (mu[cls, h]^2 + vu[cls, h]) for cls in 1:class_count)
                for k in 1:input_count
                    xnk = batch_x[n, k]
                    abs(xnk) <= 1e-12 && continue
                    grad_w = direct_weight_site_scale * discriminative_site_scale * grad_a * xnk
                    lambda_w = min(direct_weight_site_scale * discriminative_site_scale * lambda_a * xnk^2, 1e3)
                    target.w_link_xi[n, h, k] += grad_w + lambda_w * mw[h, k]
                    target.w_link_Λ[n, h, k] += lambda_w
                end
            end
        end
    end

    # Categorical softmax label site, converted into product sites on hidden x
    # and classifier weights.
    if product_classifier_site_scale > 0
        for n in 1:n_count
            prod_mean = zeros(class_count, hidden_count)
            prod_var = zeros(class_count, hidden_count)
            logits = copy(mc)
            for cls in 1:class_count, h in 1:hidden_count
                prod_mean[cls, h], prod_var[cls, h] =
                    mean_var_product(mx[n, h], vx[n, h], mu[cls, h], vu[cls, h])
                logits[cls] += prod_mean[cls, h]
            end
            xi_t, lambda_t, _ = categorical_logit_sites(logits, batch_y[n], classes)
            pseudo_y = xi_t ./ lambda_t

            if false && direct_weight_site_scale > 0
                probs = softmax_probs(logits)
                target_logits = Float64.(classes .== batch_y[n])
                logit_grad = target_logits .- probs
                for h in 1:hidden_count
                    sp_grad = sigmoid(ma[n, h])
                    grad_a = sp_grad * sum(logit_grad[cls] * mu[cls, h] for cls in 1:class_count)
                    curv_a = sp_grad^2 * sum(lambda_t[cls] * (mu[cls, h]^2 + vu[cls, h]) for cls in 1:class_count)
                    for k in 1:input_count
                        xnk = batch_x[n, k]
                        abs(xnk) <= 1e-12 && continue
                        grad_w = direct_weight_site_scale * grad_a * xnk
                        lambda_w = min(direct_weight_site_scale * curv_a * xnk^2, 1e3)
                        target.w_link_xi[n, h, k] += grad_w + lambda_w * mw[h, k]
                        target.w_link_Λ[n, h, k] += lambda_w
                    end
                end
            end

            for cls in 1:class_count
                pseudo_noise = inv(lambda_t[cls])
                for h in 1:hidden_count
                    residual = pseudo_y[cls] - (mc[cls] + sum(@view prod_mean[cls, :]) - prod_mean[cls, h])
                    noise = max(pseudo_noise + vc[cls] + sum(@view prod_var[cls, :]) - prod_var[cls, h],
                        pseudo_noise)

                    gx, hx = product_log_derivatives(mx[n, h], residual, mu[cls, h], vu[cls, h], noise)
                    xi_x, λ_x = positive_site_from_grad_hess(mx[n, h], gx, hx; max_precision=1e3)
                    target.x_cls_xi[n, h] += product_classifier_site_scale * xi_x
                    target.x_cls_Λ[n, h] += product_classifier_site_scale * λ_x

                    gu, hu = product_log_derivatives(mu[cls, h], residual, mx[n, h], vx[n, h], noise)
                    xi_u, λ_u = positive_site_from_grad_hess(mu[cls, h], gu, hu; max_precision=1e3)
                    target.u_cls_xi[n, cls, h] += product_classifier_site_scale * xi_u
                    target.u_cls_Λ[n, cls, h] += product_classifier_site_scale * λ_u
                end

                residual_c = pseudo_y[cls] - sum(@view prod_mean[cls, :])
                noise_c = max(pseudo_noise + sum(@view prod_var[cls, :]), pseudo_noise)
                target.c_cls_xi[n, cls] += product_classifier_site_scale * residual_c / noise_c
                target.c_cls_Λ[n, cls] += product_classifier_site_scale * inv(noise_c)
            end
        end
    end

    return target
end

function infer_batch_mlp_rxinfer(priors, batch_x, batch_y;
    sigma_sp2=0.05^2,
    sigma_hidden2=0.05^2,
    direct_weight_site_scale=0.5,
    discriminative_site_scale=2.0,
    product_classifier_site_scale=0.0,
    alpha=0.2,
    global_site_scale=1.0,
    projected_nesterov=false,
    nesterov_beta=0.9,
    nesterov_eps=1e-8,
    vector_transport=false,
    vector_transport_momentum=0.8,
    vector_transport_damping=1e-6,
    max_inner=1,
    tol=1e-4,
    verbose=false,
    inference_backend=:direct)
    n_count, input_count = size(batch_x)
    hidden_count = size(priors.w_m, 1)
    class_count = length(priors.classes)
    site = mlp_zero_sites(n_count, input_count, hidden_count, class_count)
    projected_nesterov && vector_transport && error("Use either projected_nesterov or vector_transport, not both.")
    previous_direction = zero_like_sites(site)
    has_previous_direction = false
    previous_update = zero_like_sites(site)
    previous_metric = diagonal_site_metric(site; damping=vector_transport_damping)
    local marginals
    last_delta = Inf

    for inner in 1:max_inner
        marginals = compute_mlp_marginals(site, priors;
            global_site_scale, inference_backend)
        target = refresh_mlp_sites(batch_x, batch_y, marginals, site;
            sigma_sp2, sigma_hidden2, direct_weight_site_scale,
            discriminative_site_scale, product_classifier_site_scale,
            classes=priors.classes)
        direction = site_direction(site, target)
        old_site = site
        if projected_nesterov
            lookahead_site = clamp_site_precisions(nesterov_projected_lookahead(
                site, direction, previous_direction, has_previous_direction;
                alpha, beta=nesterov_beta, eps=nesterov_eps))
            lookahead_marginals = compute_mlp_marginals(lookahead_site, priors;
                global_site_scale, inference_backend)
            lookahead_target = refresh_mlp_sites(batch_x, batch_y, lookahead_marginals, lookahead_site;
                sigma_sp2, sigma_hidden2, direct_weight_site_scale,
                discriminative_site_scale, product_classifier_site_scale,
                classes=priors.classes)
            lookahead_direction = site_direction(lookahead_site, lookahead_target)
            site = clamp_site_precisions(add_scaled_sites(site, lookahead_direction, alpha))
            previous_direction = direction
            has_previous_direction = true
        elseif vector_transport
            site, previous_update, previous_metric = vector_transport_site_step(
                site, direction, previous_update, previous_metric;
                alpha, momentum=vector_transport_momentum, damping=vector_transport_damping)
        else
            site = clamp_site_precisions(damp_sites(site, target, alpha))
        end
        last_delta = site_update_norm(site, old_site)
        verbose && @info "inner=$inner delta=$(round(last_delta, sigdigits=3))"
        last_delta < tol && break
    end

    marginals = compute_mlp_marginals(site, priors;
        global_site_scale, inference_backend)
    return mlp_posterior_as_priors(marginals, priors), (delta=last_delta, marginals=marginals)
end

function predict_mlp_rx(priors, input)
    hidden = softplus.(priors.w_m * input)
    logits = priors.c_m .+ priors.u_m * hidden
    probs = softmax_probs(logits)
    return priors.classes[argmax(probs)], probs
end

function evaluate_mlp_rx(priors, x, y; max_images=size(x, 2))
    n = min(max_images, size(x, 2))
    n == 0 && return 0.0
    hidden = softplus.(priors.w_m * @view(x[:, 1:n]))
    logits = priors.c_m .+ priors.u_m * hidden
    correct = 0
    for i in 1:n
        prediction = priors.classes[argmax(@view logits[:, i])]
        correct += prediction == y[i]
    end
    return correct / n
end

function train_mlp_rxinfer_demo(; ntrain=1000, nval=100, ntest=100,
    dataset=:mnist,
    image_size=nothing,
    hidden_count=32,
    batch_size=32,
    epochs=50,
    seed=1,
    alpha=0.2,
    sigma_sp2=0.05^2,
    sigma_hidden2=0.05^2,
    direct_weight_site_scale=0.5,
    discriminative_site_scale=2.0,
    product_classifier_site_scale=0.0,
    w_init_scale=0.01,
    u_init_scale=0.1,
    classes=nothing,
    projected_nesterov=true,
    nesterov_beta=0.9,
    nesterov_eps=1e-8,
    vector_transport=false,
    vector_transport_momentum=0.5,
    vector_transport_damping=1e-6,
    max_inner=3,
    eval_max_images=1000,
    inference_backend=:direct)
    data = load_flattened_image_dataset(
        dataset; ntrain, nval, ntest, seed, classes, image_size)
    classes = data.classes
    input_count = size(data.train_x, 1)
    priors = init_mlp_priors(input_count, hidden_count, classes;
        w_init_scale, u_init_scale, seed=seed + 20)
    rng = MersenneTwister(seed + 30)
    site_scale = inv(epochs)

    println("RxInfer flattened softplus MLP dataset=$(data.name) classes=$(classes)")
    println("train=$(length(data.train_y)) val=$(length(data.val_y)) test=$(length(data.test_y)) image_size=$(data.image_size) channels=$(data.channels) input=$input_count hidden=$hidden_count batch=$batch_size epochs=$epochs alpha=$alpha sigma_sp2=$sigma_sp2 sigma_hidden2=$sigma_hidden2 direct_weight_site_scale=$direct_weight_site_scale discriminative_site_scale=$discriminative_site_scale product_classifier_site_scale=$product_classifier_site_scale w_init_scale=$w_init_scale u_init_scale=$u_init_scale projected_nesterov=$projected_nesterov nesterov_beta=$nesterov_beta nesterov_eps=$nesterov_eps vector_transport=$vector_transport inference_backend=$inference_backend site_scale=$(round(site_scale, digits=4))")

    history = NamedTuple[]
    best_val_acc = -Inf
    best_priors = copy_mlp_priors(priors)
    best_epoch = 0

    for epoch in 1:epochs
        order = shuffle(rng, collect(1:length(data.train_y)))
        starts = collect(1:batch_size:length(order))
        progress = Progress(length(starts); desc="rx mlp epoch $epoch/$epochs ")
        deltas = Float64[]
        for start_idx in starts
            inds = order[start_idx:min(start_idx + batch_size - 1, end)]
            batch_x = Array(transpose(data.train_x[:, inds]))
            priors, stats = infer_batch_mlp_rxinfer(priors, batch_x, data.train_y[inds];
                sigma_sp2, sigma_hidden2, direct_weight_site_scale,
                discriminative_site_scale, product_classifier_site_scale, alpha,
                global_site_scale=site_scale,
                projected_nesterov,
                nesterov_beta,
                nesterov_eps,
                vector_transport,
                vector_transport_momentum,
                vector_transport_damping,
                max_inner,
                inference_backend)
            push!(deltas, stats.delta)
            ProgressMeter.next!(progress)
        end

        train_acc = evaluate_mlp_rx(priors, data.train_x, data.train_y;
            max_images=min(eval_max_images, length(data.train_y)))
        val_acc = evaluate_mlp_rx(priors, data.val_x, data.val_y;
            max_images=min(eval_max_images, length(data.val_y)))
        push!(history, (epoch=epoch, train_acc=train_acc,
            val_acc=val_acc, mean_delta=mean(deltas)))
        println("epoch=$epoch train_acc=$(round(train_acc, digits=3)) val_acc=$(round(val_acc, digits=3)) mean_delta=$(round(mean(deltas), sigdigits=3))")

        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_priors = copy_mlp_priors(priors)
            best_epoch = epoch
        end
    end

    test_acc = evaluate_mlp_rx(best_priors, data.test_x, data.test_y;
        max_images=min(eval_max_images, length(data.test_y)))
    println("best_val_epoch=$best_epoch best_val_acc=$(round(best_val_acc, digits=3)) test_acc=$(round(test_acc, digits=3))")
    return (priors=best_priors, history=history, data=data, test_acc=test_acc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_mlp_rxinfer_demo()
end
