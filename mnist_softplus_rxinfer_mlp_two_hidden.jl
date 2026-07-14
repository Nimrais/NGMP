using RxInfer
using LinearAlgebra
using Random
using Statistics
using ProgressMeter

# Reuse the MNIST loading, numerical helpers, and generic site optimizers from
# the one-hidden-layer experiment. Its main program is guarded, so including it
# does not start training.
include("mnist_softplus_rxinfer_mlp_flattened.jl")

# Two-hidden-layer Bayesian surrogate MLP:
#
#   a1 = W1 * input       x1 = softplus(a1)
#   a2 = W2 * x1          x2 = softplus(a2)
#   logits = c + U * x2   label ~ Categorical(softmax(logits))
#
# RxInfer performs Gaussian message passing. The linear products, Softplus
# transforms, and categorical likelihood are converted to Gaussian sites by
# refresh_two_hidden_sites below.

@model function softplus_mlp_two_hidden_surrogate(
    a1_link_y, a1_link_Λ, a1_sp_y, a1_sp_Λ,
    x1_sp_y, x1_sp_Λ, x1_link_y, x1_link_Λ,
    w1_link_y, w1_link_Λ,
    a2_link_y, a2_link_Λ, a2_sp_y, a2_sp_Λ,
    x2_sp_y, x2_sp_Λ, x2_cls_y, x2_cls_Λ,
    w2_link_y, w2_link_Λ,
    u_cls_y, u_cls_Λ, c_cls_y, c_cls_Λ,
    w1_prior_m, w1_prior_Λ,
    w2_prior_m, w2_prior_Λ,
    u_prior_m, u_prior_Λ,
    c_prior_m, c_prior_Λ,
    x_weak_precision,
    n_count, input_count, hidden1_count, hidden2_count, class_count,
)
    local hidden1_weight, hidden2_weight, classifier_weight, classifier_bias
    local hidden1_preactivation, hidden1_strength
    local hidden2_preactivation, hidden2_strength

    for h in 1:hidden1_count, k in 1:input_count
        hidden1_weight[h, k] ~ Normal(
            mean=w1_prior_m[h, k], precision=w1_prior_Λ[h, k])
    end
    for j in 1:hidden2_count, h in 1:hidden1_count
        hidden2_weight[j, h] ~ Normal(
            mean=w2_prior_m[j, h], precision=w2_prior_Λ[j, h])
    end
    for cls in 1:class_count, j in 1:hidden2_count
        classifier_weight[cls, j] ~ Normal(
            mean=u_prior_m[cls, j], precision=u_prior_Λ[cls, j])
    end
    for cls in 1:class_count
        classifier_bias[cls] ~ Normal(
            mean=c_prior_m[cls], precision=c_prior_Λ[cls])
    end

    for n in 1:n_count, h in 1:hidden1_count
        hidden1_preactivation[n, h] ~ Normal(mean=0.0, precision=x_weak_precision)
        a1_link_y[n, h] ~ Normal(
            mean=hidden1_preactivation[n, h], precision=a1_link_Λ[n, h])
        a1_sp_y[n, h] ~ Normal(
            mean=hidden1_preactivation[n, h], precision=a1_sp_Λ[n, h])

        hidden1_strength[n, h] ~ Normal(mean=0.0, precision=x_weak_precision)
        x1_sp_y[n, h] ~ Normal(
            mean=hidden1_strength[n, h], precision=x1_sp_Λ[n, h])
        x1_link_y[n, h] ~ Normal(
            mean=hidden1_strength[n, h], precision=x1_link_Λ[n, h])
    end

    for n in 1:n_count, j in 1:hidden2_count
        hidden2_preactivation[n, j] ~ Normal(mean=0.0, precision=x_weak_precision)
        a2_link_y[n, j] ~ Normal(
            mean=hidden2_preactivation[n, j], precision=a2_link_Λ[n, j])
        a2_sp_y[n, j] ~ Normal(
            mean=hidden2_preactivation[n, j], precision=a2_sp_Λ[n, j])

        hidden2_strength[n, j] ~ Normal(mean=0.0, precision=x_weak_precision)
        x2_sp_y[n, j] ~ Normal(
            mean=hidden2_strength[n, j], precision=x2_sp_Λ[n, j])
        x2_cls_y[n, j] ~ Normal(
            mean=hidden2_strength[n, j], precision=x2_cls_Λ[n, j])
    end

    for n in 1:n_count, h in 1:hidden1_count, k in 1:input_count
        w1_link_y[n, h, k] ~ Normal(
            mean=hidden1_weight[h, k], precision=w1_link_Λ[n, h, k])
    end
    for n in 1:n_count, j in 1:hidden2_count, h in 1:hidden1_count
        w2_link_y[n, j, h] ~ Normal(
            mean=hidden2_weight[j, h], precision=w2_link_Λ[n, j, h])
    end
    for n in 1:n_count, cls in 1:class_count, j in 1:hidden2_count
        u_cls_y[n, cls, j] ~ Normal(
            mean=classifier_weight[cls, j], precision=u_cls_Λ[n, cls, j])
    end
    for n in 1:n_count, cls in 1:class_count
        c_cls_y[n, cls] ~ Normal(
            mean=classifier_bias[cls], precision=c_cls_Λ[n, cls])
    end
end

function two_hidden_zero_sites(
    n_count, input_count, hidden1_count, hidden2_count, class_count,
)
    return (
        a1_link_xi=zeros(n_count, hidden1_count),
        a1_link_Λ=zeros(n_count, hidden1_count),
        a1_sp_xi=zeros(n_count, hidden1_count),
        a1_sp_Λ=zeros(n_count, hidden1_count),
        x1_sp_xi=zeros(n_count, hidden1_count),
        x1_sp_Λ=zeros(n_count, hidden1_count),
        x1_link_xi=zeros(n_count, hidden1_count),
        x1_link_Λ=zeros(n_count, hidden1_count),
        w1_link_xi=zeros(n_count, hidden1_count, input_count),
        w1_link_Λ=zeros(n_count, hidden1_count, input_count),
        a2_link_xi=zeros(n_count, hidden2_count),
        a2_link_Λ=zeros(n_count, hidden2_count),
        a2_sp_xi=zeros(n_count, hidden2_count),
        a2_sp_Λ=zeros(n_count, hidden2_count),
        x2_sp_xi=zeros(n_count, hidden2_count),
        x2_sp_Λ=zeros(n_count, hidden2_count),
        x2_cls_xi=zeros(n_count, hidden2_count),
        x2_cls_Λ=zeros(n_count, hidden2_count),
        w2_link_xi=zeros(n_count, hidden2_count, hidden1_count),
        w2_link_Λ=zeros(n_count, hidden2_count, hidden1_count),
        u_cls_xi=zeros(n_count, class_count, hidden2_count),
        u_cls_Λ=zeros(n_count, class_count, hidden2_count),
        c_cls_xi=zeros(n_count, class_count),
        c_cls_Λ=zeros(n_count, class_count),
    )
end

function init_two_hidden_priors(
    input_count, hidden1_count, hidden2_count, classes;
    w1_var=1.0, w2_var=1.0, u_var=1.0, c_var=4.0,
    w1_init_scale=sqrt(2 / (input_count + hidden1_count)),
    w2_init_scale=sqrt(2 / (hidden1_count + hidden2_count)),
    u_init_scale=sqrt(2 / (hidden2_count + length(classes))),
    seed=1,
)
    rng = MersenneTwister(seed)
    class_count = length(classes)
    return (
        w1_m=w1_init_scale .* randn(rng, hidden1_count, input_count),
        w1_Λ=fill(inv(w1_var), hidden1_count, input_count),
        w2_m=w2_init_scale .* randn(rng, hidden2_count, hidden1_count),
        w2_Λ=fill(inv(w2_var), hidden2_count, hidden1_count),
        u_m=u_init_scale .* randn(rng, class_count, hidden2_count),
        u_Λ=fill(inv(u_var), class_count, hidden2_count),
        c_m=zeros(class_count),
        c_Λ=fill(inv(c_var), class_count),
        classes=collect(classes),
    )
end

copy_two_hidden_priors(p) = (
    w1_m=copy(p.w1_m), w1_Λ=copy(p.w1_Λ),
    w2_m=copy(p.w2_m), w2_Λ=copy(p.w2_Λ),
    u_m=copy(p.u_m), u_Λ=copy(p.u_Λ),
    c_m=copy(p.c_m), c_Λ=copy(p.c_Λ), classes=copy(p.classes),
)

function two_hidden_surrogate_observation(site, priors; global_site_scale=1.0)
    pairs = map((
        (:a1_link, false), (:a1_sp, false), (:x1_sp, false), (:x1_link, false),
        (:w1_link, true), (:a2_link, false), (:a2_sp, false),
        (:x2_sp, false), (:x2_cls, false), (:w2_link, true),
        (:u_cls, true), (:c_cls, true),
    )) do (stem, global_parameter)
        xi = getfield(site, Symbol(stem, :_xi))
        lambda = getfield(site, Symbol(stem, :_Λ))
        scale = global_parameter ? global_site_scale : 1.0
        y, observation_lambda = nat_to_observations(xi, lambda; scale)
        (Symbol(stem, :_y) => y, Symbol(stem, :_Λ) => observation_lambda)
    end
    site_observations = (; Iterators.flatten(pairs)...)
    return merge(site_observations, (
        w1_prior_m=priors.w1_m, w1_prior_Λ=priors.w1_Λ,
        w2_prior_m=priors.w2_m, w2_prior_Λ=priors.w2_Λ,
        u_prior_m=priors.u_m, u_prior_Λ=priors.u_Λ,
        c_prior_m=priors.c_m, c_prior_Λ=priors.c_Λ,
    ))
end

function run_two_hidden_surrogate_bp(site, priors; global_site_scale=1.0)
    n_count, hidden1_count = size(site.a1_sp_xi)
    hidden2_count = size(site.a2_sp_xi, 2)
    input_count = size(priors.w1_m, 2)
    class_count = length(priors.classes)
    return infer(
        model=softplus_mlp_two_hidden_surrogate(
            x_weak_precision=1e-6,
            n_count=n_count,
            input_count=input_count,
            hidden1_count=hidden1_count,
            hidden2_count=hidden2_count,
            class_count=class_count,
        ),
        data=two_hidden_surrogate_observation(site, priors; global_site_scale),
        options=(limit_stack_depth=500,),
    )
end

function extract_two_hidden_marginals(result)
    p = result.posteriors
    ma1, va1 = array_mean_var(p[:hidden1_preactivation])
    mx1, vx1 = array_mean_var(p[:hidden1_strength])
    ma2, va2 = array_mean_var(p[:hidden2_preactivation])
    mx2, vx2 = array_mean_var(p[:hidden2_strength])
    mw1, vw1 = array_mean_var(p[:hidden1_weight])
    mw2, vw2 = array_mean_var(p[:hidden2_weight])
    mu, vu = array_mean_var(p[:classifier_weight])
    mc, vc = array_mean_var(p[:classifier_bias])
    return (; ma1, va1, mx1, vx1, ma2, va2, mx2, vx2,
        mw1, vw1, mw2, vw2, mu, vu, mc, vc)
end

function two_hidden_posterior_as_priors(m, old_priors)
    return (
        w1_m=m.mw1, w1_Λ=inv.(m.vw1),
        w2_m=m.mw2, w2_Λ=inv.(m.vw2),
        u_m=m.mu, u_Λ=inv.(m.vu),
        c_m=m.mc, c_Λ=inv.(m.vc),
        classes=copy(old_priors.classes),
    )
end

function refresh_two_hidden_sites(
    batch_x, batch_y, m, old_site;
    sigma_sp2=0.05^2,
    sigma_hidden1_2=0.05^2,
    sigma_hidden2_2=0.05^2,
    direct_weight_site_scale=0.5,
    discriminative_site_scale=2.0,
    classes=collect(0:(size(m.mu, 1) - 1)),
)
    n_count, input_count = size(batch_x)
    hidden1_count = size(m.mx1, 2)
    hidden2_count = size(m.mx2, 2)
    class_count = size(m.mu, 1)
    target = two_hidden_zero_sites(
        n_count, input_count, hidden1_count, hidden2_count, class_count)

    # Gaussian sites for both Softplus transforms.
    for (ma, va, mx, vx, a_stem, x_stem) in (
        (m.ma1, m.va1, m.mx1, m.vx1, :a1_sp, :x1_sp),
        (m.ma2, m.va2, m.mx2, m.vx2, :a2_sp, :x2_sp),
    )
        for n in axes(ma, 1), h in axes(ma, 2)
            old_x_xi = getfield(old_site, Symbol(x_stem, :_xi))[n, h]
            old_x_lambda = getfield(old_site, Symbol(x_stem, :_Λ))[n, h]
            x_cavity_xi = mx[n, h] / vx[n, h] - old_x_xi
            x_cavity_lambda = inv(vx[n, h]) - old_x_lambda
            mx_cavity, vx_cavity = gaussian_from_nat(x_cavity_xi, x_cavity_lambda)
            a_xi, a_lambda = positive_softplus_to_a_site(
                ma[n, h], mx_cavity, vx_cavity, sigma_sp2)
            getfield(target, Symbol(a_stem, :_xi))[n, h] = a_xi
            getfield(target, Symbol(a_stem, :_Λ))[n, h] = a_lambda

            old_a_xi = getfield(old_site, Symbol(a_stem, :_xi))[n, h]
            old_a_lambda = getfield(old_site, Symbol(a_stem, :_Λ))[n, h]
            a_cavity_xi = ma[n, h] / va[n, h] - old_a_xi
            a_cavity_lambda = inv(va[n, h]) - old_a_lambda
            ma_cavity, va_cavity = gaussian_from_nat(a_cavity_xi, a_cavity_lambda)
            x_xi, x_lambda = softplus_to_x_site(
                ma_cavity, va_cavity, sigma_sp2)
            getfield(target, Symbol(x_stem, :_xi))[n, h] = x_xi
            getfield(target, Symbol(x_stem, :_Λ))[n, h] = x_lambda
        end
    end

    # First affine layer: deterministic inputs multiplied by uncertain W1.
    for n in 1:n_count, h in 1:hidden1_count
        product_mean = zeros(input_count)
        product_var = zeros(input_count)
        total_mean = 0.0
        total_var = sigma_hidden1_2
        for k in 1:input_count
            xnk = batch_x[n, k]
            product_mean[k] = xnk * m.mw1[h, k]
            product_var[k] = xnk^2 * m.vw1[h, k]
            total_mean += product_mean[k]
            total_var += product_var[k]
        end
        target.a1_link_xi[n, h] = total_mean / total_var
        target.a1_link_Λ[n, h] = inv(total_var)

        if target.a1_sp_Λ[n, h] > SITE_EPS
            pseudo_a = target.a1_sp_xi[n, h] / target.a1_sp_Λ[n, h]
            pseudo_noise = inv(target.a1_sp_Λ[n, h])
            for k in 1:input_count
                xnk = batch_x[n, k]
                abs(xnk) <= 1e-12 && continue
                residual = pseudo_a - (total_mean - product_mean[k])
                noise = max(total_var - product_var[k] + pseudo_noise, sigma_hidden1_2)
                target.w1_link_Λ[n, h, k] = min(xnk^2 / noise, 1e3)
                target.w1_link_xi[n, h, k] = xnk * residual / noise
            end
        end
    end

    # Second affine layer: both x1 and W2 are uncertain. Moment matching sends
    # a forward site to a2; local product derivatives send sites back to x1/W2.
    for n in 1:n_count, j in 1:hidden2_count
        product_mean = zeros(hidden1_count)
        product_var = zeros(hidden1_count)
        total_mean = 0.0
        total_var = sigma_hidden2_2
        for h in 1:hidden1_count
            product_mean[h], product_var[h] = mean_var_product(
                m.mx1[n, h], m.vx1[n, h], m.mw2[j, h], m.vw2[j, h])
            total_mean += product_mean[h]
            total_var += product_var[h]
        end
        target.a2_link_xi[n, j] = total_mean / total_var
        target.a2_link_Λ[n, j] = inv(total_var)

        if target.a2_sp_Λ[n, j] > SITE_EPS
            pseudo_a = target.a2_sp_xi[n, j] / target.a2_sp_Λ[n, j]
            pseudo_noise = inv(target.a2_sp_Λ[n, j])
            for h in 1:hidden1_count
                residual = pseudo_a - (total_mean - product_mean[h])
                noise = max(
                    pseudo_noise + total_var - sigma_hidden2_2 - product_var[h],
                    pseudo_noise,
                )
                gx, hx = product_log_derivatives(
                    m.mx1[n, h], residual, m.mw2[j, h], m.vw2[j, h], noise)
                xi_x, lambda_x = positive_site_from_grad_hess(
                    m.mx1[n, h], gx, hx; max_precision=1e3)
                target.x1_link_xi[n, h] += xi_x
                target.x1_link_Λ[n, h] += lambda_x

                gw, hw = product_log_derivatives(
                    m.mw2[j, h], residual, m.mx1[n, h], m.vx1[n, h], noise)
                xi_w, lambda_w = positive_site_from_grad_hess(
                    m.mw2[j, h], gw, hw; max_precision=1e3)
                target.w2_link_xi[n, j, h] += xi_w
                target.w2_link_Λ[n, j, h] += lambda_w
            end
        end
    end

    # Supervised Gaussian sites from the softmax gradient and diagonal Fisher
    # curvature, backpropagated through U, W2, and W1.
    if discriminative_site_scale > 0
        for n in 1:n_count
            a1 = m.mw1 * @view(batch_x[n, :])
            x1 = softplus.(a1)
            sp1 = sigmoid.(a1)
            a2 = m.mw2 * x1
            x2 = softplus.(a2)
            sp2 = sigmoid.(a2)
            logits = m.mc .+ m.mu * x2
            probs = softmax_probs(logits)
            target_label = Float64.(classes .== batch_y[n])
            grad_logits = target_label .- probs
            lambda_logits = probs .* (1 .- probs)
            scale = discriminative_site_scale

            for cls in 1:class_count
                grad_c = scale * grad_logits[cls]
                lambda_c = scale * lambda_logits[cls]
                target.c_cls_xi[n, cls] += grad_c + lambda_c * m.mc[cls]
                target.c_cls_Λ[n, cls] += lambda_c
                for j in 1:hidden2_count
                    grad_u = scale * grad_logits[cls] * x2[j]
                    lambda_u = min(scale * lambda_logits[cls] * x2[j]^2, 1e3)
                    target.u_cls_xi[n, cls, j] += grad_u + lambda_u * m.mu[cls, j]
                    target.u_cls_Λ[n, cls, j] += lambda_u
                end
            end

            grad_a2 = zeros(hidden2_count)
            lambda_a2 = zeros(hidden2_count)
            for j in 1:hidden2_count
                grad_a2[j] = sp2[j] * sum(
                    grad_logits[cls] * m.mu[cls, j] for cls in 1:class_count)
                lambda_a2[j] = sp2[j]^2 * sum(
                    lambda_logits[cls] * (m.mu[cls, j]^2 + m.vu[cls, j])
                    for cls in 1:class_count)
                for h in 1:hidden1_count
                    grad_w2 = direct_weight_site_scale * scale * grad_a2[j] * x1[h]
                    lambda_w2 = min(
                        direct_weight_site_scale * scale * lambda_a2[j] * x1[h]^2,
                        1e3,
                    )
                    target.w2_link_xi[n, j, h] +=
                        grad_w2 + lambda_w2 * m.mw2[j, h]
                    target.w2_link_Λ[n, j, h] += lambda_w2
                end
            end

            for h in 1:hidden1_count
                grad_a1 = sp1[h] * sum(
                    grad_a2[j] * m.mw2[j, h] for j in 1:hidden2_count)
                lambda_a1 = sp1[h]^2 * sum(
                    lambda_a2[j] * (m.mw2[j, h]^2 + m.vw2[j, h])
                    for j in 1:hidden2_count)
                for k in 1:input_count
                    xnk = batch_x[n, k]
                    abs(xnk) <= 1e-12 && continue
                    grad_w1 = direct_weight_site_scale * scale * grad_a1 * xnk
                    lambda_w1 = min(
                        direct_weight_site_scale * scale * lambda_a1 * xnk^2,
                        1e3,
                    )
                    target.w1_link_xi[n, h, k] +=
                        grad_w1 + lambda_w1 * m.mw1[h, k]
                    target.w1_link_Λ[n, h, k] += lambda_w1
                end
            end
        end
    end
    return target
end

function infer_batch_two_hidden_mlp(
    priors, batch_x, batch_y;
    sigma_sp2=0.05^2,
    sigma_hidden1_2=0.05^2,
    sigma_hidden2_2=0.05^2,
    direct_weight_site_scale=0.5,
    discriminative_site_scale=2.0,
    alpha=0.2,
    global_site_scale=1.0,
    vector_transport=true,
    vector_transport_momentum=0.5,
    vector_transport_damping=1e-6,
    max_inner=3,
    tol=1e-4,
    verbose=false,
)
    n_count, input_count = size(batch_x)
    hidden1_count = size(priors.w1_m, 1)
    hidden2_count = size(priors.w2_m, 1)
    class_count = length(priors.classes)
    site = two_hidden_zero_sites(
        n_count, input_count, hidden1_count, hidden2_count, class_count)
    previous_update = zero_like_sites(site)
    previous_metric = diagonal_site_metric(site; damping=vector_transport_damping)
    local result, marginals
    last_delta = Inf

    for inner in 1:max_inner
        result = run_two_hidden_surrogate_bp(site, priors; global_site_scale)
        marginals = extract_two_hidden_marginals(result)
        target = refresh_two_hidden_sites(batch_x, batch_y, marginals, site;
            sigma_sp2, sigma_hidden1_2, sigma_hidden2_2,
            direct_weight_site_scale, discriminative_site_scale,
            classes=priors.classes)
        direction = site_direction(site, target)
        old_site = site
        if vector_transport
            site, previous_update, previous_metric = vector_transport_site_step(
                site, direction, previous_update, previous_metric;
                alpha, momentum=vector_transport_momentum,
                damping=vector_transport_damping)
        else
            site = clamp_site_precisions(damp_sites(site, target, alpha))
        end
        last_delta = site_update_norm(site, old_site)
        verbose && @info "inner=$inner delta=$(round(last_delta, sigdigits=3))"
        last_delta < tol && break
    end

    result = run_two_hidden_surrogate_bp(site, priors; global_site_scale)
    marginals = extract_two_hidden_marginals(result)
    return two_hidden_posterior_as_priors(marginals, priors),
        (delta=last_delta, marginals=marginals)
end

function predict_two_hidden_mlp(priors, input)
    hidden1 = softplus.(priors.w1_m * input)
    hidden2 = softplus.(priors.w2_m * hidden1)
    logits = priors.c_m .+ priors.u_m * hidden2
    probabilities = softmax_probs(logits)
    return priors.classes[argmax(probabilities)], probabilities
end

function evaluate_two_hidden_mlp(priors, x, y; max_images=size(x, 2))
    n = min(max_images, size(x, 2))
    n == 0 && return 0.0
    correct = count(1:n) do i
        prediction, _ = predict_two_hidden_mlp(priors, @view x[:, i])
        prediction == y[i]
    end
    return correct / n
end

"""Return a shuffled epoch order whose adjacent examples remain class-balanced."""
function balanced_two_hidden_epoch_order(labels, classes, rng)
    groups = [findall(==(cls), labels) for cls in classes]
    foreach(group -> shuffle!(rng, group), groups)
    positions = ones(Int, length(groups))
    order = Int[]
    sizehint!(order, length(labels))
    while length(order) < length(labels)
        class_order = shuffle(rng, collect(eachindex(groups)))
        added = false
        for class_index in class_order
            position = positions[class_index]
            if position <= length(groups[class_index])
                push!(order, groups[class_index][position])
                positions[class_index] += 1
                added = true
            end
        end
        added || break
    end
    return order
end

function train_two_hidden_mlp_rxinfer_demo(
    ; ntrain=1000, nval=100, ntest=100,
    hidden1_count=32,
    hidden2_count=32,
    batch_size=32,
    epochs=10,
    seed=1,
    alpha=0.2,
    sigma_sp2=0.05^2,
    sigma_hidden1_2=0.05^2,
    sigma_hidden2_2=0.05^2,
    direct_weight_site_scale=1.0,
    discriminative_site_scale=2.0,
    classes=collect(0:9),
    w1_init_scale=sqrt(2 / (14 * 14 + hidden1_count)),
    w2_init_scale=sqrt(2 / (hidden1_count + hidden2_count)),
    u_init_scale=sqrt(2 / (hidden2_count + length(classes))),
    vector_transport=true,
    vector_transport_momentum=0.5,
    vector_transport_damping=1e-6,
    max_inner=3,
    eval_max_images=1000,
)
    data = select_flattened_mnist(; ntrain, nval, ntest, seed, digits=classes)
    input_count = size(data.train_x, 1)
    priors = init_two_hidden_priors(
        input_count, hidden1_count, hidden2_count, classes;
        w1_init_scale, w2_init_scale, u_init_scale, seed=seed + 20)
    rng = MersenneTwister(seed + 30)
    site_scale = inv(epochs)
    history = NamedTuple[]
    best_val_acc = -Inf
    best_priors = copy_two_hidden_priors(priors)
    best_epoch = 0

    println("RxInfer two-hidden-layer Softplus MLP MNIST classes=$classes")
    println("train=$(length(data.train_y)) val=$(length(data.val_y)) test=$(length(data.test_y)) input=$input_count hidden1=$hidden1_count hidden2=$hidden2_count batch=$batch_size epochs=$epochs max_inner=$max_inner")
    println("init_scales=($(round(w1_init_scale, digits=3)), $(round(w2_init_scale, digits=3)), $(round(u_init_scale, digits=3))) direct_weight_site_scale=$direct_weight_site_scale")

    for epoch in 1:epochs
        order = balanced_two_hidden_epoch_order(data.train_y, classes, rng)
        starts = 1:batch_size:length(order)
        progress = Progress(length(starts); desc="rx 2h mlp epoch $epoch/$epochs ")
        deltas = Float64[]
        for start_index in starts
            indices = order[start_index:min(start_index + batch_size - 1, end)]
            batch_x = Array(transpose(data.train_x[:, indices]))
            priors, stats = infer_batch_two_hidden_mlp(
                priors, batch_x, data.train_y[indices];
                sigma_sp2, sigma_hidden1_2, sigma_hidden2_2,
                direct_weight_site_scale, discriminative_site_scale,
                alpha, global_site_scale=site_scale,
                vector_transport, vector_transport_momentum,
                vector_transport_damping, max_inner)
            push!(deltas, stats.delta)
            ProgressMeter.next!(progress)
        end

        train_acc = evaluate_two_hidden_mlp(
            priors, data.train_x, data.train_y;
            max_images=min(eval_max_images, length(data.train_y)))
        val_acc = evaluate_two_hidden_mlp(
            priors, data.val_x, data.val_y;
            max_images=min(eval_max_images, length(data.val_y)))
        predicted_classes = Set([
            predict_two_hidden_mlp(priors, @view(data.train_x[:, i]))[1]
            for i in 1:min(eval_max_images, length(data.train_y))
        ])
        push!(history, (; epoch, train_acc, val_acc, mean_delta=mean(deltas)))
        println("epoch=$epoch train_acc=$(round(train_acc, digits=3)) val_acc=$(round(val_acc, digits=3)) predicted_classes=$(length(predicted_classes)) mean_delta=$(round(mean(deltas), sigdigits=3))")

        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_priors = copy_two_hidden_priors(priors)
            best_epoch = epoch
        end
    end

    test_acc = evaluate_two_hidden_mlp(
        best_priors, data.test_x, data.test_y;
        max_images=min(eval_max_images, length(data.test_y)))
    println("best_val_epoch=$best_epoch best_val_acc=$(round(best_val_acc, digits=3)) test_acc=$(round(test_acc, digits=3))")
    return (; priors=best_priors, history, data, test_acc)
end

function two_hidden_smoke_test()
    rng = MersenneTwister(71)
    batch_x = randn(rng, 4, 3)
    batch_y = [0, 1, 0, 1]
    priors = init_two_hidden_priors(3, 2, 2, [0, 1]; seed=72)
    updated, stats = infer_batch_two_hidden_mlp(
        priors, batch_x, batch_y;
        max_inner=1, vector_transport=false)
    @assert size(updated.w1_m) == (2, 3)
    @assert size(updated.w2_m) == (2, 2)
    @assert size(updated.u_m) == (2, 2)
    @assert isfinite(stats.delta)
    prediction, probabilities = predict_two_hidden_mlp(updated, @view batch_x[1, :])
    @assert prediction in (0, 1)
    @assert isapprox(sum(probabilities), 1.0; atol=1e-10)
    println("two_hidden_smoke_test=passed")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--smoke-test" in ARGS
        two_hidden_smoke_test()
    else
        train_two_hidden_mlp_rxinfer_demo()
    end
end
