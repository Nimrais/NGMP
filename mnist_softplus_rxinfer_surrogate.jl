using RxInfer
using Rocket
using LinearAlgebra
using Random
using Statistics

include("mnist_softplus_ngmp_demo.jl")

# ============================================================================
# RxInfer surrogate version of the softplus image model.
#
# This script is intentionally written like poisson_surrogate_model.jl:
#
#   outer loop:
#     1. read current Gaussian marginals
#     2. compute pseudo-observation sites for non-conjugate factors
#     3. run one exact BP sweep on a fully Gaussian surrogate graph
#     4. damp sites in natural coordinates and repeat
#
# The honest model underneath is:
#
#   a[n,p,f]      ~ Normal(0, sigma_a^2)
#   x[n,p,f]      ~= softplus(a[n,p,f])
#   image[n,p,r]  ~ Normal(sum_f b[f,r] * x[n,p,f], sigma_y^2)
#   label[n]      ~ Bernoulli(sigmoid(c + sum_k u[k] * pooled_x[n,k]))
#
# The approximate model below contains none of the non-conjugate factors. It
# only contains their current Gaussian surrogate leaves. This is the same
# pattern as the Poisson example: the non-conjugacy lives in the site-refresh
# code, while RxInfer solves the frozen conjugate graph.
# ============================================================================

const SITE_EPS = 1e-8

function nat_to_observation(xi, lambda; scale = 1.0, eps = SITE_EPS)
    scaled_lambda = scale * max(lambda, 0.0)
    if scaled_lambda <= eps
        return 0.0, eps
    else
        return xi / max(lambda, eps), max(scaled_lambda, eps)
    end
end

function nat_to_observations(xi, lambda; scale = 1.0)
    y = similar(xi)
    Λ = similar(lambda)
    for i in eachindex(xi)
        y[i], Λ[i] = nat_to_observation(xi[i], lambda[i]; scale)
    end
    return y, Λ
end

function positive_softplus_to_a_site(ma, mx_cav, vx_cav, sigma_sp2)
    xi, lambda = softplus_to_a_site(ma, mx_cav, vx_cav, sigma_sp2)
    lambda = max(lambda, 0.0)
    return xi, lambda
end

@model function softplus_mnist_surrogate(
    a_site_y, a_site_Λ,
    x_sp_y, x_sp_Λ,
    x_img_y, x_img_Λ,
    b_img_y, b_img_Λ,
    x_cls_y, x_cls_Λ,
    x_smooth_y, x_smooth_Λ,
    u_cls_y, u_cls_Λ,
    c_cls_y, c_cls_Λ,
    b_prior_m, b_prior_Λ,
    u_prior_m, u_prior_Λ,
    c_prior_m, c_prior_Λ,
    sigma_a2,
    x_weak_precision,
    n_count,
    p_count,
    f_count,
    r_count,
    u_count,
)
    local b
    local u
    local a
    local x
    local c

    for f in 1:f_count, r in 1:r_count
        b[f, r] ~ Normal(mean = b_prior_m[f, r], precision = b_prior_Λ[f, r])
    end

    for k in 1:u_count
        u[k] ~ Normal(mean = u_prior_m[k], precision = u_prior_Λ[k])
    end
    c ~ Normal(mean = c_prior_m, precision = c_prior_Λ)

    for n in 1:n_count, p in 1:p_count, f in 1:f_count
        a[n, p, f] ~ Normal(mean = 0.0, precision = inv(sigma_a2))
        a_site_y[n, p, f] ~ Normal(mean = a[n, p, f], precision = a_site_Λ[n, p, f])

        x[n, p, f] ~ Normal(mean = 0.0, precision = x_weak_precision)
        x_sp_y[n, p, f] ~ Normal(mean = x[n, p, f], precision = x_sp_Λ[n, p, f])
        x_cls_y[n, p, f] ~ Normal(mean = x[n, p, f], precision = x_cls_Λ[n, p, f])
        x_smooth_y[n, p, f] ~ Normal(mean = x[n, p, f], precision = x_smooth_Λ[n, p, f])

        for r in 1:r_count
            x_img_y[n, p, f, r] ~ Normal(mean = x[n, p, f], precision = x_img_Λ[n, p, f, r])
            b_img_y[n, p, f, r] ~ Normal(mean = b[f, r], precision = b_img_Λ[n, p, f, r])
        end
    end

    for n in 1:n_count, k in 1:u_count
        u_cls_y[n, k] ~ Normal(mean = u[k], precision = u_cls_Λ[n, k])
    end

    for n in 1:n_count
        c_cls_y[n] ~ Normal(mean = c, precision = c_cls_Λ[n])
    end
end

function surrogate_observation(site, priors; global_site_scale = 1.0)
    a_y, a_Λ = nat_to_observations(site.a_xi, site.a_Λ)
    x_sp_y, x_sp_Λ = nat_to_observations(site.x_sp_xi, site.x_sp_Λ)
    x_img_y, x_img_Λ = nat_to_observations(site.x_img_xi, site.x_img_Λ)
    b_img_y, b_img_Λ = nat_to_observations(site.b_img_xi, site.b_img_Λ; scale = global_site_scale)
    x_cls_y, x_cls_Λ = nat_to_observations(site.x_cls_xi, site.x_cls_Λ)
    x_smooth_y, x_smooth_Λ = nat_to_observations(site.x_smooth_xi, site.x_smooth_Λ)
    u_cls_y, u_cls_Λ = nat_to_observations(site.u_cls_xi, site.u_cls_Λ; scale = global_site_scale)
    c_cls_y, c_cls_Λ = nat_to_observations(site.c_cls_xi, site.c_cls_Λ; scale = global_site_scale)

    return (
        a_site_y = a_y, a_site_Λ = a_Λ,
        x_sp_y = x_sp_y, x_sp_Λ = x_sp_Λ,
        x_img_y = x_img_y, x_img_Λ = x_img_Λ,
        b_img_y = b_img_y, b_img_Λ = b_img_Λ,
        x_cls_y = x_cls_y, x_cls_Λ = x_cls_Λ,
        x_smooth_y = x_smooth_y, x_smooth_Λ = x_smooth_Λ,
        u_cls_y = u_cls_y, u_cls_Λ = u_cls_Λ,
        c_cls_y = c_cls_y, c_cls_Λ = c_cls_Λ,
        b_prior_m = priors.b_m, b_prior_Λ = priors.b_Λ,
        u_prior_m = priors.u_m, u_prior_Λ = priors.u_Λ,
        c_prior_m = priors.c_m, c_prior_Λ = priors.c_Λ,
    )
end

function surrogate_dims(site, priors)
    n_count, p_count, f_count = size(site.x_sp_xi)
    r_count = size(priors.b_m, 2)
    u_count = length(priors.u_m)
    return (n_count = n_count, p_count = p_count, f_count = f_count,
            r_count = r_count, u_count = u_count)
end

function run_surrogate_bp(site, priors; sigma_a2 = 1.0, global_site_scale = 1.0)
    dims = surrogate_dims(site, priors)
    return infer(
        model = softplus_mnist_surrogate(; sigma_a2 = sigma_a2,
                                         x_weak_precision = 1e-6,
                                         dims...),
        data = surrogate_observation(site, priors; global_site_scale),
        options = (limit_stack_depth = 500,),
    )
end

mutable struct SurrogateStream
    stream::Any
    engine::Any
    latest::Dict{Symbol, Any}
    subscriptions::Vector{Any}
end

function start_surrogate_stream(site, priors; sigma_a2 = 1.0, global_site_scale = 1.0)
    first_event = surrogate_observation(site, priors; global_site_scale)
    dims = surrogate_dims(site, priors)
    stream = Subject(typeof(first_event))
    engine = infer(
        model = softplus_mnist_surrogate(; sigma_a2 = sigma_a2,
                                         x_weak_precision = 1e-6,
                                         dims...),
        datastream = stream,
        autoupdates = RxInfer.EmptyAutoUpdateSpecification,
        returnvars = (:a, :x, :b, :u, :c),
        autostart = false,
        options = (limit_stack_depth = 500,),
    )

    latest = Dict{Symbol, Any}()
    subscriptions = Any[
        subscribe!(engine.posteriors[:a], qs -> (latest[:a] = qs)),
        subscribe!(engine.posteriors[:x], qs -> (latest[:x] = qs)),
        subscribe!(engine.posteriors[:b], qs -> (latest[:b] = qs)),
        subscribe!(engine.posteriors[:u], qs -> (latest[:u] = qs)),
        subscribe!(engine.posteriors[:c], qs -> (latest[:c] = qs)),
    ]
    RxInfer.start(engine)
    return SurrogateStream(stream, engine, latest, subscriptions)
end

function stop_surrogate_stream!(state::SurrogateStream)
    RxInfer.stop(state.engine)
    foreach(unsubscribe!, state.subscriptions)
    empty!(state.subscriptions)
    return nothing
end

function run_surrogate_bp!(state::SurrogateStream, site, priors; global_site_scale = 1.0)
    empty!(state.latest)
    next!(state.stream, surrogate_observation(site, priors; global_site_scale))
    missing_keys = setdiff((:a, :x, :b, :u, :c), keys(state.latest))
    isempty(missing_keys) || error("streaming inference did not emit posteriors for $(missing_keys)")
    return extract_marginals_from_posteriors(state.latest)
end

function array_mean_var(qs)
    return mean.(qs), var.(qs)
end

function extract_marginals_from_posteriors(posteriors)
    ma, va = array_mean_var(posteriors[:a])
    mx, vx = array_mean_var(posteriors[:x])
    mb, vb = array_mean_var(posteriors[:b])
    mu, vu = array_mean_var(posteriors[:u])
    mc, vc = mean(posteriors[:c]), var(posteriors[:c])
    return (ma = ma, va = va, mx = mx, vx = vx, mb = mb, vb = vb,
            mu = mu, vu = vu, mc = mc, vc = vc)
end

function extract_marginals(result)
    return extract_marginals_from_posteriors(result.posteriors)
end

function init_priors(num_features, num_spatial_bins;
                     b_var = 1.0, u_var = 1.0, c_var = 4.0,
                     b_mean = 0.05, r_count = 4, seed = 1)
    rng = MersenneTwister(seed)
    b_m = b_mean .+ 0.01 .* randn(rng, num_features, r_count)
    b_Λ = fill(inv(b_var), num_features, r_count)
    u_count = num_features * num_spatial_bins
    u_m = 0.01 .* randn(rng, u_count)
    u_Λ = fill(inv(u_var), u_count)
    return (b_m = b_m, b_Λ = b_Λ, u_m = u_m, u_Λ = u_Λ,
            c_m = 0.0, c_Λ = inv(c_var))
end

function posterior_as_priors(m)
    return (b_m = m.mb, b_Λ = inv.(m.vb),
            u_m = m.mu, u_Λ = inv.(m.vu),
            c_m = m.mc, c_Λ = inv(m.vc))
end

function copy_priors(priors)
    return (b_m = copy(priors.b_m), b_Λ = copy(priors.b_Λ),
            u_m = copy(priors.u_m), u_Λ = copy(priors.u_Λ),
            c_m = priors.c_m, c_Λ = priors.c_Λ)
end

function zero_sites(n_count, p_count, f_count, r_count, u_count)
    return (
        a_xi = zeros(n_count, p_count, f_count),
        a_Λ = zeros(n_count, p_count, f_count),
        x_sp_xi = zeros(n_count, p_count, f_count),
        x_sp_Λ = zeros(n_count, p_count, f_count),
        x_img_xi = zeros(n_count, p_count, f_count, r_count),
        x_img_Λ = zeros(n_count, p_count, f_count, r_count),
        b_img_xi = zeros(n_count, p_count, f_count, r_count),
        b_img_Λ = zeros(n_count, p_count, f_count, r_count),
        x_cls_xi = zeros(n_count, p_count, f_count),
        x_cls_Λ = zeros(n_count, p_count, f_count),
        x_smooth_xi = zeros(n_count, p_count, f_count),
        x_smooth_Λ = zeros(n_count, p_count, f_count),
        u_cls_xi = zeros(n_count, u_count),
        u_cls_Λ = zeros(n_count, u_count),
        c_cls_xi = zeros(n_count),
        c_cls_Λ = zeros(n_count),
    )
end

function damp_sites(site, target, alpha)
    names = keys(site)
    values = map(names) do name
        (1 - alpha) .* getfield(site, name) .+ alpha .* getfield(target, name)
    end
    return NamedTuple{names}(values)
end

function add_smoothness_sites!(target, mx, smooth_precision)
    smooth_precision <= 0 && return target
    n_count, p_count, f_count = size(mx)
    side = round(Int, sqrt(p_count))
    side * side == p_count || return target

    for n in 1:n_count, row in 1:side, col in 1:side, f in 1:f_count
        p = (row - 1) * side + col
        neighbor_sum = 0.0
        neighbor_count = 0
        if row > 1
            neighbor_sum += mx[n, p - side, f]
            neighbor_count += 1
        end
        if row < side
            neighbor_sum += mx[n, p + side, f]
            neighbor_count += 1
        end
        if col > 1
            neighbor_sum += mx[n, p - 1, f]
            neighbor_count += 1
        end
        if col < side
            neighbor_sum += mx[n, p + 1, f]
            neighbor_count += 1
        end
        if neighbor_count > 0
            precision = smooth_precision * neighbor_count
            target.x_smooth_xi[n, p, f] += smooth_precision * neighbor_sum
            target.x_smooth_Λ[n, p, f] += precision
        end
    end
    return target
end

function refresh_sites(batch_x, batch_y, marginals, old_site;
                       sigma_a2 = 1.0,
                       sigma_sp2 = 0.05^2,
                       sigma_y2 = 0.18^2,
                       num_spatial_bins = 49,
                       smooth_precision = 0.0)
    p_count, r_count, n_count = size(batch_x)
    f_count = size(marginals.mx, 3)
    u_count = length(marginals.mu)
    bin_ids = patch_bin_ids(p_count; num_bins = num_spatial_bins)
    bin_counts = [count(==(q), bin_ids) for q in 1:num_spatial_bins]

    target = zero_sites(n_count, p_count, f_count, r_count, u_count)

    ma, va = marginals.ma, marginals.va
    mx, vx = marginals.mx, marginals.vx
    mb, vb = marginals.mb, marginals.vb
    mu, vu = marginals.mu, marginals.vu
    mc, vc = marginals.mc, marginals.vc
    add_smoothness_sites!(target, mx, smooth_precision)

    # Softplus sites: a -> x. The receiving a site uses the x cavity; the x
    # site uses the a prior cavity in this minimal graph.
    for n in 1:n_count, p in 1:p_count, f in 1:f_count
        x_cav_xi = mx[n, p, f] / vx[n, p, f] - old_site.x_sp_xi[n, p, f]
        x_cav_Λ = inv(vx[n, p, f]) - old_site.x_sp_Λ[n, p, f]
        mx_cav, vx_cav = gaussian_from_nat(x_cav_xi, x_cav_Λ)
        target.a_xi[n, p, f], target.a_Λ[n, p, f] =
            positive_softplus_to_a_site(ma[n, p, f], mx_cav, vx_cav, sigma_sp2)

        target.x_sp_xi[n, p, f], target.x_sp_Λ[n, p, f] =
            softplus_to_x_site(0.0, sigma_a2, sigma_sp2)
    end

    # Image likelihood product sites.
    for n in 1:n_count, p in 1:p_count, r in 1:r_count
        y = batch_x[p, r, n]
        prod_mean = zeros(f_count)
        prod_var = zeros(f_count)
        total_mean = 0.0
        total_var = sigma_y2
        for f in 1:f_count
            prod_mean[f], prod_var[f] = mean_var_product(mx[n, p, f], vx[n, p, f],
                                                         mb[f, r], vb[f, r])
            total_mean += prod_mean[f]
            total_var += prod_var[f]
        end
        for f in 1:f_count
            residual = y - (total_mean - prod_mean[f])
            noise = max(total_var - prod_var[f], sigma_y2)

            g, h = product_log_derivatives(mx[n, p, f], residual, mb[f, r], vb[f, r], noise)
            target.x_img_xi[n, p, f, r], target.x_img_Λ[n, p, f, r] =
                positive_site_from_grad_hess(mx[n, p, f], g, h; max_precision = 1e3)

            g, h = product_log_derivatives(mb[f, r], residual, mx[n, p, f], vx[n, p, f], noise)
            target.b_img_xi[n, p, f, r], target.b_img_Λ[n, p, f, r] =
                positive_site_from_grad_hess(mb[f, r], g, h; max_precision = 1e3)
        end
    end

    # Logistic label site, converted into product sites on pooled x and u.
    for n in 1:n_count
        gmean = zeros(u_count)
        gvar = zeros(u_count)
        for f in 1:f_count, q in 1:num_spatial_bins
            idx = (f - 1) * num_spatial_bins + q
            count_q = bin_counts[q]
            for p in 1:p_count
                if bin_ids[p] == q
                    gmean[idx] += mx[n, p, f] / count_q
                    gvar[idx] += vx[n, p, f] / count_q^2
                end
            end
        end

        prod_mean = zeros(u_count)
        prod_var = zeros(u_count)
        tmean = mc
        for k in 1:u_count
            prod_mean[k], prod_var[k] = mean_var_product(gmean[k], gvar[k], mu[k], vu[k])
            tmean += prod_mean[k]
        end
        xi_t, lambda_t, _ = logistic_site(tmean, batch_y[n])
        pseudo_y = xi_t / lambda_t
        pseudo_noise = inv(lambda_t)

        for f in 1:f_count, q in 1:num_spatial_bins
            idx = (f - 1) * num_spatial_bins + q
            residual = pseudo_y - (mc + sum(prod_mean) - prod_mean[idx])
            noise = max(pseudo_noise + vc + sum(prod_var) - prod_var[idx], pseudo_noise)

            gg, hh = product_log_derivatives(gmean[idx], residual, mu[idx], vu[idx], noise)
            xi_g, λ_g = positive_site_from_grad_hess(gmean[idx], gg, hh; max_precision = 1e3)
            count_q = bin_counts[q]
            for p in 1:p_count
                if bin_ids[p] == q
                    target.x_cls_xi[n, p, f] += xi_g / count_q
                    target.x_cls_Λ[n, p, f] += λ_g / count_q^2
                end
            end

            gu, hu = product_log_derivatives(mu[idx], residual, gmean[idx], gvar[idx], noise)
            target.u_cls_xi[n, idx], target.u_cls_Λ[n, idx] =
                positive_site_from_grad_hess(mu[idx], gu, hu; max_precision = 1e3)
        end

        residual_c = pseudo_y - sum(prod_mean)
        noise_c = max(pseudo_noise + sum(prod_var), pseudo_noise)
        target.c_cls_xi[n] = residual_c / noise_c
        target.c_cls_Λ[n] = inv(noise_c)
    end

    return target
end

function batch_reconstruction_mse(batch_x, m)
    p_count, r_count, n_count = size(batch_x)
    f_count = size(m.mx, 3)
    err = 0.0
    for n in 1:n_count, p in 1:p_count, r in 1:r_count
        pred = sum(m.mb[f, r] * m.mx[n, p, f] for f in 1:f_count)
        err += (batch_x[p, r, n] - pred)^2
    end
    return err / (n_count * p_count * r_count)
end

function infer_batch_rxinfer(priors, batch_x, batch_y;
                             num_features = size(priors.b_m, 1),
                             num_spatial_bins = 49,
                             sigma_a2 = 1.0,
                             sigma_sp2 = 0.05^2,
                             sigma_y2 = 0.18^2,
                             alpha = 0.08,
                             global_site_scale = 1.0,
                             smooth_precision = 0.0,
                             max_inner = 12,
                             tol = 1e-4,
                             verbose = false)
    p_count, r_count, n_count = size(batch_x)
    u_count = num_features * num_spatial_bins
    site = zero_sites(n_count, p_count, num_features, r_count, u_count)
    local result, marginals
    last_delta = Inf

    for inner in 1:max_inner
        result = run_surrogate_bp(site, priors; sigma_a2, global_site_scale)
        marginals = extract_marginals(result)
        target = refresh_sites(batch_x, batch_y, marginals, site;
                               sigma_a2, sigma_sp2, sigma_y2,
                               num_spatial_bins, smooth_precision)
        new_site = damp_sites(site, target, alpha)
        last_delta = maximum(maximum(abs.(getfield(new_site, name) .- getfield(site, name))) for name in keys(site))
        site = new_site
        verbose && @info "inner=$inner delta=$(round(last_delta, sigdigits=3))"
        last_delta < tol && break
    end

    result = run_surrogate_bp(site, priors; sigma_a2, global_site_scale)
    marginals = extract_marginals(result)
    return posterior_as_priors(marginals), (delta = last_delta,
                                            mse = batch_reconstruction_mse(batch_x, marginals),
                                            marginals = marginals)
end

function infer_batch_rxinfer_streaming(priors, batch_x, batch_y;
                                       num_features = size(priors.b_m, 1),
                                       num_spatial_bins = 49,
                                       sigma_a2 = 1.0,
                                       sigma_sp2 = 0.05^2,
                                       sigma_y2 = 0.18^2,
                                       alpha = 0.08,
                                       global_site_scale = 1.0,
                                       smooth_precision = 0.0,
                                       max_inner = 12,
                                       tol = 1e-4,
                                       verbose = false)
    p_count, r_count, n_count = size(batch_x)
    u_count = num_features * num_spatial_bins
    site = zero_sites(n_count, p_count, num_features, r_count, u_count)
    state = start_surrogate_stream(site, priors; sigma_a2, global_site_scale)
    last_delta = Inf

    try
        local marginals
        for inner in 1:max_inner
            marginals = run_surrogate_bp!(state, site, priors; global_site_scale)
            target = refresh_sites(batch_x, batch_y, marginals, site;
                                   sigma_a2, sigma_sp2, sigma_y2,
                                   num_spatial_bins, smooth_precision)
            new_site = damp_sites(site, target, alpha)
            last_delta = maximum(maximum(abs.(getfield(new_site, name) .- getfield(site, name))) for name in keys(site))
            site = new_site
            verbose && @info "inner=$inner delta=$(round(last_delta, sigdigits=3))"
            last_delta < tol && break
        end

        marginals = run_surrogate_bp!(state, site, priors; global_site_scale)
        return posterior_as_priors(marginals), (delta = last_delta,
                                                mse = batch_reconstruction_mse(batch_x, marginals),
                                                marginals = marginals)
    finally
        stop_surrogate_stream!(state)
    end
end

function infer_image_features_rx(priors, image_patches;
                                 num_features = size(priors.b_m, 1),
                                 sigma_a2 = 1.0,
                                 sigma_sp2 = 0.05^2,
                                 sigma_y2 = 0.18^2,
                                 alpha = 0.15,
                                 max_inner = 10)
    # Build a one-image surrogate with no classifier sites. This is used for
    # validation/test prediction after the global b/u/c posteriors have been
    # learned.
    batch_x = reshape(Array(image_patches), size(image_patches, 1), size(image_patches, 2), 1)
    batch_y = [0]
    p_count, r_count, n_count = size(batch_x)
    site = zero_sites(n_count, p_count, num_features, r_count, length(priors.u_m))

    local result, m
    for _ in 1:max_inner
        result = run_surrogate_bp(site, priors; sigma_a2, global_site_scale = 0.0)
        m = extract_marginals(result)
        target = zero_sites(n_count, p_count, num_features, r_count, length(priors.u_m))

        for n in 1:n_count, p in 1:p_count, f in 1:num_features
            target.x_sp_xi[n, p, f], target.x_sp_Λ[n, p, f] =
                softplus_to_x_site(0.0, sigma_a2, sigma_sp2)
            target.a_xi[n, p, f], target.a_Λ[n, p, f] =
                positive_softplus_to_a_site(m.ma[n, p, f], m.mx[n, p, f], m.vx[n, p, f], sigma_sp2)
        end

        for p in 1:p_count, r in 1:r_count
            y = batch_x[p, r, 1]
            prod_mean = [m.mx[1, p, f] * m.mb[f, r] for f in 1:num_features]
            prod_var = [m.vx[1, p, f] * m.vb[f, r] +
                        m.vx[1, p, f] * m.mb[f, r]^2 +
                        m.vb[f, r] * m.mx[1, p, f]^2 for f in 1:num_features]
            total_mean = sum(prod_mean)
            total_var = sigma_y2 + sum(prod_var)
            for f in 1:num_features
                residual = y - (total_mean - prod_mean[f])
                noise = max(total_var - prod_var[f], sigma_y2)
                g, h = product_log_derivatives(m.mx[1, p, f], residual, m.mb[f, r], m.vb[f, r], noise)
                target.x_img_xi[1, p, f, r], target.x_img_Λ[1, p, f, r] =
                    positive_site_from_grad_hess(m.mx[1, p, f], g, h; max_precision = 1e3)
            end
        end

        site = damp_sites(site, target, alpha)
    end

    result = run_surrogate_bp(site, priors; sigma_a2, global_site_scale = 0.0)
    return extract_marginals(result)
end

function refresh_image_sites(batch_x, marginals, old_site;
                             sigma_a2 = 1.0,
                             sigma_sp2 = 0.05^2,
                             sigma_y2 = 0.18^2,
                             smooth_precision = 0.0)
    p_count, r_count, n_count = size(batch_x)
    f_count = size(marginals.mx, 3)
    u_count = length(marginals.mu)
    target = zero_sites(n_count, p_count, f_count, r_count, u_count)

    ma, mx, vx = marginals.ma, marginals.mx, marginals.vx
    mb, vb = marginals.mb, marginals.vb
    add_smoothness_sites!(target, mx, smooth_precision)

    for n in 1:n_count, p in 1:p_count, f in 1:f_count
        target.x_sp_xi[n, p, f], target.x_sp_Λ[n, p, f] =
            softplus_to_x_site(0.0, sigma_a2, sigma_sp2)
        target.a_xi[n, p, f], target.a_Λ[n, p, f] =
            positive_softplus_to_a_site(ma[n, p, f], mx[n, p, f], vx[n, p, f], sigma_sp2)
    end

    for n in 1:n_count, p in 1:p_count, r in 1:r_count
        y = batch_x[p, r, n]
        prod_mean = [mx[n, p, f] * mb[f, r] for f in 1:f_count]
        prod_var = [vx[n, p, f] * vb[f, r] +
                    vx[n, p, f] * mb[f, r]^2 +
                    vb[f, r] * mx[n, p, f]^2 for f in 1:f_count]
        total_mean = sum(prod_mean)
        total_var = sigma_y2 + sum(prod_var)
        for f in 1:f_count
            residual = y - (total_mean - prod_mean[f])
            noise = max(total_var - prod_var[f], sigma_y2)
            g, h = product_log_derivatives(mx[n, p, f], residual, mb[f, r], vb[f, r], noise)
            target.x_img_xi[n, p, f, r], target.x_img_Λ[n, p, f, r] =
                positive_site_from_grad_hess(mx[n, p, f], g, h; max_precision = 1e3)
        end
    end

    return target
end

function infer_image_features_batch_rx_streaming(priors, batch_x;
                                                 num_features = size(priors.b_m, 1),
                                                 sigma_a2 = 1.0,
                                                 sigma_sp2 = 0.05^2,
                                                 sigma_y2 = 0.18^2,
                                                 alpha = 0.15,
                                                 smooth_precision = 0.0,
                                                 max_inner = 10,
                                                 tol = 1e-4)
    p_count, r_count, n_count = size(batch_x)
    site = zero_sites(n_count, p_count, num_features, r_count, length(priors.u_m))
    state = start_surrogate_stream(site, priors; sigma_a2, global_site_scale = 0.0)
    last_delta = Inf

    try
        local marginals
        for _ in 1:max_inner
            marginals = run_surrogate_bp!(state, site, priors; global_site_scale = 0.0)
            target = refresh_image_sites(batch_x, marginals, site;
                                         sigma_a2, sigma_sp2, sigma_y2,
                                         smooth_precision)
            new_site = damp_sites(site, target, alpha)
            last_delta = maximum(maximum(abs.(getfield(new_site, name) .- getfield(site, name))) for name in keys(site))
            site = new_site
            last_delta < tol && break
        end
        marginals = run_surrogate_bp!(state, site, priors; global_site_scale = 0.0)
        return marginals, last_delta
    finally
        stop_surrogate_stream!(state)
    end
end

function refresh_class_conditioned_sites(label, marginals, old_site;
                                         sigma_a2 = 1.0,
                                         sigma_sp2 = 0.05^2,
                                         num_spatial_bins = 49,
                                         label_strength = 1.0,
                                         smooth_precision = 0.0)
    n_count, p_count, f_count = size(marginals.mx)
    n_count == 1 || error("Class-conditioned generation currently expects one synthetic image.")
    u_count = length(marginals.mu)
    bin_ids = patch_bin_ids(p_count; num_bins = num_spatial_bins)
    bin_counts = [count(==(q), bin_ids) for q in 1:num_spatial_bins]
    target = zero_sites(n_count, p_count, f_count, size(old_site.x_img_xi, 4), u_count)

    ma, mx, vx = marginals.ma, marginals.mx, marginals.vx
    mu, vu = marginals.mu, marginals.vu
    mc, vc = marginals.mc, marginals.vc
    add_smoothness_sites!(target, mx, smooth_precision)

    for p in 1:p_count, f in 1:f_count
        x_cav_xi = mx[1, p, f] / vx[1, p, f] - old_site.x_sp_xi[1, p, f]
        x_cav_Λ = inv(vx[1, p, f]) - old_site.x_sp_Λ[1, p, f]
        mx_cav, vx_cav = gaussian_from_nat(x_cav_xi, x_cav_Λ)
        target.a_xi[1, p, f], target.a_Λ[1, p, f] =
            positive_softplus_to_a_site(ma[1, p, f], mx_cav, vx_cav, sigma_sp2)

        target.x_sp_xi[1, p, f], target.x_sp_Λ[1, p, f] =
            softplus_to_x_site(0.0, sigma_a2, sigma_sp2)
    end

    gmean = zeros(u_count)
    gvar = zeros(u_count)
    for f in 1:f_count, q in 1:num_spatial_bins
        idx = (f - 1) * num_spatial_bins + q
        count_q = bin_counts[q]
        for p in 1:p_count
            if bin_ids[p] == q
                gmean[idx] += mx[1, p, f] / count_q
                gvar[idx] += vx[1, p, f] / count_q^2
            end
        end
    end

    prod_mean = zeros(u_count)
    prod_var = zeros(u_count)
    tmean = mc
    for k in 1:u_count
        prod_mean[k], prod_var[k] = mean_var_product(gmean[k], gvar[k], mu[k], vu[k])
        tmean += prod_mean[k]
    end

    xi_t, lambda_t, _ = logistic_site(tmean, label)
    xi_t *= label_strength
    lambda_t *= label_strength
    pseudo_y = xi_t / lambda_t
    pseudo_noise = inv(lambda_t)

    for f in 1:f_count, q in 1:num_spatial_bins
        idx = (f - 1) * num_spatial_bins + q
        residual = pseudo_y - (mc + sum(prod_mean) - prod_mean[idx])
        noise = max(pseudo_noise + vc + sum(prod_var) - prod_var[idx], pseudo_noise)
        gg, hh = product_log_derivatives(gmean[idx], residual, mu[idx], vu[idx], noise)
        xi_g, λ_g = positive_site_from_grad_hess(gmean[idx], gg, hh; max_precision = 1e3)

        count_q = bin_counts[q]
        for p in 1:p_count
            if bin_ids[p] == q
                target.x_cls_xi[1, p, f] += xi_g / count_q
                target.x_cls_Λ[1, p, f] += λ_g / count_q^2
            end
        end
    end

    return target
end

function image_to_overlapping_patches(img14; patch_side = 3)
    grid_side = 14 - patch_side + 1
    patches = zeros(Float64, grid_side * grid_side, patch_side^2)
    idx = 1
    for i in 1:grid_side, j in 1:grid_side
        r = 1
        for di in 0:(patch_side - 1), dj in 0:(patch_side - 1)
            patches[idx, r] = img14[i + di, j + dj]
            r += 1
        end
        idx += 1
    end
    return patches
end

function select_binary_mnist_overlap(; digit0 = 0, digit1 = 1,
                                     ntrain = 256, ntest = 128, nval = 128,
                                     seed = 1, patch_side = 3)
    Random.seed!(seed)
    train = MNIST(split = :train)
    test = MNIST(split = :test)
    train_images = Float64.(train.features)
    test_images = Float64.(test.features)
    train_labels = Int.(train.targets)
    test_labels = Int.(test.targets)

    function collect_split(images, labels, nmax)
        inds = findall(l -> l == digit0 || l == digit1, labels)
        shuffle!(inds)
        inds = inds[1:min(nmax, length(inds))]
        grid_side = 14 - patch_side + 1
        x = zeros(Float64, grid_side * grid_side, patch_side^2, length(inds))
        y = zeros(Int, length(inds))
        for (k, idx) in enumerate(inds)
            img = downsample14(@view images[:, :, idx])
            x[:, :, k] .= image_to_overlapping_patches(img; patch_side)
            y[k] = labels[idx] == digit1 ? 1 : 0
        end
        return x, y
    end

    train_all_x, train_all_y = collect_split(train_images, train_labels, ntrain + nval)
    ntrain_actual = min(ntrain, length(train_all_y))
    train_x = train_all_x[:, :, 1:ntrain_actual]
    train_y = train_all_y[1:ntrain_actual]
    val_start = ntrain_actual + 1
    val_x = train_all_x[:, :, val_start:end]
    val_y = train_all_y[val_start:end]
    test_x, test_y = collect_split(test_images, test_labels, ntest)
    return BinaryMnist(train_x, train_y, val_x, val_y, test_x, test_y)
end

function patches_to_image14(patches; patch_side = 2)
    img = zeros(Float64, 14, 14)
    weight = zeros(Float64, 14, 14)
    grid_side = round(Int, sqrt(size(patches, 1)))
    stride = patch_side == 2 && grid_side == 7 ? 2 : 1
    idx = 1
    for row in 1:grid_side, col in 1:grid_side
        i = 1 + (row - 1) * stride
        j = 1 + (col - 1) * stride
        r = 1
        for di in 0:(patch_side - 1), dj in 0:(patch_side - 1)
            img[i + di, j + dj] += patches[idx, r]
            weight[i + di, j + dj] += 1.0
            r += 1
        end
        idx += 1
    end
    img ./= max.(weight, 1.0)
    return img
end

function decode_patches_from_x(mx, b_m)
    p_count, f_count = size(mx)
    r_count = size(b_m, 2)
    patches = zeros(Float64, p_count, r_count)
    for p in 1:p_count, r in 1:r_count
        patches[p, r] = sum(mx[p, f] * b_m[f, r] for f in 1:f_count)
    end
    return patches
end

function classifier_evidence_map(mx, priors; num_spatial_bins = 49)
    p_count, f_count = size(mx)
    bin_ids = patch_bin_ids(p_count; num_bins = num_spatial_bins)
    bin_counts = [count(==(q), bin_ids) for q in 1:num_spatial_bins]
    evidence = zeros(Float64, p_count)
    for f in 1:f_count, q in 1:num_spatial_bins
        idx = (f - 1) * num_spatial_bins + q
        for p in 1:p_count
            if bin_ids[p] == q
                evidence[p] += mx[p, f] * priors.u_m[idx] / bin_counts[q]
            end
        end
    end
    grid_side = round(Int, sqrt(p_count))
    return reshape(evidence, grid_side, grid_side)
end

function class_conditioned_generation(priors, label;
                                      num_spatial_bins = 49,
                                      p_count = 49,
                                      patch_side = 2,
                                      sigma_a2 = 1.0,
                                      sigma_sp2 = 0.05^2,
                                      label_strength = 1.0,
                                      smooth_precision = 0.0,
                                      alpha = 0.15,
                                      max_inner = 40,
                                      tol = 1e-4)
    num_features = size(priors.b_m, 1)
    site = zero_sites(1, p_count, num_features, size(priors.b_m, 2), length(priors.u_m))
    last_delta = Inf
    local marginals

    for inner in 1:max_inner
        result = run_surrogate_bp(site, priors; sigma_a2, global_site_scale = 0.0)
        marginals = extract_marginals(result)
        target = refresh_class_conditioned_sites(label, marginals, site;
                                                 sigma_a2, sigma_sp2,
                                                 num_spatial_bins,
                                                 label_strength, smooth_precision)
        new_site = damp_sites(site, target, alpha)
        last_delta = maximum(maximum(abs.(getfield(new_site, name) .- getfield(site, name))) for name in keys(site))
        site = new_site
        last_delta < tol && break
    end

    result = run_surrogate_bp(site, priors; sigma_a2, global_site_scale = 0.0)
    marginals = extract_marginals(result)
    mx = dropdims(marginals.mx, dims = 1)
    patches = decode_patches_from_x(mx, priors.b_m)
    image = clamp.(patches_to_image14(patches; patch_side), 0.0, 1.0)
    grid_side = round(Int, sqrt(p_count))
    x_total = reshape(sum(mx; dims = 2), grid_side, grid_side)
    evidence = classifier_evidence_map(mx, priors; num_spatial_bins)
    logit = priors.c_m + sum(evidence)
    return (label = label, image = image, patches = patches, mx = mx,
            x_total = x_total, evidence = evidence, logit = logit,
            prob1 = sigmoid(logit), delta = last_delta, marginals = marginals)
end

function save_class_generation_figure(path, priors; num_spatial_bins = 49,
                                      p_count = 49, patch_side = 2,
                                      label_strength = 1.0,
                                      smooth_precision = 0.0)
    gen0 = class_conditioned_generation(priors, 0; num_spatial_bins, p_count,
                                        patch_side, label_strength, smooth_precision)
    gen1 = class_conditioned_generation(priors, 1; num_spatial_bins, p_count,
                                        patch_side, label_strength, smooth_precision)

    @eval using Plots
    return Base.invokelatest(save_class_generation_figure_loaded, path, gen0, gen1, label_strength)
end

function save_class_generation_figure_loaded(path, gen0, gen1, label_strength)
    plt = Plots.plot(layout = (3, 2), size = (900, 1050))
    gens = (gen0, gen1)
    for (col, gen) in enumerate(gens)
        Plots.heatmap!(plt[col], reverse(gen.image; dims = 1),
                       color = :grays, aspect_ratio = :equal,
                       axis = false, title = "label=$(gen.label), p1=$(round(gen.prob1, digits=3)), strength=$(label_strength)")
        Plots.heatmap!(plt[col + 2], reverse(gen.x_total; dims = 1),
                       color = :viridis, aspect_ratio = :equal,
                       axis = false, title = "total local x")
        Plots.heatmap!(plt[col + 4], reverse(gen.evidence; dims = 1),
                       color = Plots.cgrad([:blue, :white, :red]), aspect_ratio = :equal,
                       axis = false, title = "classifier evidence")
    end
    Plots.savefig(plt, path)
    return (path = path, label0 = gen0, label1 = gen1)
end

function predict_rx(priors, image_patches; num_spatial_bins = 49)
    m = infer_image_features_rx(priors, image_patches)
    p_count = size(image_patches, 1)
    f_count = size(m.mx, 3)
    bin_ids = patch_bin_ids(p_count; num_bins = num_spatial_bins)
    bin_counts = [count(==(q), bin_ids) for q in 1:num_spatial_bins]
    pooled = zeros(length(priors.u_m))
    for f in 1:f_count, q in 1:num_spatial_bins
        idx = (f - 1) * num_spatial_bins + q
        for p in 1:p_count
            if bin_ids[p] == q
                pooled[idx] += m.mx[1, p, f] / bin_counts[q]
            end
        end
    end
    return sigmoid(priors.c_m + dot(priors.u_m, pooled))
end

function predict_batch_rx_streaming(priors, batch_x;
                                    num_spatial_bins = 49,
                                    smooth_precision = 0.0,
                                    max_inner = 10,
                                    alpha = 0.15)
    m, delta = infer_image_features_batch_rx_streaming(priors, batch_x;
                                                       smooth_precision,
                                                       max_inner,
                                                       alpha)
    n_count, p_count, f_count = size(m.mx)
    bin_ids = patch_bin_ids(p_count; num_bins = num_spatial_bins)
    bin_counts = [count(==(q), bin_ids) for q in 1:num_spatial_bins]
    probs = zeros(Float64, n_count)

    for n in 1:n_count
        pooled = zeros(length(priors.u_m))
        for f in 1:f_count, q in 1:num_spatial_bins
            idx = (f - 1) * num_spatial_bins + q
            for p in 1:p_count
                if bin_ids[p] == q
                    pooled[idx] += m.mx[n, p, f] / bin_counts[q]
                end
            end
        end
        probs[n] = sigmoid(priors.c_m + dot(priors.u_m, pooled))
    end
    return probs, delta
end

function evaluate_rx(priors, x, y; max_images = size(x, 3), num_spatial_bins = 49)
    n = min(max_images, size(x, 3))
    correct = 0
    for i in 1:n
        p = predict_rx(priors, @view x[:, :, i]; num_spatial_bins)
        correct += (p >= 0.5) == (y[i] == 1)
    end
    return correct / n
end

function evaluate_rx_streaming(priors, x, y;
                               max_images = size(x, 3),
                               num_spatial_bins = 49,
                               batch_size = 4,
                               smooth_precision = 0.0,
                               max_inner = 10,
                               alpha = 0.15,
                               log_every = 0)
    n = min(max_images, size(x, 3))
    correct = 0
    seen = 0
    deltas = Float64[]

    for (batch_idx, start_idx) in enumerate(1:batch_size:n)
        inds = start_idx:min(start_idx + batch_size - 1, n)
        probs, delta = predict_batch_rx_streaming(priors, x[:, :, inds];
                                                  num_spatial_bins,
                                                  smooth_precision,
                                                  max_inner,
                                                  alpha)
        for (j, idx) in enumerate(inds)
            correct += (probs[j] >= 0.5) == (y[idx] == 1)
            seen += 1
        end
        push!(deltas, delta)
        if log_every > 0 && (batch_idx == 1 || batch_idx % log_every == 0)
            println("eval_batch=$batch_idx seen=$seen acc=$(round(correct / seen, digits=4)) delta=$(round(delta, sigdigits=3))")
        end
    end

    return (acc = correct / seen, mean_delta = mean(deltas), n = seen)
end

function train_rxinfer_priors_only(; digit0 = 0, digit1 = 1,
                                   ntrain = 1000, num_features = 4,
                                   num_spatial_bins = 49, batch_size = 4,
                                   epochs = 1, seed = 1, alpha = 0.08,
                                   streaming = true, log_every = 25,
                                   patch_side = 2, overlap = false,
                                   smooth_precision = 0.0,
                                   max_inner = 12)
    data = overlap || patch_side != 2 ?
        select_binary_mnist_overlap(; digit0, digit1, ntrain, nval = 0, ntest = 0, seed, patch_side) :
        select_binary_mnist(; digit0, digit1, ntrain, nval = 0, ntest = 0, seed)
    priors = init_priors(num_features, num_spatial_bins;
                         r_count = size(data.train_x, 2), seed = seed + 10)
    site_scale = inv(epochs)
    rng = MersenneTwister(seed + 20)
    batch_infer = streaming ? infer_batch_rxinfer_streaming : infer_batch_rxinfer

    println("RxInfer priors-only MNIST $digit0 vs $digit1")
    println("train=$(length(data.train_y)) F=$num_features bins=$num_spatial_bins patch_side=$patch_side overlap=$(overlap || patch_side != 2) smooth_precision=$smooth_precision batch=$batch_size epochs=$epochs site_scale=$(round(site_scale, digits=4))")

    for epoch in 1:epochs
        order = shuffle(rng, collect(1:length(data.train_y)))
        for (batch_idx, start_idx) in enumerate(1:batch_size:length(order))
            inds = order[start_idx:min(start_idx + batch_size - 1, end)]
            priors, stats = batch_infer(priors, data.train_x[:, :, inds], data.train_y[inds];
                                        num_features, num_spatial_bins,
                                        alpha, global_site_scale = site_scale,
                                        smooth_precision, max_inner)
            if log_every > 0 && (batch_idx == 1 || batch_idx % log_every == 0)
                println("epoch=$epoch batch=$batch_idx mse=$(round(stats.mse, digits=5)) delta=$(round(stats.delta, sigdigits=3))")
            end
        end
    end

    return (priors = priors, data = data)
end

function train_rxinfer_demo(; digit0 = 0, digit1 = 1,
                            ntrain = 64, nval = 32, ntest = 32,
                            num_features = 4, num_spatial_bins = 49,
                            batch_size = 8, epochs = 3,
                            seed = 1, alpha = 0.08,
                            streaming = false,
                            patch_side = 2, overlap = false,
                            smooth_precision = 0.0,
                            max_inner = 12)
    data = overlap || patch_side != 2 ?
        select_binary_mnist_overlap(; digit0, digit1, ntrain, nval, ntest, seed, patch_side) :
        select_binary_mnist(; digit0, digit1, ntrain, nval, ntest, seed)
    priors = init_priors(num_features, num_spatial_bins;
                         r_count = size(data.train_x, 2), seed = seed + 10)
    site_scale = inv(epochs)
    rng = MersenneTwister(seed + 20)
    batch_infer = streaming ? infer_batch_rxinfer_streaming : infer_batch_rxinfer

    mode = streaming ? "streaming" : "static"
    println("RxInfer surrogate MNIST $digit0 vs $digit1 ($mode)")
    println("train=$(length(data.train_y)) val=$(length(data.val_y)) test=$(length(data.test_y)) F=$num_features bins=$num_spatial_bins patch_side=$patch_side overlap=$(overlap || patch_side != 2) smooth_precision=$smooth_precision batch=$batch_size epochs=$epochs site_scale=$(round(site_scale, digits=4))")

    history = NamedTuple[]
    best_val_acc = -Inf
    best_priors = copy_priors(priors)
    best_epoch = 0
    for epoch in 1:epochs
        order = shuffle(rng, collect(1:length(data.train_y)))
        for (batch_idx, start_idx) in enumerate(1:batch_size:length(order))
            inds = order[start_idx:min(start_idx + batch_size - 1, end)]
            priors, stats = batch_infer(priors, data.train_x[:, :, inds], data.train_y[inds];
                                        num_features, num_spatial_bins,
                                        alpha, global_site_scale = site_scale,
                                        smooth_precision, max_inner)
            println("epoch=$epoch batch=$batch_idx mse=$(round(stats.mse, digits=5)) delta=$(round(stats.delta, sigdigits=3))")
        end
        train_acc = evaluate_rx(priors, data.train_x, data.train_y; max_images = min(64, length(data.train_y)), num_spatial_bins)
        val_acc = evaluate_rx(priors, data.val_x, data.val_y; max_images = min(64, length(data.val_y)), num_spatial_bins)
        push!(history, (epoch = epoch, train_acc = train_acc, val_acc = val_acc))
        println("epoch=$epoch validation train_acc=$(round(train_acc, digits=3)) val_acc=$(round(val_acc, digits=3))")
        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_priors = copy_priors(priors)
            best_epoch = epoch
        end
    end

    test_acc = evaluate_rx(best_priors, data.test_x, data.test_y; max_images = min(64, length(data.test_y)), num_spatial_bins)
    println("best_val_epoch=$best_epoch best_val_acc=$(round(best_val_acc, digits=3)) holdout_test_acc=$(round(test_acc, digits=3))")
    return (priors = best_priors, history = history, data = data, test_acc = test_acc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_rxinfer_demo()
end
