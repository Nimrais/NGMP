using LinearAlgebra
using Random
using Statistics

try
    @eval using MLDatasets: MNIST
catch err
    error("""
    MLDatasets is required for this demo.

    Install it in this project with:
        julia --project=. -e 'using Pkg; Pkg.add("MLDatasets")'

    Original loading error:
    $err
    """)
end

# ============================================================================
# Small proof harness for the softplus generative MNIST idea.
#
# Model sketch for a downsampled image split into non-overlapping 2x2 patches:
#
#   a[n,p,f]         ~ Normal(0, sigma_a^2)
#   x[n,p,f]         ~ Normal(softplus(a[n,p,f]), sigma_sp^2)
#   pixel[n,p,r]     ~ Normal(sum_f b[f,r] * x[n,p,f], sigma_y^2)
#   label[n]         ~ Bernoulli(sigmoid(c + sum_f u[f] * mean_p x[n,p,f]))
#
# b, u and c are global Gaussian variables. The local a/x variables are
# re-inferred for each mini-batch. Non-conjugate factors do not use full
# quadrature here; each target message is the delta Bonnet/Price site:
#
#   Lambda* = - ell''(m),   xi* = ell'(m) + Lambda* m
#
# for the exact local BP log-message ell, evaluated at the receiving marginal
# mean. Sites are damped in natural coordinates.
#
# This is intentionally not an RxInfer custom rule yet. It is a compact
# experiment to see whether the factorization and message loop behave on real
# downsampled MNIST before investing in node/rule integration.
# ============================================================================

softplus(x) = x > 30 ? x : log1p(exp(x))
sigmoid(x) = x >= 0 ? inv(1 + exp(-x)) : (exp(x) / (1 + exp(x)))

function gaussian_from_nat(xi, lambda; min_precision = 1e-6)
    lam = max(lambda, min_precision)
    return xi / lam, inv(lam)
end

function damp_site!(xi, lambda, xi_star, lambda_star, alpha)
    xi .= (1 - alpha) .* xi .+ alpha .* xi_star
    lambda .= (1 - alpha) .* lambda .+ alpha .* lambda_star
    return nothing
end

function product_log_derivatives(t, y, m_other, v_other, noise_var)
    # ell(t) = log Normal(y | m_other * t, noise_var + v_other * t^2)
    s = max(noise_var + v_other * t^2, 1e-10)
    e = y - m_other * t
    n = m_other * e - v_other * t
    k = m_other^2 + v_other
    ell1 = n / s + v_other * t * e^2 / s^2
    ell2 = -k / s - 2v_other * t * n / s^2 +
           v_other * e^2 / s^2 - 2v_other * m_other * t * e / s^2 -
           4v_other^2 * t^2 * e^2 / s^3
    return ell1, ell2
end

function site_from_grad_hess(mean_value, grad_value, hess_value;
                             min_site_precision = -Inf,
                             max_abs_precision = 1e6)
    lambda = clamp(-hess_value, min_site_precision, max_abs_precision)
    xi = grad_value + lambda * mean_value
    return xi, lambda
end

function positive_site_from_grad_hess(mean_value, grad_value, hess_value;
                                      max_precision = 1e3)
    lambda = clamp(max(-hess_value, 0.0), 0.0, max_precision)
    xi = grad_value + lambda * mean_value
    return xi, lambda
end

function softplus_to_a_site(ma, mx_cav, vx_cav, sigma_sp2)
    # ell(a) = log Normal(mx_cav | softplus(a), sigma_sp2 + vx_cav)
    s = softplus(ma)
    g = sigmoid(ma)
    h = g * (1 - g)
    r = max(sigma_sp2 + vx_cav, 1e-8)
    ell1 = (mx_cav - s) * g / r
    ell2 = -(g^2 + (s - mx_cav) * h) / r
    return site_from_grad_hess(ma, ell1, ell2; min_site_precision = -10.0)
end

function softplus_to_x_site(ma_cav, va_cav, sigma_sp2)
    # Delta approximation to int N(x | softplus(a), sigma_sp2) q(a) da.
    s = softplus(ma_cav)
    g = sigmoid(ma_cav)
    r = max(sigma_sp2 + g^2 * va_cav, 1e-8)
    return s / r, inv(r)
end

function logistic_site(mt, label)
    p = clamp(sigmoid(mt), 1e-5, 1 - 1e-5)
    grad = label - p
    lambda = p * (1 - p)       # - second derivative of Bernoulli log-likelihood
    xi = grad + lambda * mt
    return xi, lambda, p
end

function mean_var_product(ma, va, mb, vb)
    return ma * mb, va * vb + va * mb^2 + vb * ma^2
end

struct BinaryMnist
    train_x::Array{Float64, 3}  # patches x pixel-offset x image
    train_y::Vector{Int}
    val_x::Array{Float64, 3}
    val_y::Vector{Int}
    test_x::Array{Float64, 3}
    test_y::Vector{Int}
end

function downsample14(img28)
    out = zeros(Float64, 14, 14)
    for i in 1:14, j in 1:14
        out[i, j] = mean(@view img28[(2i-1):(2i), (2j-1):(2j)])
    end
    return out
end

function image_to_patches(img14)
    patches = zeros(Float64, 49, 4)
    idx = 1
    for i in 1:2:13, j in 1:2:13
        patches[idx, 1] = img14[i, j]
        patches[idx, 2] = img14[i + 1, j]
        patches[idx, 3] = img14[i, j + 1]
        patches[idx, 4] = img14[i + 1, j + 1]
        idx += 1
    end
    return patches
end

function patch_bin_ids(p_count; num_bins = 4)
    side = round(Int, sqrt(p_count))
    side * side == p_count || return ones(Int, p_count)
    if num_bins == 1
        return ones(Int, p_count)
    elseif num_bins == p_count
        return collect(1:p_count)
    elseif num_bins != 4
        error("Only num_bins=1, num_bins=4, or num_bins=p_count are implemented in this demo.")
    end
    ids = zeros(Int, p_count)
    idx = 1
    row_cut = cld(side, 2)
    col_cut = cld(side, 2)
    for row in 1:side, col in 1:side
        ids[idx] = 1 + (row > row_cut ? 2 : 0) + (col > col_cut ? 1 : 0)
        idx += 1
    end
    return ids
end

function select_binary_mnist(; digit0 = 3, digit1 = 5, ntrain = 256, ntest = 128,
                             nval = 128, seed = 1)
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
        x = zeros(Float64, 49, 4, length(inds))
        y = zeros(Int, length(inds))
        for (k, idx) in enumerate(inds)
            img = downsample14(@view images[:, :, idx])
            x[:, :, k] .= image_to_patches(img)
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

mutable struct GlobalState
    b_xi::Matrix{Float64}
    b_lambda::Matrix{Float64}
    u_xi::Vector{Float64}
    u_lambda::Vector{Float64}
    c_xi::Float64
    c_lambda::Float64
end

function init_state(num_features; seed = 2, b_mean = 0.05, b_var = 1.0,
                    u_var = 1.0, c_var = 4.0, num_spatial_bins = 4)
    Random.seed!(seed)
    b_lambda = fill(inv(b_var), num_features, 4)
    b_xi = b_lambda .* (b_mean .+ 0.01 .* randn(num_features, 4))
    u_count = num_features * num_spatial_bins
    u_lambda = fill(inv(u_var), u_count)
    u_xi = 0.01 .* randn(u_count) .* u_lambda
    return GlobalState(b_xi, b_lambda, u_xi, u_lambda, 0.0, inv(c_var))
end

function means_vars(state::GlobalState)
    mb = state.b_xi ./ state.b_lambda
    vb = inv.(state.b_lambda)
    mu = state.u_xi ./ state.u_lambda
    vu = inv.(state.u_lambda)
    mc = state.c_xi / state.c_lambda
    vc = inv(state.c_lambda)
    return mb, vb, mu, vu, mc, vc
end

function infer_batch!(state::GlobalState, batch_x, batch_y;
                      num_features = size(state.b_xi, 1),
                      sigma_a2 = 1.0,
                      sigma_sp2 = 0.05^2,
                      sigma_y2 = 0.18^2,
                      alpha = 0.08,
                      global_site_scale = 1.0,
                      max_inner = 35,
                      tol = 1e-4,
                      positive_product_sites = true,
                      verbose = false)
    p_count, r_count, n_count = size(batch_x)
    f_count = num_features
    u_count = length(state.u_xi)
    num_spatial_bins = max(1, div(u_count, f_count))
    bin_ids = patch_bin_ids(p_count; num_bins = num_spatial_bins)
    bin_counts = [count(==(q), bin_ids) for q in 1:num_spatial_bins]

    # Global base natural parameters are the posterior from previous batches.
    b_base_xi = copy(state.b_xi)
    b_base_lambda = copy(state.b_lambda)
    u_base_xi = copy(state.u_xi)
    u_base_lambda = copy(state.u_lambda)
    c_base_xi = state.c_xi
    c_base_lambda = state.c_lambda

    # Local marginals and sites for this batch.
    a_xi_prior = 0.0
    a_lambda_prior = inv(sigma_a2)
    ma = zeros(n_count, p_count, f_count)
    va = fill(sigma_a2, n_count, p_count, f_count)
    mx = fill(softplus(0.0), n_count, p_count, f_count)
    vx = fill(0.25, n_count, p_count, f_count)

    sp_a_xi = zeros(n_count, p_count, f_count)
    sp_a_lambda = zeros(n_count, p_count, f_count)
    sp_x_xi = zeros(n_count, p_count, f_count)
    sp_x_lambda = zeros(n_count, p_count, f_count)

    img_x_xi = zeros(n_count, p_count, f_count, r_count)
    img_x_lambda = zeros(n_count, p_count, f_count, r_count)
    img_b_xi = zeros(n_count, p_count, f_count, r_count)
    img_b_lambda = zeros(n_count, p_count, f_count, r_count)

    cls_x_xi = zeros(n_count, p_count, f_count)
    cls_x_lambda = zeros(n_count, p_count, f_count)
    cls_u_xi = zeros(n_count, u_count)
    cls_u_lambda = zeros(n_count, u_count)
    cls_c_xi = zeros(n_count)
    cls_c_lambda = zeros(n_count)

    last_delta = Inf
    batch_accuracy = 0.0
    batch_mse = 0.0

    for inner in 1:max_inner
        old_mb, _, old_mu, _, old_mc, _ = means_vars(state)
        old_mx = copy(mx)

        # Current global marginals include this batch's damped sites.
        state.b_xi .= b_base_xi .+ global_site_scale .* dropdims(sum(img_b_xi; dims = (1, 2)); dims = (1, 2))
        state.b_lambda .= b_base_lambda .+ global_site_scale .* dropdims(sum(img_b_lambda; dims = (1, 2)); dims = (1, 2))
        state.u_xi .= u_base_xi .+ global_site_scale .* vec(sum(cls_u_xi; dims = 1))
        state.u_lambda .= u_base_lambda .+ global_site_scale .* vec(sum(cls_u_lambda; dims = 1))
        state.c_xi = c_base_xi + global_site_scale * sum(cls_c_xi)
        state.c_lambda = c_base_lambda + global_site_scale * sum(cls_c_lambda)
        state.b_lambda .= max.(state.b_lambda, 1e-5)
        state.u_lambda .= max.(state.u_lambda, 1e-5)
        state.c_lambda = max(state.c_lambda, 1e-5)
        mb, vb, mu, vu, mc, vc = means_vars(state)

        # Local marginals from current local sites.
        for n in 1:n_count, p in 1:p_count, f in 1:f_count
            ax = a_xi_prior + sp_a_xi[n, p, f]
            al = a_lambda_prior + sp_a_lambda[n, p, f]
            ma[n, p, f], va[n, p, f] = gaussian_from_nat(ax, al)

            xx = sp_x_xi[n, p, f] + sum(@view img_x_xi[n, p, f, :]) + cls_x_xi[n, p, f]
            xl = sp_x_lambda[n, p, f] + sum(@view img_x_lambda[n, p, f, :]) + cls_x_lambda[n, p, f]
            mx[n, p, f], vx[n, p, f] = gaussian_from_nat(xx, xl)
        end

        # Softplus factor sites.
        sp_a_xi_star = similar(sp_a_xi)
        sp_a_lambda_star = similar(sp_a_lambda)
        sp_x_xi_star = similar(sp_x_xi)
        sp_x_lambda_star = similar(sp_x_lambda)
        for n in 1:n_count, p in 1:p_count, f in 1:f_count
            # cavity for x into softplus = marginal minus old softplus-to-x site
            x_cav_xi = mx[n, p, f] / vx[n, p, f] - sp_x_xi[n, p, f]
            x_cav_lambda = inv(vx[n, p, f]) - sp_x_lambda[n, p, f]
            mx_cav, vx_cav = gaussian_from_nat(x_cav_xi, x_cav_lambda)
            sp_a_xi_star[n, p, f], sp_a_lambda_star[n, p, f] =
                softplus_to_a_site(ma[n, p, f], mx_cav, vx_cav, sigma_sp2)

            # cavity for a into softplus = prior, because no other a factors exist
            sp_x_xi_star[n, p, f], sp_x_lambda_star[n, p, f] =
                softplus_to_x_site(0.0, sigma_a2, sigma_sp2)
        end
        damp_site!(sp_a_xi, sp_a_lambda, sp_a_xi_star, sp_a_lambda_star, alpha)
        damp_site!(sp_x_xi, sp_x_lambda, sp_x_xi_star, sp_x_lambda_star, alpha)

        # Image likelihood product sites.
        img_x_xi_star = similar(img_x_xi)
        img_x_lambda_star = similar(img_x_lambda)
        img_b_xi_star = similar(img_b_xi)
        img_b_lambda_star = similar(img_b_lambda)
        fill!(img_x_xi_star, 0.0); fill!(img_x_lambda_star, 0.0)
        fill!(img_b_xi_star, 0.0); fill!(img_b_lambda_star, 0.0)

        for n in 1:n_count, p in 1:p_count, r in 1:r_count
            y = batch_x[p, r, n]
            prod_mean = zeros(f_count)
            prod_var = zeros(f_count)
            xb_mean_total = 0.0
            xb_var_total = sigma_y2
            for f in 1:f_count
                prod_mean[f], prod_var[f] = mean_var_product(mx[n, p, f], vx[n, p, f],
                                                             mb[f, r], vb[f, r])
                xb_mean_total += prod_mean[f]
                xb_var_total += prod_var[f]
            end
            for f in 1:f_count
                other_mean = xb_mean_total - prod_mean[f]
                other_var = max(xb_var_total - prod_var[f], sigma_y2)
                residual = y - other_mean

                # x receiving site, using b cavity approximately as current b marginal.
                g, h = product_log_derivatives(mx[n, p, f], residual, mb[f, r], vb[f, r], other_var)
                img_x_xi_star[n, p, f, r], img_x_lambda_star[n, p, f, r] =
                    positive_product_sites ?
                    positive_site_from_grad_hess(mx[n, p, f], g, h; max_precision = 1e3) :
                    site_from_grad_hess(mx[n, p, f], g, h; min_site_precision = -5.0)

                # b receiving site, using x cavity approximately as current x marginal.
                g, h = product_log_derivatives(mb[f, r], residual, mx[n, p, f], vx[n, p, f], other_var)
                img_b_xi_star[n, p, f, r], img_b_lambda_star[n, p, f, r] =
                    positive_product_sites ?
                    positive_site_from_grad_hess(mb[f, r], g, h; max_precision = 1e3) :
                    site_from_grad_hess(mb[f, r], g, h; min_site_precision = -5.0)
            end
        end
        damp_site!(img_x_xi, img_x_lambda, img_x_xi_star, img_x_lambda_star, alpha)
        damp_site!(img_b_xi, img_b_lambda, img_b_xi_star, img_b_lambda_star, alpha)

        # Binary softmax/logistic classifier sites through pooled x.
        cls_x_xi_star = similar(cls_x_xi)
        cls_x_lambda_star = similar(cls_x_lambda)
        cls_u_xi_star = similar(cls_u_xi)
        cls_u_lambda_star = similar(cls_u_lambda)
        cls_c_xi_star = similar(cls_c_xi)
        cls_c_lambda_star = similar(cls_c_lambda)
        fill!(cls_x_xi_star, 0.0); fill!(cls_x_lambda_star, 0.0)
        fill!(cls_u_xi_star, 0.0); fill!(cls_u_lambda_star, 0.0)
        fill!(cls_c_xi_star, 0.0); fill!(cls_c_lambda_star, 0.0)

        correct = 0
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
            tvar = vc
            for idx in 1:u_count
                prod_mean[idx], prod_var[idx] = mean_var_product(gmean[idx], gvar[idx], mu[idx], vu[idx])
                tmean += prod_mean[idx]
                tvar += prod_var[idx]
            end
            xi_t, lambda_t, prob = logistic_site(tmean, batch_y[n])
            correct += (prob >= 0.5) == (batch_y[n] == 1)
            pseudo_y = xi_t / lambda_t
            pseudo_noise = inv(lambda_t)

            for f in 1:f_count, q in 1:num_spatial_bins
                idx = (f - 1) * num_spatial_bins + q
                other_mean = mc + sum(prod_mean) - prod_mean[idx]
                other_var = max(pseudo_noise + vc + sum(prod_var) - prod_var[idx], pseudo_noise)
                residual = pseudo_y - other_mean

                gg, hh = product_log_derivatives(gmean[idx], residual, mu[idx], vu[idx], other_var)
                xi_g, lambda_g =
                    positive_product_sites ?
                    positive_site_from_grad_hess(gmean[idx], gg, hh; max_precision = 1e3) :
                    site_from_grad_hess(gmean[idx], gg, hh; min_site_precision = -1.0)
                # Approximate pooled g by distributing the site over its patches.
                count_q = bin_counts[q]
                for p in 1:p_count
                    if bin_ids[p] == q
                        cls_x_xi_star[n, p, f] += xi_g / count_q
                        cls_x_lambda_star[n, p, f] += lambda_g / count_q^2
                    end
                end

                gu, hu = product_log_derivatives(mu[idx], residual, gmean[idx], gvar[idx], other_var)
                cls_u_xi_star[n, idx], cls_u_lambda_star[n, idx] =
                    positive_product_sites ?
                    positive_site_from_grad_hess(mu[idx], gu, hu; max_precision = 1e3) :
                    site_from_grad_hess(mu[idx], gu, hu; min_site_precision = -1.0)
            end

            other_mean = sum(prod_mean)
            other_var = max(pseudo_noise + sum(prod_var), pseudo_noise)
            cls_c_lambda_star[n] = inv(other_var)
            cls_c_xi_star[n] = (pseudo_y - other_mean) / other_var
        end
        batch_accuracy = correct / n_count
        damp_site!(cls_x_xi, cls_x_lambda, cls_x_xi_star, cls_x_lambda_star, alpha)
        damp_site!(cls_u_xi, cls_u_lambda, cls_u_xi_star, cls_u_lambda_star, alpha)
        damp_site!(cls_c_xi, cls_c_lambda, cls_c_xi_star, cls_c_lambda_star, alpha)

        # Diagnostics and convergence.
        state.b_xi .= b_base_xi .+ global_site_scale .* dropdims(sum(img_b_xi; dims = (1, 2)); dims = (1, 2))
        state.b_lambda .= max.(b_base_lambda .+ global_site_scale .* dropdims(sum(img_b_lambda; dims = (1, 2)); dims = (1, 2)), 1e-5)
        state.u_xi .= u_base_xi .+ global_site_scale .* vec(sum(cls_u_xi; dims = 1))
        state.u_lambda .= max.(u_base_lambda .+ global_site_scale .* vec(sum(cls_u_lambda; dims = 1)), 1e-5)
        state.c_xi = c_base_xi + global_site_scale * sum(cls_c_xi)
        state.c_lambda = max(c_base_lambda + global_site_scale * sum(cls_c_lambda), 1e-5)
        mb, _, mu, _, mc, _ = means_vars(state)

        sqerr = 0.0
        total_pixels = n_count * p_count * r_count
        for n in 1:n_count, p in 1:p_count, r in 1:r_count
            pred = sum(mb[f, r] * mx[n, p, f] for f in 1:f_count)
            sqerr += (batch_x[p, r, n] - pred)^2
        end
        batch_mse = sqerr / total_pixels

        delta_global = maximum(abs.([vec(mb .- old_mb); mu .- old_mu; mc - old_mc]))
        delta_local = maximum(abs.(mx .- old_mx))
        last_delta = max(delta_global, delta_local)
        verbose && @info "inner $inner mse=$(round(batch_mse, digits=5)) acc=$(round(batch_accuracy, digits=3)) delta=$(round(last_delta, sigdigits=3))"
        last_delta < tol && break
    end

    return (delta = last_delta, mse = batch_mse, accuracy = batch_accuracy)
end

function infer_local_for_prediction(state::GlobalState, image_patches;
                                    num_features = size(state.b_xi, 1),
                                    sigma_a2 = 1.0,
                                    sigma_sp2 = 0.05^2,
                                    sigma_y2 = 0.18^2,
                                    alpha = 0.35,
                                    positive_product_sites = true,
                                    max_inner = 25)
    # Reuse infer_batch! on a one-image unlabeled pseudo-batch with a neutral
    # label site disabled by setting classifier globals to tiny values would be
    # overkill. Here we infer x from image + softplus only, then classify with
    # the current global classifier.
    p_count, r_count = size(image_patches)
    f_count = num_features
    u_count = length(state.u_xi)
    num_spatial_bins = max(1, div(u_count, f_count))
    bin_ids = patch_bin_ids(p_count; num_bins = num_spatial_bins)
    bin_counts = [count(==(q), bin_ids) for q in 1:num_spatial_bins]
    mb, vb, _, _, _, _ = means_vars(state)

    ma = zeros(p_count, f_count)
    va = fill(sigma_a2, p_count, f_count)
    mx = fill(softplus(0.0), p_count, f_count)
    vx = fill(0.25, p_count, f_count)
    sp_a_xi = zeros(p_count, f_count); sp_a_lambda = zeros(p_count, f_count)
    sp_x_xi = zeros(p_count, f_count); sp_x_lambda = zeros(p_count, f_count)
    img_x_xi = zeros(p_count, f_count, r_count)
    img_x_lambda = zeros(p_count, f_count, r_count)

    for _ in 1:max_inner
        for p in 1:p_count, f in 1:f_count
            ma[p, f], va[p, f] = gaussian_from_nat(sp_a_xi[p, f], inv(sigma_a2) + sp_a_lambda[p, f])
            mx[p, f], vx[p, f] = gaussian_from_nat(sp_x_xi[p, f] + sum(@view img_x_xi[p, f, :]),
                                                   sp_x_lambda[p, f] + sum(@view img_x_lambda[p, f, :]))
        end
        for p in 1:p_count, f in 1:f_count
            sx, sl = softplus_to_x_site(0.0, sigma_a2, sigma_sp2)
            sp_x_xi[p, f] = (1 - alpha) * sp_x_xi[p, f] + alpha * sx
            sp_x_lambda[p, f] = (1 - alpha) * sp_x_lambda[p, f] + alpha * sl
            ax, al = softplus_to_a_site(ma[p, f], mx[p, f], vx[p, f], sigma_sp2)
            sp_a_xi[p, f] = (1 - alpha) * sp_a_xi[p, f] + alpha * ax
            sp_a_lambda[p, f] = (1 - alpha) * sp_a_lambda[p, f] + alpha * al
        end
        for p in 1:p_count, r in 1:r_count
            y = image_patches[p, r]
            prod_mean = [mx[p, f] * mb[f, r] for f in 1:f_count]
            prod_var = [vx[p, f] * vb[f, r] + vx[p, f] * mb[f, r]^2 + vb[f, r] * mx[p, f]^2 for f in 1:f_count]
            total_mean = sum(prod_mean)
            total_var = sigma_y2 + sum(prod_var)
            for f in 1:f_count
                residual = y - (total_mean - prod_mean[f])
                noise = max(total_var - prod_var[f], sigma_y2)
                g, h = product_log_derivatives(mx[p, f], residual, mb[f, r], vb[f, r], noise)
                xi, lam =
                    positive_product_sites ?
                    positive_site_from_grad_hess(mx[p, f], g, h; max_precision = 1e3) :
                    site_from_grad_hess(mx[p, f], g, h; min_site_precision = -5.0)
                img_x_xi[p, f, r] = (1 - alpha) * img_x_xi[p, f, r] + alpha * xi
                img_x_lambda[p, f, r] = (1 - alpha) * img_x_lambda[p, f, r] + alpha * lam
            end
        end
    end

    _, _, mu, _, mc, _ = means_vars(state)
    gmean = zeros(u_count)
    for f in 1:f_count, q in 1:num_spatial_bins
        idx = (f - 1) * num_spatial_bins + q
        count_q = bin_counts[q]
        for p in 1:p_count
            if bin_ids[p] == q
                gmean[idx] += mx[p, f] / count_q
            end
        end
    end
    logit = mc + dot(mu, gmean)
    return sigmoid(logit), gmean
end

function evaluate(state::GlobalState, x, y; max_images = size(x, 3))
    n = min(max_images, size(x, 3))
    correct = 0
    probs = zeros(n)
    for i in 1:n
        probs[i], _ = infer_local_for_prediction(state, @view x[:, :, i])
        correct += (probs[i] >= 0.5) == (y[i] == 1)
    end
    return correct / n, mean(abs.(probs .- y[1:n]))
end

function train_demo(; digit0 = 3, digit1 = 5, ntrain = 128, nval = 64, ntest = 64,
                    num_features = 4, batch_size = 16, epochs = 2,
                    seed = 1, verbose_inner = false, num_spatial_bins = 4,
                    alpha = 0.08, positive_product_sites = true,
                    global_site_scale = nothing)
    data = select_binary_mnist(; digit0, digit1, ntrain, nval, ntest, seed)
    state = init_state(num_features; seed = seed + 1, num_spatial_bins)
    rng = MersenneTwister(seed + 2)
    site_scale = isnothing(global_site_scale) ? inv(epochs) : global_site_scale

    println("Loaded binary MNIST digits $digit0 vs $digit1")
    println("train=$(length(data.train_y)) val=$(length(data.val_y)) test=$(length(data.test_y)) features=$num_features spatial_bins=$num_spatial_bins batch=$batch_size site_scale=$(round(site_scale, digits=4))")

    history = NamedTuple[]

    for epoch in 1:epochs
        order = shuffle(rng, collect(1:length(data.train_y)))
        for (batch_idx, start_idx) in enumerate(1:batch_size:length(order))
            inds = order[start_idx:min(start_idx + batch_size - 1, end)]
            stats = infer_batch!(state, data.train_x[:, :, inds], data.train_y[inds];
                                 num_features, alpha, positive_product_sites,
                                 global_site_scale = site_scale,
                                 verbose = verbose_inner)
            println("epoch=$epoch batch=$batch_idx n=$(length(inds)) mse=$(round(stats.mse, digits=5)) acc=$(round(stats.accuracy, digits=3)) delta=$(round(stats.delta, sigdigits=3))")
        end
        train_acc, train_mae = evaluate(state, data.train_x, data.train_y; max_images = min(128, length(data.train_y)))
        val_acc, val_mae = evaluate(state, data.val_x, data.val_y; max_images = min(128, length(data.val_y)))
        test_acc, test_mae = evaluate(state, data.test_x, data.test_y; max_images = min(128, length(data.test_y)))
        push!(history, (epoch = epoch, train_acc = train_acc, val_acc = val_acc,
                        test_acc = test_acc, train_mae = train_mae, val_mae = val_mae,
                        test_mae = test_mae))
        println("epoch=$epoch eval train_acc=$(round(train_acc, digits=3)) val_acc=$(round(val_acc, digits=3)) test_acc=$(round(test_acc, digits=3)) train_mae=$(round(train_mae, digits=3)) val_mae=$(round(val_mae, digits=3)) test_mae=$(round(test_mae, digits=3))")
    end

    mb, vb, mu, vu, mc, vc = means_vars(state)
    println("\nlearned kernel means b[f, 1:4]:")
    for f in 1:num_features
        println("feature $f: ", round.(mb[f, :]; digits = 3))
    end
    println("classifier mean c=$(round(mc, digits=3)), u[f,bin]=")
    println(round.(reshape(mu, num_spatial_bins, num_features)'; digits = 3))
    best_val = history[argmax(map(h -> h.val_acc, history))]
    println("best validation epoch=$(best_val.epoch) val_acc=$(round(best_val.val_acc, digits=3)) holdout_test_acc_at_that_run=$(round(best_val.test_acc, digits=3))")
    return (state = state, history = history, data = data)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_demo()
end
