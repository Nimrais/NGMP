using RxInfer
using Distributions
using FastGaussQuadrature
using LinearAlgebra
using Random
using Statistics
using Plots

# ============================================================================
# Stage 1: noise recovery in a state-space model (univariate).
#
#     z_1 ~ N(m0, v0),   z_k = z_{k-1} + N(0, q),   y_k ~ N(z_k, 1/τ),
#     τ ~ Gamma(a0, b0),  k = 1..N.
#
# Why the SSM is the right demonstration for the precision edge: in the iid
# model the cavity variance of the latent shrinks as 1/N, so the non-affine
# part of the τ-message dies with more data. In a chain, the smoothing
# variance of z_k is bounded below by its steady state -- more data does NOT
# make the local cavity tighter -- so the beyond-mean-field correction stays
# O(1) per factor at any chain length.
#
# Methods (the conjugate carrier -- an RTS sweep -- is the SAME hand-rolled
# smoother for both, so the wall-clock comparison is algorithm vs algorithm):
#   VMP    structured mean-field q(z_{1:N}) q(τ): smoother with R = 1/E[τ],
#          conjugate Gamma update with smoothed moments. (Cross-checked
#          against RxInfer's structured-constraints VMP, mirroring the
#          "Kalman filtering and smoothing" / system-identification examples.)
#   NG-BP  projected exact-BP messages with cavities: Student-t toward each
#          z_k (Bonnet-Price, Gauss-Hermite), "strange gamma" toward τ with
#          the FULL dof block (Gauss-Legendre quadrature -- stable in 1-D).
#   Exact  p(τ | y) on a grid via the Kalman marginal likelihood (closed
#          form given τ); exact z marginals as the induced mixture.
# ============================================================================

const GH_NODES, GH_WEIGHTS = gausshermite(64)
const GL_NODES, GL_WEIGHTS = gausslegendre(400)

# RTS smoother for a random walk with per-step observation variance R[k].
# Returns smoothed means/variances and the marginal log-likelihood.
function rts(y, R, m0, v0, qw)
    N = length(y)
    mf = zeros(N); vf = zeros(N); ll = 0.0
    m̄, v̄ = m0, v0
    for k in 1:N
        S  = v̄ + R[k]
        ll += logpdf(Normal(m̄, sqrt(S)), y[k])
        K  = v̄ / S
        mf[k] = m̄ + K * (y[k] - m̄)
        vf[k] = (1 - K) * v̄
        m̄, v̄ = mf[k], vf[k] + qw
    end
    ms = copy(mf); vs = copy(vf)
    for k in (N-1):-1:1
        C = vf[k] / (vf[k] + qw)
        ms[k] = mf[k] + C * (ms[k+1] - mf[k])
        vs[k] = vf[k] + C^2 * (vs[k+1] - (vf[k] + qw))
    end
    return (m = ms, v = vs, loglik = ll)
end

# Structured mean-field VMP in closed form.
function vmp_ssm(y; m0, v0, qw, a0, b0, iters = 200, tol = 1e-12)
    N = length(y)
    a, b = a0 + N / 2, b0
    local sm
    Eτ = a0 / b0
    for it in 1:iters
        sm = rts(y, fill(1 / Eτ, N), m0, v0, qw)
        b  = b0 + 0.5 * sum(@. (y - sm.m)^2 + sm.v)
        δ  = abs(a / b - Eτ) / (a / b)
        Eτ = a / b
        δ < tol && break
    end
    return (m = sm.m, v = sm.v, a = a, b = b)
end

# Projection of the Student-t log-message onto the Gaussian z_k edge.
function project_to_z(y, ã, b̃, m, v)
    xs = m .+ sqrt(2v) .* GH_NODES
    ws = GH_WEIGHTS ./ sqrt(π)
    d  = xs .- y
    ℓ′ = @. -(2ã + 1) * d / (2b̃ + d^2)
    ℓ″ = @. -(2ã + 1) * (2b̃ - d^2) / (2b̃ + d^2)^2
    Λstar = -sum(ws .* ℓ″)
    ξstar = sum(ws .* ℓ′) + m * Λstar
    return ξstar, Λstar
end

# Projection of the exact log-message onto the Gamma τ edge (full dof block,
# covariance form η = F⁻¹ Cov[T, ℓ], Gauss-Legendre against q(τ)).
function project_to_τ(y, m̃, ṽ, a, b)
    qτ = Gamma(a, 1 / b)
    lo, hi = quantile(qτ, 1e-9), quantile(qτ, 1 - 1e-9)
    τs = (hi + lo) / 2 .+ (hi - lo) / 2 .* GL_NODES
    ws = (hi - lo) / 2 .* GL_WEIGHTS .* pdf.(qτ, τs)
    Z  = sum(ws)
    ℓ  = @. -0.5 * log(ṽ + 1 / τs) - (y - m̃)^2 / (2 * (ṽ + 1 / τs))
    T1, T2 = log.(τs), τs
    ET1, ET2, Eℓ = sum(ws .* T1) / Z, sum(ws .* T2) / Z, sum(ws .* ℓ) / Z
    c = [sum(ws .* T1 .* ℓ) / Z - ET1 * Eℓ,
         sum(ws .* T2 .* ℓ) / Z - ET2 * Eℓ]
    F = [SpecialFunctions.trigamma(a) 1/b; 1/b a/b^2]
    η = F \ c
    return η[1], -η[2]                                   # Δa*, Δb*
end
import SpecialFunctions

function ngbp_ssm(y; m0, v0, qw, a0, b0,
                  α = 0.5, iters = 150, tol = 1e-9, mf_init = 25)
    N = length(y)
    ξ  = zeros(N); Λ  = zeros(N)
    Δa = zeros(N); Δb = zeros(N)
    m  = zeros(N); v  = fill(v0, N)
    a, b = a0, b0
    λold = [0.0, 0.0]
    local sm
    for iter in 1:iters
        ξs = similar(ξ); Λs = similar(Λ); Δas = similar(Δa); Δbs = similar(Δb)
        if iter <= mf_init
            Eτ = a / b
            @. ξs  = Eτ * y;   @. Λs  = Eτ
            @. Δas = 0.5;      @. Δbs = 0.5 * ((y - m)^2 + v)
            αt = 1.0
        else
            for k in 1:N
                ã, b̃ = a - Δa[k], b - Δb[k]              # τ cavity
                ξs[k], Λs[k] = project_to_z(y[k], ã, b̃, m[k], v[k])
                λ̃ = 1 / v[k] - Λ[k]                      # z_k cavity
                m̃ = (m[k] / v[k] - ξ[k]) / λ̃
                Δas[k], Δbs[k] = project_to_τ(y[k], m̃, 1 / λ̃, a, b)
            end
            αt = α
        end
        ξ  = (1 - αt) .* ξ  .+ αt .* ξs
        Λ  = (1 - αt) .* Λ  .+ αt .* Λs
        Δa = (1 - αt) .* Δa .+ αt .* Δas
        Δb = (1 - αt) .* Δb .+ αt .* Δbs

        # exact conjugate sweep: RTS on the pseudo-observation chain + Gamma
        sm = rts(ξ ./ Λ, 1 ./ Λ, m0, v0, qw)
        m, v = sm.m, sm.v
        a = a0 + sum(Δa)
        b = b0 + sum(Δb)

        λnew = [a, b]
        Δλ = maximum(abs.(λnew .- λold) ./ (1 .+ abs.(λnew)))
        λold = λnew
        Δλ < tol && iter > mf_init + 1 && break
    end
    return (m = m, v = v, a = a, b = b)
end

# Exact posterior: τ grid × Kalman marginal likelihood; z marginals as the
# induced mixture over the grid.
function exact_ssm(y; m0, v0, qw, a0, b0, nτ = 1200)
    N  = length(y)
    τs = exp.(range(log(1e-3), log(1e2); length = nτ))
    lp = zeros(nτ)
    mz = zeros(nτ, N); vz = zeros(nτ, N)
    for (j, τ) in enumerate(τs)
        sm = rts(y, fill(1 / τ, N), m0, v0, qw)
        lp[j] = logpdf(Gamma(a0, 1 / b0), τ) + sm.loglik
        mz[j, :] .= sm.m; vz[j, :] .= sm.v
    end
    lp .-= maximum(lp)
    w  = exp.(lp)
    dτ = [τs[2] - τs[1]; (τs[3:end] .- τs[1:end-2]) ./ 2; τs[end] - τs[end-1]]
    w  = w .* dτ; w ./= sum(w)
    mτ = sum(w .* τs); vτ = sum(w .* τs .^ 2) - mτ^2
    mx = vec(sum(w .* mz; dims = 1))
    sx = vec(sum(w .* (vz .+ mz .^ 2); dims = 1)) .- mx .^ 2
    return (τs = τs, w = w ./ dτ, wn = w, mτ = mτ, vτ = vτ, mz = mx, vz = sx)
end


# ----------------------------------------------------------------------------
begin
    Random.seed!(11)
    N, qw, τtrue = 500, 0.25, 1.0
    m0, v0, a0, b0 = 0.0, 100.0, 1.0, 1.0
    z = cumsum(sqrt(qw) .* randn(N)) .+ 1.0
    y = z .+ randn(N) ./ sqrt(τtrue)

    te = @elapsed exact = exact_ssm(y; m0, v0, qw, a0, b0)
    tv = @elapsed vmp   = vmp_ssm(y;  m0, v0, qw, a0, b0)
    tn = @elapsed ngbp  = ngbp_ssm(y; m0, v0, qw, a0, b0)

    # RxInfer structured-VMP anchor (same constraints idiom as the RxInfer
    # "system identification" example)
    @model function ssm_model(y, m0, v0, qw, a0, b0)
        τ ~ Gamma(shape = a0, rate = b0)
        z[1] ~ Normal(mean = m0, variance = v0)
        y[1] ~ Normal(mean = z[1], precision = τ)
        for k in 2:length(y)
            z[k] ~ Normal(mean = z[k-1], variance = qw)
            y[k] ~ Normal(mean = z[k], precision = τ)
        end
    end
    cons = @constraints begin
        q(z, τ) = q(z)q(τ)
    end
    init = @initialization begin
        q(τ) = GammaShapeRate(1.0, 1.0)
    end
    t_rx = @elapsed rx = infer(
        model = ssm_model(m0 = m0, v0 = v0, qw = qw, a0 = a0, b0 = b0),
        data = (y = y,), constraints = cons, initialization = init,
        iterations = 50, options = (limit_stack_depth = 300,),
    )
    qτ_rx = rx.posteriors[:τ][end]

    println("q(τ)   exact : mean $(round(exact.mτ, digits=4))  sd $(round(sqrt(exact.vτ), digits=4))")
    println("q(τ)   VMP   : mean $(round(vmp.a/vmp.b, digits=4))  sd $(round(sqrt(vmp.a)/vmp.b, digits=4))   (RxInfer: mean $(round(mean(qτ_rx), digits=4))  sd $(round(std(qτ_rx), digits=4)))")
    println("q(τ)   NG-BP : mean $(round(ngbp.a/ngbp.b, digits=4))  sd $(round(sqrt(ngbp.a)/ngbp.b, digits=4))")
    println("var(τ) ratio vs exact   VMP $(round((vmp.a/vmp.b^2)/exact.vτ, digits=3))   NG-BP $(round((ngbp.a/ngbp.b^2)/exact.vτ, digits=3))")
    println("var(z) ratio vs exact   VMP $(round(mean(vmp.v ./ exact.vz), digits=3))   NG-BP $(round(mean(ngbp.v ./ exact.vz), digits=3))")
    println("log q(τ*)  VMP $(round(logpdf(Gamma(vmp.a, 1/vmp.b), τtrue), digits=3))   NG-BP $(round(logpdf(Gamma(ngbp.a, 1/ngbp.b), τtrue), digits=3))   exact $(round(log(exact.w[argmin(abs.(exact.τs .- τtrue))]), digits=3))")
    println("runtimes   exact-grid $(round(te, digits=2))s   VMP $(round(tv, digits=3))s   NG-BP $(round(tn, digits=2))s   RxInfer-VMP $(round(t_rx, digits=2))s")

    # --- plots ---
    sel = exact.w .> 1e-6 * maximum(exact.w)
    p1 = plot(exact.τs[sel], exact.w[sel]; lw = 3, color = :black, label = "exact p(τ|y)",
              xlabel = "τ", ylabel = "density", title = "Observation precision (N = $N)")
    plot!(p1, exact.τs[sel], pdf.(Gamma(vmp.a, 1/vmp.b), exact.τs[sel]);
          lw = 2.5, ls = :dash, color = :crimson, label = "VMP")
    plot!(p1, exact.τs[sel], pdf.(Gamma(ngbp.a, 1/ngbp.b), exact.τs[sel]);
          lw = 2.5, color = :dodgerblue, label = "NG-BP")
    vline!(p1, [τtrue]; color = :gray, ls = :dot, label = "true τ")

    rng = 200:260
    p2 = plot(rng, z[rng]; lw = 2, color = :black, label = "true z",
              xlabel = "k", ylabel = "z_k", title = "Smoothed state (zoom)")
    scatter!(p2, rng, y[rng]; ms = 2.5, color = :gray, label = "y")
    plot!(p2, rng, exact.mz[rng]; ribbon = 1.96 .* sqrt.(exact.vz[rng]),
          fillalpha = 0.2, lw = 2, color = :black, ls = :dot, label = "exact ±1.96σ")
    plot!(p2, rng, ngbp.m[rng]; ribbon = 1.96 .* sqrt.(ngbp.v[rng]),
          fillalpha = 0.25, lw = 2, color = :dodgerblue, label = "NG-BP ±1.96σ")

    p3 = bar(["VMP" "NG-BP"], [(vmp.a/vmp.b^2)/exact.vτ (ngbp.a/ngbp.b^2)/exact.vτ];
             color = [:crimson :dodgerblue], legend = false,
             ylabel = "var(τ)/exact", title = "τ-uncertainty calibration")
    hline!(p3, [1.0]; color = :black, ls = :dash)

    p4 = bar(["exact grid", "RxInfer VMP", "NG-BP", "VMP"],
             [te, t_rx, tn, tv]; color = [:black, :gray, :dodgerblue, :crimson],
             legend = false, yscale = :log10,
             ylabel = "seconds (log)", title = "Wall-clock")

    plt = plot(p1, p2, p3, p4; layout = (2, 2), size = (1150, 800))
    savefig(plt, joinpath(@__DIR__, "ssm_ng_bp_1d.png"))
    display(plt)
end
