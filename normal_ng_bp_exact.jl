using RxInfer            # re-exports ExponentialFamily.jl (fisherinformation, gradlogpartition, ...)
using Distributions
using FastGaussQuadrature
using Random
using Statistics
using LinearAlgebra
using Plots

# ============================================================================
# Natural-gradient BP for NormalMeanPrecision observations -- WITHOUT the
# mean-field cluster at the observation node.
#
# Model:
#     x ~ N(m0, v0),   τ ~ Gamma(a0, b0),   y_i ~ N(x, 1/τ),  i = 1..N
#
# Only the EDGE marginals are form-constrained, q(x) = N(m, v) and
# q(τ) = Gamma(a, b); the observation-node beliefs stay free. The projected
# message is therefore built from the EXACT BP log-message, not the tilted
# (mean-field) one. The exact message integrates the factor against the
# message arriving on its other edge -- the CAVITY, i.e. the product of
# everything on that edge except the factor's own current message
# (natural-parameter subtraction from the edge marginal):
#
#   to the x edge (cavity Gamma(ã, b̃) = (a_q − Δa_i, b_q − Δb_i) on τ):
#     μ_{i→x}(x) = ∫ N(y_i|x, 1/τ) Ga(τ; ã, b̃) dτ          ... a Student-t:
#     ℓ_i(x)     = −(ã + ½) log(1 + (x − y_i)²/(2b̃)) + const
#
#   to the τ edge (cavity N(m̃, ṽ),  Λ̃ = Λ_q − Λ_i,  ξ̃ = ξ_q − ξ_i  on x):
#     μ_{i→τ}(τ) = ∫ N(y_i|x, 1/τ) N(x; m̃, ṽ) dx = N(y_i; m̃, ṽ + 1/τ)
#     ℓ_i(τ)     = −½ log(ṽ + 1/τ) − (y_i − m̃)² / (2(ṽ + 1/τ)) + const
#
# Neither log-message lies in the receiving family (that is what breaks
# conjugate VMP), so the tangent projection is nontrivial:
#
#     η*_{i→edge} = Π^{λ}[ℓ_i] = ∇_μ E_{q_λ}[ℓ_i] = F(λ)⁻¹ Cov_{q_λ}[T, ℓ_i],
#
# evaluated at the FULL receiving marginal q_λ (the cavity enters the
# computation of ℓ, the projection point is the marginal -- unlike EP, which
# moment-matches the cavity-tilted distribution). We use both equivalent
# forms of the projection:
#
#   x edge (Gaussian, T = (x, x²), mean coords (m, s)) -- Bonnet/Price:
#     Λ*_i = −E_q[ℓ_i″(x)]            (expected curvature of the t log-pdf;
#     ξ*_i =  E_q[ℓ_i′(x)] + m Λ*_i    down-weights outliers, unlike E[τ])
#
#   τ edge (Gamma, T = (log τ, τ)) -- covariance form with the Fisher metric
#   from ExponentialFamily.jl:
#     (Δa*_i, −Δb*_i) = fisherinformation(q_τ) \ Cov_{q_τ}[T, ℓ_i]
#     (as ṽ → 0 this recovers the VMP message  ½ log τ − ½(y_i−m̃)² τ)
#
# Outer loop is the same as in the Poisson/mean-field examples: project at
# the current marginals, damp the message natural parameters
# η ← (1−α) η + α η*, run one exact BP sweep on the conjugate surrogate
# carrier (Gaussian pseudo-observation on the x edge, Gamma pseudo-likelihood
# τ^{Δa_i} e^{−Δb_i τ} on the τ edge), repeat. The fixed point is NOT the
# mean-field one: the node beliefs keep the x–τ coupling, which shows up as
# better-calibrated posteriors in the small-data regime.
# ============================================================================

@model function approx_model(ytilde, Λ, s, Δa, m0, v0, a0, b0)
    x ~ Normal(mean = m0, variance = v0)
    τ ~ GammaShapeRate(a0, b0)
    for i in 1:length(ytilde)
        ytilde[i] ~ Normal(mean = x, precision = Λ[i])
        s[i] ~ GammaShapeRate(Δa[i], τ)
    end
end

const GH_NODES, GH_WEIGHTS = gausshermite(64)
const GL_NODES, GL_WEIGHTS = gausslegendre(400)

# Projection of the Student-t log-message onto the Gaussian x edge,
# Bonnet/Price form, Gauss-Hermite under q(x) = N(m, v).
function project_to_x(y, ã, b̃, m, v)
    xs = m .+ sqrt(2v) .* GH_NODES
    ws = GH_WEIGHTS ./ sqrt(π)
    d  = xs .- y
    ℓ′ = @. -(2ã + 1) * d / (2b̃ + d^2)
    ℓ″ = @. -(2ã + 1) * (2b̃ - d^2) / (2b̃ + d^2)^2
    Λstar = -sum(ws .* ℓ″)
    ξstar = sum(ws .* ℓ′) + m * Λstar
    return ξstar, Λstar
end

# Projection of the exact log-message onto the Gamma τ edge, covariance form
# η* = F(λ)⁻¹ Cov[T, ℓ], Gauss-Legendre against q(τ) = Gamma(a, b).
function project_to_τ(y, m̃, ṽ, a, b)
    qτ = Gamma(a, 1 / b)                       # shape-scale for Distributions
    lo, hi = quantile(qτ, 1e-9), quantile(qτ, 1 - 1e-9)
    τs = (hi + lo) / 2 .+ (hi - lo) / 2 .* GL_NODES
    ws = (hi - lo) / 2 .* GL_WEIGHTS .* pdf.(qτ, τs)
    Z  = sum(ws)
    ℓ  = @. -0.5 * log(ṽ + 1 / τs) - (y - m̃)^2 / (2 * (ṽ + 1 / τs))
    T1, T2 = log.(τs), τs
    ET1, ET2, Eℓ = sum(ws .* T1) / Z, sum(ws .* T2) / Z, sum(ws .* ℓ) / Z
    cov = [sum(ws .* T1 .* ℓ) / Z - ET1 * Eℓ,
           sum(ws .* T2 .* ℓ) / Z - ET2 * Eℓ]
    F  = fisherinformation(convert(ExponentialFamilyDistribution, GammaShapeRate(a, b)))
    η  = F \ cov
    return η[1], -η[2]                          # Δa*, Δb*
end

# Mean-field free energy of the original model evaluated at (q(x), q(τ)) --
# not the objective of this scheme, reported only for comparison with VMP.
function meanfield_F(ys, qx, qτ, m0, v0, a0, b0)
    m, v  = mean(qx), var(qx)
    Elogτ = gradlogpartition(convert(ExponentialFamilyDistribution, convert(GammaShapeRate, qτ)))[1]
    energy = sum(@. 0.5 * log(2π) - 0.5 * Elogτ + 0.5 * mean(qτ) * ((ys - m)^2 + v))
    return energy + kldivergence(Normal(m, sqrt(v)), Normal(m0, sqrt(v0))) +
           kldivergence(Gamma(shape(qτ), 1 / rate(qτ)), Gamma(a0, 1 / b0))
end

function ng_bp_exact(ys;
                     m0       = 0.0,
                     v0       = 100.0,
                     a0       = 1.0,
                     b0       = 1.0,
                     max_iter = 500,
                     tol      = 1e-8,
                     α        = 0.5,
                     mf_init  = 30,
                     verbose  = true)
    N = length(ys)

    # persistent message natural parameters
    ξ  = zeros(N);  Λ  = zeros(N)               # → x edge:  exp(ξx − Λx²/2)
    Δa = zeros(N);  Δb = zeros(N)               # → τ edge:  τ^Δa e^{−Δb τ}

    qx = NormalMeanVariance(m0, v0)
    qτ = GammaShapeRate(a0, b0)

    # Note: free_energy=true is off on the surrogate sweep. The BP messages
    # are well-defined for any natural parameters, but RxInfer's average
    # energy of the GammaShapeRate pseudo-node evaluates loggamma(Δa_i),
    # which rejects the (legitimate) slightly-negative shape increments that
    # occur during warm-up. Convergence is monitored on the edge-marginal
    # natural parameters instead; the mean-field F is tracked for comparison.
    Fmf  = Float64[]                            # mean-field F (comparison)
    λold = Float64[m0 / v0, -1 / (2v0), a0 - 1, -b0]
    local result

    for iter in 1:max_iter
        m, v   = mean(qx), var(qx)
        ξq, Λq = m / v, 1 / v
        aq, bq = shape(qτ), rate(qτ)

        # (A) per-factor message targets. The first `mf_init` iterations use
        # the tilted (mean-field) projection at full step -- i.e. plain VMP --
        # purely as initialization: the exact-BP projection point must be in
        # the bulk of the data before the Student-t curvatures behave.
        ξstar = similar(ξ); Λstar = similar(Λ)
        Δastar = similar(Δa); Δbstar = similar(Δb)
        if iter <= mf_init
            τ̄ = aq / bq
            @. ξstar  = τ̄ * ys;    @. Λstar  = τ̄
            @. Δastar = 0.5;       @. Δbstar = 0.5 * ((ys - m)^2 + v)
        else
            # cavity → exact BP log-message → tangent projection
            for i in 1:N
                ã, b̃ = aq - Δa[i], bq - Δb[i]              # τ cavity
                ξstar[i], Λstar[i] = project_to_x(ys[i], ã, b̃, m, v)
                Λ̃, ξ̃ = Λq - Λ[i], ξq - ξ[i]                # x cavity
                Δastar[i], Δbstar[i] = project_to_τ(ys[i], ξ̃ / Λ̃, 1 / Λ̃, aq, bq)
            end
        end

        # (B) natural-gradient step in message space
        αt = iter <= mf_init ? 1.0 : α
        ξ  = (1 - αt) .* ξ  .+ αt .* ξstar
        Λ  = (1 - αt) .* Λ  .+ αt .* Λstar
        Δa = (1 - αt) .* Δa .+ αt .* Δastar
        Δb = (1 - αt) .* Δb .+ αt .* Δbstar

        # (C) one exact BP sweep on the conjugate surrogate
        result = infer(
            model = approx_model(m0 = m0, v0 = v0, a0 = a0, b0 = b0),
            data  = (ytilde = ξ ./ Λ, Λ = Λ, s = Δb, Δa = Δa),
        )
        qx = convert(NormalMeanVariance, result.posteriors[:x])
        qτ = result.posteriors[:τ]

        # (D) convergence on the edge-marginal natural parameters
        λnew = [mean(qx) / var(qx), -1 / (2var(qx)), shape(qτ) - 1, -rate(qτ)]
        Δλ   = maximum(abs.(λnew .- λold))
        λold = λnew
        push!(Fmf, meanfield_F(ys, qx, qτ, m0, v0, a0, b0))

        verbose && @info "iter $iter   F_mf = $(round(Fmf[end], sigdigits=8))   Δλ = $(round(Δλ, sigdigits=3))"

        if Δλ < tol && iter > mf_init + 1
            verbose && @info "converged in $iter iterations (Δλ < $tol, α = $α)"
            break
        end
    end

    return (qx = qx, qτ = qτ,
            meanfield_F = Fmf,
            ξ = ξ, Λ = Λ, Δa = Δa, Δb = Δb)
end

# Exact posterior by 1-D marginalization: x integrates out in closed form
# given τ, leaving p(τ | y) on a grid; p(x | y) is the induced mixture.
function exact_posterior(ys, m0, v0, a0, b0; nτ = 4000)
    N, S1, S2 = length(ys), sum(ys), sum(abs2, ys)
    τs = exp.(range(log(1e-5), log(50.0); length = nτ))
    P  = @. 1 / v0 + N * τs                     # posterior precision of x | τ
    h  = @. m0 / v0 + τs * S1
    logp = @. (a0 - 1) * log(τs) - b0 * τs +
              (N / 2) * log(τs) - 0.5 * τs * S2 + h^2 / (2P) - 0.5 * log(P)
    logp .-= maximum(logp)
    w  = exp.(logp)
    dτ = [τs[2] - τs[1]; (τs[3:end] .- τs[1:end-2]) ./ 2; τs[end] - τs[end-1]]
    w  = w .* dτ; w ./= sum(w)
    μx, vx = h ./ P, 1 ./ P                     # p(x | τ_k, y)
    mx = sum(w .* μx)
    sx = sum(w .* (vx .+ μx .^ 2)) - mx^2
    mτ = sum(w .* τs)
    vτ = sum(w .* τs .^ 2) - mτ^2
    pdfx = xg -> sum(w .* pdf.(Normal.(μx, sqrt.(vx)), xg))
    pdfτ = (τs = τs, w = w ./ dτ)
    return (mx = mx, vx = sx, mτ = mτ, vτ = vτ, pdfx = pdfx, pdfτ = pdfτ)
end


# ----------------------------------------------------------------------------
# Demo: small-data regime, where mean-field is visibly off and the projected
# exact-BP messages should improve calibration. Compare NG-BP, mean-field
# VMP, and the exact posterior.
# ----------------------------------------------------------------------------
begin
    Random.seed!(42)
    N  = 20
    ys = rand(NormalMeanVariance(10.0, 10.0), N)

    m0, v0, a0, b0 = 0.0, 100.0, 1.0, 1.0

    exact = exact_posterior(ys, m0, v0, a0, b0)

    # --- mean-field VMP reference ---------------------------------------------
    @model function normal_node_vmp(y, m0, v0, a0, b0)
        x ~ NormalMeanVariance(m0, v0)
        τ ~ GammaShapeRate(a0, b0)
        for i in 1:length(y)
            y[i] ~ NormalMeanPrecision(x, τ)
        end
    end
    @initialization function marginal_init()
        q(x) = NormalMeanVariance(0, 1)
    end
    vmp = infer(
        data = (y = ys,),
        model = normal_node_vmp(m0 = m0, v0 = v0, a0 = a0, b0 = b0),
        constraints = MeanField(),
        iterations = 100,
        initialization = marginal_init(),
    )
    qx_vmp = convert(NormalMeanVariance, vmp.posteriors[:x][end])
    qτ_vmp = vmp.posteriors[:τ][end]

    # --- natural-gradient BP with exact-BP-projected messages -----------------
    res = ng_bp_exact(ys; m0, v0, a0, b0, α = 0.5)

    fmt(d) = "mean $(round(mean(d), digits = 4)), var $(round(var(d), digits = 5))"
    println("\nq(x)  exact : mean $(round(exact.mx, digits = 4)), var $(round(exact.vx, digits = 5))")
    println("q(x)  NG-BP : ", fmt(res.qx))
    println("q(x)  VMP   : ", fmt(qx_vmp))
    println("\nq(τ)  exact : mean $(round(exact.mτ, digits = 5)), var $(round(exact.vτ, digits = 7))")
    println("q(τ)  NG-BP : ", fmt(res.qτ))
    println("q(τ)  VMP   : ", fmt(qτ_vmp))
    println("\nmean-field F at fixed point   NG-BP / VMP : ",
            round(res.meanfield_F[end], digits = 6), " / ",
            round(meanfield_F(ys, qx_vmp, qτ_vmp, m0, v0, a0, b0), digits = 6))

    # --- plots ----------------------------------------------------------------
    # (1) posterior of the mean
    xg = range(exact.mx - 5sqrt(exact.vx), exact.mx + 5sqrt(exact.vx); length = 400)
    p1 = plot(xg, exact.pdfx.(xg); lw = 3, color = :black, label = "exact",
              xlabel = "x", ylabel = "density", title = "Posterior of the mean")
    plot!(p1, xg, pdf.(Normal(mean(res.qx), std(res.qx)), xg);
          lw = 2.5, color = :dodgerblue, label = "NG-BP (projected exact BP)")
    plot!(p1, xg, pdf.(Normal(mean(qx_vmp), std(qx_vmp)), xg);
          lw = 2, ls = :dash, color = :crimson, label = "mean-field VMP")

    # (2) posterior of the precision
    mask = exact.pdfτ.w .> 1e-8 * maximum(exact.pdfτ.w)
    p2 = plot(exact.pdfτ.τs[mask], exact.pdfτ.w[mask];
              lw = 3, color = :black, label = "exact",
              xlabel = "τ", ylabel = "density", title = "Posterior of the precision")
    plot!(p2, exact.pdfτ.τs[mask],
          pdf.(Gamma(shape(res.qτ), 1 / rate(res.qτ)), exact.pdfτ.τs[mask]);
          lw = 2.5, color = :dodgerblue, label = "NG-BP")
    plot!(p2, exact.pdfτ.τs[mask],
          pdf.(Gamma(shape(qτ_vmp), 1 / rate(qτ_vmp)), exact.pdfτ.τs[mask]);
          lw = 2, ls = :dash, color = :crimson, label = "mean-field VMP")

    # (3) mean-field F along the NG-BP trajectory: NOT the objective of this
    #     scheme, so it need not end below the VMP line -- shown to make the
    #     "different fixed point" visible
    p3 = plot(res.meanfield_F; lw = 2, marker = :circle, ms = 3,
              color = :dodgerblue, label = "NG-BP trajectory",
              xlabel = "outer iteration", ylabel = "mean-field F",
              title = "Convergence (α = 0.5)")
    hline!(p3, [meanfield_F(ys, qx_vmp, qτ_vmp, m0, v0, a0, b0)];
           color = :crimson, ls = :dash, lw = 2, label = "VMP fixed point")

    # (4) per-observation message precisions: NG-BP adapts to the residual,
    #     mean-field sends the same E[τ] to every observation
    resid = abs.(ys .- mean(res.qx))
    p4 = scatter(resid, res.Λ; ms = 6, color = :dodgerblue,
                 label = "NG-BP  Λ_i", xlabel = "|y_i − m|",
                 ylabel = "message precision",
                 title = "Outlier down-weighting")
    hline!(p4, [mean(qτ_vmp)]; color = :crimson, ls = :dash, lw = 2,
           label = "VMP  E[τ] (same ∀ i)")

    plt = plot(p1, p2, p3, p4; layout = (2, 2), size = (1100, 800))
    savefig(plt, joinpath(@__DIR__, "normal_ng_bp_posterior.png"))
    display(plt)
end
