using RxInfer            # re-exports ExponentialFamily.jl (logpartition, fisherinformation, ...)
using Distributions
using Random
using Statistics
using Plots

# ============================================================================
# Natural-gradient BP for NormalMeanPrecision observations.
#
# Model (same as naive_mean_field.jl):
#     x ~ N(m0, v0)
#     τ ~ Gamma(a0, b0)              (shape-rate)
#     y_i ~ N(x, 1/τ),  i = 1..N
#
# Form constraints on the two latent edges:
#     q(x) = N(m, v)        T_x(x) = (x, x²),      mean coords (m, s), s = m² + v
#     q(τ) = Gamma(a, b)    T_τ(τ) = (log τ, τ),   mean coords (E[log τ], E[τ])
#
# Each observation factor f_i(x, τ) = N(y_i | x, 1/τ) touches BOTH constrained
# edges, so (unlike the unary Poisson leaf) we put a mean-field cluster {x},{τ}
# at the node and project the TILTED log-messages (paper, SVMP remark):
#
#   ℓ̃_{i→x}(x) = E_{q(τ)}[log f_i] = τ̄ y_i x − (τ̄/2) x²            + const
#   ℓ̃_{i→τ}(τ) = E_{q(x)}[log f_i] = ½ log τ − ½((y_i−m)² + v) τ   + const
#
# with τ̄ = E_q[τ] = a/b. The message natural parameter is the mean-coordinate
# gradient of the expected log-message,  η* = ∇_μ E_q[ℓ̃]:
#
#   x-edge:  E_{q(x)}[ℓ̃_{i→x}] = τ̄ y_i m − (τ̄/2) s
#            η*_{i→x} = (τ̄ y_i, −τ̄/2)
#            →  Gaussian pseudo-observation  N(ỹ_i | x, R_i),
#               ỹ_i = y_i,  R_i = 1/τ̄
#
#   τ-edge:  E_{q(τ)}[ℓ̃_{i→τ}] = ½ E[log τ] − Δb*_i E[τ]
#            η*_{i→τ} = (1/2, −Δb*_i),   Δb*_i = ((y_i − m)² + v)/2
#            →  Gamma pseudo-observation  ∝ τ^{1/2} exp(−Δb*_i τ)
#
# Both tilted log-messages are already affine in the receiving sufficient
# statistics, so the tangent projection Π = F⁻¹ Cov[T, ·] is the IDENTITY:
# this is the conjugate-computation special case of natural-gradient MP
# (Khan's CVI), and the undamped fixed-point iteration is exactly mean-field
# VMP (naive_mean_field.jl). What survives from the Poisson example is the
# algorithmic shape:
#
#   (A) project the exact/tilted BP log-messages at the current edge marginals
#   (B) damp the message NATURAL PARAMETERS:  η ← (1−α) η + α η*
#   (C) one exact BP sweep on the resulting fully conjugate surrogate model
#   (D) repeat until the free energy stops moving
#
# Step (B) with step size α is the natural-gradient / mirror-descent step;
# α = 1 recovers (parallel) VMP coordinate ascent.
# ============================================================================

# Conjugate surrogate model for one outer iteration. Every observation factor
# is replaced by two leaf factors on the FFG:
#   - a Gaussian pseudo-observation ỹ_i with precision Λ_i on the x edge,
#   - a Gamma pseudo-observation s_i = Δb_i on the τ edge; the GammaShapeRate
#     node with observed out = Δb_i and observed shape = Δa_i sends the exact
#     BP message  ∝ τ^{Δa_i} exp(−Δb_i τ)  to its rate.
# The surrogate graph is a forest, so one `infer` call is an exact BP sweep.
@model function approx_model(ytilde, Λ, s, Δa, m0, v0, a0, b0)
    x ~ Normal(mean = m0, variance = v0)
    τ ~ GammaShapeRate(a0, b0)
    for i in 1:length(ytilde)
        ytilde[i] ~ Normal(mean = x, precision = Λ[i])
        s[i] ~ GammaShapeRate(Δa[i], τ)
    end
end

# Mean-field variational free energy of the ORIGINAL (non-surrogate) model,
# F[q] = Σ_i E_q[−log N(y_i | x, 1/τ)] + KL(q(x)‖p(x)) + KL(q(τ)‖p(τ)),
# in closed form. E[log τ] is the first mean parameter of q(τ), i.e. the first
# component of ∇A(η) — provided by ExponentialFamily.jl as gradlogpartition.
function true_free_energy(ys, qx, qτ, m0, v0, a0, b0)
    m, v  = mean(qx), var(qx)
    τ̄     = mean(qτ)
    μτ    = gradlogpartition(convert(ExponentialFamilyDistribution, qτ))
    Elogτ = μτ[1]                       # E[log τ] = ∂A/∂η₁,  T_τ = (log τ, τ)
    energy = sum(@. 0.5 * log(2π) - 0.5 * Elogτ + 0.5 * τ̄ * ((ys - m)^2 + v))
    klx = kldivergence(Normal(m, sqrt(v)), Normal(m0, sqrt(v0)))
    klτ = kldivergence(Gamma(shape(qτ), 1 / rate(qτ)), Gamma(a0, 1 / b0))
    return energy + klx + klτ
end

# Natural-gradient BP with damping in natural-parameter space.
# Per-observation surrogate messages are carried across outer iterations as
# natural parameters:
#   x edge:  (ξ_i, Λ_i)   meaning  exp(ξ_i x − Λ_i x²/2)
#   τ edge:  (Δa_i, Δb_i) meaning  τ^{Δa_i} exp(−Δb_i τ)
# Starting from (0, 0) gives the same "soft warm-up" as the damped Poisson
# smoother: the first sweep injects only fraction α of the evidence.
function ng_bp_normal(ys;
                      m0       = 0.0,
                      v0       = 1.0,
                      a0       = 1.0,
                      b0       = 1.0,
                      max_iter = 200,
                      tol      = 1e-8,
                      α        = 0.25,
                      verbose  = true)
    N = length(ys)

    # persistent natural-parameter state of the surrogate messages
    ξ  = zeros(N);  Λ  = zeros(N)       # → x edge
    Δa = zeros(N);  Δb = zeros(N)       # → τ edge

    # initial edge marginals = priors
    qx = NormalMeanVariance(m0, v0)
    qτ = GammaShapeRate(a0, b0)

    Fhist  = Float64[]                  # true mean-field free energy
    Fbethe = Float64[]                  # BFE of the conjugate surrogate
    local result

    for iter in 1:max_iter
        # (A) project the tilted log-messages at the current edge marginals:
        #     η* = ∇_μ E_q[ℓ̃]  (projection is the identity here, see header)
        m, v = mean(qx), var(qx)
        τ̄    = mean(qτ)
        ξstar  = τ̄ .* ys
        Λstar  = fill(τ̄, N)
        Δastar = fill(0.5, N)
        Δbstar = 0.5 .* ((ys .- m) .^ 2 .+ v)

        # (B) natural-gradient step: geometric message damping
        #     μ̂^{(t)} ∝ (μ̂^{(t−1)})^{1−α} (μ̂*)^{α}  in natural coordinates
        ξ  = (1 - α) .* ξ  .+ α .* ξstar
        Λ  = (1 - α) .* Λ  .+ α .* Λstar
        Δa = (1 - α) .* Δa .+ α .* Δastar
        Δb = (1 - α) .* Δb .+ α .* Δbstar

        # (C) one exact BP sweep on the conjugate surrogate
        result = infer(
            model = approx_model(m0 = m0, v0 = v0, a0 = a0, b0 = b0),
            data  = (ytilde = ξ ./ Λ, Λ = Λ, s = Δb, Δa = Δa),
            free_energy = true,
        )
        qx = convert(NormalMeanVariance, result.posteriors[:x])
        qτ = result.posteriors[:τ]

        # (D) convergence on the true mean-field free energy
        F = true_free_energy(ys, qx, qτ, m0, v0, a0, b0)
        push!(Fhist, F)
        push!(Fbethe, result.free_energy[end])
        ΔF = length(Fhist) < 2 ? Inf : abs(Fhist[end] - Fhist[end-1])

        verbose && @info "iter $iter   F = $(round(F, sigdigits=8))   ΔF = $(round(ΔF, sigdigits=3))"

        if ΔF < tol
            verbose && @info "converged in $iter iterations (|ΔF| < $tol, α = $α)"
            break
        end
    end

    return (qx = qx, qτ = qτ,
            free_energy = Fhist, bethe = Fbethe,
            ξ = ξ, Λ = Λ, Δa = Δa, Δb = Δb)
end


# ----------------------------------------------------------------------------
# Demo: same setting as naive_mean_field.jl. Compare against RxInfer's
# mean-field VMP, sweep the step size α, and verify the main theorem
#     λ_edge = Σ_a η_{a→edge}
# numerically in natural coordinates via ExponentialFamily.jl.
# ----------------------------------------------------------------------------
begin
    Random.seed!(42)
    N  = 1000
    ys = rand(NormalMeanVariance(10.0, 10.0), N)     # true mean 10, precision 0.1

    m0, v0, a0, b0 = 0.0, 1.0, 1.0, 1.0

    # --- reference: naive mean-field VMP (as in naive_mean_field.jl) ---------
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
        iterations = 50,
        initialization = marginal_init(),
    )
    qx_vmp = convert(NormalMeanVariance, vmp.posteriors[:x][end])
    qτ_vmp = vmp.posteriors[:τ][end]
    F_vmp  = true_free_energy(ys, qx_vmp, qτ_vmp, m0, v0, a0, b0)

    # --- natural-gradient BP for several step sizes ---------------------------
    αs   = [0.1, 0.25, 0.5, 1.0]
    runs = Dict(α => ng_bp_normal(ys; m0, v0, a0, b0, α, verbose = false) for α in αs)
    res  = runs[0.25]

    println("\nposterior q(x)  (NG-BP, α=0.25) : ", res.qx)
    println("posterior q(x)  (VMP reference) : ", qx_vmp)
    println("posterior q(τ)  (NG-BP, α=0.25) : ", res.qτ)
    println("posterior q(τ)  (VMP reference) : ", qτ_vmp)
    println("E[τ] NG-BP / VMP / truth        : ",
            round(mean(res.qτ), digits = 5), " / ",
            round(mean(qτ_vmp), digits = 5), " / 0.1")
    println("final F NG-BP / VMP             : ",
            round(res.free_energy[end], digits = 6), " / ",
            round(F_vmp, digits = 6))

    # --- main theorem check: λ_edge = λ_prior + Σ_i η_{i→edge} ----------------
    # Natural coordinates via ExponentialFamily.jl. For the Gaussian edge
    # η = (ξ, −Λ/2); for the Gamma edge η = (a−1, −b).
    λx  = getnaturalparameters(convert(ExponentialFamilyDistribution, res.qx))
    λx′ = [m0 / v0 + sum(res.ξ), -(1 / v0 + sum(res.Λ)) / 2]
    λτ  = getnaturalparameters(convert(ExponentialFamilyDistribution, convert(GammaShapeRate, res.qτ)))
    λτ′ = [a0 - 1 + sum(res.Δa), -(b0 + sum(res.Δb))]
    println("\nstationarity λ = Σ η  (x edge): ", maximum(abs.(λx .- λx′)))
    println("stationarity λ = Σ η  (τ edge): ", maximum(abs.(λτ .- λτ′)))

    # --- plots ----------------------------------------------------------------
    # (1) free-energy trajectories for different natural-gradient step sizes
    p1 = plot(xlabel = "outer iteration", ylabel = "F",
              title = "True mean-field free energy", legend = :topright)
    for α in αs
        plot!(p1, runs[α].free_energy; lw = 2, marker = :circle, ms = 2,
              label = "α = $α")
    end
    hline!(p1, [F_vmp]; color = :black, ls = :dash, label = "VMP fixed point")

    # (2) surrogate-model BFE (what RxInfer reports on the conjugate forest)
    p2 = plot(xlabel = "outer iteration", ylabel = "F (surrogate)",
              title = "BFE of the conjugate surrogate", legend = :topright)
    for α in αs
        plot!(p2, runs[α].bethe; lw = 2, marker = :circle, ms = 2,
              label = "α = $α")
    end

    # (3) posterior over the mean: NG-BP vs VMP
    xs = range(mean(res.qx) - 5std(res.qx), mean(res.qx) + 5std(res.qx); length = 300)
    p3 = plot(xs, pdf.(Normal(mean(res.qx), std(res.qx)), xs);
              lw = 3, color = :dodgerblue, label = "NG-BP q(x)",
              xlabel = "x", ylabel = "density", title = "Posterior of the mean")
    plot!(p3, xs, pdf.(Normal(mean(qx_vmp), std(qx_vmp)), xs);
          lw = 2, ls = :dash, color = :black, label = "VMP q(x)")
    vline!(p3, [10.0]; color = :red, ls = :dot, label = "true mean")

    # (4) posterior over the precision: NG-BP vs VMP
    τs = range(0.07, 0.13; length = 300)
    p4 = plot(τs, pdf.(Gamma(shape(res.qτ), 1 / rate(res.qτ)), τs);
              lw = 3, color = :dodgerblue, label = "NG-BP q(τ)",
              xlabel = "τ", ylabel = "density", title = "Posterior of the precision")
    plot!(p4, τs, pdf.(Gamma(shape(qτ_vmp), 1 / rate(qτ_vmp)), τs);
          lw = 2, ls = :dash, color = :black, label = "VMP q(τ)")
    vline!(p4, [0.1]; color = :red, ls = :dot, label = "true precision")

    plt = plot(p1, p2, p3, p4; layout = (2, 2), size = (1100, 800))
    savefig(plt, joinpath(@__DIR__, "normal_surrogate_posterior.png"))
    display(plt)
end
