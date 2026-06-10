using RxInfer
using Distributions
using LinearAlgebra
using Random
using Statistics
using Plots

# ============================================================================
# Mean-field VMP baseline for the multivariate Normal with unknown mean AND
# unknown precision matrix (multivariate analogue of naive_mean_field.jl):
#
#     x ~ N(m0, V0)                       (mean vector, dim d)
#     W ~ Wishart(ν0, S0)                 (precision matrix, E[W] = ν0 S0)
#     y_i ~ N(x, W⁻¹),  i = 1..N
#
# Pattern follows RxInferExamples.jl "Basic Examples/Incomplete Data":
# MeanField() constraints q(x) q(W) and an explicit initialization of q(W).
# This is the baseline that natural-gradient BP (projected exact-BP messages,
# matrix-T toward x, "strange Wishart" toward W) is meant to improve.
#
# Ground truth: Gibbs sampling -- both full conditionals are conjugate,
#     x | W, y ~ N( Σn (V0⁻¹ m0 + W Σᵢ yᵢ),  Σn ),  Σn = (V0⁻¹ + N W)⁻¹
#     W | x, y ~ Wishart( ν0 + N, (S0⁻¹ + Σᵢ (yᵢ−x)(yᵢ−x)ᵀ)⁻¹ )
# so the sampler is exact and gives calibrated posterior moments.
# ============================================================================

@model function mv_normal_mf(y, m0, V0, ν0, S0)
    x ~ MvNormal(mean = m0, covariance = V0)
    W ~ Wishart(ν0, S0)
    for i in eachindex(y)
        y[i] ~ MvNormal(mean = x, precision = W)
    end
end

function vmp_fit(ys; m0, V0, ν0, S0, iterations = 50)
    init = @initialization begin
        q(W) = Wishart(ν0, S0)
    end
    return infer(
        model = mv_normal_mf(m0 = m0, V0 = V0, ν0 = ν0, S0 = S0),
        data = (y = ys,),
        constraints = MeanField(),
        initialization = init,
        iterations = iterations,
        free_energy = true,
    )
end

function gibbs_fit(ys; m0, V0, ν0, S0, nsamples = 50_000, burnin = 5_000)
    d, N = length(m0), length(ys)
    P0, Sy = inv(V0), sum(ys)
    S0inv  = inv(S0)
    x  = copy(m0)
    W  = ν0 .* S0
    xs = Vector{Vector{Float64}}()
    Ws = Vector{Matrix{Float64}}()
    for it in 1:(burnin + nsamples)
        Σn = Symmetric(inv(P0 + N .* W))
        x  = rand(MvNormal(Σn * (P0 * m0 .+ W * Sy), Σn))
        R  = sum((y .- x) * (y .- x)' for y in ys)
        W  = rand(Wishart(ν0 + N, Matrix(Symmetric(inv(S0inv .+ R)))))
        if it > burnin
            push!(xs, x); push!(Ws, Matrix(W))
        end
    end
    return (xs = xs, Ws = Ws)
end

# 2σ-equivalent ellipse of a 2D Gaussian (mean μ, covariance Σ)
function ellipse(μ, Σ; level = 2.0, n = 200)
    L = cholesky(Symmetric(Σ)).L
    θ = range(0, 2π; length = n)
    pts = [μ .+ level .* (L * [cos(t), sin(t)]) for t in θ]
    return first.(pts), last.(pts)
end


# ----------------------------------------------------------------------------
# Demo: d = 2, correlated noise, small N so the mean-field error is visible.
# ----------------------------------------------------------------------------
begin
    Random.seed!(42)
    d, N  = 2, 20
    μtrue = [10.0, -5.0]
    Σtrue = [10.0 6.0; 6.0 8.0]
    Wtrue = inv(Σtrue)
    ys    = [rand(MvNormal(μtrue, Σtrue)) for _ in 1:N]

    m0, V0 = zeros(d), 100.0 .* diageye(d)
    ν0, S0 = d + 1.0, diageye(d)

    result = vmp_fit(ys; m0, V0, ν0, S0)
    qx = result.posteriors[:x][end]
    qW = result.posteriors[:W][end]

    gibbs = gibbs_fit(ys; m0, V0, ν0, S0)
    gx_mean = mean(gibbs.xs)
    gx_cov  = cov(reduce(hcat, gibbs.xs)')
    gW_mean = mean(gibbs.Ws)

    println("posterior mean of x   VMP   : ", round.(mean(qx), digits = 4))
    println("posterior mean of x   Gibbs : ", round.(gx_mean, digits = 4))
    println("true mean                   : ", μtrue)
    println("\ncov of q(x)           VMP   : ", round.(cov(qx), digits = 5))
    println("cov of p(x|y)         Gibbs : ", round.(gx_cov, digits = 5))
    println("variance ratio VMP/Gibbs    : ",
            round.(diag(cov(qx)) ./ diag(gx_cov), digits = 3))
    println("\nE[W]                  VMP   : ", round.(mean(qW), digits = 5))
    println("E[W]                  Gibbs : ", round.(gW_mean, digits = 5))
    println("W true                      : ", round.(Wtrue, digits = 5))
    println("\nfinal free energy           : ", round(result.free_energy[end], digits = 5))

    # --- plots ----------------------------------------------------------------
    # (1) data + estimated observation ellipse
    e_true = ellipse(μtrue, Σtrue)
    e_vmp  = ellipse(mean(qx), inv(mean(qW)))
    p1 = scatter(first.(ys), last.(ys); ms = 4, color = :gray, label = "data",
                 xlabel = "y₁", ylabel = "y₂", title = "Observation model (2σ)")
    plot!(p1, e_true...; lw = 2.5, color = :black, label = "true N(μ, Σ)")
    plot!(p1, e_vmp...;  lw = 2.5, color = :dodgerblue, label = "VMP  N(E[x], E[W]⁻¹)")

    # (2) posterior of the mean: Gibbs cloud vs mean-field q(x)
    sub = gibbs.xs[1:25:end]
    e_qx = ellipse(mean(qx), cov(qx))
    e_gx = ellipse(gx_mean, gx_cov)
    p2 = scatter(first.(sub), last.(sub); ms = 2, alpha = 0.25, color = :gray,
                 label = "Gibbs p(x|y)", xlabel = "x₁", ylabel = "x₂",
                 title = "Posterior of the mean (2σ)")
    plot!(p2, e_gx...; lw = 2.5, color = :black, label = "Gibbs")
    plot!(p2, e_qx...; lw = 2.5, color = :dodgerblue, label = "mean-field q(x)")

    # (3) free energy convergence
    p3 = plot(result.free_energy; lw = 2, marker = :circle, ms = 3,
              color = :dodgerblue, label = "BFE",
              xlabel = "VMP iteration", ylabel = "F",
              title = "Free energy convergence")

    # (4) marginal std of the precision entries: VMP vs Gibbs
    labels = ["W₁₁", "W₁₂", "W₂₂"]
    idx = [(1, 1), (1, 2), (2, 2)]
    sd_vmp   = [sqrt(var(qW)[i, j]) for (i, j) in idx]
    sd_gibbs = [std([Wk[i, j] for Wk in gibbs.Ws]) for (i, j) in idx]
    p4 = groupedbar = bar(1:3, sd_gibbs; bar_width = 0.35, color = :gray,
                          label = "Gibbs", xticks = (1:3, labels),
                          ylabel = "posterior std",
                          title = "Uncertainty of precision entries")
    bar!(p4, (1:3) .+ 0.37, sd_vmp; bar_width = 0.35, color = :dodgerblue,
         label = "VMP q(W)")

    plt = plot(p1, p2, p3, p4; layout = (2, 2), size = (1100, 800))
    savefig(plt, joinpath(@__DIR__, "mv_normal_mean_field.png"))
    display(plt)
end
