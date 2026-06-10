using RxInfer
using Distributions
using Random
using Statistics
using Plots

# Linear-Gaussian chain with pseudo-observations y_k and per-step
# observation variances τs_k (= R_k). One `infer` call on this tree
# runs an exact forward-backward (RTS) sweep.
@model function approx_model(y, τs, σ, m0, v0)
    x[1] ~ Normal(mean = m0, variance = v0)
    y[1] ~ Normal(mean = x[1], variance = τs[1])
    for i in 2:length(y)
        x[i] ~ Normal(mean = x[i-1], variance = σ)
        y[i] ~ Normal(mean = x[i],   variance = τs[i])
    end
end

# Iterated Bethe / EKS smoother for a Poisson-observed log-rate chain:
#     y_k ~ Poisson(exp(z_k))
#     z_k = z_{k-1} + N(0, σ)
# Outer loop:
#   (A) build Gaussian surrogate (pseudo-obs ỹ, variance R) for every
#       Poisson factor at the current marginal (m_k, v_k)
#   (B) one BP sweep on the linear-Gaussian chain (exact on a tree)
#   (C) read off new marginals
#   (D) repeat until convergence
function bethe_poisson_smoother(y_counts;
                                σ        = 0.1,
                                m0       = 0.0,
                                v0       = 10.0,
                                max_iter = 100,
                                tol      = 1e-6)
    N    = length(y_counts)
    mbar = log.(y_counts .+ 1.0)         # initial marginal means
    vbar = fill(1.0, N)                  # initial marginal variances
    local result, mnew, vnew
    Fhist = Float64[]                    # Bethe Free Energy trajectory

    for iter in 1:max_iter
        # (A) refresh the Gaussian surrogate for every Poisson factor:
        #     λ̄_k    = E_q[exp(z_k)] = exp(m_k + v_k/2)
        #     ỹ_k    = m_k + (y_k - λ̄_k) / λ̄_k
        #     R_k    = 1 / λ̄_k
        λbar = exp.(mbar .+ vbar ./ 2)
        ytilde = mbar .+ (y_counts .- λbar) ./ λbar
        τs     = 1.0 ./ λbar

        # (B) one BP sweep on the linearized chain.
        result = infer(
            model = approx_model(σ = σ, m0 = m0, v0 = v0),
            data  = (y = ytilde, τs = τs),
            free_energy = true,
            options = (limit_stack_depth = 100,)
        )

        # (C) read off new q(z_k) = Normal(m_k, v_k)
        post = result.posteriors[:x]
        mnew = map(mean, post)
        vnew = map(var, post)
        mbar, vbar = mnew, vnew

        # Convergence on the Bethe Free Energy of the linearized model.
        F = isempty(result.free_energy) ? NaN : result.free_energy[end]
        push!(Fhist, F)
        ΔF = length(Fhist) < 2 ? Inf : abs(Fhist[end] - Fhist[end-1])

        @info "iter $iter   F = $(round(F, sigdigits=6))   ΔF = $(round(ΔF, sigdigits=3))"

        if ΔF < tol
            @info "converged in $iter iterations (|ΔF| < $tol)"
            break
        end
    end

    return (means = mbar, variances = vbar,
            free_energy = Fhist, result = result)
end


# ============================================================================
# Damped variant: damping in NATURAL-PARAMETER space.
#
# Carries the Poisson surrogate γ_k as persistent state (η^γ_k, Λ^γ_k) across
# outer iterations. Each iteration replaces it by the geometric product
#
#     γ_k_new  ∝  γ_k_old ^ (1-α)  *  γ_k_target ^ α
#
# which in natural form is the convex combination
#
#     Λ^γ_k  ←  (1-α) Λ^γ_k  +  α Λ*_k
#     η^γ_k  ←  (1-α) η^γ_k  +  α η*_k
#
# with target  Λ*_k = λ̄_k,  η*_k = y_k - λ̄_k + λ̄_k m̄_k,  λ̄_k = exp(m̄_k + v̄_k/2).
# Starting from (η^γ, Λ^γ) = (0, 0) gives a "soft warm-up": the first sweep
# injects only fraction α of the Poisson evidence. At a fixed point the
# damping disappears and stationarity matches the un-damped algorithm.
# ============================================================================
function bethe_damped_poisson_smoother(y_counts;
                                       σ        = 0.1,
                                       m0       = 0.0,
                                       v0       = 10.0,
                                       max_iter = 200,
                                       tol      = 1e-6,
                                       α        = 0.1)
    N    = length(y_counts)
    mbar = log.(y_counts .+ 1.0)
    vbar = fill(1.0, N)

    # persistent natural-parameter state of γ_k
    ηγ = zeros(N)
    Λγ = zeros(N)

    local result, mnew, vnew
    Fhist = Float64[]

    for iter in 1:max_iter
        # (A) target surrogate γ*_k in canonical form, evaluated at current marginal
        λbar  = exp.(mbar .+ vbar ./ 2)
        Λstar = λbar
        ηstar = y_counts .- λbar .+ λbar .* mbar

        # (B) damped mix in natural-parameter space  (power product of Gaussians)
        Λγ = (1 - α) .* Λγ .+ α .* Λstar
        ηγ = (1 - α) .* ηγ .+ α .* ηstar

        # (C) convert back to (ỹ, R) only at the moment of feeding RxInfer
        ytilde = ηγ ./ Λγ
        τs     = 1.0 ./ Λγ

        # (D) one BP sweep on the linearized chain
        result = infer(
            model = approx_model(σ = σ, m0 = m0, v0 = v0),
            data  = (y = ytilde, τs = τs),
            free_energy = true,
            options = (limit_stack_depth = 100,)
        )

        # (E) read new marginals
        post = result.posteriors[:x]
        mnew = map(mean, post)
        vnew = map(var, post)
        mbar, vbar = mnew, vnew

        F = isempty(result.free_energy) ? NaN : result.free_energy[end]
        push!(Fhist, F)
        ΔF = length(Fhist) < 2 ? Inf : abs(Fhist[end] - Fhist[end-1])

        @info "iter $iter   F = $(round(F, sigdigits=6))   ΔF = $(round(ΔF, sigdigits=3))"

        if ΔF < tol
            @info "converged in $iter iterations (|ΔF| < $tol, α = $α)"
            break
        end
    end

    return (means = mbar, variances = vbar,
            free_energy = Fhist, result = result,
            eta_gamma = ηγ, lambda_gamma = Λγ)
end


# ----------------------------------------------------------------------------
# Simulate a log-Gaussian Cox / Poisson state-space chain, run the smoother,
# and visualize how well the posteriors recover the latent random walk.
# Wrapped in begin ... end so the whole block runs as a single expression.
# ----------------------------------------------------------------------------
begin
    N      = 1000
    σ_true = 0.1
    seed   = 42

    Random.seed!(seed)
    z_true = cumsum(sqrt(σ_true) .* randn(N))
    y = map(z -> rand(Poisson(exp(z))), z_true)

    res = bethe_damped_poisson_smoother(y; σ = σ_true,
                                           m0 = 0.0, v0 = 10.0,
                                           max_iter = 300, tol = 1e-6, α=0.25)

    m = res.means
    v = res.variances
    s = sqrt.(v)
    F = res.free_energy

    rmse_z  = sqrt(mean((m .- z_true).^2))
    covered = mean((z_true .>= m .- 1.96 .* s) .& (z_true .<= m .+ 1.96 .* s))
    println("\nRMSE on z      : ", round(rmse_z,  digits = 4))
    println("95% coverage   : ", round(covered, digits = 3))
    println("final F        : ", round(F[end],   digits = 4))

    t      = 1:N
    λ_true = exp.(z_true)
    λ_post = exp.(m .+ v ./ 2)
    resid  = (z_true .- m) ./ s
    xs     = range(-4, 4; length = 200)

    # (1) log-rate space: true z_k vs posterior mean ± 1.96σ
    p1 = plot(t, z_true; label = "true z_k", lw = 2, color = :black,
              xlabel = "k", ylabel = "log-rate z_k",
              title  = "Latent log-rate: posterior vs truth")
    plot!(p1, t, m; ribbon = 1.96 .* s, fillalpha = 0.25,
          label = "posterior mean ± 1.96σ",
          color = :dodgerblue, lw = 2)

    # (2) rate space: true rate, counts, posterior expected rate
    p2 = scatter(t, y; label = "counts y_k", ms = 3, color = :gray,
                 xlabel = "k", ylabel = "rate / count",
                 title  = "Rate space")
    plot!(p2, t, λ_true; label = "true rate exp(z_k)", color = :black,     lw = 2)
    plot!(p2, t, λ_post; label = "E_q[exp(z_k)]",      color = :dodgerblue, lw = 2)

    # (3) calibration of posterior std
    p3 = histogram(resid; bins = 30, normalize = :pdf,
                   label = "(z_true − m)/σ",
                   xlabel = "standardized residual", ylabel = "density",
                   title  = "Posterior calibration",
                   color = :dodgerblue, alpha = 0.6)
    plot!(p3, xs, pdf.(Normal(0, 1), xs); label = "N(0,1)",
          color = :black, lw = 2)

    # (4) Bethe Free Energy convergence trajectory
    p4 = plot(1:length(F), F; lw = 2, marker = :circle, ms = 4,
              color = :dodgerblue, label = "F (linearized BFE)",
              xlabel = "outer iteration", ylabel = "F",
              title  = "Bethe Free Energy convergence")

    plt = plot(p1, p2, p3, p4; layout = (4, 1), size = (900, 1200),
               legend = :topleft)
    savefig(plt, joinpath(@__DIR__, "rxinfer_posterior.png"))
    display(plt)

    println("\nfirst 10 latent log-rates:")
    println("  true     : ", round.(z_true[1:10], digits = 3))
    println("  smoothed : ", round.(m[1:10],      digits = 3))
    println("  std-err  : ", round.(s[1:10],      digits = 3))
    println("  counts   : ", y[1:10])
end

# This experiment shows that there is indeed optimal momentum value if you will run this experiment with α=0.01 and smaller almost no momentum
# the result will be very bad model if you are using some good momentum value like α=0.4 it actually will converge to a good model and BFE is convering as well
# if α=0.75 the model is good but BFE does not converge and has periodic behaviour near the optimum 