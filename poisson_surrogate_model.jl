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
                                tol      = 1e-6,
                                verbose  = true,
                                snapshot_every = 0)
    N    = length(y_counts)
    mbar = log.(y_counts .+ 1.0)         # initial marginal means
    vbar = fill(1.0, N)                  # initial marginal variances
    local result, mnew, vnew
    Fhist = Float64[]                    # Bethe Free Energy trajectory
    Δqhist = Float64[]                   # marginal-update trajectory
    snapshots = NamedTuple[]

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
        Δq = max(maximum(abs.(mnew .- mbar)), maximum(abs.(vnew .- vbar)))
        push!(Δqhist, Δq)
        mbar, vbar = mnew, vnew

        # Keep the changing-surrogate BFE for diagnostics. Convergence is
        # measured on the posterior marginals because the surrogate changes.
        F = isempty(result.free_energy) ? NaN : result.free_energy[end]
        push!(Fhist, F)
        verbose && @info "iter $iter   F_surrogate = $(round(F, sigdigits=6))   Δq = $(round(Δq, sigdigits=3))"

        save_snapshot = snapshot_every > 0 && iter % snapshot_every == 0
        if save_snapshot || Δq < tol || iter == max_iter
            push!(snapshots, (iteration = iter, means = copy(mbar),
                              variances = copy(vbar)))
        end

        if Δq < tol
            verbose && @info "converged in $iter iterations (Δq < $tol)"
            break
        end
    end

    return (means = mbar, variances = vbar,
            free_energy = Fhist, update_norm = Δqhist,
            iterations = length(Δqhist), converged = Δqhist[end] < tol,
            snapshots = snapshots,
            result = result)
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
                                       α        = 0.1,
                                       verbose  = true,
                                       snapshot_every = 0)
    N    = length(y_counts)
    mbar = log.(y_counts .+ 1.0)
    vbar = fill(1.0, N)

    # persistent natural-parameter state of γ_k
    ηγ = zeros(N)
    Λγ = zeros(N)

    local result, mnew, vnew
    Fhist = Float64[]
    Δqhist = Float64[]
    snapshots = NamedTuple[]

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
        Δq = max(maximum(abs.(mnew .- mbar)), maximum(abs.(vnew .- vbar)))
        push!(Δqhist, Δq)
        mbar, vbar = mnew, vnew

        F = isempty(result.free_energy) ? NaN : result.free_energy[end]
        push!(Fhist, F)
        verbose && @info "iter $iter   F_surrogate = $(round(F, sigdigits=6))   Δq = $(round(Δq, sigdigits=3))"

        save_snapshot = snapshot_every > 0 && iter % snapshot_every == 0
        if save_snapshot || Δq < tol || iter == max_iter
            push!(snapshots, (iteration = iter, means = copy(mbar),
                              variances = copy(vbar)))
        end

        if Δq < tol
            verbose && @info "converged in $iter iterations (Δq < $tol, α = $α)"
            break
        end
    end

    return (means = mbar, variances = vbar,
            free_energy = Fhist, update_norm = Δqhist,
            iterations = length(Δqhist), converged = Δqhist[end] < tol,
            snapshots = snapshots,
            result = result,
            eta_gamma = ηγ, lambda_gamma = Λγ)
end


# ----------------------------------------------------------------------------
# Simulate one data set and compare plain and damped Bethe smoothers using
# identical priors, initialization, tolerance, and iteration budget.
# Wrapped in begin ... end so the whole block runs as a single expression.
# ----------------------------------------------------------------------------
begin
    N      = 1000
    σ_true = 0.1
    seed   = 42

    Random.seed!(seed)
    z_true = cumsum(sqrt(σ_true) .* randn(N))
    y = map(z -> rand(Poisson(exp(z))), z_true)

    common = (;σ = σ_true, m0 = 0.0, v0 = 10.0,
              max_iter = 200, tol = 1e-6, verbose = false,
              snapshot_every = 50)

    # Compile both paths before timing so method order does not dominate the
    # runtime comparison.
    warm_y = y[1:10]
    bethe_poisson_smoother(warm_y; σ = σ_true, max_iter = 1, verbose = false)
    bethe_damped_poisson_smoother(warm_y; σ = σ_true, max_iter = 1,
                                  α = 0.25, verbose = false)

    plain_t  = @timed bethe_poisson_smoother(y; common...)
    damped_t = @timed bethe_damped_poisson_smoother(y; common..., α = 0.25)
    plain, damped = plain_t.value, damped_t.value

    function metrics(res, elapsed)
        s = sqrt.(res.variances)
        return (rmse = sqrt(mean((res.means .- z_true) .^ 2)),
                coverage = mean(abs.(z_true .- res.means) .<= 1.96 .* s),
                iterations = res.iterations, converged = res.converged,
                seconds = elapsed)
    end
    plain_metrics  = metrics(plain, plain_t.time)
    damped_metrics = metrics(damped, damped_t.time)

    println("\nPoisson smoother comparison (same data and stopping rule)")
    println(rpad("method", 14), rpad("RMSE", 10), rpad("coverage", 12),
            rpad("iters", 8), rpad("seconds", 10), "converged")
    for (name, x) in (("Bethe", plain_metrics), ("Damped (0.25)", damped_metrics))
        println(rpad(name, 14), rpad(string(round(x.rmse, digits = 4)), 10),
                rpad(string(round(x.coverage, digits = 3)), 12),
                rpad(string(x.iterations), 8),
                rpad(string(round(x.seconds, digits = 3)), 10), x.converged)
    end

    t      = 1:N
    λ_true = exp.(z_true)
    s_plain, s_damped = sqrt.(plain.variances), sqrt.(damped.variances)

    # (1) log-rate space: true z_k vs posterior mean ± 1.96σ
    p1 = plot(t, z_true; label = "true z_k", lw = 2, color = :black,
              xlabel = "k", ylabel = "log-rate z_k",
              title  = "Latent log-rate")
    plot!(p1, t, plain.means; ribbon = 1.96 .* s_plain, fillalpha = 0.12,
          label = "Bethe", color = :darkorange, lw = 1.5)
    plot!(p1, t, damped.means; ribbon = 1.96 .* s_damped, fillalpha = 0.12,
          label = "damped Bethe (α=0.25)", color = :dodgerblue, lw = 1.5)

    # (2) rate space: true rate, counts, posterior expected rate
    p2 = scatter(t, y; label = "counts y_k", ms = 3, color = :gray,
                 xlabel = "k", ylabel = "rate / count",
                 title  = "Rate space")
    plot!(p2, t, λ_true; label = "true rate exp(z_k)", color = :black,     lw = 2)
    plot!(p2, t, exp.(plain.means .+ plain.variances ./ 2);
          label = "Bethe rate", color = :darkorange, lw = 1.5)
    plot!(p2, t, exp.(damped.means .+ damped.variances ./ 2);
          label = "damped rate", color = :dodgerblue, lw = 1.5)

    # The common stopping statistic is directly comparable across methods.
    p3 = plot(plain.update_norm; yscale = :log10, lw = 2,
              color = :darkorange, label = "Bethe",
              xlabel = "outer iteration", ylabel = "max marginal change",
              title = "Convergence")
    plot!(p3, damped.update_norm; lw = 2, color = :dodgerblue,
          label = "damped Bethe (α=0.25)")

    # This BFE belongs to a different Gaussian surrogate each iteration. It is
    # useful diagnostically, but it is not a shared objective or stopping rule.
    p4 = plot(plain.free_energy; lw = 2, color = :darkorange, label = "Bethe",
              xlabel = "outer iteration", ylabel = "surrogate BFE",
              title = "Changing-surrogate BFE")
    plot!(p4, damped.free_energy; lw = 2, color = :dodgerblue,
          label = "damped Bethe (α=0.25)")

    plt = plot(p1, p2, p3, p4; layout = (4, 1), size = (900, 1200),
               legend = :topright, dpi = 600)
    savefig(plt, joinpath(@__DIR__, "poisson_bethe_comparison.png"))
    savefig(plt, joinpath(@__DIR__, "poisson_bethe_comparison.pdf"))
    display(plt)

    # Progress snapshots expose how damping changes the path, not just the
    # final answer. After one method converges, its final state is held fixed
    # in subsequent panels while the other method continues iterating.
    function latest_snapshot(snapshots, iteration)
        eligible = filter(s -> s.iteration <= iteration, snapshots)
        return isempty(eligible) ? first(snapshots) : last(eligible)
    end

    checkpoint_iters = collect(50:50:max(plain.iterations, damped.iterations))
    progress_plots = Any[]
    for iter in checkpoint_iters
        plain_snapshot = latest_snapshot(plain.snapshots, iter)
        damped_snapshot = latest_snapshot(damped.snapshots, iter)
        plain_ci = 1.96 .* sqrt.(plain_snapshot.variances)
        damped_ci = 1.96 .* sqrt.(damped_snapshot.variances)
        p = plot(t, z_true; color = :black, lw = 1.5, label = "truth",
                 xlabel = "k", ylabel = "z_k", title = "Iteration $iter")
        plot!(p, t, plain_snapshot.means; ribbon = plain_ci,
              color = :darkorange, fillalpha = 0.12, lw = 1.5,
              label = "Bethe 95% CI @$(plain_snapshot.iteration)")
        plot!(p, t, damped_snapshot.means; ribbon = damped_ci,
              color = :dodgerblue, fillalpha = 0.12, lw = 1.5,
              label = "damped 95% CI @$(damped_snapshot.iteration)")
        push!(progress_plots, p)
    end

    p_energy = plot(plain.free_energy; lw = 2, color = :darkorange,
                    label = "Bethe", xlabel = "outer iteration",
                    ylabel = "surrogate BFE", title = "Free energy")
    plot!(p_energy, damped.free_energy; lw = 2, color = :dodgerblue,
          label = "damped Bethe (α=0.25)")
    checkpoint_rows = cld(length(progress_plots), 2)
    checkpoint_plt = plot(progress_plots...;
                          layout = (checkpoint_rows, 2), legend = :topright)
    progress_layout = @layout [checkpoints{0.72h}
                               energy{0.28h}]
    progress_plt = plot(checkpoint_plt, p_energy; layout = progress_layout,
                        size = (1200, 320 * checkpoint_rows + 380),
                        legend = :topright, dpi = 600)
    savefig(progress_plt, joinpath(@__DIR__, "poisson_bethe_progress.png"))
    savefig(progress_plt, joinpath(@__DIR__, "poisson_bethe_progress.pdf"))
    display(progress_plt)

    println("\nfirst 10 latent log-rates:")
    println("  true     : ", round.(z_true[1:10], digits = 3))
    println("  Bethe    : ", round.(plain.means[1:10],  digits = 3))
    println("  damped   : ", round.(damped.means[1:10], digits = 3))
    println("  counts   : ", y[1:10])
end
