# =============================================================================
# Validation: cavi_inner_infer_structured vs RxInfer under q(w, z)q(τ)
# =============================================================================
# The script's constraints changed to structured q(w,z)q(τ) (2026-07-03).
# Compares posteriors at several iteration counts / schedules, and times the
# structured RxInfer path (which routes :y through the AR companion rule and
# builds a dense 66×66 joint per node — expected slower than mean-field).
# =============================================================================

using Distributions, LinearAlgebra, Printf, Statistics, Random
using RxInfer
using ProbabilisticEnsembling

include("cavi_inner.jl")

@model function inner_exp_ensemble(n_forecasters, n_obs, features, w_priors, τ_priors, obsz, Rz)
    local w, z, τ
    for i in 1:n_forecasters
        w[i] ~ w_priors[i]
        τ[i] ~ τ_priors[i]
    end
    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i]) where {meta = LowRankMeta()}
            obsz[i, j] ~ NormalMeanVariance(z[i, j], Rz[i, j])
        end
    end
end

@constraints function inner_constraints()   # the script's NEW factorization
    q(w, z, τ) = q(w, z)q(τ)
end

inner_init(w_priors, τ_priors) = @initialization begin
    q(w) = w_priors
    q(τ) = τ_priors
    q(z) = NormalMeanVariance(0.0, 1.0)
end

const nf = 7
const nfeat = 65

rng = MersenneTwister(1)
no = 800
features = [randn(rng, nfeat) for _ in 1:no]
obsz = randn(MersenneTwister(2), nf, no) .* 0.5 .+ 2.0
Rz = fill(0.7, nf, no)
w0 = [MvNormalMeanScalePrecision(zeros(nfeat), 0.01) for _ in 1:nf]
τ0 = [GammaShapeRate(1.0, 1e-3) for _ in 1:nf]

run_rxinfer(iters) = infer(model = inner_exp_ensemble(n_forecasters = nf, n_obs = no,
                               w_priors = w0, τ_priors = τ0),
                           data = (features = features, obsz = obsz, Rz = Rz),
                           constraints = inner_constraints(),
                           initialization = inner_init(w0, τ0),
                           iterations = iters, free_energy = false,
                           options = (limit_stack_depth = 200,))

function compare(res, cavi)
    qw = res.posteriors[:w][end]; qτ = res.posteriors[:τ][end]; qz = res.posteriors[:z][end]
    dμw = maximum(maximum(abs, mean(qw[i]) - mean(cavi.w[i])) for i in 1:nf)
    dΣw = maximum(maximum(abs, cov(qw[i]) - cov(cavi.w[i])) for i in 1:nf)
    dmz = maximum(abs(mean(qz[i, j]) - mean(cavi.z[i, j])) for i in 1:nf, j in 1:no)
    dvz = maximum(abs(var(qz[i, j]) - var(cavi.z[i, j])) for i in 1:nf, j in 1:no)
    da = maximum(abs(shape(qτ[i]) - shape(cavi.τ[i])) for i in 1:nf)
    db = maximum(abs(rate(qτ[i]) / rate(cavi.τ[i]) - 1) for i in 1:nf)
    return dμw, dΣw, dmz, dvz, da, db
end

println("== structured q(w,z)q(τ): posterior agreement (max |Δ|; Δrate relative) ==")
@printf("%5s  %-8s  %10s %10s %10s %10s %8s %10s\n",
        "iters", "schedule", "E[w]", "cov(w)", "E[z]", "V[z]", "shape(τ)", "rate(τ)")
for iters in (1, 2, 5, 50)
    res = run_rxinfer(iters)
    for sch in ((:wz, :τ), (:τ, :wz))
        cavi = cavi_inner_infer_structured(features, obsz, Rz, w0, τ0; iterations = iters, schedule = sch)
        d = compare(res, cavi)
        @printf("%5d  %-8s  %10.2e %10.2e %10.2e %10.2e %8.1e %10.2e\n",
                iters, join(String.(sch), ""), d...)
    end
    println()
end

println("== timing (iterations = 5, structured) ==")
run_rxinfer(2)
t_rx = @elapsed run_rxinfer(5)
cavi_inner_infer_structured(features, obsz, Rz, w0, τ0; iterations = 2)
t_cv = @elapsed cavi_inner_infer_structured(features, obsz, Rz, w0, τ0; iterations = 5)
@printf("no=%d:  RxInfer %.3f s (mean-field was 3.78 s)   CAVI %.5f s   speedup %.0f×\n",
        no, t_rx, t_cv, t_rx / t_cv)

no_full = 3398
features_f = [randn(rng, nfeat) for _ in 1:no_full]
obsz_f = randn(MersenneTwister(3), nf, no_full) .* 0.5 .+ 2.0
Rz_f = fill(0.7, nf, no_full)
cavi_inner_infer_structured(features_f, obsz_f, Rz_f, w0, τ0; iterations = 2)
t_cvf = @elapsed cavi_inner_infer_structured(features_f, obsz_f, Rz_f, w0, τ0; iterations = 5)
@printf("no=%d (full scale): CAVI %.4f s per outer → 40 outers ≈ %.1f s; RxInfer projected ≈ %.0f min\n",
        no_full, t_cvf, 40 * t_cvf, 40 * t_rx * (no_full / no) / 60)
