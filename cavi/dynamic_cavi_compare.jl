# =============================================================================
# β `dynamic` model: NGMP with CAVI inner solvers (structured vs mean-field)
# =============================================================================
# Same construction as dynamic_ngmp_surrogate.jl (which used RxInfer with the
# structured constraint q(w,z)q(τ) and stock softdot rules), but the inner
# conjugate z-model is solved by cavi_inner_infer_structured / cavi_inner_infer
# — validated exact vs (bug-free) RxInfer, ~1000× faster. β is learned by the
# conjugate Gamma update from E[γ]; prediction uses the additive noise floor
# V_i = E[1/γ_i] + κ·E[β_i] (κ=1 principled).
#
# Targets: VMP dynamic(β) [paper model] mean-ll −0.3141; paper-reported NLL 0.412.
# =============================================================================

using JLD2, Distributions, LinearAlgebra, Printf, Statistics
using RxInfer            # distribution types
using ProbabilisticEnsembling
import ClosedFormExpectations: ClosedWilliamsProduct, Logpdf, LogGamma

include("cavi_inner.jl")

const CACHE = get(ENV, "NGMP_DYN_CACHE",
    "/private/tmp/claude-501/-Users-mykola-repos-papers-Lukashchuk-TMLR-Natural-gradient-message-passing/c7fe75dc-4431-49db-9114-8ffd50615e1b/scratchpad/etth1_h336_dynamic_prep.jld2")

function project_loggamma_to_normal(α, β, m, σ)
    ∂μ, ∂σ = mean(ClosedWilliamsProduct(), Logpdf(LogGamma(α, β; check_args = false)), Normal(m, σ))
    Λ = -∂σ / σ
    ξ = ∂μ + m * Λ
    return ξ, Λ
end

# moment-matched LogNormal(m,v) → Gamma(a,b): E[γ]=exp(m+v/2), Var/E² = eᵛ−1
function zforward_gamma(m, v)
    em = exp(clamp(v, 1e-6, 12.0)) - 1
    a = 1 / em
    b = exp(-m - v / 2) / em
    return a, b
end

# metrics with 95% CIs: ±1.96·SE over test points (delta method for rmse,
# binomial for cov95)
function metrics(μ, σ, y; quantiles = (0.1, 0.9))
    n = length(y)
    ll_terms = [logpdf(Normal(μ[j], σ[j]), y[j]) for j in 1:n]
    zc = 1.959963984540054
    ae = abs.(μ .- y); sq = (μ .- y) .^ 2
    covi = (y .>= μ .- zc .* σ) .& (y .<= μ .+ zc .* σ)
    pint = zeros(n)
    for q in quantiles
        zq = quantile(Normal(), q); qhat = μ .+ zq .* σ
        pint .+= max.(q .* (y .- qhat), (q - 1) .* (y .- qhat)) ./ length(quantiles)
    end
    mae = mean(ae); rmse = sqrt(mean(sq)); ll = mean(ll_terms); p = mean(covi)
    return (; mae, mae_ci = zc * std(ae) / sqrt(n),
            rmse, rmse_ci = zc * std(sq) / sqrt(n) / (2 * rmse),
            ll, ll_ci = zc * std(ll_terms) / sqrt(n), ll_std = std(ll_terms),
            cov95 = p, cov_ci = zc * sqrt(p * (1 - p) / n),
            pinball = mean(pint), pin_ci = zc * std(pint) / sqrt(n))
end
fmt(v, c) = @sprintf("%.4f±%.4f", v, c)
function print_row(name, m)
    @printf("%-30s  %15s  %15s  %16s  %15s  %15s\n", name,
            fmt(m.mae, m.mae_ci), fmt(m.rmse, m.rmse_ci), fmt(m.ll, m.ll_ci),
            fmt(m.cov95, m.cov_ci), fmt(m.pinball, m.pin_ci))
end

# forward z-messages (softdot marginal with vacuous pseudo-obs) — direct formula,
# ≡ infer_qz(..., obsz=0, Rz=1e12, iters=1): mz = f'μ_w, vz = f'Σ_w f + 1/E[τ]
function forward_z(features, w, τ)
    nf, no = length(w), length(features)
    M = zeros(nf, no); V = zeros(nf, no)
    for i in 1:nf
        μ = mean(w[i]); Σ = cov(w[i]); invτ̄ = 1 / mean(τ[i])
        for j in 1:no
            f = features[j]
            M[i, j] = dot(f, μ)
            V[i, j] = max(dot(f, Σ, f) + invτ̄, 1e-12)
        end
    end
    return M, V
end

function dyn_predict(features, predictions, w, τ, Eβ; κ = 1.0)
    mz, vz = forward_z(features, w, τ)
    nf, no = size(mz)
    V = exp.(.-mz .+ vz ./ 2) .+ κ .* reshape(Float64.(Eβ), nf, 1)
    P = clamp.(1 ./ V, 1e-6, 1e6)
    μ = Vector{Float64}(undef, no); σ = Vector{Float64}(undef, no)
    for j in 1:no
        τc = sum(@view P[:, j])
        μ[j] = sum(P[i, j] * predictions[i, j] for i in 1:nf) / τc
        σ[j] = sqrt(1 / τc)
    end
    return μ, σ
end

# NGMP training of w, τ, β — same loop as dynamic_ngmp_surrogate.jl, inner solve
# replaced by a CAVI solver (structured or mean-field)
function dyn_train(solver, features, predictions, y, w0, τ0, β_shape0, β_rate0;
                   outer = 40, inner = 10, α_d = 0.2, β_m = 0.5, verbose = true)
    nf, no = length(w0), length(y)
    r = [max((y[j] - predictions[i, j])^2 / 2, 1e-8) for i in 1:nf, j in 1:no]
    Eβ = fill(β_shape0 / β_rate0, nf)
    obsz = [log(1.5) - log(r[i, j] + Eβ[i]) for i in 1:nf, j in 1:no]
    Rz = fill(1 / 1.5, nf, no); ξ = obsz ./ Rz; Λ = 1 ./ Rz
    vξ = zeros(nf, no); vΛ = zeros(nf, no)
    local res
    for t in 1:outer
        res = solver(features, obsz, Rz, w0, τ0; iterations = inner)
        mz = map(mean, res.z); vz = map(var, res.z)
        # z→β: E[γ] from q(γ) = zforward × Gamma(1,Eβ) × y-lik Gamma(3/2,r)
        Eγ = Matrix{Float64}(undef, nf, no)
        for j in 1:no, i in 1:nf
            a, b = zforward_gamma(clamp(mz[i, j], -30, 30), clamp(vz[i, j], 1e-6, 8))
            Eγ[i, j] = (a + 0.5) / (b + Eβ[i] + r[i, j])
        end
        for i in 1:nf
            Eβ[i] = (β_shape0 + no) / (β_rate0 + sum(@view Eγ[i, :]))
        end
        # β→z: re-project LogGamma(α=1/(Eβ+r), 3/2) at q(z); momentum + anti-windup
        ηξ = similar(ξ); ηΛ = similar(Λ)
        for j in 1:no, i in 1:nf
            mc = clamp(mz[i, j], -30, 30); vc = clamp(vz[i, j], 1e-6, 8)
            ξt, Λt = project_loggamma_to_normal(1 / (Eβ[i] + r[i, j]), 1.5, mc, sqrt(vc))
            ηξ[i, j] = ξt; ηΛ[i, j] = clamp(Λt, 1e-6, 1e3)
        end
        @. vξ = β_m * vξ + α_d * (ηξ - ξ); @. ξ += vξ
        @. vΛ = β_m * vΛ + α_d * (ηΛ - Λ); @. Λ += vΛ
        Λc = clamp.(Λ, 1e-3, 1e3); nobsz = clamp.(ξ ./ Λc, -25.0, 25.0)
        Δ = maximum(abs.(obsz .- nobsz))
        ξ .= nobsz .* Λc; Λ .= Λc; obsz .= nobsz; Rz .= 1 ./ Λc
        verbose && t % 10 == 0 &&
            (@printf("    outer %2d  Δ=%.3e  meanΛ=%.3f  meanEβ=%.4g  mean(τ)=%.3g\n",
                     t, Δ, mean(Λc), mean(Eβ), mean(mean.(res.τ))); flush(stdout))
    end
    return res.w, res.τ, Eβ
end

# =============================================================================
d = JLD2.load(CACHE)
nf = d["n_forecasters"]
ftest = [Vector{Float64}(d["features_test"][:, j]) for j in 1:size(d["features_test"], 2)]
ptest = d["predictions_test"]; ytest = d["y_test"]
fval = [Vector{Float64}(d["features_val"][:, j]) for j in 1:size(d["features_val"], 2)]
pval = d["predictions_val"]; yval = d["y_val"]
w_vmp = [MvNormalMeanCovariance(d["w_means"][i], Symmetric(Matrix(d["w_covs"][i]))) for i in 1:nf]
τ_vmp = [GammaShapeRate(d["τ_shape"][i], d["τ_rate"][i]) for i in 1:nf]
Eβ_vmp = [d["β_shape"][i] / d["β_rate"][i] for i in 1:nf]
vmpm = d["vmp_metrics"]
nfeat = d["priors_w_nfeat"]
w0 = [MvNormalMeanScalePrecision(zeros(nfeat), Float64(d["priors_w_scale"])) for _ in 1:nf]
τ0 = [GammaShapeRate(Float64(d["priors_τ_shape"]), Float64(d["priors_τ_rate"])) for _ in 1:nf]

println("== ETTh1 h336 OT — β dynamic model, n_val=$(length(yval)), n_test=$(length(ytest)), nf=$nf ==")
println("   (95% CIs, ±1.96·SE over the $(length(ytest)) test points)")
@printf("%-30s  %15s  %15s  %16s  %15s  %15s\n", "method", "mae", "rmse", "mean-ll", "cov95", "pinball")
# VMP reference: only aggregates cached; CI on mean-ll from its nll_std, binomial for cov95
let n = length(ytest), zc = 1.959963984540054, p = vmpm.ci95_target_overlap
    @printf("%-30s  %15s  %15s  %16s  %15s  %15s\n", "VMP dynamic(β) [paper]",
            @sprintf("%.4f", vmpm.mae), @sprintf("%.4f", vmpm.rmse),
            fmt(vmpm.nll, zc * vmpm.nll_std / sqrt(n)), fmt(p, zc * sqrt(p * (1 - p) / n)), "-")
end

# regime (a): predict-only from the VMP-trained posteriors
μa, σa = dyn_predict(ftest, ptest, w_vmp, τ_vmp, Eβ_vmp; κ = 1.0)
print_row("NGMP predict-only κ=1", metrics(μa, σa, ytest))

# regime (b): NGMP-train w, τ, β with each inner solver
for (name, solver) in (("structured", cavi_inner_infer_structured), ("mean-field", cavi_inner_infer))
    @printf("\n== NGMP train (%s inner CAVI) on %d val obs ==\n", name, length(yval))
    t = @elapsed wb, τb, Eβb = dyn_train(solver, fval, pval, yval, w0, τ0,
                                         Float64(d["priors_β_shape"]), Float64(d["priors_β_rate"]))
    @printf("  train %.1f s   E[β] = %s   mean(τ) = %s\n", t,
            round.(Eβb, sigdigits = 3), round.(mean.(τb), sigdigits = 3))
    for κ in (1.0, 0.5)
        μb, σb = dyn_predict(ftest, ptest, wb, τb, Eβb; κ)
        print_row("NGMP trained ($name) κ=$κ", metrics(μb, σb, ytest))
    end
end
