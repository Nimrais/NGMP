# =============================================================================
# Predictive comparison: NGMP with mean-field vs structured inner CAVI
# =============================================================================
# Same NGMP outer loop as dynamic_exp_ngmp_surrogate.jl (project LogGamma leaves
# at q(z) → damped step → inner solve), but the inner conjugate model is solved
# by cavi_inner_infer (q(w)q(z)q(τ)) or cavi_inner_infer_structured (q(w,z)q(τ))
# instead of RxInfer `infer`. Trains both on full val, predicts on test with the
# harmonic and MC rules, prints PE metrics side by side.
#
# Prediction messages use the direct formula N(f'μ_w, f'Σ_w f + 1/E[τ]) —
# validated identical to the @call_rule softdot(:y)/Exp(:out) path (0.64 µs vs
# 222 µs per message).
# =============================================================================

using JLD2, Distributions, LinearAlgebra, Printf, Statistics, Random
using RxInfer
using ProbabilisticEnsembling
import ClosedFormExpectations: ClosedWilliamsProduct, Logpdf, LogGamma
using YAML

include("cavi_inner.jl")

cfg = YAML.load_file("sessions/dynamic/vae/dynamic_ETTh1_336.yaml")
spec = ProbabilisticEnsembling._parse_spec(cfg)
y_val, y_test, predictions_val, predictions_test, features_val, features_test =
    ProbabilisticEnsembling.before_rxinfer(spec);

function project_loggamma_to_normal(μ, m, σ)
    ∂μ, ∂σ = mean(ClosedWilliamsProduct(), Logpdf(μ), Normal(m, σ))
    Λ = -∂σ / σ
    ξ = ∂μ + m * Λ
    return ξ, Λ
end

function form_loggamma_messages(predictions, ys; αmax = 1e6)
    nf, n = size(predictions, 1), length(ys)
    q_prediction = PointMass.(predictions)
    q_y = [PointMass(y) for y in ys]
    μ_likelihood_γ = [
        @call_rule NormalMeanPrecision(:τ, Marginalisation) (q_out = q_y[j], q_μ = q_prediction[i, j])
        for i in 1:nf, j in 1:n
    ]
    μz = [
        @call_rule Exp(:in, Marginalisation) (m_out = μ_likelihood_γ[i, j],)
        for i in 1:nf, j in 1:n
    ]
    capα(m) = LogGamma(min(m.α, αmax), m.β; check_args = false)
    return capα.(μz)
end

# NGMP outer loop with a pluggable inner solver (mean-field or structured CAVI)
function ngmp_train_cavi(solver, features, predictions, y, w0, τ0;
                         outer = 40, inner = 5, α = 0.2, verbose = true)
    nf, no = size(predictions)
    μz = form_loggamma_messages(predictions, y)
    m = [log(μz[i, j].β * μz[i, j].α) for i in 1:nf, j in 1:no]
    v = ones(nf, no)
    ξ = zeros(nf, no); Λ = zeros(nf, no)
    local res
    for t in 1:outer
        ηξ = similar(ξ); ηΛ = similar(Λ)
        for j in 1:no, i in 1:nf
            ηξ[i, j], ηΛ[i, j] = project_loggamma_to_normal(μz[i, j], m[i, j], sqrt(v[i, j]))
        end
        rΛ = norm(ηΛ .- Λ) / max(norm(Λ), 1e-12)
        @. ξ += α * (ηξ - ξ)
        @. Λ += α * (ηΛ - Λ)
        res = solver(features, ξ ./ Λ, 1 ./ Λ, w0, τ0; iterations = inner)
        m = map(mean, res.z); v = map(var, res.z)
        verbose && t % 10 == 0 && (@printf("    outer %2d  rΛ=%.2e  meanΛ=%.3f  meanE[z]=%.3f  meanv=%.3f  mean(τ)=%.3f\n",
            t, rΛ, mean(Λ), mean(m), mean(v), mean(mean.(res.τ))); flush(stdout))
    end
    return res.w, res.τ
end

# direct forward z-messages (≡ @call_rule softdot(:y)/Exp(:out), validated)
function form_lognormal_messages_direct(w, τ, features)
    nf, no = length(w), length(features)
    M = zeros(nf, no); V = zeros(nf, no)
    for i in 1:nf
        μ = mean(w[i]); Σ = cov(w[i]); invτ̄ = 1 / mean(τ[i])
        for j in 1:no
            f = features[j]
            M[i, j] = dot(f, μ)
            V[i, j] = max(dot(f, Σ, f) + invτ̄, eps())
        end
    end
    return M, V
end

function ngmp_predict(w, τ, features, predictions; rule = :harmonic, S = 512, rng = MersenneTwister(7))
    M, V = form_lognormal_messages_direct(w, τ, features)
    nf, no = size(M)
    preds = Vector{Any}(undef, no)
    for j in 1:no
        if rule === :harmonic
            γh = exp.(M[:, j] .- V[:, j] ./ 2)
            μj = sum(γh .* predictions[:, j]) / sum(γh)
            preds[j] = NormalMeanVariance(μj, 1 / sum(γh))
        elseif rule === :mc
            μsum = 0.0; μ2sum = 0.0; vsum = 0.0
            for _ in 1:S
                Σγ = 0.0; Σγp = 0.0
                for i in 1:nf
                    γ = exp(M[i, j] + sqrt(V[i, j]) * randn(rng))
                    Σγ += γ; Σγp += γ * predictions[i, j]
                end
                μγ = Σγp / Σγ
                μsum += μγ; μ2sum += μγ^2; vsum += 1 / Σγ
            end
            μ̄ = μsum / S
            preds[j] = NormalMeanVariance(μ̄, vsum / S + (μ2sum / S - μ̄^2))
        end
    end
    return preds
end

# per-point 95% CIs: ±1.96·SE over the n test points (delta method for rmse,
# binomial for cov95); mc-rule preds are deterministic (fixed rng seed)
function metrics_ci(preds, y)
    n = length(y)
    μ = [mean(p) for p in preds]; σ = [std(p) for p in preds]
    ae = abs.(μ .- y); sq = (μ .- y) .^ 2
    llt = [logpdf(Normal(μ[j], σ[j]), y[j]) for j in 1:n]
    zc = 1.959963984540054
    covi = (y .>= μ .- zc .* σ) .& (y .<= μ .+ zc .* σ)
    mae = mean(ae); rmse = sqrt(mean(sq)); ll = mean(llt); p = mean(covi)
    return (; mae, mae_ci = zc * std(ae) / sqrt(n),
            rmse, rmse_ci = zc * std(sq) / sqrt(n) / (2 * rmse),
            ll, ll_ci = zc * std(llt) / sqrt(n),
            cov = p, cov_ci = zc * sqrt(p * (1 - p) / n),
            ciw = mean(2 .* zc .* σ))
end
fmt(v, c) = @sprintf("%.4f±%.4f", v, c)
function print_row(name, preds, y)
    m = metrics_ci(preds, y)
    @printf("%-26s  %15s  %15s  %16s  %15s  %7.3f\n", name,
            fmt(m.mae, m.mae_ci), fmt(m.rmse, m.rmse_ci), fmt(m.ll, m.ll_ci), fmt(m.cov, m.cov_ci), m.ciw)
end

nf    = size(predictions_val, 1)
nfeat = cfg["params"]["priors"]["w"]["n_features"]
w0 = [MvNormalMeanScalePrecision(zeros(nfeat), cfg["params"]["priors"]["w"]["scale"]) for _ in 1:nf]
τ0 = [GammaShapeRate(cfg["params"]["priors"]["τ"]["shape"], cfg["params"]["priors"]["τ"]["rate"]) for _ in 1:nf]

structured_mfτ(args...; kw...) = cavi_inner_infer_structured(args...; τ_scatter = :marginal, kw...)
variants = (("mean-field", cavi_inner_infer), ("structured", cavi_inner_infer_structured),
            ("structured+mfτ", structured_mfτ))
trained = Dict{String,Any}()
for (name, solver) in variants
    @printf("== training %s CAVI on %d val obs (outer=40, inner=5, α=0.2) ==\n", name, length(y_val))
    t = @elapsed w, τ = ngmp_train_cavi(solver, features_val, predictions_val, y_val, w0, τ0)
    trained[name] = (w, τ)
    @printf("  train time %.1f s   mean(τ) = %s\n", t, round.(mean.(τ), sigdigits = 3))
end

println("\n== prediction on ", length(y_test), " test obs (95% CIs, ±1.96·SE over test points) ==")
@printf("%-26s  %15s  %15s  %16s  %15s  %7s\n", "method", "mae", "rmse", "logpdf", "cov95", "ci_w")
for (name, _) in variants, rule in (:mc, :harmonic)
    w, τ = trained[name]
    print_row("$name $rule", ngmp_predict(w, τ, features_test, predictions_test; rule), y_test)
end
# --- (a) mix-and-match: structured w,z posterior + mean-field τ at prediction ---
for rule in (:mc, :harmonic)
    w, _ = trained["structured"]; _, τmf = trained["mean-field"]
    print_row("struct-w+mfτ $rule", ngmp_predict(w, τmf, features_test, predictions_test; rule), y_test)
end

# --- (b) additive noise floor at prediction: V_i = E[1/γ_i] + κ·E[β_i], with E[β]
# taken from the β `dynamic` model's VMP fit (the principled floor this model lacks)
const DYN_CACHE = "/private/tmp/claude-501/-Users-mykola-repos-papers-Lukashchuk-TMLR-Natural-gradient-message-passing/c7fe75dc-4431-49db-9114-8ffd50615e1b/scratchpad/etth1_h336_dynamic_prep.jld2"
Eβ_vmp = try
    dd = JLD2.load(DYN_CACHE)
    [dd["β_shape"][i] / dd["β_rate"][i] for i in 1:dd["n_forecasters"]]
catch e
    @warn "no dynamic cache, skipping β-floor variant" e
    nothing
end

function ngmp_predict_floor(w, τ, features, predictions, Eβ; κ = 1.0)
    M, V = form_lognormal_messages_direct(w, τ, features)
    nf, no = size(M)
    preds = Vector{Any}(undef, no)
    for j in 1:no
        P = [1 / (exp(-(M[i, j] - V[i, j] / 2)) + κ * Eβ[i]) for i in 1:nf]   # 1/(E[1/γ]+κE[β])
        μj = sum(P .* predictions[:, j]) / sum(P)
        preds[j] = NormalMeanVariance(μj, 1 / sum(P))
    end
    return preds
end

if Eβ_vmp !== nothing && length(Eβ_vmp) == nf
    for (name, _) in variants, κ in (1.0,)
        w, τ = trained[name]
        print_row("$name +βfloor κ=$κ",
                  ngmp_predict_floor(w, τ, features_test, predictions_test, Eβ_vmp; κ), y_test)
    end
end

# references (measured earlier on this split; no per-point data → point estimates only)
@printf("%-26s  %15s  %15s  %16s  %15s  %7s\n", "VMP dyn_exp",    "0.3032", "0.3692", "-57.5779", "0.1722", "0.157")
@printf("%-26s  %15s  %15s  %16s  %15s  %7s\n", "VMP dyn(β) ref", "0.2467", "0.3105", "-0.3141",  "0.9891", "1.606")

# --- paired per-point comparisons (shared test set → much tighter than unpaired) ---
function paired(nameA, predsA, nameB, predsB, y)
    n = length(y); zc = 1.959963984540054
    μA = mean.(predsA); μB = mean.(predsB)
    dae = abs.(μA .- y) .- abs.(μB .- y)
    dll = [logpdf(Normal(μA[j], std(predsA[j])), y[j]) - logpdf(Normal(μB[j], std(predsB[j])), y[j]) for j in 1:n]
    @printf("%-46s  Δmae %+.4f±%.4f   Δll %+.4f±%.4f\n",
            "$nameA − $nameB", mean(dae), zc * std(dae) / sqrt(n), mean(dll), zc * std(dll) / sqrt(n))
end

println("\n== paired per-point differences (negative Δmae / positive Δll favor the first) ==")
p_mf_mc   = ngmp_predict(trained["mean-field"]..., features_test, predictions_test; rule = :mc)
p_st_mc   = ngmp_predict(trained["structured"]..., features_test, predictions_test; rule = :mc)
p_mix_mc  = ngmp_predict(trained["structured"][1], trained["mean-field"][2], features_test, predictions_test; rule = :mc)
if Eβ_vmp !== nothing && length(Eβ_vmp) == nf
    p_mf_fl = ngmp_predict_floor(trained["mean-field"]..., features_test, predictions_test, Eβ_vmp)
    p_st_fl = ngmp_predict_floor(trained["structured"]..., features_test, predictions_test, Eβ_vmp)
    paired("mean-field+βfloor", p_mf_fl, "structured+βfloor", p_st_fl, y_test)
    paired("structured+βfloor", p_st_fl, "structured mc (no floor)", p_st_mc, y_test)
end
paired("structured mc", p_st_mc, "mean-field mc", p_mf_mc, y_test)
paired("struct-w+mfτ mc", p_mix_mc, "mean-field mc", p_mf_mc, y_test)
