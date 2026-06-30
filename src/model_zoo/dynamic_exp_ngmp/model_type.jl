export DynamicExpNGMP
# ============================================================================

# ============================================================================

import ProbabilisticEnsembling as PE
import ProbabilisticEnsembling: ModelType, create_based_on_symbol, model_type_name,
    parse_priors, extract_prediction_priors, parse_mvn_mean_scale_precision_priors
using ExponentialFamily: GammaShapeRate

"""
DynamicExpNGMP — the natural-gradient message passing (NGMP) surrogate version
of ProbabilisticEnsembling's `dynamic_exp` model.

Generative model is IDENTICAL to dynamic_exp (per forecaster i, obs j):
      z[i,j] ~ softdot(features[j], w[i], τ[i])
      γ[i,j] = exp(z[i,j])
      y[j]   ~ NormalMeanPrecision(predictions[i,j], γ[i,j])
What changes is inference: instead of VMP with a I-projection
projection through the Exp node, the deterministic γ=exp(z) link is replaced by
a conjugate Gaussian pseudo-observation on z whose natural parameters are the
EXACT Fisher-tangent projection of the y→z log-message (LogGamma), recomputed at
the receiving marginal q(z) in an outer fixed-point loop.

This file mirrors src/model_zoo/dynamic_exp/model_type.jl from
ProbabilisticEnsembling and registers the model with the same YAML pipeline.
"""
struct DynamicExpNGMP <: ModelType end

# YAML `model_type: dynamic_exp_ngmp`  ->  DynamicExpNGMP()
create_based_on_symbol(::Val{:dynamic_exp_ngmp}) = DynamicExpNGMP()
model_type_name(::DynamicExpNGMP) = "dynamic_exp_ngmp"

# ---------------------------------------------------------------------------
# Priors — identical to dynamic_exp (w, τ). The NGMP outer
# loop hyperparameters (damping/momentum/outer iters) are read in pipeline.jl
# from the same `priors`/spec block (TODO: decide whether they live under
# `priors` or a dedicated `ngmp:` YAML block).
# ---------------------------------------------------------------------------
function parse_priors(::DynamicExpNGMP, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()
    τ_cfg = cfg["τ"]
    priors[:τ] = [GammaShapeRate(τ_cfg["shape"], τ_cfg["rate"]) for _ = 1:n_forecasters]
    priors[:w] = parse_mvn_mean_scale_precision_priors(cfg["w"], n_forecasters; prior_name = "w")
    return priors
end

# ---------------------------------------------------------------------------
# Reconstruct prediction priors from a saved run (predict_from_trained_ensemble).
# Same schema as dynamic_exp: w, τ posteriors only.
# ---------------------------------------------------------------------------
function extract_prediction_priors(::DynamicExpNGMP, saved)
    _to_gamma(d) = GammaShapeRate(d.a, d.b)
    return Dict{Symbol,Any}(
        :w => saved["w_posteriors"],
        :τ => map(_to_gamma, saved["τ_posteriors"]),
    )
end
