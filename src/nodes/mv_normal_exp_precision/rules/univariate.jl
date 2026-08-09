# Univariate (d = 1) twins of the MvNormalExpPrecision rules, with SCALAR
# edges: y ~ N(μ, exp(s)⁻¹) where y, μ, s are univariate. This is the node's
# role inside per-observation deep-kernel hierarchies — the fused replacement
# of the `precision ~ Exp(score); y ~ softdot(f, v, precision)` link: the Gamma
# edge and the Exp node disappear, `s` (= the old score) feeds the likelihood
# directly. Same math as the multivariate rules at d = 1; the site toward `s`
# IS the scalar `ExpGammaSiteMessage`, so the closed-form projection and
# damping paths are reused verbatim.

import ExponentialFamily: NormalMeanPrecision
import BayesBase: mean_var

_mnep_mean_vardiag(q::UnivariateNormalDistributionsFamily) = mean_var(q)

# toward y:  N(E[μ], ρ⁻¹),  ρ = E[e^s] = exp(m + v/2)
@rule MvNormalExpPrecision(:out, Marginalisation) (q_μ::Any, q_s::UnivariateNormalDistributionsFamily, meta::Any) = begin
    m, v = mean_var(q_s)
    return NormalMeanPrecision(mean(q_μ), exp(_mnep_clamp_exponent(m + v / 2)))
end

# toward μ:  N(E[y], ρ⁻¹)
@rule MvNormalExpPrecision(:μ, Marginalisation) (q_out::Any, q_s::UnivariateNormalDistributionsFamily, meta::Any) = begin
    m, v = mean_var(q_s)
    return NormalMeanPrecision(mean(q_out), exp(_mnep_clamp_exponent(m + v / 2)))
end

# NGMP toward s: the exact mean-field site ½s − ½E[(y−μ)²]e^s is the scalar
# ExpGammaSiteMessage — project through the EXISTING scalar ClosedForm
# Williams-product path and damp.
@rule MvNormalExpPrecision(:s, NaturalGradientMessage) (
    q_out::Any,
    q_μ::Any,
    q_s::UnivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    E = _mnep_expected_square_residuals(q_out, q_μ)
    exact = Logpdf(ExpGammaSiteMessage(one(E) / 2, E / 2))
    site = project(resolve_projection(getprojection(vconstraint)), q_s, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@average_energy MvNormalExpPrecision (q_out::Any, q_μ::Any, q_s::UnivariateNormalDistributionsFamily, meta::Any) = begin
    m, v = mean_var(q_s)
    ρ = exp(_mnep_clamp_exponent(m + v / 2))
    E = _mnep_expected_square_residuals(q_out, q_μ)
    return log(2π) / 2 - m / 2 + ρ * E / 2
end
