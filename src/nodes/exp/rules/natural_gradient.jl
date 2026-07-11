# Natural-gradient rules for ProbabilisticEnsembling's smooth positive Exp node.
# They are the inverse orientation of the Log-node rules: a Gaussian input pushes
# forward to LogNormal on the Gamma output, while a Gamma output pulls back to
# LogGamma on the Gaussian input.

import ProbabilisticEnsembling: Exp

@rule Exp(:out, NaturalGradientMessage) (
    m_in::UnivariateGaussianDistributionsFamily,
    q_out::GammaDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    exact = Logpdf(LogNormal(mean(m_in), std(m_in)))
    site = project(resolve_projection(getprojection(vconstraint)), q_out, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule Exp(:in, NaturalGradientMessage) (
    m_out::GammaDistributionsFamily,
    q_in::UnivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    exact = Logpdf(LogGamma(scale(m_out), shape(m_out); check_args = false))
    site = project(resolve_projection(getprojection(vconstraint)), q_in, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end
