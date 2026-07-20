import ExponentialFamily: MvNormalWeightedMeanPrecision
import LinearAlgebra: Diagonal, eigen

# Deterministic-node free-energy scoring marginal. Rebuild q(in) from its
# Gaussian cavity and the same exact backward Fisher site used by the message
# rule, then make only the scoring precision proper if a transient site is
# locally non-concave.
@marginalrule MvResidualSine(:in) (
    m_out::MultivariateNormalDistributionsFamily,
    m_in::MultivariateNormalDistributionsFamily,
    meta::ResidualSineMeta,
) = begin
    xi_out, Lambda_out = weightedmean_precision(m_out)
    exact = Logpdf(MvResidualSineGaussianBackwardMessage(xi_out, Lambda_out, meta))
    site = project(TangentProjection(type = ClosedForm), m_in, exact)
    eta = getnaturalparameters(site)
    d = length(mean(m_in))
    xi_site = eta[1:d]
    Lambda_site = -2 .* reshape(eta[(d + 1):end], d, d)
    xi_cavity, Lambda_cavity = weightedmean_precision(m_in)
    xi_q = xi_cavity .+ xi_site
    Lambda_q = Symmetric(Matrix(Lambda_cavity) .+ Lambda_site)
    return MvNormalWeightedMeanPrecision(
        xi_q,
        _mv_residual_sine_scoring_precision(Lambda_q),
    )
end

function _mv_residual_sine_scoring_precision(precision)
    decomposition = eigen(Matrix(precision))
    return decomposition.vectors *
           Diagonal(max.(decomposition.values, sqrt(eps(Float64)))) *
           decomposition.vectors'
end
