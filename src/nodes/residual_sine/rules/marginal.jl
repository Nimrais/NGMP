# Deterministic-node free-energy scoring marginal. Rebuild q(in) from its
# Gaussian cavity and the same exact backward Fisher site used by the message
# rule (the univariate analogue of the MvResidualSine marginal), flooring only
# the scoring precision if a transient site is locally non-concave.
@marginalrule ResidualSine(:in) (
    m_out::UnivariateGaussianDistributionsFamily,
    m_in::UnivariateGaussianDistributionsFamily,
    meta::ResidualSineMeta,
) = begin
    xi_out, Lambda_out = weightedmean_precision(m_out)
    exact = Logpdf(ResidualSineGaussianBackwardMessage(xi_out, Lambda_out, meta))
    site = project(TangentProjection(type = ClosedForm), m_in, exact)
    site_parameters = getnaturalparameters(site)
    cavity_weighted_mean, cavity_precision = weightedmean_precision(m_in)
    weighted_mean = cavity_weighted_mean + site_parameters[1]
    precision = cavity_precision - 2 * site_parameters[2]
    scoring_precision = max(float(precision), sqrt(eps(Float64)))
    return NormalWeightedMeanPrecision(weighted_mean, scoring_precision)
end
