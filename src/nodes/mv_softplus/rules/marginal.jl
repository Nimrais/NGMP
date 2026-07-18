import ExponentialFamily: MvNormalWeightedMeanPrecision
import LinearAlgebra: Diagonal, eigen

# ReactiveMP scores unary deterministic nodes as -H(q_in). Rebuild the local
# multivariate Gaussian input marginal from both cavities: q(in) = cavity ×
# UT-projected site, the Mv analogue of `@marginalrule Softplus(:in)`.
@marginalrule MvSoftplus(:in) (
    m_out::MultivariateNormalDistributionsFamily,
    m_in::MultivariateNormalDistributionsFamily,
    meta::Any,
) = begin
    ξ_out, Λ_out = weightedmean_precision(m_out)
    exact = Logpdf(MvSoftplusGaussianBackwardMessage(ξ_out, Λ_out))
    site = project(TangentProjection(type = Unscented), m_in, exact)
    η = getnaturalparameters(site)
    d = length(mean(m_in))
    ξ_site = η[1:d]
    Λ_site = -2 .* reshape(η[(d + 1):end], d, d)
    ξ_cavity, Λ_cavity = weightedmean_precision(m_in)
    ξ_q = ξ_cavity .+ ξ_site
    Λ_q = Symmetric(Matrix(Λ_cavity) .+ Λ_site)

    # This marginal is used only by the deterministic free-energy scorer. A
    # locally non-concave projected site can be improper while the variable
    # belief remains valid; floor the eigenvalues to keep -H(q_in) finite
    # (multivariate analogue of the scalar `max(precision, sqrt(eps))`).
    F = eigen(Matrix(Λ_q))
    Λ_scoring = F.vectors * Diagonal(max.(F.values, sqrt(eps(Float64)))) * F.vectors'
    return MvNormalWeightedMeanPrecision(ξ_q, Λ_scoring)
end
