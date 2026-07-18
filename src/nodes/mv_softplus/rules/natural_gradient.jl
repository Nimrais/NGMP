import ExponentialFamily: MultivariateNormalDistributionsFamily, MvNormalMeanCovariance

# Forward (:out): moment-matched Gaussian pushforward of m_in through
# softplus, via the 2d+1 scaled-UT points of m_in (all mapped into the
# positive orthant by construction, all weights positive -> PSD covariance).
# The EXACT pushforward log-density (MvSoftplusForwardMessage) is supported
# only on the positive orthant, and nothing constrains the receiving Gaussian
# q(out) to stay inside it: as soon as one cubature point of q(out) leaves the
# orthant the exact Williams product is -Inf-contaminated (E_q[l] = -infinity),
# so the exact-log-message tangent projection is undefined there. The
# moment-matched pushforward is the projection of the GAUSSIANIZED forward
# message (projecting a quadratic log-message recovers it exactly) and is
# well-defined for any receiving marginal.
@rule MvSoftplus(:out, NaturalGradientMessage) (
    m_in::MultivariateNormalDistributionsFamily,
    q_out::MultivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    α, β, κ = UnscentedTransforms.ut_parameters(UnscentedTransform)
    m, V = mean_cov(m_in)
    x, wm, wc = UnscentedTransforms.gaussian_sigma_points(α, β, κ, m, V)
    s = map(xk -> _softplus.(xk), x)
    μ = sum(wm .* s)
    Σ = sum(wc[k] .* ((s[k] .- μ) * (s[k] .- μ)') for k in eachindex(s))
    Σ = (Σ .+ Σ') ./ 2
    site = MvNormalMeanCovariance(μ, Σ)
    return NaturalGradientMP.apply_damping!(meta, site)
end

# Backward (:in): genuine tangent projection of the EXACT backward
# log-message (finite on all of R^d - no support issue) onto the Gaussian
# input edge at the current marginal q(in).
@rule MvSoftplus(:in, NaturalGradientMessage) (
    m_out::MultivariateNormalDistributionsFamily,
    q_in::MultivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    ξ, Λ = weightedmean_precision(m_out)
    exact = Logpdf(MvSoftplusGaussianBackwardMessage(ξ, Λ))
    site = project(resolve_projection(getprojection(vconstraint)), q_in, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end
