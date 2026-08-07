# Natural-gradient rule toward the log-precision vector s of
# MvNormalExpPrecision, plus its EXACT closed-form tangent projection.
#
# The exact mean-field site is log μ(s) = Σⱼ (cⱼsⱼ − bⱼe^{sⱼ}) — independent
# ExpGamma-shaped coordinates — so the multivariate Williams product decomposes
# into the scalar identity of `ExpGammaSiteMessage` per coordinate: with
# q(sⱼ) = N(mⱼ, σⱼ²) and ρⱼ = e^{mⱼ + σⱼ²/2},
#
#     ∂mⱼ E[ℓ] = cⱼ − bⱼρⱼ,      ∂σⱼ E[ℓ] = −σⱼ·bⱼρⱼ,
#
# mapped through the Gaussian Fisher (Λ = −∂σ/σ, ξ = ∂m + mΛ) coordinate-wise:
#
#     Λⱼ = bⱼρⱼ   (> 0 always — the site is proper by construction),
#     ξⱼ = cⱼ + (mⱼ − 1)·Λⱼ.
#
# Only diag(V) of q(s) is read; the site is a diagonal-precision Gaussian that
# adds onto a dense prior/marginal without restriction.

import ExponentialFamily: ExponentialFamilyDistribution
import LinearAlgebra: vec

function project(
    ::TangentProjection{<:ClosedForm},
    q::_MvGaussianProjectionPoint,
    f::Logpdf{<:MvExpGammaSiteMessage},
)
    m, V = _mv_mean_cov(q)
    message = f.dist
    length(m) == length(message.rates) || throw(DimensionMismatch(
        "MvExpGammaSiteMessage has $(length(message.rates)) coordinates; " *
        "the receiving Gaussian has $(length(m))",
    ))
    Λ = message.rates .* _mnep_rho(m, diag(V))
    ξ = message.log_coefficients .+ (m .- 1) .* Λ
    return ExponentialFamilyDistribution(
        MvNormalMeanCovariance,
        vcat(ξ, vec(Matrix(Diagonal(-Λ ./ 2)))),
        nothing,
        nothing,
    )
end

# Marginal dependencies arrive in interface order (out, μ), with the receiving
# edge's own marginal q_s injected last by NGMPDependencies. `q_out::Any` /
# `q_μ::Any` cover both PointMass data and latent Gaussian marginals — the
# expected square residual handles either through _mnep_mean_vardiag.
@rule MvNormalExpPrecision(:s, NaturalGradientMessage) (
    q_out::Any,
    q_μ::Any,
    q_s::MultivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    E = _mnep_expected_square_residuals(q_out, q_μ)
    exact = Logpdf(MvExpGammaSiteMessage(fill(one(eltype(E)) / 2, length(E)), E ./ 2))
    site = project(resolve_projection(getprojection(vconstraint)), q_s, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end
