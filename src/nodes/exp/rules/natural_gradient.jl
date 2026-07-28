# Natural-gradient rules for ProbabilisticEnsembling's smooth positive Exp node.
# They are the inverse orientation of the Log-node rules: a Gaussian input pushes
# forward to LogNormal on the Gamma output, while a Gamma output pulls back to
# LogGamma on the Gaussian input.

import ProbabilisticEnsembling: Exp

"""
Unnormalised message obtained by pulling a Gamma-family SITE through
`out = exp(in)`.

If the message on `out` has natural parameters

    log μ(out) = ηlog * log(out) - rate * out + const,

then the deterministic factor sends

    log μ(in) = ηlog * in - rate * exp(in) + const.

This is deliberately not represented as a `LogGamma` distribution. A factor
message is a site and may have either natural parameter outside the proper
Gamma domain; moreover, a deterministic pullback has no change-of-variables
Jacobian, so its linear coefficient is `shape - 1`, not `shape`.
"""
struct ExpGammaSiteMessage{T<:Real} <: ClosedFormExpectations.Expression
    log_coefficient::T
    rate::T
end

function ExpGammaSiteMessage(log_coefficient::Real, rate_parameter::Real)
    coefficient, rate_value = promote(log_coefficient, rate_parameter)
    return ExpGammaSiteMessage{typeof(coefficient)}(coefficient, rate_value)
end

function Base.log(message::ExpGammaSiteMessage, input::Real)
    return message.log_coefficient * input - message.rate * exp(input)
end

function (message::ExpGammaSiteMessage)(input::Real)
    return exp(log(message, input))
end

# Exact Williams product under a Gaussian projection point. For
# ℓ(x) = c*x - b*exp(x) and q = Normal(m, σ²),
#
#   ∂m E[ℓ] = c - b*exp(m + σ²/2)
#   ∂σ E[ℓ] = -σ*b*exp(m + σ²/2).
function ClosedFormExpectations.mean(
    ::ClosedWilliamsProduct,
    objective::Logpdf{<:ExpGammaSiteMessage},
    q::Distributions.Normal,
)
    message = objective.dist
    exponential_term = message.rate * exp(mean(q) + abs2(std(q)) / 2)
    return (
        message.log_coefficient - exponential_term,
        -std(q) * exponential_term,
    )
end

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
    exact = Logpdf(ExpGammaSiteMessage(shape(m_out) - 1, rate(m_out)))
    site = project(resolve_projection(getprojection(vconstraint)), q_in, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end
