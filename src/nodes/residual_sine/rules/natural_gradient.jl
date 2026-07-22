# Natural-gradient rules for the scalar `ResidualSine` node. Forward uses the
# exact pushforward moments on the default (ClosedForm) path — an assumed-density
# update, always a proper Gaussian — or a tangent projection of the exact
# pushforward log-density under Unscented/Quadrature/DeltaApproximation.
# Backward is the exact Gaussian Fisher projection of the exact BP log-message
# on the ClosedForm path, with the generic sigma-point/quadrature projections
# available as ablations. All sites are damped through `apply_damping!`, so the
# messages entering downstream Gaussian nodes (`ManyPlus`, `softdot`) are plain
# `NormalWeightedMeanPrecision` and stay proper under damping-only (`beta = 0`)
# updates.

function DerivativeEnhancedFunction(
    p::Logpdf{<:ResidualSineForwardMessage},
    expansion_point::Real,
)
    return DerivativeEnhancedFunction(
        p,
        expansion_point,
        y -> first(_residual_sine_forward_logderivatives(p.dist, y)),
        y -> last(_residual_sine_forward_logderivatives(p.dist, y)),
    )
end

function project(
    ::TangentProjection{<:DeltaApproximation},
    q::_GaussianProjectionPoint,
    exact::Logpdf{<:ResidualSineForwardMessage},
)
    q_ef = q isa ExponentialFamilyDistribution ? q : convert(ExponentialFamilyDistribution, q)
    m, _ = _mean_var(q_ef)
    weighted_mean, message_precision = project_to_normal(
        DerivativeEnhancedFunction(exact, m), q_ef,
    )
    natural = promote(weighted_mean, -message_precision / 2)
    return ExponentialFamilyDistribution(
        NormalMeanVariance, collect(natural), nothing, nothing,
    )
end

# The exact Stein projection IS the closed form for this backward message; the
# generic ClosedForm method would look for a ClosedWilliamsProduct that does not
# exist for this expression.
function project(
    ::TangentProjection{<:ClosedForm},
    q::_GaussianProjectionPoint,
    exact::Logpdf{<:ResidualSineGaussianBackwardMessage},
)
    return _project_residual_sine_backward_1d(q, exact.dist)
end

"""
    _residual_sine_forward_site(activation, projection, m_in, q_out)

Gaussian forward site for `ResidualSine`, selected by dispatch on the
projection strategy. `ClosedForm` matches the exact pushforward moments (an
assumed-density update, independent of `q_out`); every other strategy is a
genuine `q_out`-dependent tangent projection of the exact pushforward
log-density.
"""
function _residual_sine_forward_site(
    activation::ResidualSineMeta,
    ::TangentProjection{<:ClosedForm},
    m_in::UnivariateGaussianDistributionsFamily,
    ::_GaussianProjectionPoint,
)
    transformed_mean, transformed_variance =
        _residual_sine_mean_var_1d(mean(m_in), var(m_in), activation)
    return NormalMeanVariance(transformed_mean, transformed_variance)
end

function _residual_sine_forward_site(
    activation::ResidualSineMeta,
    projection::TangentProjection,
    m_in::UnivariateGaussianDistributionsFamily,
    q_out::_GaussianProjectionPoint,
)
    exact = Logpdf(ResidualSineForwardMessage(mean(m_in), var(m_in), activation))
    return project(projection, q_out, exact)
end

@rule ResidualSine(:out, NaturalGradientMessage) (
    m_in::UnivariateGaussianDistributionsFamily,
    q_out::UnivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    activation = _residual_sine_meta(meta)
    projection = resolve_projection(getprojection(vconstraint))
    site = _residual_sine_forward_site(activation, projection, m_in, q_out)
    return NaturalGradientMP.apply_damping!(meta, site)
end

@rule ResidualSine(:in, NaturalGradientMessage) (
    m_out::UnivariateGaussianDistributionsFamily,
    q_in::UnivariateNormalDistributionsFamily,
    meta::NGMPEdgeState,
) = begin
    activation = _residual_sine_meta(meta)
    xi, Lambda = weightedmean_precision(m_out)
    exact = Logpdf(ResidualSineGaussianBackwardMessage(xi, Lambda, activation))
    site = project(resolve_projection(getprojection(vconstraint)), q_in, exact)
    return NaturalGradientMP.apply_damping!(meta, site)
end

# Plain sum-product forward: the exact pushforward moments, undamped. Used by
# graphs without NGMP dependencies (sanity checks, forward-only prediction).
@rule ResidualSine(:out, Marginalisation) (
    m_in::UnivariateGaussianDistributionsFamily,
    meta::ResidualSineMeta,
) = begin
    transformed_mean, transformed_variance =
        _residual_sine_mean_var_1d(mean(m_in), var(m_in), meta)
    return NormalMeanVariance(transformed_mean, transformed_variance)
end
