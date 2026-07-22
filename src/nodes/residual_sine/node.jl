export ResidualSine

"""
    ResidualSine

Scalar deterministic activation `out = phi(in)`,
`phi(x) = x + (rho / omega) * sin(omega * x)`, with univariate Gaussian edges
on both sides — the 1D counterpart of `MvResidualSine`, configured by the same
`ResidualSineMeta`. Because `phi` is a diffeomorphism of the real line, both
the exact pushforward moments and the exact backward Gaussian Fisher
projection are closed-form; no sigma points are required on the default path.
"""
struct ResidualSine end

@node ResidualSine Deterministic [out, in]

# Scalar trigonometric moments under N(m, v): the d = 1 case of
# `_mv_residual_sine_trig_moments`.
function _residual_sine_trig_moments_1d(m::Real, v::Real, activation::ResidualSineMeta)
    omega = activation.omega
    attenuation = exp(-omega^2 * v / 2)
    attenuation2 = exp(-2 * omega^2 * v)
    sine = attenuation * sin(omega * m)
    cosine = attenuation * cos(omega * m)
    sine_sine = (1 - attenuation2 * cos(2 * omega * m)) / 2      # E[sin²(ωx)]
    cosine_cosine = (1 + attenuation2 * cos(2 * omega * m)) / 2  # E[cos²(ωx)]
    cosine_sine = attenuation2 * sin(2 * omega * m) / 2          # E[cos(ωx) sin(ωx)]
    return (; sine, cosine, sine_sine, cosine_cosine, cosine_sine)
end

"""Exact mean and variance of `phi(X)`, `X ~ N(m, v)`."""
function _residual_sine_mean_var_1d(m::Real, v::Real, activation::ResidualSineMeta)
    mf, vf = promote(float(m), float(v))
    moments = _residual_sine_trig_moments_1d(mf, vf, activation)
    rho = activation.rho
    scale = rho / activation.omega
    transformed_mean = mf + scale * moments.sine
    transformed_variance = vf + 2 * rho * vf * moments.cosine +
        scale^2 * (moments.sine_sine - moments.sine^2)
    return transformed_mean, transformed_variance
end

"""
Exact forward pushforward log-density for `Y = phi(X)`, `X ~ N(mean, variance)`.
The inverse is evaluated with the safeguarded Newton iteration shared with the
multivariate node; the Jacobian is analytic.
"""
struct ResidualSineForwardMessage{T <: Real, M <: ResidualSineMeta} <:
       ClosedFormExpectations.Expression
    mean::T
    variance::T
    activation::M
end

function ResidualSineForwardMessage(mean::Real, variance::Real, activation::ResidualSineMeta)
    m, v = promote(float(mean), float(variance))
    return ResidualSineForwardMessage(m, v, activation)
end

function Base.log(message::ResidualSineForwardMessage, y::Real)
    x = _inverse_residual_sine(y, message.activation)
    lognormal = -(log(2π * message.variance) + (x - message.mean)^2 / message.variance) / 2
    logjacobian = -log(_residual_sine_prime(x, message.activation))
    return lognormal + logjacobian
end

(message::ResidualSineForwardMessage)(y::Real) = exp(log(message, y))

# Analytic derivatives of the exact forward log-message via the implicit
# function theorem — the d = 1 case of `_mv_residual_sine_forward_logderivatives`.
function _residual_sine_forward_logderivatives(
    message::ResidualSineForwardMessage,
    y::Real,
)
    activation = message.activation
    x = _inverse_residual_sine(y, activation)
    a = _residual_sine_prime(x, activation)
    b = _residual_sine_second(x, activation)
    c = _residual_sine_third(x, activation)
    residual_precision = (x - message.mean) / message.variance
    inverse_a = inv(a)
    gradient = -residual_precision * inverse_a - b * inverse_a^2
    hessian = -inverse_a^2 / message.variance +
        b * residual_precision * inverse_a^3 -
        c * inverse_a^3 +
        2 * b^2 * inverse_a^4
    return gradient, hessian
end

"""Exact backward log-message from a univariate Gaussian output cavity in
information form: `l(x) = xi * phi(x) - Lambda * phi(x)^2 / 2`."""
struct ResidualSineGaussianBackwardMessage{T <: Real, M <: ResidualSineMeta} <:
       ClosedFormExpectations.Expression
    xi::T
    Lambda::T
    activation::M
end

function ResidualSineGaussianBackwardMessage(
    xi::Real,
    Lambda::Real,
    activation::ResidualSineMeta,
)
    xi_float, Lambda_float = promote(float(xi), float(Lambda))
    return ResidualSineGaussianBackwardMessage(xi_float, Lambda_float, activation)
end

function Base.log(message::ResidualSineGaussianBackwardMessage, x::Real)
    transformed = _residual_sine(x, message.activation)
    return message.xi * transformed - message.Lambda * transformed^2 / 2
end

(message::ResidualSineGaussianBackwardMessage)(x::Real) = exp(log(message, x))

"""
Exact Gaussian Fisher tangent projection of the scalar residual-sine backward
log-message at `q = N(m, v)` — the d = 1 case of
`_project_mv_residual_sine_backward`, using the Gaussian Stein identities
`E[grad l]` and `E[Hessian l]` with analytic trigonometric moments.
"""
function _project_residual_sine_backward_1d(
    q::_GaussianProjectionPoint,
    message::ResidualSineGaussianBackwardMessage,
)
    normal_q = convert(Distributions.Normal, q)
    m = Distributions.mean(normal_q)
    v = Distributions.var(normal_q)
    activation = message.activation
    moments = _residual_sine_trig_moments_1d(m, v, activation)
    rho = activation.rho
    omega = activation.omega
    scale = rho / omega
    xi = message.xi
    Lambda = message.Lambda

    expected_d = 1 + rho * moments.cosine                                # E[phi']
    expected_e = -rho * omega * moments.sine                             # E[phi'']
    expected_d_phi = m + scale * moments.sine +
        rho * (m * moments.cosine - omega * v * moments.sine) +
        rho * scale * moments.cosine_sine                                # E[phi' phi]
    expected_e_phi = -rho * omega * (
        m * moments.sine + omega * v * moments.cosine +
        scale * moments.sine_sine
    )                                                                    # E[phi'' phi]
    expected_dd = 1 + 2 * rho * moments.cosine +
        rho^2 * moments.cosine_cosine                                    # E[(phi')²]

    expected_gradient = expected_d * xi - Lambda * expected_d_phi
    expected_hessian = -Lambda * expected_dd + expected_e * xi - Lambda * expected_e_phi
    site_precision = -expected_hessian
    site_weighted_mean = expected_gradient + site_precision * m
    natural = promote(site_weighted_mean, -site_precision / 2)
    return ExponentialFamilyDistribution(
        NormalMeanVariance, collect(natural), nothing, nothing,
    )
end
