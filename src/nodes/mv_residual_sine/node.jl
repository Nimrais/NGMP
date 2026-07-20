export MvResidualSine, ResidualSineMeta

import BayesBase: mean_cov
import ExponentialFamily: weightedmean_precision
import LinearAlgebra: Cholesky, Diagonal, I, Symmetric, cholesky, diag, dot, logdet

"""
    ResidualSineMeta(; rho = 0.9, omega = 1.0)

Configuration of the elementwise residual-sine activation

`phi(x) = x + (rho / omega) * sin(omega * x)`.

`0 < rho < 1` makes `phi` a smooth increasing bijection from `R` to `R`, with
`1 - rho <= phi'(x) <= 1 + rho`.

The forward approximation is deliberately not stored here: it is a property
of the NGMP edge constraint, not of the activation. Select either
`TangentProjection(type = DeltaApproximation)` for the touching quadratic of
the exact pushforward log-density, or `TangentProjection(type = ClosedForm)`
for analytic Gaussian moment matching. The forward rule dispatches on that
strategy type.

The moment arm is an assumed-density update, not a Fisher tangent projection.
Both arms use the exact closed-form Gaussian Fisher projection backward.
"""
struct ResidualSineMeta{T <: Real}
    rho::T
    omega::T
end

function ResidualSineMeta(;
    rho::Real = 0.9,
    omega::Real = 1.0,
)
    0 < rho < 1 || throw(ArgumentError("rho must lie strictly between zero and one"))
    omega > 0 || throw(ArgumentError("omega must be positive"))
    rho_float, omega_float = promote(float(rho), float(omega))
    return ResidualSineMeta(rho_float, omega_float)
end

"""
    MvResidualSine

Deterministic elementwise activation `out = phi.(in)` with Gaussian input and
output edges. Unlike softplus, `phi` is a diffeomorphism of the whole real line,
so the exact forward density and its local Gaussian tangent approximation are
defined at every finite output expansion point.
"""
struct MvResidualSine end

@node MvResidualSine Deterministic [out, in]

@inline _residual_sine(x::Real, meta::ResidualSineMeta) =
    x + (meta.rho / meta.omega) * sin(meta.omega * x)
@inline _residual_sine_prime(x::Real, meta::ResidualSineMeta) =
    1 + meta.rho * cos(meta.omega * x)
@inline _residual_sine_second(x::Real, meta::ResidualSineMeta) =
    -meta.rho * meta.omega * sin(meta.omega * x)
@inline _residual_sine_third(x::Real, meta::ResidualSineMeta) =
    -meta.rho * meta.omega^2 * cos(meta.omega * x)

function _residual_sine_meta(state::NGMPEdgeState)
    state.usermeta isa ResidualSineMeta || throw(ArgumentError(
        "MvResidualSine NGMP rules require ResidualSineMeta as node metadata; " *
        "got $(typeof(state.usermeta))",
    ))
    return state.usermeta
end

_residual_sine_meta(meta::ResidualSineMeta) = meta

"""
    _inverse_residual_sine(y, meta)

Invert the monotone residual-sine map with safeguarded Newton iteration. Since
`abs(phi(x) - x) <= rho / omega`, the unique root is bracketed by
`y +/- rho / omega`; bisection is used whenever a Newton step leaves it.
"""
function _inverse_residual_sine(y::Real, meta::ResidualSineMeta)
    T = promote_type(typeof(float(y)), typeof(meta.rho), typeof(meta.omega))
    target = T(y)
    radius = T(meta.rho / meta.omega)
    lower = target - radius
    upper = target + radius
    x = target
    tolerance = 16 * eps(T) * (one(T) + abs(target))

    for _ in 1:64
        residual = _residual_sine(x, meta) - target
        abs(residual) <= tolerance && return x

        if residual > zero(T)
            upper = x
        else
            lower = x
        end

        candidate = x - residual / _residual_sine_prime(x, meta)
        if !isfinite(candidate) || !(lower < candidate < upper)
            candidate = (lower + upper) / 2
        end
        x = candidate
    end

    residual = _residual_sine(x, meta) - target
    abs(residual) <= sqrt(eps(T)) * (one(T) + abs(target)) && return x
    throw(ErrorException("residual-sine inverse did not converge for y = $y"))
end

"""
Exact forward pushforward log-density for `Y = phi.(X)`, `X ~ N(mean,
covariance)`. The inverse is evaluated numerically but its gradient and Hessian
are obtained analytically from the implicit-function theorem.
"""
struct MvResidualSineForwardMessage{T <: Real, M <: ResidualSineMeta} <:
       ClosedFormExpectations.Expression
    mean::Vector{T}
    chol::Cholesky{T, Matrix{T}}
    logdet_cov::T
    activation::M
end

function MvResidualSineForwardMessage(
    mean::AbstractVector,
    covariance::AbstractMatrix,
    activation::ResidualSineMeta,
)
    m = collect(float.(mean))
    C = cholesky(Symmetric(Matrix(float.(covariance))))
    return MvResidualSineForwardMessage(m, C, logdet(C), activation)
end

function Base.log(message::MvResidualSineForwardMessage, y::AbstractVector)
    length(y) == length(message.mean) || throw(DimensionMismatch(
        "residual-sine forward point has length $(length(y)); expected $(length(message.mean))",
    ))
    x = map(yk -> _inverse_residual_sine(yk, message.activation), y)
    residual = x .- message.mean
    lognormal = -(
        length(y) * log(2pi) + message.logdet_cov +
        dot(residual, message.chol \ residual)
    ) / 2
    logjacobian = -sum(xk -> log(_residual_sine_prime(xk, message.activation)), x)
    return lognormal + logjacobian
end

(message::MvResidualSineForwardMessage)(y::AbstractVector) = exp(log(message, y))

"""Exact backward log-message from a Gaussian output cavity in information form."""
struct MvResidualSineGaussianBackwardMessage{T <: Real, M <: ResidualSineMeta} <:
       ClosedFormExpectations.Expression
    xi::Vector{T}
    Lambda::Matrix{T}
    activation::M
end

function MvResidualSineGaussianBackwardMessage(
    xi::AbstractVector,
    Lambda::AbstractMatrix,
    activation::ResidualSineMeta,
)
    xi_float = collect(float.(xi))
    return MvResidualSineGaussianBackwardMessage(
        xi_float,
        Matrix(float.(Lambda)),
        activation,
    )
end

function Base.log(message::MvResidualSineGaussianBackwardMessage, x::AbstractVector)
    transformed = map(xk -> _residual_sine(xk, message.activation), x)
    return dot(message.xi, transformed) -
           dot(transformed, message.Lambda * transformed) / 2
end

(message::MvResidualSineGaussianBackwardMessage)(x::AbstractVector) = exp(log(message, x))

function _mv_residual_sine_forward_logderivatives(
    message::MvResidualSineForwardMessage,
    y::AbstractVector,
)
    d = length(message.mean)
    length(y) == d || throw(DimensionMismatch(
        "residual-sine expansion point has length $(length(y)); expected $d",
    ))
    activation = message.activation
    x = map(yk -> _inverse_residual_sine(yk, activation), y)
    a = _residual_sine_prime.(x, Ref(activation))
    b = _residual_sine_second.(x, Ref(activation))
    c = _residual_sine_third.(x, Ref(activation))
    invcov = Matrix(message.chol \ Matrix{eltype(message.mean)}(I, d, d))
    residual_precision = invcov * (x .- message.mean)
    inverse_a = inv.(a)

    gradient = .-residual_precision .* inverse_a .- b .* inverse_a .^ 2
    hessian = -Diagonal(inverse_a) * invcov * Diagonal(inverse_a) +
        Diagonal(
            b .* residual_precision .* inverse_a .^ 3 .-
            c .* inverse_a .^ 3 .+
            2 .* b .^ 2 .* inverse_a .^ 4,
        )
    return gradient, Matrix((hessian .+ hessian') ./ 2)
end

# Joint Gaussian trigonometric moments used by both exact moment matching and
# the exact backward Fisher projection. `cos_sin[i, j]` is
# E[cos(omega X_i) sin(omega X_j)].
function _mv_residual_sine_trig_moments(
    mean::AbstractVector,
    covariance::AbstractMatrix,
    activation::ResidualSineMeta,
)
    m = collect(float.(mean))
    V = Matrix(float.(covariance))
    d = length(m)
    size(V) == (d, d) || throw(DimensionMismatch(
        "covariance has size $(size(V)); expected ($d, $d)",
    ))
    omega = activation.omega
    marginal_attenuation = exp.(-0.5 .* omega^2 .* diag(V))
    sine = marginal_attenuation .* sin.(omega .* m)
    cosine = marginal_attenuation .* cos.(omega .* m)
    sine_sine = zeros(promote_type(eltype(m), eltype(V)), d, d)
    cosine_cosine = similar(sine_sine)
    cosine_sine = similar(sine_sine)

    @inbounds for i in 1:d, j in 1:d
        variance_minus = max(V[i, i] + V[j, j] - 2 * V[i, j], zero(eltype(V)))
        variance_plus = max(V[i, i] + V[j, j] + 2 * V[i, j], zero(eltype(V)))
        attenuation_minus = exp(-0.5 * omega^2 * variance_minus)
        attenuation_plus = exp(-0.5 * omega^2 * variance_plus)
        angle_minus = omega * (m[i] - m[j])
        angle_plus = omega * (m[i] + m[j])

        expected_cos_minus = attenuation_minus * cos(angle_minus)
        expected_cos_plus = attenuation_plus * cos(angle_plus)
        expected_sin_plus = attenuation_plus * sin(angle_plus)
        expected_sin_j_minus_i = attenuation_minus * sin(-angle_minus)

        sine_sine[i, j] = (expected_cos_minus - expected_cos_plus) / 2
        cosine_cosine[i, j] = (expected_cos_minus + expected_cos_plus) / 2
        cosine_sine[i, j] = (expected_sin_plus + expected_sin_j_minus_i) / 2
    end

    return (; sine, cosine, sine_sine, cosine_cosine, cosine_sine)
end

"""Exact mean and covariance of the residual-sine transform of a Gaussian."""
function _mv_residual_sine_mean_cov(
    mean::AbstractVector,
    covariance::AbstractMatrix,
    activation::ResidualSineMeta,
)
    m = collect(float.(mean))
    V = Matrix(float.(covariance))
    moments = _mv_residual_sine_trig_moments(m, V, activation)
    rho = activation.rho
    scale = rho / activation.omega
    transformed_mean = m .+ scale .* moments.sine
    d = length(m)
    transformed_covariance = similar(V)
    @inbounds for i in 1:d, j in 1:d
        transformed_covariance[i, j] = V[i, j] +
            rho * V[i, j] * (moments.cosine[i] + moments.cosine[j]) +
            scale^2 * (moments.sine_sine[i, j] - moments.sine[i] * moments.sine[j])
    end
    transformed_covariance = Matrix((transformed_covariance .+ transformed_covariance') ./ 2)
    return transformed_mean, transformed_covariance
end

"""
Exact Gaussian Fisher tangent projection of the residual-sine backward
log-message. Gaussian Stein identities reduce the projection to E[grad ell]
and E[Hessian ell]; all required sine/cosine moments are analytic.
"""
function _project_mv_residual_sine_backward(
    q::_MvGaussianProjectionPoint,
    message::MvResidualSineGaussianBackwardMessage,
)
    m, V = _mv_mean_cov(q)
    m = collect(float.(m))
    V = Matrix(float.(V))
    moments = _mv_residual_sine_trig_moments(m, V, message.activation)
    rho = message.activation.rho
    omega = message.activation.omega
    scale = rho / omega
    xi = message.xi
    Lambda = Matrix((message.Lambda .+ message.Lambda') ./ 2)
    d = length(m)

    expected_d = 1 .+ rho .* moments.cosine
    expected_e = .-rho .* omega .* moments.sine
    expected_d_phi = zeros(promote_type(eltype(m), eltype(V), eltype(xi)), d, d)
    expected_e_phi = similar(expected_d_phi)
    expected_dd = similar(expected_d_phi)

    @inbounds for i in 1:d, j in 1:d
        expected_d_phi[i, j] = m[j] + scale * moments.sine[j] +
            rho * (m[j] * moments.cosine[i] - omega * V[j, i] * moments.sine[i]) +
            rho * scale * moments.cosine_sine[i, j]
        expected_e_phi[i, j] = -rho * omega * (
            m[j] * moments.sine[i] + omega * V[j, i] * moments.cosine[i] +
            scale * moments.sine_sine[i, j]
        )
        expected_dd[i, j] = 1 + rho * (moments.cosine[i] + moments.cosine[j]) +
            rho^2 * moments.cosine_cosine[i, j]
    end

    expected_gradient = similar(m)
    expected_hessian = -Lambda .* expected_dd
    @inbounds for i in 1:d
        expected_gradient[i] = expected_d[i] * xi[i] -
            sum(Lambda[i, j] * expected_d_phi[i, j] for j in 1:d)
        expected_hessian[i, i] += expected_e[i] * xi[i] -
            sum(Lambda[i, j] * expected_e_phi[i, j] for j in 1:d)
    end

    expected_hessian = Matrix((expected_hessian .+ expected_hessian') ./ 2)
    site_precision = -expected_hessian
    site_weighted_mean = expected_gradient .+ site_precision * m
    return ExponentialFamilyDistribution(
        MvNormalMeanCovariance,
        vcat(site_weighted_mean, vec(-site_precision ./ 2)),
        nothing,
        nothing,
    )
end
