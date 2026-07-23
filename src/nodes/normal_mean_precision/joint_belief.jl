# Score-only local belief for a fully structured NormalMeanPrecision factor.
# Outbound messages continue to use the rules in rules/natural_gradient.jl.

const _NORMAL_MEAN_PRECISION_JOINT_QUADRATURE_ORDER = 64
const _NORMAL_MEAN_PRECISION_JOINT_LOG_RADIUS = 12.0
const _NORMAL_MEAN_PRECISION_JOINT_LOG_NODES = range(
    -_NORMAL_MEAN_PRECISION_JOINT_LOG_RADIUS,
    _NORMAL_MEAN_PRECISION_JOINT_LOG_RADIUS;
    length = _NORMAL_MEAN_PRECISION_JOINT_QUADRATURE_ORDER,
)
const _NORMAL_MEAN_PRECISION_JOINT_LOG_STEP =
    2 * _NORMAL_MEAN_PRECISION_JOINT_LOG_RADIUS /
    (_NORMAL_MEAN_PRECISION_JOINT_QUADRATURE_ORDER - 1)
const _NORMAL_MEAN_PRECISION_LOG2PI = log(2π)
const _NORMAL_MEAN_PRECISION_ENDPOINT_LOGWEIGHT = -log(2.0)

struct NormalMeanPrecisionJointStatistics{T <: AbstractFloat}
    log_normalizer::T
    average_energy::T
    entropy::T
end

"""
    NormalMeanPrecisionJointBelief

Normalized local belief induced by two Gaussian cavity messages, one Gamma
cavity message, and a `NormalMeanPrecision` factor. It is used only by RxInfer's
free-energy scorer. Expensive statistics are evaluated lazily and cached for the
lifetime of the local belief.
"""
mutable struct NormalMeanPrecisionJointBelief{T <: AbstractFloat}
    const out_mean::T
    const out_variance::T
    const μ_mean::T
    const μ_variance::T
    const τ_shape::T
    const τ_rate::T
    statistics::Union{Nothing, NormalMeanPrecisionJointStatistics{T}}
end

function NormalMeanPrecisionJointBelief(
    m_out::UnivariateNormalDistributionsFamily,
    m_μ::UnivariateNormalDistributionsFamily,
    m_τ::GammaDistributionsFamily,
)
    out_mean, out_variance = mean_var(m_out)
    μ_mean, μ_variance = mean_var(m_μ)
    τ_shape, τ_rate = shape(m_τ), rate(m_τ)
    parameters = promote(
        float(out_mean),
        float(out_variance),
        float(μ_mean),
        float(μ_variance),
        float(τ_shape),
        float(τ_rate),
    )
    T = eltype(parameters)
    return NormalMeanPrecisionJointBelief{T}(parameters..., nothing)
end

@inline function _normal_mean_precision_logistic_log1pexp(x)
    if x >= zero(x)
        expmx = exp(-x)
        return inv(one(x) + expmx), x + log1p(expmx)
    end
    expx = exp(x)
    return expx / (one(x) + expx), log1p(expx)
end

function _compute_normal_mean_precision_joint_statistics(
    belief::NormalMeanPrecisionJointBelief{T},
) where {T}
    m_out = belief.out_mean
    v_out = belief.out_variance
    m_μ = belief.μ_mean
    v_μ = belief.μ_variance
    a = belief.τ_shape
    b = belief.τ_rate

    if !(
        isfinite(m_out) &&
        isfinite(m_μ) &&
        isfinite(v_out) &&
        isfinite(v_μ) &&
        isfinite(a) &&
        isfinite(b) &&
        v_out > zero(T) &&
        v_μ > zero(T) &&
        a > zero(T) &&
        b > zero(T)
    )
        throw(DomainError(
            (m_out, v_out, m_μ, v_μ, a, b),
            "NormalMeanPrecision joint scoring requires finite means and positive finite variances, shape, and rate",
        ))
    end

    V = v_out + v_μ
    residual = m_out - m_μ
    residual_scale = abs2(residual) / V
    logV = log(V)
    logb = log(b)
    log2π = T(_NORMAL_MEAN_PRECISION_LOG2PI)
    half = inv(T(2))
    oneT = one(T)

    log_gamma_constant = a * logb - T(loggamma(a))
    logτ_mean = T(SpecialFunctions.digamma(a)) - logb
    logτ_std = sqrt(T(SpecialFunctions.trigamma(a)))
    logτ_step = T(_NORMAL_MEAN_PRECISION_JOINT_LOG_STEP) * logτ_std
    endpoint_logweight = T(_NORMAL_MEAN_PRECISION_ENDPOINT_LOGWEIGHT)

    maximum_logweight = T(-Inf)
    weight_sum = zero(T)
    energy_sum = zero(T)
    log_gamma_sum = zero(T)
    log_likelihood_sum = zero(T)
    log_precision_denominator_sum = zero(T)

    for index in eachindex(_NORMAL_MEAN_PRECISION_JOINT_LOG_NODES)
        standardized_logτ = T(_NORMAL_MEAN_PRECISION_JOINT_LOG_NODES[index])
        logτ = logτ_mean + logτ_std * standardized_logτ
        logVτ = logV + logτ
        precision_fraction, log_precision_denominator =
            _normal_mean_precision_logistic_log1pexp(logVτ)
        log_conditional_variance = log_precision_denominator - logτ

        rate_term = exp(logb + logτ)
        log_gamma_density =
            log_gamma_constant + (a - oneT) * logτ - rate_term
        log_likelihood = -half * (
            log2π + log_conditional_variance +
            residual_scale * precision_fraction
        )
        logweight = log_gamma_density + logτ + log_likelihood
        if index == firstindex(_NORMAL_MEAN_PRECISION_JOINT_LOG_NODES) ||
           index == lastindex(_NORMAL_MEAN_PRECISION_JOINT_LOG_NODES)
            logweight += endpoint_logweight
        end

        if !isfinite(logweight)
            logweight == T(Inf) && error(
                "NormalMeanPrecision joint quadrature produced an infinite weight",
            )
            continue
        end

        conditional_squared_error =
            precision_fraction +
            residual_scale * precision_fraction * (oneT - precision_fraction)
        energy = half * (log2π - logτ + conditional_squared_error)

        if logweight > maximum_logweight
            rescale = isfinite(maximum_logweight) ?
                      exp(maximum_logweight - logweight) : zero(T)
            weight_sum = weight_sum * rescale + oneT
            energy_sum = energy_sum * rescale + energy
            log_gamma_sum = log_gamma_sum * rescale + log_gamma_density
            log_likelihood_sum =
                log_likelihood_sum * rescale + log_likelihood
            log_precision_denominator_sum =
                log_precision_denominator_sum * rescale +
                log_precision_denominator
            maximum_logweight = logweight
        else
            weight = exp(logweight - maximum_logweight)
            weight_sum += weight
            energy_sum += weight * energy
            log_gamma_sum += weight * log_gamma_density
            log_likelihood_sum += weight * log_likelihood
            log_precision_denominator_sum +=
                weight * log_precision_denominator
        end
    end

    if !(isfinite(weight_sum) && weight_sum > zero(T))
        error("NormalMeanPrecision joint quadrature has no finite mass")
    end

    inverse_weight_sum = inv(weight_sum)
    log_normalizer = maximum_logweight + log(weight_sum) + log(logτ_step)
    average_energy = energy_sum * inverse_weight_sum
    expected_log_gamma = log_gamma_sum * inverse_weight_sum
    expected_log_likelihood = log_likelihood_sum * inverse_weight_sum
    expected_log_precision_denominator =
        log_precision_denominator_sum * inverse_weight_sum

    τ_entropy =
        log_normalizer - expected_log_gamma - expected_log_likelihood
    center_entropy = half * (
        log2π + oneT + log(v_out) + log(v_μ) - logV
    )
    conditional_difference_entropy = half * (
        log2π + oneT + logV - expected_log_precision_denominator
    )
    joint_entropy = center_entropy + τ_entropy + conditional_difference_entropy

    return NormalMeanPrecisionJointStatistics{T}(
        log_normalizer,
        average_energy,
        joint_entropy,
    )
end

function _normal_mean_precision_joint_statistics!(
    belief::NormalMeanPrecisionJointBelief,
)
    statistics = belief.statistics
    if statistics === nothing
        computed = _compute_normal_mean_precision_joint_statistics(belief)
        belief.statistics = computed
        return computed
    end
    return statistics
end

BayesBase.entropy(belief::NormalMeanPrecisionJointBelief) =
    _normal_mean_precision_joint_statistics!(belief).entropy

@marginalrule NormalMeanPrecision(:out_μ_τ) (
    m_out::UnivariateNormalDistributionsFamily,
    m_μ::UnivariateNormalDistributionsFamily,
    m_τ::GammaDistributionsFamily,
    meta::Any,
) = begin
    return NormalMeanPrecisionJointBelief(m_out, m_μ, m_τ)
end

@average_energy NormalMeanPrecision (
    q_out_μ_τ::NormalMeanPrecisionJointBelief,
    meta::Any,
) = begin
    return _normal_mean_precision_joint_statistics!(q_out_μ_τ).average_energy
end

# Local belief for the dynamic-ensemble consensus factor
#
#     out ~ NormalMeanPrecision(fixed_mean, τ)
#
# when `out` and `τ` belong to the same variational cluster.  Conditional on τ,
# the product between the Gaussian cavity on `out` and the factor is Gaussian,
# so only a scalar integral over log(τ) remains.  This is deliberately separate
# from `NormalMeanPrecisionJointBelief`: the fixed mean has no entropy and the
# joint belief is two-dimensional rather than three-dimensional.

struct FixedMeanNormalPrecisionJointStatistics{T <: AbstractFloat}
    log_normalizer::T
    average_energy::T
    entropy::T
end

mutable struct FixedMeanNormalPrecisionJointBelief{T <: AbstractFloat}
    const out_mean::T
    const out_variance::T
    const fixed_mean::T
    const τ_shape::T
    const τ_rate::T
    statistics::Union{Nothing, FixedMeanNormalPrecisionJointStatistics{T}}
end

function FixedMeanNormalPrecisionJointBelief(
    m_out::UnivariateNormalDistributionsFamily,
    m_μ::PointMass,
    m_τ::GammaDistributionsFamily,
)
    out_mean, out_variance = mean_var(m_out)
    parameters = promote(
        float(out_mean),
        float(out_variance),
        float(mean(m_μ)),
        float(shape(m_τ)),
        float(rate(m_τ)),
    )
    T = eltype(parameters)
    return FixedMeanNormalPrecisionJointBelief{T}(parameters..., nothing)
end

function _compute_fixed_mean_normal_precision_joint_statistics(
    belief::FixedMeanNormalPrecisionJointBelief{T},
) where {T}
    m_out = belief.out_mean
    v_out = belief.out_variance
    fixed_mean = belief.fixed_mean
    a = belief.τ_shape
    b = belief.τ_rate

    if !(
        isfinite(m_out) &&
        isfinite(v_out) &&
        isfinite(fixed_mean) &&
        isfinite(a) &&
        isfinite(b) &&
        v_out > zero(T) &&
        a > zero(T) &&
        b > zero(T)
    )
        throw(DomainError(
            (m_out, v_out, fixed_mean, a, b),
            "fixed-mean NormalMeanPrecision joint scoring requires finite " *
            "means and positive finite variances, shape, and rate",
        ))
    end

    residual = m_out - fixed_mean
    logb = log(b)
    log2π = T(_NORMAL_MEAN_PRECISION_LOG2PI)
    half = inv(T(2))
    oneT = one(T)
    log_gamma_constant = a * logb - T(loggamma(a))
    logτ_mean = T(SpecialFunctions.digamma(a)) - logb
    logτ_std = sqrt(T(SpecialFunctions.trigamma(a)))
    logτ_step = T(_NORMAL_MEAN_PRECISION_JOINT_LOG_STEP) * logτ_std
    endpoint_logweight = T(_NORMAL_MEAN_PRECISION_ENDPOINT_LOGWEIGHT)

    maximum_logweight = T(-Inf)
    weight_sum = zero(T)
    energy_sum = zero(T)
    log_gamma_overlap_sum = zero(T)
    conditional_entropy_sum = zero(T)

    for index in eachindex(_NORMAL_MEAN_PRECISION_JOINT_LOG_NODES)
        standardized_logτ = T(_NORMAL_MEAN_PRECISION_JOINT_LOG_NODES[index])
        logτ = logτ_mean + logτ_std * standardized_logτ
        τ = exp(logτ)
        vτ = v_out + inv(τ)
        log_overlap = -half * (log2π + log(vτ) + abs2(residual) / vτ)
        log_gamma_density =
            log_gamma_constant + (a - oneT) * logτ - exp(logb + logτ)
        logweight = log_gamma_density + log_overlap + logτ
        if index == firstindex(_NORMAL_MEAN_PRECISION_JOINT_LOG_NODES) ||
           index == lastindex(_NORMAL_MEAN_PRECISION_JOINT_LOG_NODES)
            logweight += endpoint_logweight
        end

        if !isfinite(logweight)
            logweight == T(Inf) && error(
                "fixed-mean NormalMeanPrecision quadrature produced an infinite weight",
            )
            continue
        end

        conditional_variance = inv(inv(v_out) + τ)
        conditional_mean = conditional_variance * (
            m_out / v_out + τ * fixed_mean
        )
        conditional_squared_error =
            conditional_variance + abs2(conditional_mean - fixed_mean)
        energy = half * (
            log2π - logτ + τ * conditional_squared_error
        )
        conditional_entropy = half * (
            log2π + oneT + log(conditional_variance)
        )
        log_gamma_overlap = log_gamma_density + log_overlap

        if logweight > maximum_logweight
            rescale = isfinite(maximum_logweight) ?
                      exp(maximum_logweight - logweight) : zero(T)
            weight_sum = weight_sum * rescale + oneT
            energy_sum = energy_sum * rescale + energy
            log_gamma_overlap_sum =
                log_gamma_overlap_sum * rescale + log_gamma_overlap
            conditional_entropy_sum =
                conditional_entropy_sum * rescale + conditional_entropy
            maximum_logweight = logweight
        else
            weight = exp(logweight - maximum_logweight)
            weight_sum += weight
            energy_sum += weight * energy
            log_gamma_overlap_sum += weight * log_gamma_overlap
            conditional_entropy_sum += weight * conditional_entropy
        end
    end

    if !(isfinite(weight_sum) && weight_sum > zero(T))
        error("fixed-mean NormalMeanPrecision joint quadrature has no finite mass")
    end

    inverse_weight_sum = inv(weight_sum)
    log_normalizer = maximum_logweight + log(weight_sum) + log(logτ_step)
    average_energy = energy_sum * inverse_weight_sum
    τ_entropy =
        log_normalizer - log_gamma_overlap_sum * inverse_weight_sum
    joint_entropy =
        τ_entropy + conditional_entropy_sum * inverse_weight_sum

    return FixedMeanNormalPrecisionJointStatistics{T}(
        log_normalizer,
        average_energy,
        joint_entropy,
    )
end

function _fixed_mean_normal_precision_joint_statistics!(
    belief::FixedMeanNormalPrecisionJointBelief,
)
    statistics = belief.statistics
    if statistics === nothing
        computed =
            _compute_fixed_mean_normal_precision_joint_statistics(belief)
        belief.statistics = computed
        return computed
    end
    return statistics
end

BayesBase.entropy(belief::FixedMeanNormalPrecisionJointBelief) =
    _fixed_mean_normal_precision_joint_statistics!(belief).entropy

@marginalrule NormalMeanPrecision(:out_τ) (
    m_out::UnivariateNormalDistributionsFamily,
    m_μ::PointMass,
    m_τ::GammaDistributionsFamily,
    meta::Any,
) = begin
    return FixedMeanNormalPrecisionJointBelief(m_out, m_μ, m_τ)
end

# A data-valued mean lies outside q(out, τ), so GraphPPL exposes it as a
# PointMass marginal after the two within-cluster cavity messages.
@marginalrule NormalMeanPrecision(:out_τ) (
    m_out::UnivariateNormalDistributionsFamily,
    m_τ::GammaDistributionsFamily,
    q_μ::PointMass,
    meta::Any,
) = begin
    return FixedMeanNormalPrecisionJointBelief(m_out, q_μ, m_τ)
end

# Symmetric observed-output specialization.  In a regression likelihood
#
#     observed_y ~ NormalMeanPrecision(μ, τ),
#
# a structured q(μ, τ) cluster has exactly the same local density as the
# fixed-mean q(out, τ) consensus case above after swapping the two Gaussian
# location interfaces.  Reuse the same one-dimensional log-precision
# quadrature rather than introducing a second belief implementation.
@marginalrule NormalMeanPrecision(:μ_τ) (
    m_μ::UnivariateNormalDistributionsFamily,
    m_τ::GammaDistributionsFamily,
    q_out::PointMass,
    meta::Any,
) = begin
    return FixedMeanNormalPrecisionJointBelief(m_μ, q_out, m_τ)
end

@average_energy NormalMeanPrecision (
    q_out_τ::FixedMeanNormalPrecisionJointBelief,
    q_μ::PointMass,
    meta::Any,
) = begin
    return _fixed_mean_normal_precision_joint_statistics!(q_out_τ).average_energy
end

@average_energy NormalMeanPrecision (
    q_out::PointMass,
    q_μ_τ::FixedMeanNormalPrecisionJointBelief,
    meta::Any,
) = begin
    return _fixed_mean_normal_precision_joint_statistics!(q_μ_τ).average_energy
end
