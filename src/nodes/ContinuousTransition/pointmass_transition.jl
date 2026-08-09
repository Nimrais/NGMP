# Exact ContinuousTransition rules for a FIXED (PointMass) transition matrix —
# the design-matrix case y = A x + N(0, W⁻¹) with A = reshape(a, dy, dx) data
# (e.g. a block of feature rows mapping shared weights x to a vector of latent
# per-observation means). The stock LinearReshapeMeta rules dispatch on a
# Gaussian q_a and would carry a dense all-zero parameter covariance through
# mean_cov(q_a); these twins are the exact Va = 0 specializations.

@rule ContinuousTransition(:y, Marginalisation) (
    m_x::MultivariateNormalDistributionsFamily,
    q_a::PointMass,
    q_W::Any,
    meta::LinearReshapeMeta,
) = begin
    A = _linear_reshape_matrix(meta, mean(q_a))
    mx, Vx = mean_cov(m_x)
    return MvNormalMeanCovariance(A * mx, A * Vx * A' + ReactiveMP.cholinv(mean(q_W)))
end

@rule ContinuousTransition(:x, Marginalisation) (
    m_y::MultivariateNormalDistributionsFamily,
    q_a::PointMass,
    q_W::Any,
    meta::LinearReshapeMeta,
) = begin
    A = _linear_reshape_matrix(meta, mean(q_a))
    my, Wy = mean_precision(m_y)
    mW = mean(q_W)
    WymW = Wy - Wy * ReactiveMP.cholinv(Wy + mW) * Wy
    return MvNormalWeightedMeanPrecision(A' * (WymW * my), A' * WymW * A)
end

# mean-field toward x — the stock rule materializes mean_cov(q_a), which for a
# PointMass design matrix would be a huge all-zero parameter covariance
@rule ContinuousTransition(:x, Marginalisation) (
    q_y::Any,
    q_a::PointMass,
    q_W::Any,
    meta::LinearReshapeMeta,
) = begin
    A = _linear_reshape_matrix(meta, mean(q_a))
    mW = mean(q_W)
    return MvNormalWeightedMeanPrecision(A' * (mW * mean(q_y)), A' * mW * A)
end

@marginalrule ContinuousTransition(:y_x) (
    m_y::MultivariateNormalDistributionsFamily,
    m_x::MultivariateNormalDistributionsFamily,
    q_a::PointMass,
    q_W::Any,
    meta::LinearReshapeMeta,
) = begin
    A = _linear_reshape_matrix(meta, mean(q_a))
    mW = mean(q_W)
    ξy, Wy = weightedmean_precision(m_y)
    ξx, Wx = weightedmean_precision(m_x)
    return MvNormalWeightedMeanPrecision(
        [ξy; ξx],
        [(Wy + mW) (-(mW * A)); (-(A' * mW)) (Wx + A' * mW * A)],
    )
end
