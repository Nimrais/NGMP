export LinearReshapeMeta

import LinearAlgebra

"""
    LinearReshapeMeta(output_dim, input_dim)

Metadata for a `ContinuousTransition` whose parameter vector is reshaped in
Julia's column-major order into an `output_dim × input_dim` matrix.  It is
equivalent to `CTMeta(a -> reshape(a, output_dim, input_dim))`, but dispatches
rules that exploit the selector structure of the reshape Jacobian.
"""
struct LinearReshapeMeta
    output_dim::Int
    input_dim::Int

    function LinearReshapeMeta(output_dim::Integer, input_dim::Integer)
        output_dim > 0 || throw(ArgumentError("output_dim must be positive"))
        input_dim > 0 || throw(ArgumentError("input_dim must be positive"))
        return new(Int(output_dim), Int(input_dim))
    end
end

@inline _linear_reshape_parameter_count(meta::LinearReshapeMeta) =
    meta.output_dim * meta.input_dim

function _linear_reshape_check_parameter(meta::LinearReshapeMeta, a)
    expected = _linear_reshape_parameter_count(meta)
    actual = length(a)
    actual == expected || throw(
        DimensionMismatch(
            "LinearReshapeMeta($(meta.output_dim), $(meta.input_dim)) requires " *
            "length(a) == $expected, got $actual",
        ),
    )
    return a
end

function _linear_reshape_check_covariance(meta::LinearReshapeMeta, Va)
    expected = _linear_reshape_parameter_count(meta)
    size(Va) == (expected, expected) || throw(
        DimensionMismatch(
            "LinearReshapeMeta($(meta.output_dim), $(meta.input_dim)) requires " *
            "a covariance of size ($expected, $expected), got $(size(Va))",
        ),
    )
    return Va
end

@inline function _linear_reshape_matrix(meta::LinearReshapeMeta, a)
    _linear_reshape_check_parameter(meta, a)
    return reshape(a, meta.output_dim, meta.input_dim)
end

@inline _linear_reshape_parameter_index(meta::LinearReshapeMeta, row, column) =
    row + meta.output_dim * (column - 1)

# Compute Σᵢⱼ W[j, i] F_j Va F_i' directly from the row covariance
# blocks of Va.  F_i is the selector Jacobian for row i of reshape(a, dy, dx).
function _linear_reshape_input_uncertainty(meta::LinearReshapeMeta, W, Va)
    _linear_reshape_check_covariance(meta, Va)
    dy, dx = meta.output_dim, meta.input_dim
    T = promote_type(eltype(W), eltype(Va))
    result = zeros(T, dx, dx)

    @inbounds for i in 1:dy
        for j in 1:dy
            Wij = W[j, i]
            for column in 1:dx
                pi = _linear_reshape_parameter_index(meta, i, column)
                for row in 1:dx
                    pj = _linear_reshape_parameter_index(meta, j, row)
                    result[row, column] += Wij * Va[pj, pi]
                end
            end
        end
    end

    return result
end

# Compute the contribution of parameter uncertainty to E[A Exx A'].
# Entry (i, j) is tr(Exx * F_j Va F_i').
function _linear_reshape_output_uncertainty(meta::LinearReshapeMeta, Exx, Va)
    _linear_reshape_check_covariance(meta, Va)
    dy = meta.output_dim
    rows = ntuple(i -> i:dy:_linear_reshape_parameter_count(meta), dy)
    T = promote_type(eltype(Exx), eltype(Va))
    result = zeros(T, dy, dy)

    @inbounds for i in 1:dy
        for j in 1:dy
            result[i, j] = ReactiveMP.mul_trace(Exx, @view(Va[rows[j], rows[i]]))
        end
    end

    return result
end

function _linear_reshape_delta(my, Vy, mx, Vx, Vyx, A, Va, meta::LinearReshapeMeta)
    G1 = my * my' + Vy
    G2 = (my * mx' + Vyx) * A'
    G3 = transpose(G2)
    Exx = ReactiveMP.rank1update(Vx, mx)
    EAxxA = A * Exx * A' + _linear_reshape_output_uncertainty(meta, Exx, Va)
    return G1 - (G2 + G3) .+ LinearAlgebra.Symmetric(EAxxA)
end

# VMP: structured message to y.
@rule ContinuousTransition(:y, Marginalisation) (
    m_x::MultivariateNormalDistributionsFamily,
    q_a::MultivariateNormalDistributionsFamily,
    q_W::Any,
    meta::LinearReshapeMeta,
) = begin
    ma = mean(q_a)
    A = _linear_reshape_matrix(meta, ma)
    mx, Vx = mean_cov(m_x)
    mW = mean(q_W)

    Vy = A * Vx * A' + ReactiveMP.cholinv(mW)
    my = A * mx
    return MvNormalMeanCovariance(my, Vy)
end

# VMP: mean-field message to y.
@rule ContinuousTransition(:y, Marginalisation) (
    q_x::Any, q_a::Any, q_W::Any, meta::LinearReshapeMeta
) = begin
    A = _linear_reshape_matrix(meta, mean(q_a))
    return MvNormalMeanPrecision(A * mean(q_x), mean(q_W))
end

# VMP: structured message to x.
@rule ContinuousTransition(:x, Marginalisation) (
    m_y::MultivariateNormalDistributionsFamily,
    q_a::MultivariateNormalDistributionsFamily,
    q_W::Any,
    meta::LinearReshapeMeta,
) = begin
    ma, Va = mean_cov(q_a)
    A = _linear_reshape_matrix(meta, ma)
    my, Wy = mean_precision(m_y)
    mW = mean(q_W)

    WymW = Wy - Wy * ReactiveMP.cholinv(Wy + mW) * Wy
    Xi = A' * WymW * A + _linear_reshape_input_uncertainty(meta, mW, Va)
    z = A' * WymW * my
    return MvNormalWeightedMeanPrecision(z, Xi)
end

# VMP: mean-field message to x.
@rule ContinuousTransition(:x, Marginalisation) (
    q_y::Any, q_a::Any, q_W::Any, meta::LinearReshapeMeta
) = begin
    ma, Va = mean_cov(q_a)
    A = _linear_reshape_matrix(meta, ma)
    my = mean(q_y)
    mW = mean(q_W)

    Xi = A' * mW * A + _linear_reshape_input_uncertainty(meta, mW, Va)
    z = A' * mW * my
    return MvNormalWeightedMeanPrecision(z, Xi)
end

# VMP: structured message to a.
@rule ContinuousTransition(:a, Marginalisation) (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_a::MultivariateNormalDistributionsFamily,
    q_W::Any,
    meta::LinearReshapeMeta,
) = begin
    _linear_reshape_check_parameter(meta, mean(q_a))
    dy = meta.output_dim
    myx, Vyx = mean_cov(q_y_x)
    mx, Vx = @views myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my = @view myx[1:dy]
    Cyx = @view Vyx[1:dy, (dy + 1):end]
    mW = mean(q_W)

    Eyx = ReactiveMP.rank1update(Cyx, my, mx)
    Exx = ReactiveMP.rank1update(Vx, mx)
    xi = vec(transpose(mW) * Eyx)
    Lambda = LinearAlgebra.kron(Exx, transpose(mW))
    return MvNormalWeightedMeanPrecision(xi, Lambda)
end

# VMP: mean-field message to a.
@rule ContinuousTransition(:a, Marginalisation) (
    q_y::Any, q_x::Any, q_a::Any, q_W::Any, meta::LinearReshapeMeta
) = begin
    _linear_reshape_check_parameter(meta, mean(q_a))
    my = mean(q_y)
    mx, Vx = mean_cov(q_x)
    mW = mean(q_W)

    Eyx = my * mx'
    Exx = ReactiveMP.rank1update(Vx, mx)
    xi = vec(transpose(mW) * Eyx)
    Lambda = LinearAlgebra.kron(Exx, transpose(mW))
    return MvNormalWeightedMeanPrecision(xi, Lambda)
end

# VMP: structured message to W.
@rule ContinuousTransition(:W, Marginalisation) (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_a::MultivariateNormalDistributionsFamily,
    meta::LinearReshapeMeta,
) = begin
    ma, Va = mean_cov(q_a)
    A = _linear_reshape_matrix(meta, ma)
    dy = meta.output_dim
    myx, Vyx = mean_cov(q_y_x)

    mx, Vx = @views myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = @views myx[1:dy], Vyx[1:dy, 1:dy]
    Cyx = @view Vyx[1:dy, (dy + 1):end]
    Delta = _linear_reshape_delta(my, Vy, mx, Vx, Cyx, A, Va, meta)
    return WishartFast(dy + 2, Delta)
end

# VMP: mean-field message to W.
@rule ContinuousTransition(:W, Marginalisation) (
    q_y::Any, q_x::Any, q_a::Any, meta::LinearReshapeMeta
) = begin
    ma, Va = mean_cov(q_a)
    A = _linear_reshape_matrix(meta, ma)
    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_x)
    Cyx = zeros(eltype(ma), meta.output_dim, meta.input_dim)
    Delta = _linear_reshape_delta(my, Vy, mx, Vx, Cyx, A, Va, meta)
    return WishartFast(meta.output_dim + 2, Delta)
end

# Structured q(y, x) marginal.
@marginalrule ContinuousTransition(:y_x) (
    m_y::MultivariateNormalDistributionsFamily,
    m_x::MultivariateNormalDistributionsFamily,
    q_a::Any,
    q_W::Any,
    meta::LinearReshapeMeta,
) = begin
    ma, Va = mean_cov(q_a)
    A = _linear_reshape_matrix(meta, ma)
    mW = mean(q_W)
    xiy, Wy = weightedmean_precision(m_y)
    xix, Wx = weightedmean_precision(m_x)

    W11 = Wy + mW
    W12 = -(mW * A)
    W21 = -(A' * mW)
    Xi = Wx + _linear_reshape_input_uncertainty(meta, mW, Va)
    W22 = Xi + A' * mW * A
    return MvNormalWeightedMeanPrecision([xiy; xix], [W11 W12; W21 W22])
end

# Average energy: structured q(y, x).
@average_energy ContinuousTransition (
    q_y_x::Any, q_a::Any, q_W::Any, meta::LinearReshapeMeta
) = begin
    ma, Va = mean_cov(q_a)
    A = _linear_reshape_matrix(meta, ma)
    myx, Vyx = mean_cov(q_y_x)
    mW = mean(q_W)
    dy = meta.output_dim
    n = div(ndims(q_y_x), 2)

    mx, Vx = @views myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = @views myx[1:dy], Vyx[1:dy, 1:dy]
    Cyx = @view Vyx[1:dy, (dy + 1):end]
    g1 = -A * Cyx'
    g2 = g1'
    uncertainty = _linear_reshape_input_uncertainty(meta, mW, Va)
    xxt = mx * mx'
    trWSU = LinearAlgebra.tr(uncertainty)
    trkronxxWSU = ReactiveMP.mul_trace(xxt, uncertainty)

    return n / 2 * ReactiveMP.log2π - mean(LinearAlgebra.logdet, q_W) +
           (
        LinearAlgebra.tr(
            mW * (
                A * Vx * A' + g1 + g2 + Vy +
                (A * mx - my) * (A * mx - my)'
            ),
        ) + trWSU + trkronxxWSU
    ) / 2
end

# Average energy: mean-field q(y)q(x).
@average_energy ContinuousTransition (
    q_y::Any, q_x::Any, q_a::Any, q_W::Any, meta::LinearReshapeMeta
) = begin
    ma, Va = mean_cov(q_a)
    A = _linear_reshape_matrix(meta, ma)
    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_x)
    mW = mean(q_W)
    n = div(ndims(q_y), 2)
    uncertainty = _linear_reshape_input_uncertainty(meta, mW, Va)
    xxt = mx * mx'
    trWSU = LinearAlgebra.tr(uncertainty)
    trkronxxWSU = ReactiveMP.mul_trace(xxt, uncertainty)

    return n / 2 * ReactiveMP.log2π - mean(LinearAlgebra.logdet, q_W) +
           (
        LinearAlgebra.tr(
            mW * (A * Vx * A' + Vy + (A * mx - my) * (A * mx - my)'),
        ) + trWSU + trkronxxWSU
    ) / 2
end
