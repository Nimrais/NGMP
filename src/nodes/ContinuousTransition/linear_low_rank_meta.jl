export LinearLowRankMeta

import LinearAlgebra

raw"""
    LinearLowRankMeta(U, V)
    LinearLowRankMeta(a0_diagonal, U, V)

Metadata for a `ContinuousTransition` with transition matrix

```math
A(a) = A_0 + U \operatorname{Diag}(a) V^\mathsf{T}.
```

`U` has size `output_dim × rank`, `V` has size `input_dim × rank`, and the
parameter vector `a` has length `rank`.  `A_0` is a fixed rectangular diagonal
matrix represented by `a0_diagonal`, whose length must be
`min(output_dim, input_dim)`.  The two-argument constructor uses an exactly
zero `A_0`.

This is equivalent to
`CTMeta(a -> A0 + U * Diagonal(a) * V')`, but its rules exploit the fixed
rank-one basis matrices `U[:, k] * V[:, k]'` and do not construct their dense
Jacobians.
"""
struct LinearLowRankMeta{
    D <: AbstractVector, UM <: AbstractMatrix, VM <: AbstractMatrix
}
    a0_diagonal::D
    U::UM
    V::VM
    function LinearLowRankMeta(a0_diagonal::D, U::UM, V::VM) where {
        D <: AbstractVector, UM <: AbstractMatrix, VM <: AbstractMatrix
    }
        output_dim, rank = size(U)
        input_dim, v_rank = size(V)
        output_dim > 0 || throw(ArgumentError("U must have at least one row"))
        input_dim > 0 || throw(ArgumentError("V must have at least one row"))
        rank > 0 || throw(ArgumentError("the low-rank dimension must be positive"))
        v_rank == rank || throw(
            DimensionMismatch(
                "U and V must have the same number of columns, got $rank and $v_rank",
            ),
        )
        expected_diagonal = min(output_dim, input_dim)
        length(a0_diagonal) == expected_diagonal || throw(
            DimensionMismatch(
                "the A0 diagonal must have length $expected_diagonal, " *
                "got $(length(a0_diagonal))",
            ),
        )
        return new{D, UM, VM}(a0_diagonal, U, V)
    end
end

function LinearLowRankMeta(U::AbstractMatrix, V::AbstractMatrix)
    T = promote_type(eltype(U), eltype(V))
    diagonal = zeros(T, min(size(U, 1), size(V, 1)))
    return LinearLowRankMeta(diagonal, U, V)
end

@inline _linear_low_rank_output_dim(meta::LinearLowRankMeta) = size(meta.U, 1)
@inline _linear_low_rank_input_dim(meta::LinearLowRankMeta) = size(meta.V, 1)
@inline _linear_low_rank_rank(meta::LinearLowRankMeta) = size(meta.U, 2)

function _linear_low_rank_check_parameter(meta::LinearLowRankMeta, a)
    expected = _linear_low_rank_rank(meta)
    actual = length(a)
    actual == expected || throw(
        DimensionMismatch(
            "LinearLowRankMeta requires length(a) == $expected, got $actual",
        ),
    )
    return a
end

function _linear_low_rank_check_covariance(meta::LinearLowRankMeta, Va)
    expected = _linear_low_rank_rank(meta)
    size(Va) == (expected, expected) || throw(
        DimensionMismatch(
            "LinearLowRankMeta requires an a covariance of size " *
            "($expected, $expected), got $(size(Va))",
        ),
    )
    return Va
end

# U * Diagonal(a), without constructing Diagonal(a).
@inline function _linear_low_rank_scaled_U(meta::LinearLowRankMeta, a)
    _linear_low_rank_check_parameter(meta, a)
    return meta.U .* transpose(a)
end

# Materialize the mean transition matrix only for diagnostics/tests.  The
# inference rules below use factorized products instead.
function _linear_low_rank_matrix(meta::LinearLowRankMeta, a)
    B = _linear_low_rank_scaled_U(meta, a)
    T = promote_type(eltype(meta.a0_diagonal), eltype(B), eltype(meta.V))
    result = B * transpose(meta.V)
    @inbounds for i in eachindex(meta.a0_diagonal)
        result[i, i] += convert(T, meta.a0_diagonal[i])
    end
    return result
end

# Compute mean(A) * x without materializing mean(A).
function _linear_low_rank_mul(meta::LinearLowRankMeta, a, x::AbstractVector)
    _linear_low_rank_check_parameter(meta, a)
    size(x, 1) == _linear_low_rank_input_dim(meta) || throw(
        DimensionMismatch(
            "LinearLowRankMeta requires an input of length " *
            "$(_linear_low_rank_input_dim(meta)), got $(length(x))",
        ),
    )
    result = meta.U * (a .* (transpose(meta.V) * x))
    @inbounds for i in eachindex(meta.a0_diagonal)
        result[i] += meta.a0_diagonal[i] * x[i]
    end
    return result
end

# Compute mean(A) * X without materializing mean(A).
function _linear_low_rank_mul(meta::LinearLowRankMeta, a, X::AbstractMatrix)
    _linear_low_rank_check_parameter(meta, a)
    size(X, 1) == _linear_low_rank_input_dim(meta) || throw(
        DimensionMismatch(
            "LinearLowRankMeta requires an input matrix with " *
            "$(_linear_low_rank_input_dim(meta)) rows, got $(size(X, 1))",
        ),
    )
    projected = transpose(meta.V) * X
    projected .*= a
    result = meta.U * projected
    @inbounds for i in eachindex(meta.a0_diagonal)
        @views result[i, :] .+= meta.a0_diagonal[i] .* X[i, :]
    end
    return result
end

# Compute mean(A)' * y without materializing mean(A).
function _linear_low_rank_tmul(meta::LinearLowRankMeta, a, y::AbstractVector)
    _linear_low_rank_check_parameter(meta, a)
    length(y) == _linear_low_rank_output_dim(meta) || throw(
        DimensionMismatch(
            "LinearLowRankMeta requires an output of length " *
            "$(_linear_low_rank_output_dim(meta)), got $(length(y))",
        ),
    )
    result = meta.V * (a .* (transpose(meta.U) * y))
    @inbounds for i in eachindex(meta.a0_diagonal)
        result[i] += meta.a0_diagonal[i] * y[i]
    end
    return result
end

# Compute mean(A)' * R * mean(A).  The result is necessarily a dense input
# precision block in ReactiveMP's current Gaussian message representation.
function _linear_low_rank_AtRA(meta::LinearLowRankMeta, a, R)
    B = _linear_low_rank_scaled_U(meta, a)
    RB = R * B
    result = meta.V * (transpose(B) * RB) * transpose(meta.V)

    input_dim = _linear_low_rank_input_dim(meta)
    T = promote_type(eltype(result), eltype(meta.a0_diagonal), eltype(R))
    cross = zeros(T, input_dim, _linear_low_rank_rank(meta))
    @inbounds for i in eachindex(meta.a0_diagonal)
        @views cross[i, :] .= meta.a0_diagonal[i] .* RB[i, :]
    end
    result .+= cross * transpose(meta.V)
    result .+= meta.V * transpose(cross)

    @inbounds for j in eachindex(meta.a0_diagonal)
        for i in eachindex(meta.a0_diagonal)
            result[i, j] += meta.a0_diagonal[i] * R[i, j] * meta.a0_diagonal[j]
        end
    end
    return result
end

# Compute mean(A) * S * mean(A)' for a symmetric input second moment S.
function _linear_low_rank_ASAt(meta::LinearLowRankMeta, a, S)
    B = _linear_low_rank_scaled_U(meta, a)
    SV = S * meta.V
    result = B * (transpose(meta.V) * SV) * transpose(B)

    output_dim = _linear_low_rank_output_dim(meta)
    T = promote_type(eltype(result), eltype(meta.a0_diagonal), eltype(S))
    cross = zeros(T, output_dim, _linear_low_rank_rank(meta))
    @inbounds for i in eachindex(meta.a0_diagonal)
        @views cross[i, :] .= meta.a0_diagonal[i] .* SV[i, :]
    end
    result .+= cross * transpose(B)
    result .+= B * transpose(cross)

    @inbounds for j in eachindex(meta.a0_diagonal)
        for i in eachindex(meta.a0_diagonal)
            result[i, j] += meta.a0_diagonal[i] * S[i, j] * meta.a0_diagonal[j]
        end
    end
    return result
end

# Compute L * mean(A), used for the off-diagonal joint-precision blocks.
function _linear_low_rank_leftmul(meta::LinearLowRankMeta, a, L)
    B = _linear_low_rank_scaled_U(meta, a)
    result = (L * B) * transpose(meta.V)
    @inbounds for i in eachindex(meta.a0_diagonal)
        @views result[:, i] .+= meta.a0_diagonal[i] .* L[:, i]
    end
    return result
end

# Compute E[yx'] * mean(A)', used in the expected residual covariance.
function _linear_low_rank_right_transpose(meta::LinearLowRankMeta, a, Eyx)
    B = _linear_low_rank_scaled_U(meta, a)
    result = (Eyx * meta.V) * transpose(B)
    @inbounds for i in eachindex(meta.a0_diagonal)
        @views result[:, i] .+= meta.a0_diagonal[i] .* Eyx[:, i]
    end
    return result
end

# Parameter-uncertainty contribution to E[A' W A].
function _linear_low_rank_input_uncertainty(meta::LinearLowRankMeta, W, Va)
    _linear_low_rank_check_covariance(meta, Va)
    gram = transpose(meta.U) * W * meta.U
    return meta.V * (Va .* gram) * transpose(meta.V)
end

# Parameter-uncertainty contribution to E[A S A'].
function _linear_low_rank_output_uncertainty(meta::LinearLowRankMeta, S, Va)
    _linear_low_rank_check_covariance(meta, Va)
    gram = transpose(meta.V) * S * meta.V
    return meta.U * (Va .* gram) * transpose(meta.U)
end

function _linear_low_rank_offset_residual(meta::LinearLowRankMeta, Eyx, Exx)
    result = copy(Eyx)
    @inbounds for i in eachindex(meta.a0_diagonal)
        @views result[i, :] .-= meta.a0_diagonal[i] .* Exx[i, :]
    end
    return result
end

function _linear_low_rank_parameter_message(meta::LinearLowRankMeta, Eyx, Exx, W)
    residual = _linear_low_rank_offset_residual(meta, Eyx, Exx)
    UW = transpose(meta.U) * W
    xi = LinearAlgebra.diag(UW * residual * meta.V)
    Lambda = (UW * meta.U) .* (transpose(meta.V) * Exx * meta.V)
    return MvNormalWeightedMeanPrecision(xi, Lambda)
end

function _linear_low_rank_delta(my, Vy, mx, Vx, Cyx, ma, Va, meta)
    Eyy = ReactiveMP.rank1update(Vy, my)
    Eyx = ReactiveMP.rank1update(Cyx, my, mx)
    Exx = ReactiveMP.rank1update(Vx, mx)
    cross = _linear_low_rank_right_transpose(meta, ma, Eyx)
    EAxxA = _linear_low_rank_ASAt(meta, ma, Exx) +
            _linear_low_rank_output_uncertainty(meta, Exx, Va)
    return Eyy - (cross + transpose(cross)) .+ LinearAlgebra.Symmetric(EAxxA)
end

# VMP: structured message to y.
@rule ContinuousTransition(:y, Marginalisation) (
    m_x::MultivariateNormalDistributionsFamily,
    q_a::MultivariateNormalDistributionsFamily,
    q_W::Any,
    meta::LinearLowRankMeta,
) = begin
    ma = mean(q_a)
    _linear_low_rank_check_parameter(meta, ma)
    mx, Vx = mean_cov(m_x)
    mW = mean(q_W)

    my = _linear_low_rank_mul(meta, ma, mx)
    Vy = _linear_low_rank_ASAt(meta, ma, Vx) + ReactiveMP.cholinv(mW)
    return MvNormalMeanCovariance(my, Vy)
end

# VMP: mean-field message to y.
@rule ContinuousTransition(:y, Marginalisation) (
    q_x::Any, q_a::Any, q_W::Any, meta::LinearLowRankMeta
) = begin
    ma = mean(q_a)
    return MvNormalMeanPrecision(
        _linear_low_rank_mul(meta, ma, mean(q_x)), mean(q_W)
    )
end

# VMP: structured message to x.
@rule ContinuousTransition(:x, Marginalisation) (
    m_y::MultivariateNormalDistributionsFamily,
    q_a::MultivariateNormalDistributionsFamily,
    q_W::Any,
    meta::LinearLowRankMeta,
) = begin
    ma, Va = mean_cov(q_a)
    _linear_low_rank_check_parameter(meta, ma)
    my, Wy = mean_precision(m_y)
    mW = mean(q_W)

    WymW = Wy - Wy * ReactiveMP.cholinv(Wy + mW) * Wy
    Xi = _linear_low_rank_AtRA(meta, ma, WymW) +
         _linear_low_rank_input_uncertainty(meta, mW, Va)
    z = _linear_low_rank_tmul(meta, ma, WymW * my)
    return MvNormalWeightedMeanPrecision(z, Xi)
end

# VMP: mean-field message to x.
@rule ContinuousTransition(:x, Marginalisation) (
    q_y::Any, q_a::Any, q_W::Any, meta::LinearLowRankMeta
) = begin
    ma, Va = mean_cov(q_a)
    _linear_low_rank_check_parameter(meta, ma)
    my = mean(q_y)
    mW = mean(q_W)

    Xi = _linear_low_rank_AtRA(meta, ma, mW) +
         _linear_low_rank_input_uncertainty(meta, mW, Va)
    z = _linear_low_rank_tmul(meta, ma, mW * my)
    return MvNormalWeightedMeanPrecision(z, Xi)
end

# VMP: structured message to a.
@rule ContinuousTransition(:a, Marginalisation) (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_a::MultivariateNormalDistributionsFamily,
    q_W::Any,
    meta::LinearLowRankMeta,
) = begin
    _linear_low_rank_check_parameter(meta, mean(q_a))
    dy = _linear_low_rank_output_dim(meta)
    myx, Vyx = mean_cov(q_y_x)
    mx, Vx = @views myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my = @view myx[1:dy]
    Cyx = @view Vyx[1:dy, (dy + 1):end]

    Eyx = ReactiveMP.rank1update(Cyx, my, mx)
    Exx = ReactiveMP.rank1update(Vx, mx)
    return _linear_low_rank_parameter_message(meta, Eyx, Exx, mean(q_W))
end

# VMP: mean-field message to a.
@rule ContinuousTransition(:a, Marginalisation) (
    q_y::Any, q_x::Any, q_a::Any, q_W::Any, meta::LinearLowRankMeta
) = begin
    _linear_low_rank_check_parameter(meta, mean(q_a))
    my = mean(q_y)
    mx, Vx = mean_cov(q_x)
    Eyx = my * transpose(mx)
    Exx = ReactiveMP.rank1update(Vx, mx)
    return _linear_low_rank_parameter_message(meta, Eyx, Exx, mean(q_W))
end

# VMP: structured message to W.
@rule ContinuousTransition(:W, Marginalisation) (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_a::MultivariateNormalDistributionsFamily,
    meta::LinearLowRankMeta,
) = begin
    ma, Va = mean_cov(q_a)
    _linear_low_rank_check_parameter(meta, ma)
    dy = _linear_low_rank_output_dim(meta)
    myx, Vyx = mean_cov(q_y_x)

    mx, Vx = @views myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = @views myx[1:dy], Vyx[1:dy, 1:dy]
    Cyx = @view Vyx[1:dy, (dy + 1):end]
    Delta = _linear_low_rank_delta(my, Vy, mx, Vx, Cyx, ma, Va, meta)
    return WishartFast(dy + 2, Delta)
end

# VMP: mean-field message to W.
@rule ContinuousTransition(:W, Marginalisation) (
    q_y::Any, q_x::Any, q_a::Any, meta::LinearLowRankMeta
) = begin
    ma, Va = mean_cov(q_a)
    _linear_low_rank_check_parameter(meta, ma)
    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_x)
    Cyx = zeros(eltype(ma), _linear_low_rank_output_dim(meta), length(mx))
    Delta = _linear_low_rank_delta(my, Vy, mx, Vx, Cyx, ma, Va, meta)
    return WishartFast(_linear_low_rank_output_dim(meta) + 2, Delta)
end

# Structured q(y, x) marginal.
@marginalrule ContinuousTransition(:y_x) (
    m_y::MultivariateNormalDistributionsFamily,
    m_x::MultivariateNormalDistributionsFamily,
    q_a::Any,
    q_W::Any,
    meta::LinearLowRankMeta,
) = begin
    ma, Va = mean_cov(q_a)
    _linear_low_rank_check_parameter(meta, ma)
    mW = mean(q_W)
    xiy, Wy = weightedmean_precision(m_y)
    xix, Wx = weightedmean_precision(m_x)

    W11 = Wy + mW
    W12 = -_linear_low_rank_leftmul(meta, ma, mW)
    W21 = -_linear_low_rank_leftmul(meta, ma, transpose(mW))'
    W22 = Wx + _linear_low_rank_AtRA(meta, ma, mW) +
          _linear_low_rank_input_uncertainty(meta, mW, Va)
    return MvNormalWeightedMeanPrecision([xiy; xix], [W11 W12; W21 W22])
end

function _linear_low_rank_uncertainty_energy(meta, mx, mW, Va)
    gram = transpose(meta.U) * mW * meta.U
    middle = Va .* gram
    VtV = transpose(meta.V) * meta.V
    projected_mean = transpose(meta.V) * mx
    trWSU = LinearAlgebra.tr(middle * VtV)
    trkronxxWSU = LinearAlgebra.dot(projected_mean, middle * projected_mean)
    return trWSU, trkronxxWSU
end

# Average energy: structured q(y, x).
@average_energy ContinuousTransition (
    q_y_x::Any, q_a::Any, q_W::Any, meta::LinearLowRankMeta
) = begin
    ma, Va = mean_cov(q_a)
    _linear_low_rank_check_parameter(meta, ma)
    myx, Vyx = mean_cov(q_y_x)
    mW = mean(q_W)
    dy = _linear_low_rank_output_dim(meta)
    n = div(ndims(q_y_x), 2)

    mx, Vx = @views myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = @views myx[1:dy], Vyx[1:dy, 1:dy]
    Cyx = @view Vyx[1:dy, (dy + 1):end]
    g1 = -_linear_low_rank_mul(meta, ma, transpose(Cyx))
    mean_residual = _linear_low_rank_mul(meta, ma, mx) - my
    trWSU, trkronxxWSU = _linear_low_rank_uncertainty_energy(meta, mx, mW, Va)

    return n / 2 * ReactiveMP.log2π - mean(LinearAlgebra.logdet, q_W) +
           (
        LinearAlgebra.tr(
            mW * (
                _linear_low_rank_ASAt(meta, ma, Vx) + g1 + transpose(g1) + Vy +
                mean_residual * transpose(mean_residual)
            ),
        ) + trWSU + trkronxxWSU
    ) / 2
end

# Average energy: mean-field q(y)q(x).
@average_energy ContinuousTransition (
    q_y::Any, q_x::Any, q_a::Any, q_W::Any, meta::LinearLowRankMeta
) = begin
    ma, Va = mean_cov(q_a)
    _linear_low_rank_check_parameter(meta, ma)
    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_x)
    mW = mean(q_W)
    n = div(ndims(q_y), 2)
    mean_residual = _linear_low_rank_mul(meta, ma, mx) - my
    trWSU, trkronxxWSU = _linear_low_rank_uncertainty_energy(meta, mx, mW, Va)

    return n / 2 * ReactiveMP.log2π - mean(LinearAlgebra.logdet, q_W) +
           (
        LinearAlgebra.tr(
            mW * (
                _linear_low_rank_ASAt(meta, ma, Vx) + Vy +
                mean_residual * transpose(mean_residual)
            ),
        ) + trWSU + trkronxxWSU
    ) / 2
end
