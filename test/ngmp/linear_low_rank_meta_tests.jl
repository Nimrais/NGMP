using Test

using Distributions
using LinearAlgebra: Diagonal, I, Symmetric, diag, logdet, norm, tr
using Random
using ReactiveMP
using RxInfer
using SurrogateModelling

import ReactiveMP: @call_marginalrule, @call_rule, Marginal

@model function linear_low_rank_ct_toy(
    y, features, priors, feature_cov, meta_map, meta_pred, sp_deps, sp_damping
)
    a_map ~ priors[:a_map]
    a_pred ~ priors[:a_pred]
    theta ~ priors[:theta]
    P ~ priors[:P]
    Gamma2 ~ priors[:Gamma2]
    gamma_obs ~ priors[:gamma_obs]
    for i in eachindex(y)
        x_f[i] ~ MvNormalMeanCovariance(features[i], feature_cov)
        h1[i] ~ ContinuousTransition(x_f[i], a_map, P) where {meta = meta_map}
        s[i] ~ MvSoftplus(h1[i]) where {dependencies = sp_deps, meta = sp_damping}
        h2[i] ~ ContinuousTransition(s[i], a_pred, Gamma2) where {meta = meta_pred}
        y[i] ~ softdot(theta, h2[i], gamma_obs)
    end
end

@constraints function linear_low_rank_ct_toy_constraints()
    q(x_f, h1, s, h2, a_map, a_pred, P, Gamma2, theta, gamma_obs) =
        q(x_f, h1)q(s, h2)q(a_map)q(a_pred)q(P)q(Gamma2)q(theta)q(gamma_obs)
end

function _low_rank_test_spd(rng, dimension; ridge = 0.5)
    factor = randn(rng, dimension, dimension)
    return Matrix(Symmetric(factor * factor' + ridge * I))
end

function _low_rank_test_A0(diagonal, dy, dx)
    result = zeros(eltype(diagonal), dy, dx)
    @inbounds for i in eachindex(diagonal)
        result[i, i] = diagonal[i]
    end
    return result
end

function _low_rank_test_same_gaussian(left, right; atol = 1e-10)
    @test weightedmean(left) ≈ weightedmean(right) atol = atol rtol = atol
    @test precision(left) ≈ precision(right) atol = atol rtol = atol
end

function _low_rank_test_same_wishart(left, right; atol = 1e-10)
    @test left.ν ≈ right.ν atol = atol rtol = atol
    @test left.invS ≈ right.invS atol = atol rtol = atol
    @test mean(left) ≈ mean(right) atol = atol rtol = atol
end

function _low_rank_test_average_energy(meta, names, distributions)
    marginals = map(distribution -> Marginal(distribution, false, false), distributions)
    return score(AverageEnergy(), ContinuousTransition, names, marginals, meta)
end

function _low_rank_test_inputs(rng, dy, dx, rank)
    ma = randn(rng, rank)
    Va = _low_rank_test_spd(rng, rank; ridge = 0.8)
    q_a = MvNormalMeanCovariance(ma, Va)

    my, mx = randn(rng, dy), randn(rng, dx)
    Vy = _low_rank_test_spd(rng, dy; ridge = 0.9)
    Vx = _low_rank_test_spd(rng, dx; ridge = 0.7)
    q_y = MvNormalMeanCovariance(my, Vy)
    q_x = MvNormalMeanCovariance(mx, Vx)

    Vjoint = _low_rank_test_spd(rng, dy + dx; ridge = 1.1)
    q_y_x = MvNormalMeanCovariance([my; mx], Vjoint)
    @test !iszero(norm(@view Vjoint[1:dy, (dy + 1):end]))

    m_y = MvNormalWeightedMeanPrecision(
        randn(rng, dy), _low_rank_test_spd(rng, dy; ridge = 1.2)
    )
    m_x = MvNormalWeightedMeanPrecision(
        randn(rng, dx), _low_rank_test_spd(rng, dx; ridge = 1.3)
    )
    W = _low_rank_test_spd(rng, dy; ridge = 1.4)
    q_W = PointMass(W)
    return (; q_a, q_y, q_x, q_y_x, m_y, m_x, q_W)
end

function _low_rank_test_equivalent_rules(generic_meta, specialized_meta, data)
    generic_y_structured = @call_rule ContinuousTransition(:y, Marginalisation) (
        m_x = data.m_x, q_a = data.q_a, q_W = data.q_W, meta = generic_meta
    )
    specialized_y_structured = @call_rule ContinuousTransition(:y, Marginalisation) (
        m_x = data.m_x, q_a = data.q_a, q_W = data.q_W, meta = specialized_meta
    )
    _low_rank_test_same_gaussian(generic_y_structured, specialized_y_structured)

    generic_y_mean_field = @call_rule ContinuousTransition(:y, Marginalisation) (
        q_x = data.q_x, q_a = data.q_a, q_W = data.q_W, meta = generic_meta
    )
    specialized_y_mean_field = @call_rule ContinuousTransition(:y, Marginalisation) (
        q_x = data.q_x, q_a = data.q_a, q_W = data.q_W, meta = specialized_meta
    )
    _low_rank_test_same_gaussian(generic_y_mean_field, specialized_y_mean_field)

    generic_x_structured = @call_rule ContinuousTransition(:x, Marginalisation) (
        m_y = data.m_y, q_a = data.q_a, q_W = data.q_W, meta = generic_meta
    )
    specialized_x_structured = @call_rule ContinuousTransition(:x, Marginalisation) (
        m_y = data.m_y, q_a = data.q_a, q_W = data.q_W, meta = specialized_meta
    )
    _low_rank_test_same_gaussian(generic_x_structured, specialized_x_structured)

    generic_x_mean_field = @call_rule ContinuousTransition(:x, Marginalisation) (
        q_y = data.q_y, q_a = data.q_a, q_W = data.q_W, meta = generic_meta
    )
    specialized_x_mean_field = @call_rule ContinuousTransition(:x, Marginalisation) (
        q_y = data.q_y, q_a = data.q_a, q_W = data.q_W, meta = specialized_meta
    )
    _low_rank_test_same_gaussian(generic_x_mean_field, specialized_x_mean_field)

    generic_a_structured = @call_rule ContinuousTransition(:a, Marginalisation) (
        q_y_x = data.q_y_x,
        q_a = data.q_a,
        q_W = data.q_W,
        meta = generic_meta,
    )
    specialized_a_structured = @call_rule ContinuousTransition(:a, Marginalisation) (
        q_y_x = data.q_y_x,
        q_a = data.q_a,
        q_W = data.q_W,
        meta = specialized_meta,
    )
    _low_rank_test_same_gaussian(generic_a_structured, specialized_a_structured)

    generic_a_mean_field = @call_rule ContinuousTransition(:a, Marginalisation) (
        q_y = data.q_y,
        q_x = data.q_x,
        q_a = data.q_a,
        q_W = data.q_W,
        meta = generic_meta,
    )
    specialized_a_mean_field = @call_rule ContinuousTransition(:a, Marginalisation) (
        q_y = data.q_y,
        q_x = data.q_x,
        q_a = data.q_a,
        q_W = data.q_W,
        meta = specialized_meta,
    )
    _low_rank_test_same_gaussian(generic_a_mean_field, specialized_a_mean_field)

    generic_W_structured = @call_rule ContinuousTransition(:W, Marginalisation) (
        q_y_x = data.q_y_x, q_a = data.q_a, meta = generic_meta
    )
    specialized_W_structured = @call_rule ContinuousTransition(:W, Marginalisation) (
        q_y_x = data.q_y_x, q_a = data.q_a, meta = specialized_meta
    )
    _low_rank_test_same_wishart(generic_W_structured, specialized_W_structured)

    generic_W_mean_field = @call_rule ContinuousTransition(:W, Marginalisation) (
        q_y = data.q_y, q_x = data.q_x, q_a = data.q_a, meta = generic_meta
    )
    specialized_W_mean_field = @call_rule ContinuousTransition(:W, Marginalisation) (
        q_y = data.q_y, q_x = data.q_x, q_a = data.q_a, meta = specialized_meta
    )
    _low_rank_test_same_wishart(generic_W_mean_field, specialized_W_mean_field)

    generic_joint = @call_marginalrule ContinuousTransition(:y_x) (
        m_y = data.m_y,
        m_x = data.m_x,
        q_a = data.q_a,
        q_W = data.q_W,
        meta = generic_meta,
    )
    specialized_joint = @call_marginalrule ContinuousTransition(:y_x) (
        m_y = data.m_y,
        m_x = data.m_x,
        q_a = data.q_a,
        q_W = data.q_W,
        meta = specialized_meta,
    )
    _low_rank_test_same_gaussian(generic_joint, specialized_joint)

    generic_structured_energy = _low_rank_test_average_energy(
        generic_meta,
        Val{(:y_x, :a, :W)}(),
        (data.q_y_x, data.q_a, data.q_W),
    )
    specialized_structured_energy = _low_rank_test_average_energy(
        specialized_meta,
        Val{(:y_x, :a, :W)}(),
        (data.q_y_x, data.q_a, data.q_W),
    )
    @test generic_structured_energy ≈ specialized_structured_energy atol = 1e-10 rtol = 1e-10

    generic_mean_field_energy = _low_rank_test_average_energy(
        generic_meta,
        Val{(:y, :x, :a, :W)}(),
        (data.q_y, data.q_x, data.q_a, data.q_W),
    )
    specialized_mean_field_energy = _low_rank_test_average_energy(
        specialized_meta,
        Val{(:y, :x, :a, :W)}(),
        (data.q_y, data.q_x, data.q_a, data.q_W),
    )
    @test generic_mean_field_energy ≈ specialized_mean_field_energy atol = 1e-10 rtol = 1e-10
end

@testset "LinearLowRankMeta" begin
    @testset "construction and dimensions" begin
        U = ones(3, 2)
        V = ones(4, 2)
        zero_meta = LinearLowRankMeta(U, V)
        @test zero_meta.a0_diagonal == zeros(3)
        @test size(zero_meta.U) == (3, 2)
        @test size(zero_meta.V) == (4, 2)

        diagonal = [1.0, 2.0, 3.0]
        meta = LinearLowRankMeta(diagonal, U, V)
        a = [0.4, -0.2]
        A0 = _low_rank_test_A0(diagonal, 3, 4)
        @test SurrogateModelling._linear_low_rank_matrix(meta, a) ≈
              A0 + U * Diagonal(a) * V'

        @test_throws ArgumentError LinearLowRankMeta(zeros(0, 2), ones(3, 2))
        @test_throws ArgumentError LinearLowRankMeta(ones(3, 0), ones(4, 0))
        @test_throws DimensionMismatch LinearLowRankMeta(ones(3, 2), ones(4, 1))
        @test_throws DimensionMismatch LinearLowRankMeta(zeros(2), U, V)
    end

    @testset "zero A0 rules match CTMeta for dy=$dy, dx=$dx, rank=$rank" for
        (dy, dx, rank) in ((1, 3, 1), (2, 3, 2), (3, 2, 2), (4, 5, 2), (3, 3, 3))
        rng = MersenneTwister(2_000 + 100dy + 10dx + rank)
        U = randn(rng, dy, rank)
        V = randn(rng, dx, rank)
        A0 = zeros(dy, dx)
        generic_meta = CTMeta(a -> A0 + U * Diagonal(a) * V')
        specialized_meta = LinearLowRankMeta(U, V)
        data = _low_rank_test_inputs(rng, dy, dx, rank)
        _low_rank_test_equivalent_rules(generic_meta, specialized_meta, data)
    end

    @testset "nonzero rectangular A0 for dy=$dy, dx=$dx" for
        (dy, dx) in ((2, 3), (3, 2))
        rng = MersenneTwister(3_500 + 10dy + dx)
        rank = 2
        diagonal = collect(range(0.7, -0.4; length = min(dy, dx)))
        A0 = _low_rank_test_A0(diagonal, dy, dx)
        U = randn(rng, dy, rank)
        V = randn(rng, dx, rank)
        generic_meta = CTMeta(a -> A0 + U * Diagonal(a) * V')
        specialized_meta = LinearLowRankMeta(diagonal, U, V)
        data = _low_rank_test_inputs(rng, dy, dx, rank)

        # Rules that use the full companion matrix already handle A0 correctly
        # in generic CTMeta.
        generic_y = @call_rule ContinuousTransition(:y, Marginalisation) (
            q_x = data.q_x, q_a = data.q_a, q_W = data.q_W, meta = generic_meta
        )
        specialized_y = @call_rule ContinuousTransition(:y, Marginalisation) (
            q_x = data.q_x,
            q_a = data.q_a,
            q_W = data.q_W,
            meta = specialized_meta,
        )
        _low_rank_test_same_gaussian(generic_y, specialized_y)

        generic_x = @call_rule ContinuousTransition(:x, Marginalisation) (
            q_y = data.q_y, q_a = data.q_a, q_W = data.q_W, meta = generic_meta
        )
        specialized_x = @call_rule ContinuousTransition(:x, Marginalisation) (
            q_y = data.q_y,
            q_a = data.q_a,
            q_W = data.q_W,
            meta = specialized_meta,
        )
        _low_rank_test_same_gaussian(generic_x, specialized_x)

        specialized_W = @call_rule ContinuousTransition(:W, Marginalisation) (
            q_y = data.q_y,
            q_x = data.q_x,
            q_a = data.q_a,
            meta = specialized_meta,
        )

        ma, Va = mean_cov(data.q_a)
        my, Vy = mean_cov(data.q_y)
        mx, Vx = mean_cov(data.q_x)
        Exx = Vx + mx * mx'
        Eyx = my * mx'
        Eyy = Vy + my * my'
        mean_A = A0 + U * Diagonal(ma) * V'
        output_uncertainty = U * (Va .* (V' * Exx * V)) * U'
        expected_delta = Eyy - Eyx * mean_A' - mean_A * Eyx' +
                         mean_A * Exx * mean_A' + output_uncertainty
        @test specialized_W.invS ≈ expected_delta atol = 1e-10 rtol = 1e-10

        myx, Vyx = mean_cov(data.q_y_x)
        joint_my = @view myx[1:dy]
        joint_mx = @view myx[(dy + 1):end]
        joint_Vy = @view Vyx[1:dy, 1:dy]
        joint_Vx = @view Vyx[(dy + 1):end, (dy + 1):end]
        joint_Cyx = @view Vyx[1:dy, (dy + 1):end]
        joint_Eyy = joint_Vy + joint_my * joint_my'
        joint_Eyx = joint_Cyx + joint_my * joint_mx'
        joint_Exx = joint_Vx + joint_mx * joint_mx'
        joint_output_uncertainty = U * (Va .* (V' * joint_Exx * V)) * U'
        expected_joint_delta = joint_Eyy - joint_Eyx * mean_A' -
                               mean_A * joint_Eyx' + mean_A * joint_Exx * mean_A' +
                               joint_output_uncertainty
        specialized_structured_W = @call_rule ContinuousTransition(:W, Marginalisation) (
            q_y_x = data.q_y_x, q_a = data.q_a, meta = specialized_meta
        )
        @test specialized_structured_W.invS ≈ expected_joint_delta atol = 1e-10 rtol = 1e-10

        generic_joint = @call_marginalrule ContinuousTransition(:y_x) (
            m_y = data.m_y,
            m_x = data.m_x,
            q_a = data.q_a,
            q_W = data.q_W,
            meta = generic_meta,
        )
        specialized_joint = @call_marginalrule ContinuousTransition(:y_x) (
            m_y = data.m_y,
            m_x = data.m_x,
            q_a = data.q_a,
            q_W = data.q_W,
            meta = specialized_meta,
        )
        _low_rank_test_same_gaussian(generic_joint, specialized_joint)

        # The affine a-update must use y - A0*x.  This direct reference is the
        # mathematically correct rule; generic CTMeta currently drops A0 here.
        W = mean(data.q_W)
        expected_xi = diag(U' * W * (Eyx - A0 * Exx) * V)
        expected_precision = (U' * W * U) .* (V' * Exx * V)
        specialized_a = @call_rule ContinuousTransition(:a, Marginalisation) (
            q_y = data.q_y,
            q_x = data.q_x,
            q_a = data.q_a,
            q_W = data.q_W,
            meta = specialized_meta,
        )
        @test weightedmean(specialized_a) ≈ expected_xi atol = 1e-10 rtol = 1e-10
        @test precision(specialized_a) ≈ expected_precision atol = 1e-10 rtol = 1e-10

        expected_joint_xi = diag(U' * W * (joint_Eyx - A0 * joint_Exx) * V)
        expected_joint_precision = (U' * W * U) .* (V' * joint_Exx * V)
        specialized_structured_a = @call_rule ContinuousTransition(:a, Marginalisation) (
            q_y_x = data.q_y_x,
            q_a = data.q_a,
            q_W = data.q_W,
            meta = specialized_meta,
        )
        @test weightedmean(specialized_structured_a) ≈ expected_joint_xi atol = 1e-10 rtol = 1e-10
        @test precision(specialized_structured_a) ≈ expected_joint_precision atol = 1e-10 rtol = 1e-10

        generic_mean_field_energy = _low_rank_test_average_energy(
            generic_meta,
            Val{(:y, :x, :a, :W)}(),
            (data.q_y, data.q_x, data.q_a, data.q_W),
        )
        specialized_mean_field_energy = _low_rank_test_average_energy(
            specialized_meta,
            Val{(:y, :x, :a, :W)}(),
            (data.q_y, data.q_x, data.q_a, data.q_W),
        )
        @test generic_mean_field_energy ≈ specialized_mean_field_energy atol = 1e-10 rtol = 1e-10

        generic_structured_energy = _low_rank_test_average_energy(
            generic_meta,
            Val{(:y_x, :a, :W)}(),
            (data.q_y_x, data.q_a, data.q_W),
        )
        specialized_structured_energy = _low_rank_test_average_energy(
            specialized_meta,
            Val{(:y_x, :a, :W)}(),
            (data.q_y_x, data.q_a, data.q_W),
        )
        @test generic_structured_energy ≈ specialized_structured_energy atol = 1e-10 rtol = 1e-10
    end

    @testset "point-mass input avoids dense feature covariance" begin
        rng = MersenneTwister(4_219)
        dy, dx, rank = 6, 65, 10
        diagonal = randn(rng, min(dy, dx))
        meta = LinearLowRankMeta(
            diagonal,
            randn(rng, dy, rank),
            randn(rng, dx, rank),
        )
        data = _low_rank_test_inputs(rng, dy, dx, rank)
        x = randn(rng, dx)
        q_x = PointMass(x)
        my, Vy = mean_cov(data.q_y)
        ma, Va = mean_cov(data.q_a)
        W = mean(data.q_W)
        Exx = x * x'
        Eyx = my * x'

        point_a = @call_rule ContinuousTransition(:a, Marginalisation) (
            q_y = data.q_y,
            q_x = q_x,
            q_a = data.q_a,
            q_W = data.q_W,
            meta = meta,
        )
        dense_a = SurrogateModelling._linear_low_rank_parameter_message(
            meta,
            Eyx,
            Exx,
            W,
        )
        _low_rank_test_same_gaussian(point_a, dense_a)

        point_W = @call_rule ContinuousTransition(:W, Marginalisation) (
            q_y = data.q_y,
            q_x = q_x,
            q_a = data.q_a,
            meta = meta,
        )
        dense_delta = SurrogateModelling._linear_low_rank_delta(
            my,
            Vy,
            x,
            zeros(dx, dx),
            zeros(dy, dx),
            ma,
            Va,
            meta,
        )
        @test point_W.invS ≈ dense_delta atol = 1e-10 rtol = 1e-10

        point_energy = _low_rank_test_average_energy(
            meta,
            Val{(:y, :x, :a, :W)}(),
            (data.q_y, q_x, data.q_a, data.q_W),
        )
        n = div(ndims(data.q_y), 2)
        expected_energy = n / 2 * ReactiveMP.log2π - logdet(W) +
                          tr(W * dense_delta) / 2
        @test point_energy ≈ expected_energy atol = 1e-10 rtol = 1e-10

        SurrogateModelling._linear_low_rank_pointmass_parameter_message(
            meta,
            my,
            x,
            W,
        )
        allocated = @allocated SurrogateModelling._linear_low_rank_pointmass_parameter_message(
            meta,
            my,
            x,
            W,
        )
        @test allocated < sizeof(Float64) * dx^2
    end

    @testset "parameter length validation" begin
        meta = LinearLowRankMeta(ones(2, 2), ones(3, 2))
        q_a = MvNormalMeanCovariance(zeros(3), Matrix(Diagonal(ones(3))))
        q_x = MvNormalMeanCovariance(zeros(3), Matrix(Diagonal(ones(3))))
        q_W = PointMass(Matrix{Float64}(I, 2, 2))
        @test_throws DimensionMismatch @call_rule ContinuousTransition(
            :y, Marginalisation
        ) (q_x = q_x, q_a = q_a, q_W = q_W, meta = meta)
    end

    @testset "natural-gradient adapter" begin
        rng = MersenneTwister(4_811)
        dy, dx, rank = 2, 3, 2
        meta = LinearLowRankMeta(randn(rng, dy, rank), randn(rng, dx, rank))
        data = _low_rank_test_inputs(rng, dy, dx, rank)
        state = NGMPEdgeState(meta; damping = DampingMeta(alpha = 1.0, beta = 0.0))
        target = @call_rule ContinuousTransition(:a, Marginalisation) (
            q_y_x = data.q_y_x,
            q_a = data.q_a,
            q_W = data.q_W,
            meta = meta,
        )
        message = @call_rule ContinuousTransition(:a, NaturalGradientMessage) (
            q_y_x = data.q_y_x,
            q_a = data.q_a,
            q_W = data.q_W,
            meta = state,
        )
        _low_rank_test_same_gaussian(target, message)
        @test state.usermeta === meta
        @test state.nfired == 1
    end

    @testset "parameter precision scales with rank" begin
        rng = MersenneTwister(5_109)
        dy, dx, rank = 8, 12, 3
        meta = LinearLowRankMeta(randn(rng, dy, rank), randn(rng, dx, rank))
        data = _low_rank_test_inputs(rng, dy, dx, rank)
        message = @call_rule ContinuousTransition(:a, Marginalisation) (
            q_y = data.q_y,
            q_x = data.q_x,
            q_a = data.q_a,
            q_W = data.q_W,
            meta = meta,
        )
        @test length(weightedmean(message)) == rank
        @test size(precision(message)) == (rank, rank)
        @test rank < dy * dx
    end


    @testset "two-iteration CT/MvSoftplus integration equivalence" begin
        dy, dx, map_rank, pred_rank = 2, 3, 2, 2
        rng = MersenneTwister(6_203)
        U_map, V_map = randn(rng, dy, map_rank), randn(rng, dx, map_rank)
        U_pred, V_pred = randn(rng, dy, pred_rank), randn(rng, dy, pred_rank)
        features = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
        y = [0.02, 0.96, 1.01, -0.01]
        nu = dy + 2.0
        priors = Dict{Symbol, Any}(
            :a_map => MvNormalMeanCovariance(
                0.3 .* randn(rng, map_rank), Diagonal(ones(map_rank))
            ),
            :a_pred => MvNormalMeanCovariance(
                0.3 .* randn(rng, pred_rank), Diagonal(ones(pred_rank))
            ),
            :theta => MvNormalMeanCovariance(zeros(dy), Diagonal(ones(dy))),
            :P => ExponentialFamily.WishartFast(
                nu, Matrix(Diagonal(fill(nu / 10.0, dy)))
            ),
            :Gamma2 => ExponentialFamily.WishartFast(
                nu, Matrix(Diagonal(fill(nu / 10.0, dy)))
            ),
            :gamma_obs => GammaShapeRate(1.0, 1.0),
        )

        function low_rank_initialization()
            return @initialization begin
                q(a_map) = priors[:a_map]
                q(a_pred) = priors[:a_pred]
                q(theta) = priors[:theta]
                q(P) = priors[:P]
                q(Gamma2) = priors[:Gamma2]
                q(gamma_obs) = priors[:gamma_obs]
                q(h1) = MvNormalMeanCovariance(zeros(dy), Diagonal(ones(dy)))
                q(s) = MvNormalMeanCovariance(
                    fill(log(2.0), dy), Diagonal(fill(0.04, dy))
                )
                q(h2) = MvNormalMeanCovariance(zeros(dy), Diagonal(ones(dy)))
            end
        end

        dependencies = NGMPDependencies(
            out = nothing,
            in = nothing,
            projection = TangentProjection(type = Unscented),
        )
        function run_low_rank_integration(meta_map, meta_pred)
            return infer(
                model = linear_low_rank_ct_toy(
                    priors = priors,
                    feature_cov = Matrix(Diagonal(fill(1e-4, dx))),
                    meta_map = meta_map,
                    meta_pred = meta_pred,
                    sp_deps = dependencies,
                    sp_damping = DampingMeta(alpha = 0.2, beta = 0.0, max_step = 1.0),
                ),
                data = (y = y, features = features),
                constraints = linear_low_rank_ct_toy_constraints(),
                initialization = low_rank_initialization(),
                iterations = 2,
                free_energy = true,
                showprogress = false,
                options = (limit_stack_depth = 100,),
            )
        end

        zero_map = zeros(dy, dx)
        zero_pred = zeros(dy, dy)
        generic = run_low_rank_integration(
            CTMeta(a -> zero_map + U_map * Diagonal(a) * V_map'),
            CTMeta(a -> zero_pred + U_pred * Diagonal(a) * V_pred'),
        )
        specialized = run_low_rank_integration(
            LinearLowRankMeta(U_map, V_map), LinearLowRankMeta(U_pred, V_pred)
        )

        @test generic.free_energy ≈ specialized.free_energy atol = 1e-8 rtol = 1e-8
        @test keys(generic.posteriors) == keys(specialized.posteriors)

        function test_low_rank_posterior_tree(left, right)
            if left isa AbstractArray
                @test size(left) == size(right)
                for index in eachindex(left, right)
                    test_low_rank_posterior_tree(left[index], right[index])
                end
            else
                @test mean(left) ≈ mean(right) atol = 1e-8 rtol = 1e-8
                left_second = try
                    cov(left)
                catch
                    var(left)
                end
                right_second = try
                    cov(right)
                catch
                    var(right)
                end
                @test left_second ≈ right_second atol = 1e-8 rtol = 1e-8
            end
        end
        for key in keys(generic.posteriors)
            test_low_rank_posterior_tree(
                generic.posteriors[key], specialized.posteriors[key]
            )
        end
    end
end
