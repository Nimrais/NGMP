import LinearAlgebra: Diagonal, I, Symmetric, norm
import ReactiveMP: @call_marginalrule

@model function linear_reshape_ct_toy(
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

@constraints function linear_reshape_ct_toy_constraints()
    q(x_f, h1, s, h2, a_map, a_pred, P, Gamma2, theta, gamma_obs) =
        q(x_f, h1)q(s, h2)q(a_map)q(a_pred)q(P)q(Gamma2)q(theta)q(gamma_obs)
end

function _linear_reshape_spd(rng, dimension; ridge = 0.5)
    factor = randn(rng, dimension, dimension)
    return Matrix(Symmetric(factor * factor' + ridge * I))
end

function _test_same_gaussian(left, right; atol = 1e-10)
    @test weightedmean(left) ≈ weightedmean(right) atol = atol rtol = atol
    @test precision(left) ≈ precision(right) atol = atol rtol = atol
end

function _test_same_wishart(left, right; atol = 1e-10)
    # WishartFast stores the natural precision parameter directly as invS.
    @test left.ν ≈ right.ν atol = atol rtol = atol
    @test left.invS ≈ right.invS atol = atol rtol = atol
    @test mean(left) ≈ mean(right) atol = atol rtol = atol
end

function _ct_average_energy(meta, names, distributions)
    marginals = map(distribution -> Marginal(distribution, false, false), distributions)
    return score(AverageEnergy(), ContinuousTransition, names, marginals, meta)
end

@testset "LinearReshapeMeta" begin
    @test_throws ArgumentError LinearReshapeMeta(0, 2)
    @test_throws ArgumentError LinearReshapeMeta(2, 0)

    @testset "all reshape rules match CTMeta for dy=$dy, dx=$dx" for (dy, dx) in
                                                                        ((1, 3), (2, 3), (3, 2), (2, 2))
        rng = MersenneTwister(1_000 + 10dy + dx)
        da = dy * dx
        generic_meta = CTMeta(a -> reshape(a, dy, dx))
        reshape_meta = LinearReshapeMeta(dy, dx)

        ma = randn(rng, da)
        Va = _linear_reshape_spd(rng, da; ridge = 0.8)
        q_a = MvNormalMeanCovariance(ma, Va)

        my, mx = randn(rng, dy), randn(rng, dx)
        Vy = _linear_reshape_spd(rng, dy; ridge = 0.9)
        Vx = _linear_reshape_spd(rng, dx; ridge = 0.7)
        q_y = MvNormalMeanCovariance(my, Vy)
        q_x = MvNormalMeanCovariance(mx, Vx)

        # A full random joint covariance guarantees nonzero y/x cross-covariance.
        Vjoint = _linear_reshape_spd(rng, dy + dx; ridge = 1.1)
        q_y_x = MvNormalMeanCovariance([my; mx], Vjoint)
        @test !iszero(norm(@view Vjoint[1:dy, (dy + 1):end]))

        m_y = MvNormalWeightedMeanPrecision(
            randn(rng, dy), _linear_reshape_spd(rng, dy; ridge = 1.2)
        )
        m_x = MvNormalWeightedMeanPrecision(
            randn(rng, dx), _linear_reshape_spd(rng, dx; ridge = 1.3)
        )
        W = _linear_reshape_spd(rng, dy; ridge = 1.4)
        q_W = PointMass(W)

        generic_y_structured = @call_rule ContinuousTransition(:y, Marginalisation) (
            m_x = m_x, q_a = q_a, q_W = q_W, meta = generic_meta
        )
        reshape_y_structured = @call_rule ContinuousTransition(:y, Marginalisation) (
            m_x = m_x, q_a = q_a, q_W = q_W, meta = reshape_meta
        )
        _test_same_gaussian(generic_y_structured, reshape_y_structured)

        generic_y_mean_field = @call_rule ContinuousTransition(:y, Marginalisation) (
            q_x = q_x, q_a = q_a, q_W = q_W, meta = generic_meta
        )
        reshape_y_mean_field = @call_rule ContinuousTransition(:y, Marginalisation) (
            q_x = q_x, q_a = q_a, q_W = q_W, meta = reshape_meta
        )
        _test_same_gaussian(generic_y_mean_field, reshape_y_mean_field)

        generic_x_structured = @call_rule ContinuousTransition(:x, Marginalisation) (
            m_y = m_y, q_a = q_a, q_W = q_W, meta = generic_meta
        )
        reshape_x_structured = @call_rule ContinuousTransition(:x, Marginalisation) (
            m_y = m_y, q_a = q_a, q_W = q_W, meta = reshape_meta
        )
        _test_same_gaussian(generic_x_structured, reshape_x_structured)

        generic_x_mean_field = @call_rule ContinuousTransition(:x, Marginalisation) (
            q_y = q_y, q_a = q_a, q_W = q_W, meta = generic_meta
        )
        reshape_x_mean_field = @call_rule ContinuousTransition(:x, Marginalisation) (
            q_y = q_y, q_a = q_a, q_W = q_W, meta = reshape_meta
        )
        _test_same_gaussian(generic_x_mean_field, reshape_x_mean_field)

        generic_a_structured = @call_rule ContinuousTransition(:a, Marginalisation) (
            q_y_x = q_y_x, q_a = q_a, q_W = q_W, meta = generic_meta
        )
        reshape_a_structured = @call_rule ContinuousTransition(:a, Marginalisation) (
            q_y_x = q_y_x, q_a = q_a, q_W = q_W, meta = reshape_meta
        )
        _test_same_gaussian(generic_a_structured, reshape_a_structured)

        generic_a_mean_field = @call_rule ContinuousTransition(:a, Marginalisation) (
            q_y = q_y, q_x = q_x, q_a = q_a, q_W = q_W, meta = generic_meta
        )
        reshape_a_mean_field = @call_rule ContinuousTransition(:a, Marginalisation) (
            q_y = q_y, q_x = q_x, q_a = q_a, q_W = q_W, meta = reshape_meta
        )
        _test_same_gaussian(generic_a_mean_field, reshape_a_mean_field)

        generic_W_structured = @call_rule ContinuousTransition(:W, Marginalisation) (
            q_y_x = q_y_x, q_a = q_a, meta = generic_meta
        )
        reshape_W_structured = @call_rule ContinuousTransition(:W, Marginalisation) (
            q_y_x = q_y_x, q_a = q_a, meta = reshape_meta
        )
        _test_same_wishart(generic_W_structured, reshape_W_structured)

        generic_W_mean_field = @call_rule ContinuousTransition(:W, Marginalisation) (
            q_y = q_y, q_x = q_x, q_a = q_a, meta = generic_meta
        )
        reshape_W_mean_field = @call_rule ContinuousTransition(:W, Marginalisation) (
            q_y = q_y, q_x = q_x, q_a = q_a, meta = reshape_meta
        )
        _test_same_wishart(generic_W_mean_field, reshape_W_mean_field)

        generic_joint = @call_marginalrule ContinuousTransition(:y_x) (
            m_y = m_y, m_x = m_x, q_a = q_a, q_W = q_W, meta = generic_meta
        )
        reshape_joint = @call_marginalrule ContinuousTransition(:y_x) (
            m_y = m_y, m_x = m_x, q_a = q_a, q_W = q_W, meta = reshape_meta
        )
        _test_same_gaussian(generic_joint, reshape_joint)

        generic_structured_energy = _ct_average_energy(
            generic_meta, Val{(:y_x, :a, :W)}(), (q_y_x, q_a, q_W)
        )
        reshape_structured_energy = _ct_average_energy(
            reshape_meta, Val{(:y_x, :a, :W)}(), (q_y_x, q_a, q_W)
        )
        @test generic_structured_energy ≈ reshape_structured_energy atol = 1e-10 rtol = 1e-10

        generic_mean_field_energy = _ct_average_energy(
            generic_meta, Val{(:y, :x, :a, :W)}(), (q_y, q_x, q_a, q_W)
        )
        reshape_mean_field_energy = _ct_average_energy(
            reshape_meta, Val{(:y, :x, :a, :W)}(), (q_y, q_x, q_a, q_W)
        )
        @test generic_mean_field_energy ≈ reshape_mean_field_energy atol = 1e-10 rtol = 1e-10
    end

    @testset "parameter length validation" begin
        meta = LinearReshapeMeta(2, 3)
        q_a = MvNormalMeanCovariance(zeros(5), Matrix(Diagonal(ones(5))))
        q_x = MvNormalMeanCovariance(zeros(3), Matrix(Diagonal(ones(3))))
        q_W = PointMass(Matrix{Float64}(I, 2, 2))
        @test_throws DimensionMismatch @call_rule ContinuousTransition(:y, Marginalisation) (
            q_x = q_x, q_a = q_a, q_W = q_W, meta = meta
        )
    end

    @testset "two-iteration CT/MvSoftplus integration equivalence" begin
        d_h, d_f = 2, 3
        rng = MersenneTwister(8_101)
        features = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
        y = [0.02, 0.96, 1.01, -0.01]
        nu = d_h + 2.0
        priors = Dict{Symbol, Any}(
            :a_map => MvNormalMeanCovariance(
                0.5 .* randn(rng, d_h * d_f), Diagonal(ones(d_h * d_f))
            ),
            :a_pred => MvNormalMeanCovariance(
                0.5 .* randn(rng, d_h * d_h), Diagonal(ones(d_h * d_h))
            ),
            :theta => MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h))),
            :P => ExponentialFamily.WishartFast(
                nu, Matrix(Diagonal(fill(nu / 10.0, d_h)))
            ),
            :Gamma2 => ExponentialFamily.WishartFast(
                nu, Matrix(Diagonal(fill(nu / 10.0, d_h)))
            ),
            :gamma_obs => GammaShapeRate(1.0, 1.0),
        )

        function run_with_meta(meta_map, meta_pred)
            initialization = @initialization begin
                q(a_map) = priors[:a_map]
                q(a_pred) = priors[:a_pred]
                q(theta) = priors[:theta]
                q(P) = priors[:P]
                q(Gamma2) = priors[:Gamma2]
                q(gamma_obs) = priors[:gamma_obs]
                q(h1) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
                q(s) = MvNormalMeanCovariance(
                    fill(log(2.0), d_h), Diagonal(fill(0.04, d_h))
                )
                q(h2) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
            end
            dependencies = NGMPDependencies(
                out = nothing,
                in = nothing,
                projection = TangentProjection(type = Unscented),
            )
            return infer(
                model = linear_reshape_ct_toy(
                    priors = priors,
                    feature_cov = Matrix(Diagonal(fill(1e-4, d_f))),
                    meta_map = meta_map,
                    meta_pred = meta_pred,
                    sp_deps = dependencies,
                    sp_damping = DampingMeta(alpha = 0.2, beta = 0.0, max_step = 1.0),
                ),
                data = (y = y, features = features),
                constraints = linear_reshape_ct_toy_constraints(),
                initialization = initialization,
                iterations = 2,
                free_energy = true,
                showprogress = false,
                options = (limit_stack_depth = 100,),
            )
        end

        generic = run_with_meta(
            CTMeta(a -> reshape(a, d_h, d_f)),
            CTMeta(a -> reshape(a, d_h, d_h)),
        )
        specialized = run_with_meta(
            LinearReshapeMeta(d_h, d_f), LinearReshapeMeta(d_h, d_h)
        )

        @test generic.free_energy ≈ specialized.free_energy atol = 1e-8 rtol = 1e-8
        @test keys(generic.posteriors) == keys(specialized.posteriors)

        function test_posterior_tree(left, right)
            if left isa AbstractArray
                @test size(left) == size(right)
                for index in eachindex(left, right)
                    test_posterior_tree(left[index], right[index])
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
            test_posterior_tree(generic.posteriors[key], specialized.posteriors[key])
        end
    end
end
