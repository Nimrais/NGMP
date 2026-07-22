import LinearAlgebra
import LinearAlgebra: Diagonal, dot, diagm
import ReactiveMP: @call_marginalrule
import SurrogateModelling:
    NaturalGradientMP,
    MvSoftplusForwardMessage,
    MvSoftplusGaussianBackwardMessage,
    SoftplusForwardMessage,
    SoftplusGaussianBackwardMessage,
    _softplus

struct MvQuadraticLogMessage <: ClosedFormExpectations.Expression
    xi::Vector{Float64}
    Lambda::Matrix{Float64}
end

Base.log(p::MvQuadraticLogMessage, x::AbstractVector) =
    dot(p.xi, x) - dot(x, p.Lambda * x) / 2

@model function mv_softplus_ct_toy(y, features, priors, feature_cov, meta_map, meta_pred, sp_deps, sp_damping)
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

@constraints function mv_softplus_ct_toy_constraints()
    q(x_f, h1, s, h2, a_map, a_pred, P, Gamma2, theta, gamma_obs) =
        q(x_f, h1)q(s, h2)q(a_map)q(a_pred)q(P)q(Gamma2)q(theta)q(gamma_obs)
end

@testset "MvSoftplus deterministic NGMP node" begin
    @testset "exact log-message expressions" begin
        # Diagonal covariance: the forward pushforward factorizes over elements.
        m = [0.3, -0.5]
        vs = [0.8, 1.4]
        forward = MvSoftplusForwardMessage(m, Diagonal(vs))
        for y in ([0.4, 0.9], [1e-6, 2.0], [3.0, 0.05])
            expected = sum(log(SoftplusForwardMessage(m[k], vs[k]), y[k]) for k in 1:2)
            @test log(forward, y) ≈ expected atol = 1e-12
        end
        @test log(forward, [0.5, 0.0]) == -Inf
        @test log(forward, [-0.1, 0.5]) == -Inf

        # Backward info form drops the Gaussian normalizer: compare log
        # DIFFERENCES against the scalar closed form for d = 1.
        mean_out, var_out = 0.8, 0.4
        scalar = SoftplusGaussianBackwardMessage(mean_out, var_out)
        mv = MvSoftplusGaussianBackwardMessage([mean_out / var_out], fill(1 / var_out, 1, 1))
        for (x1, x2) in ((-2.0, 0.5), (0.0, 3.0), (1.2, -0.7))
            @test log(mv, [x1]) - log(mv, [x2]) ≈ log(scalar, x1) - log(scalar, x2) atol = 1e-12
        end
    end

    @testset "d = 1 projection matches the scalar sigma-point rule" begin
        m, v = 0.4, 0.7
        mean_out, var_out = 1.3, 0.5
        scalar_site = project(
            TangentProjection(type = Unscented),
            NormalMeanVariance(m, v),
            Logpdf(SoftplusGaussianBackwardMessage(mean_out, var_out)),
        )
        mv_site = project(
            TangentProjection(type = Unscented),
            MvNormalMeanCovariance([m], fill(v, 1, 1)),
            Logpdf(MvSoftplusGaussianBackwardMessage([mean_out / var_out], fill(1 / var_out, 1, 1))),
        )
        ηs = getnaturalparameters(scalar_site)
        ηmv = getnaturalparameters(mv_site)
        @test mv_site isa ExponentialFamilyDistribution{MvNormalMeanCovariance}
        @test ηmv[1] ≈ ηs[1] atol = 1e-12
        @test ηmv[2] ≈ ηs[2] atol = 1e-12
    end

    @testset "quadratic log-messages are recovered exactly (cross terms included)" begin
        xi = [0.7, -0.3]
        Lambda = [1.2 0.3; 0.3 0.9]
        q = MvNormalMeanCovariance([0.1, -0.2], [0.5 0.1; 0.1 0.8])
        site = project(TangentProjection(type = Unscented), q, Logpdf(MvQuadraticLogMessage(xi, Lambda)))
        η = getnaturalparameters(site)
        @test η[1:2] ≈ xi atol = 1e-10
        @test -2 .* reshape(η[3:end], 2, 2) ≈ Lambda atol = 1e-10

        @test_throws Exception project(
            TangentProjection(type = ClosedForm), q, Logpdf(MvQuadraticLogMessage(xi, Lambda))
        )
        @test_throws Exception project(
            TangentProjection(type = DeltaApproximation), q, Logpdf(MvQuadraticLogMessage(xi, Lambda))
        )
    end

    @testset "MvNormal damping bridges round-trip improper sites" begin
        proper = MvNormalWeightedMeanPrecision([0.5, -1.0], [2.0 0.3; 0.3 1.5])
        improper = MvNormalWeightedMeanPrecision([0.5, -1.0], [-0.4 0.1; 0.1 0.2])
        for message in (proper, improper)
            T, η = NaturalGradientMP.natural_parameters(message)
            @test T === MvNormalMeanCovariance
            back = NaturalGradientMP.from_natural(T, η)
            @test back isa MvNormalWeightedMeanPrecision
            @test weightedmean(back) ≈ weightedmean(message) atol = 1e-12
            @test precision(back) ≈ precision(message) atol = 1e-12
        end

        # First fire of the damped update sends α · η_target.
        α = 0.2
        state = NGMPEdgeState(DampingMeta(alpha = α, beta = 0.0, max_step = 1e6))
        sent = NaturalGradientMP.apply_damping!(state, proper)
        @test sent isa MvNormalWeightedMeanPrecision
        @test weightedmean(sent) ≈ α .* weightedmean(proper) atol = 1e-12
        @test precision(sent) ≈ α .* precision(proper) atol = 1e-12
        @test state.nfired == 1
    end

    @testset "NGMP rules fire in both directions" begin
        m_in = MvNormalMeanCovariance([0.2, -0.4], [0.6 0.1; 0.1 0.9])
        q_out = MvNormalMeanCovariance(fill(log(2.0), 2), Diagonal(fill(0.04, 2)))
        state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        forward = @call_rule MvSoftplus(
            :out,
            NaturalGradientMessage(TangentProjection(type = Unscented)),
        ) (m_in = m_in, q_out = q_out, meta = state)
        @test forward isa MvNormalWeightedMeanPrecision
        @test all(isfinite, weightedmean(forward))
        @test all(isfinite, precision(forward))
        @test state.nfired == 1

        m_out = MvNormalWeightedMeanPrecision([1.5, 0.8], [2.0 0.0; 0.0 1.2])
        q_in = MvNormalMeanCovariance([0.1, -0.3], Diagonal(fill(0.8, 2)))
        state_in = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        backward = @call_rule MvSoftplus(
            :in,
            NaturalGradientMessage(TangentProjection(type = Unscented)),
        ) (m_out = m_out, q_in = q_in, meta = state_in)
        @test backward isa MvNormalWeightedMeanPrecision
        @test all(isfinite, weightedmean(backward))
        @test all(isfinite, precision(backward))
        @test state_in.nfired == 1
    end

    @testset "delta forward projection uses q(out)" begin
        m_in = MvNormalMeanCovariance([0.2, -0.4], [0.6 0.1; 0.1 0.9])
        q_out_near = MvNormalMeanCovariance([0.7, 0.8], Diagonal(fill(0.04, 2)))
        q_out_far = MvNormalMeanCovariance([1.1, 1.3], Diagonal(fill(0.04, 2)))

        forward_near = @call_rule MvSoftplus(
            :out,
            NaturalGradientMessage(TangentProjection(type = DeltaApproximation)),
        ) (m_in = m_in, q_out = q_out_near, meta = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0)))
        forward_far = @call_rule MvSoftplus(
            :out,
            NaturalGradientMessage(TangentProjection(type = DeltaApproximation)),
        ) (m_in = m_in, q_out = q_out_far, meta = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0)))

        _, η_near = NaturalGradientMP.natural_parameters(forward_near)
        _, η_far = NaturalGradientMP.natural_parameters(forward_far)
        @test all(isfinite, η_near)
        @test all(isfinite, η_far)
        @test !isapprox(η_near, η_far; atol = 1e-8, rtol = 1e-8)

        q_out_invalid = MvNormalMeanCovariance([0.7, -0.1], Diagonal(fill(0.04, 2)))
        @test_throws DomainError @call_rule MvSoftplus(
            :out,
            NaturalGradientMessage(TangentProjection(type = DeltaApproximation)),
        ) (m_in = m_in, q_out = q_out_invalid, meta = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0)))
    end

    @testset "scoring marginal is proper with finite entropy" begin
        for m_out in (
            MvNormalWeightedMeanPrecision([1.5, 0.8], [2.0 0.0; 0.0 1.2]),
            # A wide, badly-placed cavity that drives the projected site improper.
            MvNormalWeightedMeanPrecision([-40.0, -40.0], [4.0 0.0; 0.0 4.0]),
        )
            local_marginal = @call_marginalrule MvSoftplus(:in) (
                m_out = m_out,
                m_in = MvNormalMeanCovariance([0.1, -0.3], Diagonal(fill(0.8, 2))),
                meta = DampingMeta(alpha = 0.2, beta = 0.0),
            )
            @test local_marginal isa MvNormalWeightedMeanPrecision
            @test isfinite(entropy(local_marginal))
        end
    end

    @testset "smoke integration: CTransition ∘ MvSoftplus ∘ CTransition ∘ softdot" begin
        d_h = 2
        d_f = 3
        rng = Random.MersenneTwister(7)
        raw = [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.2, 0.8], [1.0, 0.9, 0.1]]
        y = [Float64(xor(f[2] > 0.5, f[3] > 0.5)) + 0.05 * randn(rng) for f in raw]
        ν = d_h + 2.0
        priors = Dict{Symbol, Any}(
            :a_map => MvNormalMeanCovariance(0.5 .* randn(rng, d_h * d_f), Diagonal(ones(d_h * d_f))),
            :a_pred => MvNormalMeanCovariance(0.5 .* randn(rng, d_h * d_h), Diagonal(ones(d_h * d_h))),
            :theta => MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h))),
            # WishartFast(ν, invS): mean = ν·inv(invS) = 10·I. Plain Wishart breaks
            # the free-energy scorer (no closed-form Wishart-Wishart KL in Distributions).
            :P => ExponentialFamily.WishartFast(ν, Matrix(Diagonal(fill(ν / 10.0, d_h)))),
            :Gamma2 => ExponentialFamily.WishartFast(ν, Matrix(Diagonal(fill(ν / 10.0, d_h)))),
            :gamma_obs => GammaShapeRate(1.0, 1.0),
        )
        initialization = @initialization begin
            q(a_map) = priors[:a_map]
            q(a_pred) = priors[:a_pred]
            q(theta) = priors[:theta]
            q(P) = priors[:P]
            q(Gamma2) = priors[:Gamma2]
            q(gamma_obs) = priors[:gamma_obs]
            q(h1) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
            q(s) = MvNormalMeanCovariance(fill(log(2.0), d_h), Diagonal(fill(0.04, d_h)))
            q(h2) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        end
        iterations = 3
        sp_deps = NGMPDependencies(
            out = nothing,
            in = nothing,
            projection = TangentProjection(type = Unscented),
        )
        result = infer(
            model = mv_softplus_ct_toy(
                priors = priors,
                feature_cov = Matrix(Diagonal(fill(1e-4, d_f))),
                meta_map = CTMeta(a -> reshape(a, d_h, d_f)),
                meta_pred = CTMeta(a -> reshape(a, d_h, d_h)),
                sp_deps = sp_deps,
                sp_damping = DampingMeta(alpha = 0.2, beta = 0.0, max_step = 1.0),
            ),
            data = (y = y, features = raw),
            constraints = mv_softplus_ct_toy_constraints(),
            initialization = initialization,
            iterations = iterations,
            free_energy = true,
            options = (limit_stack_depth = 100,),
        )

        @test all(isfinite, result.free_energy)
        @test all(isfinite, mean(last(result.posteriors[:a_map])))
        @test all(isfinite, mean(last(result.posteriors[:a_pred])))
        @test all(isfinite, mean(last(result.posteriors[:theta])))
        @test length(sp_deps.states) == 2 * length(y)
        @test all(state -> state.nfired == iterations, sp_deps.states)
    end
end
