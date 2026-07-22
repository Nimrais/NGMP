import LinearAlgebra
import LinearAlgebra: Diagonal, Symmetric, diag, dot
import ReactiveMP: @call_marginalrule
import ExponentialFamily:
    MvNormalMeanCovariance,
    MvNormalWeightedMeanPrecision,
    NaturalParametersSpace,
    getlogpartition,
    getfisherinformation,
    weightedmean_precision
import ExponentialFamily: WishartFast
import SurrogateModelling:
    NaturalGradientMP,
    MvInverseSoftplusNormal,
    MvSoftplusForwardMessage,
    MvSoftplusGaussianBackwardMessage,
    _softplus,
    _inverse_softplus

# A y-space quadratic log-message that is NOT typed as a Gaussian, to exercise
# the generic SoftplusPullback wrapper path of the projection.
struct MvISNQuadraticInY <: ClosedFormExpectations.Expression
    xi::Vector{Float64}
    Lambda::Matrix{Float64}
end

Base.log(p::MvISNQuadraticInY, y::AbstractVector) =
    dot(p.xi, y) - dot(y, p.Lambda * y) / 2

@model function mvisn_ct_toy(y, features, priors, feature_cov, meta_map, meta_pred, ct2_deps, sp_deps, sp_damping)
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
        h2[i] ~ ContinuousTransition(s[i], a_pred, Gamma2) where {
            dependencies = ct2_deps, meta = meta_pred
        }
        y[i] ~ softdot(theta, h2[i], gamma_obs)
    end
end

# The exact arm: the s—h2 boundary is explicitly mean-field because q(s) lives
# in the positive-orthant pushforward family, not in the Gaussian joint.
@constraints function mvisn_ct_toy_constraints()
    q(x_f, h1, s, h2, a_map, a_pred, P, Gamma2, theta, gamma_obs) =
        q(x_f, h1)q(s)q(h2)q(a_map)q(a_pred)q(P)q(Gamma2)q(theta)q(gamma_obs)
end

@testset "MvInverseSoftplusNormal distribution" begin
    rng = Random.MersenneTwister(2026)
    μ = [0.3, -0.5, 1.1]
    A = randn(rng, 3, 3)
    Σ = Matrix(Symmetric(A * A' ./ 3 .+ 0.5 .* LinearAlgebra.I(3)))
    dist = MvInverseSoftplusNormal(μ, Σ)

    @testset "logpdf is the exact softplus pushforward" begin
        forward = MvSoftplusForwardMessage(μ, Σ)
        for y in ([0.4, 0.9, 1.3], [1e-6, 2.0, 0.7], [3.0, 0.05, 0.2])
            @test logpdf(dist, y) ≈ log(forward, y) atol = 1e-12
        end
        @test logpdf(dist, [0.5, -0.1, 1.0]) == -Inf
        @test logpdf(dist, [0.5, 0.0, 1.0]) == -Inf
        @test Distributions.median(dist) ≈ _softplus.(μ) atol = 1e-12
        @test insupport(dist, [0.1, 0.1, 0.1])
        @test !insupport(dist, [0.1, 0.1])
    end

    @testset "EF form shares the Gaussian natural structure" begin
        ef = convert(ExponentialFamilyDistribution, dist)
        @test ef isa ExponentialFamilyDistribution{MvInverseSoftplusNormal}
        η = getnaturalparameters(ef)
        gaussian_η = getnaturalparameters(
            convert(ExponentialFamilyDistribution, MvNormalMeanCovariance(μ, Σ)),
        )
        @test η ≈ gaussian_η atol = 1e-10

        back = convert(Distribution, ef)
        @test back isa MvInverseSoftplusNormal
        @test back.μ ≈ μ atol = 1e-10
        @test back.Σ ≈ Σ atol = 1e-10

        y = [0.4, 0.9, 1.3]
        @test logpdf(ef, y) ≈ logpdf(dist, y) atol = 1e-10

        # Log-partition and Fisher are delegated, never re-derived.
        @test getlogpartition(NaturalParametersSpace(), MvInverseSoftplusNormal)(η) ≈
              getlogpartition(NaturalParametersSpace(), MvNormalMeanCovariance)(η)
        @test getfisherinformation(NaturalParametersSpace(), MvInverseSoftplusNormal)(η) ≈
              getfisherinformation(NaturalParametersSpace(), MvNormalMeanCovariance)(η)
    end

    @testset "cubature moments and entropy match Monte Carlo" begin
        n = 200_000
        draws = rand(rng, dist, n)
        m, C = BayesBase.mean_cov(dist)
        @test m ≈ vec(mean(draws, dims = 2)) atol = 2e-2
        @test C ≈ cov(draws') rtol = 5e-2
        @test mean(dist) ≈ m
        @test var(dist) ≈ diag(C)
        mc_entropy = -mean(logpdf(dist, draws[:, k]) for k in 1:50_000)
        @test entropy(dist) ≈ mc_entropy atol = 2e-2
        # EF form computes the same moments straight from the naturals.
        ef = convert(ExponentialFamilyDistribution, dist)
        @test mean(ef) ≈ m atol = 1e-10
        @test cov(ef) ≈ C atol = 1e-10
        @test entropy(ef) ≈ entropy(dist) atol = 1e-10
    end

    @testset "products add natural parameters" begin
        ef = convert(ExponentialFamilyDistribution, dist)
        η = getnaturalparameters(ef)
        site1 = ExponentialFamilyDistribution(MvInverseSoftplusNormal, η, nothing, nothing)
        site2 = ExponentialFamilyDistribution(MvInverseSoftplusNormal, 0.5 .* η, nothing, nothing)
        p = prod(BayesBase.GenericProd(), site1, site2)
        @test p isa ExponentialFamilyDistribution{MvInverseSoftplusNormal}
        @test getnaturalparameters(p) ≈ 1.5 .* η atol = 1e-12

        other = MvInverseSoftplusNormal(fill(0.2, 3), Matrix(Diagonal(fill(0.7, 3))))
        pd = prod(BayesBase.GenericProd(), dist, other)
        @test pd isa MvInverseSoftplusNormal
        expected = getnaturalparameters(convert(ExponentialFamilyDistribution, dist)) .+
                   getnaturalparameters(convert(ExponentialFamilyDistribution, other))
        @test getnaturalparameters(convert(ExponentialFamilyDistribution, pd)) ≈ expected atol = 1e-10

        mixed = prod(BayesBase.GenericProd(), dist, site2)
        @test mixed isa ExponentialFamilyDistribution{MvInverseSoftplusNormal}
        @test getnaturalparameters(mixed) ≈ 1.5 .* η atol = 1e-10
    end
end

@testset "MvInverseSoftplusNormal tangent projections" begin
    μ = [0.1, -0.4]
    Σ = [0.5 0.1; 0.1 0.8]
    q = MvInverseSoftplusNormal(μ, Σ)
    ξ = [1.5, 0.8]
    Λ = [2.0 0.3; 0.3 1.2]
    msg = MvNormalWeightedMeanPrecision(ξ, Λ)

    @testset "projection ≡ latent Gaussian projection of the analytic pullback" begin
        for strategy in (
            TangentProjection(type = Unscented),
            TangentProjection(type = DeltaApproximation),
        )
            site = project(strategy, q, Logpdf(msg))
            latent = project(
                strategy,
                MvNormalMeanCovariance(μ, Σ),
                Logpdf(MvSoftplusGaussianBackwardMessage(ξ, Λ)),
            )
            @test site isa ExponentialFamilyDistribution{MvInverseSoftplusNormal}
            @test getnaturalparameters(site) ≈ getnaturalparameters(latent) atol = 1e-12

            # The EF-site form of the belief projects identically.
            site_ef = project(
                strategy,
                convert(ExponentialFamilyDistribution, q),
                Logpdf(msg),
            )
            @test getnaturalparameters(site_ef) ≈ getnaturalparameters(site) atol = 1e-10
        end
    end

    @testset "generic pullback wrapper matches the analytic specialization" begin
        generic = project(
            TangentProjection(type = Unscented),
            q,
            Logpdf(MvISNQuadraticInY(ξ, Λ)),
        )
        analytic = project(TangentProjection(type = Unscented), q, Logpdf(msg))
        @test getnaturalparameters(generic) ≈ getnaturalparameters(analytic) atol = 1e-10
    end

    @testset "unsupported strategies raise informative errors" begin
        @test_throws Exception project(TangentProjection(type = ClosedForm), q, Logpdf(msg))
        @test_throws Exception project(
            TangentProjection(type = DeltaApproximation), q, Logpdf(MvISNQuadraticInY(ξ, Λ))
        )
    end
end

@testset "MvInverseSoftplusNormal NGMP rules" begin
    m_in = MvNormalMeanCovariance([0.2, -0.4], [0.6 0.1; 0.1 0.9])
    ξ_in, Λ_in = weightedmean_precision(m_in)
    η_in = vcat(ξ_in, vec(-Λ_in ./ 2))
    q_out_a = MvInverseSoftplusNormal(zeros(2), Matrix(Diagonal(fill(0.16, 2))))
    q_out_b = MvInverseSoftplusNormal(fill(1.0, 2), Matrix(Diagonal(fill(0.5, 2))))

    @testset "forward :out is the exact in-family pushforward" begin
        sites = map((q_out_a, q_out_b)) do q_out
            @call_rule MvSoftplus(
                :out,
                NaturalGradientMessage(TangentProjection(type = Unscented)),
            ) (m_in = m_in, q_out = q_out, meta = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0)))
        end
        for site in sites
            @test site isa ExponentialFamilyDistribution{MvInverseSoftplusNormal}
            @test getnaturalparameters(site) ≈ η_in atol = 1e-12
        end
        # Exactness: the site does not depend on the receiving belief.
        @test getnaturalparameters(sites[1]) ≈ getnaturalparameters(sites[2]) atol = 1e-12
    end

    @testset "backward :in is the exact Gaussian tilt with the same naturals" begin
        m_out = ExponentialFamilyDistribution(MvInverseSoftplusNormal, η_in, nothing, nothing)
        q_in = MvNormalMeanCovariance([0.1, -0.3], Diagonal(fill(0.8, 2)))
        site = @call_rule MvSoftplus(
            :in,
            NaturalGradientMessage(TangentProjection(type = Unscented)),
        ) (m_out = m_out, q_in = q_in, meta = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0)))
        T, η = NaturalGradientMP.natural_parameters(site)
        @test T === MvNormalMeanCovariance
        @test η ≈ η_in atol = 1e-12
    end

    @testset "scoring marginal combines cavity and softplus-normal tilt" begin
        m_out = ExponentialFamilyDistribution(MvInverseSoftplusNormal, η_in, nothing, nothing)
        m_in_cavity = MvNormalMeanCovariance([0.1, -0.3], Diagonal(fill(0.8, 2)))
        local_marginal = @call_marginalrule MvSoftplus(:in) (
            m_out = m_out, m_in = m_in_cavity, meta = nothing,
        )
        @test local_marginal isa MvNormalWeightedMeanPrecision
        ξ_cavity, Λ_cavity = weightedmean_precision(m_in_cavity)
        @test weightedmean(local_marginal) ≈ ξ_cavity .+ ξ_in atol = 1e-12
        @test precision(local_marginal) ≈ Matrix(Λ_cavity) .+ Λ_in atol = 1e-12
        @test isfinite(entropy(local_marginal))
    end

    @testset "ContinuousTransition :x projects its VMP target onto the edge" begin
        d = 2
        meta = LinearReshapeMeta(d, d)
        q_y = MvNormalMeanCovariance([0.4, -0.2], Diagonal(fill(0.3, d)))
        q_a = MvNormalMeanCovariance(fill(0.5, d * d), Diagonal(fill(0.1, d * d)))
        q_W = WishartFast(d + 2.0, Matrix(Diagonal(fill(1.0, d))))
        q_x = MvInverseSoftplusNormal(zeros(d), Matrix(Diagonal(fill(0.16, d))))

        stock = @call_rule ContinuousTransition(:x, Marginalisation) (
            q_y = q_y, q_a = q_a, q_W = q_W, meta = meta,
        )
        expected = project(TangentProjection(type = Unscented), q_x, Logpdf(stock))

        site = @call_rule ContinuousTransition(
            :x,
            NaturalGradientMessage(TangentProjection(type = Unscented)),
        ) (
            q_y = q_y, q_x = q_x, q_a = q_a, q_W = q_W,
            meta = NGMPEdgeState(meta; damping = DampingMeta(alpha = 1.0, beta = 0.0)),
        )
        @test site isa ExponentialFamilyDistribution{MvInverseSoftplusNormal}
        @test getnaturalparameters(site) ≈ getnaturalparameters(expected) atol = 1e-12
    end
end

@testset "smoke integration: exact MvInverseSoftplusNormal s-edge" begin
    d_h = 2
    d_f = 3
    rng = Random.MersenneTwister(11)
    raw = [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.2, 0.8], [1.0, 0.9, 0.1]]
    y = [Float64(xor(f[2] > 0.5, f[3] > 0.5)) + 0.05 * randn(rng) for f in raw]
    ν = d_h + 2.0
    priors = Dict{Symbol, Any}(
        :a_map => MvNormalMeanCovariance(0.5 .* randn(rng, d_h * d_f), Diagonal(ones(d_h * d_f))),
        :a_pred => MvNormalMeanCovariance(0.5 .* randn(rng, d_h * d_h), Diagonal(ones(d_h * d_h))),
        :theta => MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h))),
        :P => WishartFast(ν, Matrix(Diagonal(fill(ν / 10.0, d_h)))),
        :Gamma2 => WishartFast(ν, Matrix(Diagonal(fill(ν / 10.0, d_h)))),
        :gamma_obs => GammaShapeRate(1e4, 1.0),
    )
    damping = DampingMeta(alpha = 0.2, beta = 0.0, max_step = 1.0)
    sp_deps = NGMPDependencies(
        out = nothing, in = nothing,
        projection = TangentProjection(type = Unscented),
    )
    ct2_deps = NGMPDependencies(
        a = nothing, x = nothing,
        projection = TangentProjection(type = Unscented),
        damping = damping,
    )
    initialization = @initialization begin
        q(a_map) = priors[:a_map]
        q(a_pred) = priors[:a_pred]
        q(theta) = priors[:theta]
        q(P) = priors[:P]
        q(Gamma2) = priors[:Gamma2]
        q(gamma_obs) = priors[:gamma_obs]
        q(h1) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        q(s) = MvInverseSoftplusNormal(zeros(d_h), Matrix(Diagonal(fill(0.16, d_h))))
        q(h2) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
    end
    result = infer(
        model = mvisn_ct_toy(
            priors = priors,
            feature_cov = Matrix(Diagonal(fill(1e-4, d_f))),
            meta_map = LinearReshapeMeta(d_h, d_f),
            meta_pred = LinearReshapeMeta(d_h, d_h),
            ct2_deps = ct2_deps,
            sp_deps = sp_deps,
            sp_damping = damping,
        ),
        data = (y = y, features = raw),
        constraints = mvisn_ct_toy_constraints(),
        initialization = initialization,
        iterations = 5,
        free_energy = true,
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    @test all(isfinite, result.free_energy)
    q_s = last(result.posteriors[:s])
    @test all(marginal -> marginal isa Union{
        MvInverseSoftplusNormal,
        ExponentialFamilyDistribution{MvInverseSoftplusNormal},
    }, q_s)
    @test all(marginal -> all(isfinite, mean(marginal)) && all(>(0), mean(marginal)), q_s)
    @test all(marginal -> all(isfinite, mean(marginal)), last(result.posteriors[:h1]))
end
