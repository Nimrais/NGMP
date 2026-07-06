import SurrogateModelling: UnscentedTransforms, NormalPrecisionMessage, project_to_gamma
import SurrogateModelling.UnscentedTransforms: gaussian_sigma_points, moment_matched_sigma_points, ut_parameters, UnscentedTransform

@testset "unscented-transform tangent projection" begin
    @testset "sigma points and weights" begin
        # classical scaled UT, GH(3)-equivalent defaults: m ± √3σ, weights (2/3, 1/6, 1/6)
        m, v = 0.3, 0.8
        x, wm, wc = gaussian_sigma_points(1.0, 0.0, 2.0, m, v)
        @test collect(x) ≈ [m, m + sqrt(3v), m - sqrt(3v)]
        @test collect(wm) ≈ [2 / 3, 1 / 6, 1 / 6]
        @test collect(wc) ≈ [2 / 3, 1 / 6, 1 / 6]
        @test sum(wm) ≈ 1 && sum(wc) ≈ 1

        # generalized UT with Gaussian moments (μ3 = 0, μ4 = 3v²) degenerates to the same rule
        xg, wg = moment_matched_sigma_points(m, v, 0.0, 3 * v^2)
        @test sort(collect(xg)) ≈ sort(collect(x))
        @test sum(wg) ≈ 1
        @test collect(wg) ≈ [2 / 3, 1 / 6, 1 / 6]

        # generalized UT reproduces the matched moments exactly (skewed case)
        μ3, μ4 = 0.5, 3 * v^2 + 0.4
        xs, ws = moment_matched_sigma_points(m, v, μ3, μ4)
        @test sum(ws) ≈ 1
        @test sum(ws .* xs) ≈ m
        @test sum(ws .* (xs .- m) .^ 2) ≈ v
        @test sum(ws .* (xs .- m) .^ 3) ≈ μ3
        @test sum(ws .* (xs .- m) .^ 4) ≈ μ4

        # parameter recovery from the type domain (bare type = GH(3) defaults)
        @test ut_parameters(UnscentedTransform) == (1.0, 0.0, 2.0)
        @test ut_parameters(UnscentedTransform{0.5, 1.0, 3.0}) == (0.5, 1.0, 3.0)
    end

    @testset "ReactiveMP.Unscented bridge into TangentProjection" begin
        @test TangentProjection(type = Unscented) === TangentProjection{UnscentedTransform}()
        @test TangentProjection(type = Unscented(alpha = 1.0, beta = 0.0, kappa = 2.0)) ===
              TangentProjection{UnscentedTransform{1.0, 0.0, 2.0}}()
    end

    @testset "Gaussian edge: UT(1,0,2) coincides with 3-point Gauss-Hermite" begin
        q = NormalMeanVariance(0.4, 0.7)
        f = Logpdf(LogGamma(2.0, 3.0; check_args = false))
        ηut = getnaturalparameters(project(TangentProjection(type = Unscented), q, f))
        ηq3 = getnaturalparameters(project(TangentProjection(type = Quadrature(3)), q, f))
        @test ηut ≈ ηq3 atol = 1e-12
        # and stays within striking distance of the exact closed form
        ηcf = getnaturalparameters(project(TangentProjection(type = ClosedForm), q, f))
        @test maximum(abs.(ηut .- ηcf)) < 0.1
    end

    @testset "Gamma edge: log-space GenUT is the middle ground (beats delta at every width)" begin
        p = Logpdf(NormalPrecisionMessage(1.3, 0.4, 0.6))
        for (a, b) in ((1.5, 2.0), (3.5, 2.0), (10.0, 5.0), (200.0, 100.0))
            qg = GammaShapeRate(a, b)
            ref = getnaturalparameters(project(TangentProjection(type = Quadrature(4096)), qg, p))
            ηut = getnaturalparameters(project(TangentProjection(type = Unscented), qg, p))
            Δa, Δb = project_to_gamma(p.dist, convert(ExponentialFamilyDistribution, qg))
            ηdelta = [Δa, -Δb]
            ut_err = maximum(abs.(ηut .- ref) ./ abs.(ref))
            delta_err = maximum(abs.(ηdelta .- ref) ./ abs.(ref))
            @test ut_err < delta_err            # strictly closer to the quadrature truth
            @test ut_err < 0.1                  # and accurate in absolute terms
        end
    end

    @testset "rules pick the strategy from the NaturalGradientMessage" begin
        kw = (m_μ = NormalMeanVariance(0.4, 0.6), q_out = PointMass(1.3), q_τ = GammaShapeRate(1.5, 2.0))
        st() = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        r_ut = @call_rule NormalMeanPrecision(:τ, NaturalGradientMessage(TangentProjection(type = Unscented))) (
            m_μ = kw.m_μ, q_out = kw.q_out, q_τ = kw.q_τ, meta = st()
        )
        η = getnaturalparameters(project(
            TangentProjection(type = Unscented), kw.q_τ,
            Logpdf(NormalPrecisionMessage(mean(kw.q_out), mean(kw.m_μ), var(kw.m_μ)))
        ))
        @test shape(r_ut) ≈ η[1] + 1
        @test rate(r_ut) ≈ -η[2]

        # DeltaApproximation = the analytic-derivative second-order projection
        r_delta = @call_rule NormalMeanPrecision(:τ, NaturalGradientMessage(TangentProjection(type = DeltaApproximation))) (
            m_μ = kw.m_μ, q_out = kw.q_out, q_τ = kw.q_τ, meta = st()
        )
        Δa, Δb = project_to_gamma(
            NormalPrecisionMessage(mean(kw.q_out), mean(kw.m_μ), var(kw.m_μ)),
            convert(ExponentialFamilyDistribution, kw.q_τ)
        )
        @test shape(r_delta) ≈ Δa + 1
        @test rate(r_delta) ≈ Δb
        @test !(shape(r_delta) ≈ shape(r_ut))   # the strategies genuinely differ here

        # the DEFAULT (ClosedForm) means an exact Williams product, which these
        # messages do not have — it must error informatively, not silently degrade
        @test_throws ErrorException @call_rule NormalMeanPrecision(:τ, NaturalGradientMessage) (
            m_μ = kw.m_μ, q_out = kw.q_out, q_τ = kw.q_τ, meta = st()
        )
    end
end
