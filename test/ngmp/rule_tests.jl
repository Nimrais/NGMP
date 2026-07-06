@testset "PoissonExp NGMP rules" begin
    @testset "NGMP rule matches the closed-form surrogate (alpha = 1, beta = 0)" begin
        for y in (0, 1, 7, 114), m in (-1.0, 0.0, 1.3), v in (0.1, 1.0, 4.0)
            state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
            msg = @call_rule PoissonExp(:in, NaturalGradientMessage) (
                q_out = PointMass(y), q_in = NormalMeanVariance(m, v), meta = state
            )
            xiref, Lambdaref = poisson_surrogate_reference(y, m, v)
            @test weightedmean(msg) ≈ xiref
            @test precision(msg) ≈ Lambdaref
            @test state.nfired == 1
        end
    end

    @testset "damping recursion matches manual heavy-ball updates" begin
        alpha, beta = 0.5, 0.2
        state = NGMPEdgeState(DampingMeta(alpha = alpha, beta = beta))
        xi, Lambda, vxi, vLambda = 0.0, 0.0, 0.0, 0.0
        y = 5
        marginals = [NormalMeanVariance(0.5, 2.0), NormalMeanVariance(1.1, 0.7), NormalMeanVariance(1.4, 0.3)]
        for (t, q) in enumerate(marginals)
            msg = @call_rule PoissonExp(:in, NaturalGradientMessage) (
                q_out = PointMass(y), q_in = q, meta = state
            )
            etaxi, etaLambda = poisson_surrogate_reference(y, mean(q), var(q))
            vxi = beta * vxi + alpha * (etaxi - xi)
            xi += vxi
            vLambda = beta * vLambda + alpha * (etaLambda - Lambda)
            Lambda += vLambda
            @test weightedmean(msg) ≈ xi
            @test precision(msg) ≈ Lambda
            @test state.nfired == t
        end
    end

    @testset "default damping parameters without user meta" begin
        @test NaturalGradientMP.damping_parameters(NGMPEdgeState(nothing)) == (0.5, 0.2)
        @test NaturalGradientMP.damping_parameters(NGMPEdgeState(DampingMeta(alpha = 0.7, beta = 0.1))) == (0.7, 0.1)
    end

    @testset "prediction rule" begin
        m, v = 1.3, 0.4
        pred = @call_rule PoissonExp(:out, Marginalisation) (q_in = NormalMeanVariance(m, v), meta = nothing)
        @test pred isa Poisson
        @test rate(pred) ≈ exp(m + v / 2)
    end

    @testset "plain marginal rule returns the exact non-Gaussian expression" begin
        expr = @call_rule PoissonExp(:in, Marginalisation) (q_out = PointMass(3),)
        @test expr isa PoissonExpression
        @test expr(0.7) ≈ 3 * 0.7 - exp(0.7) - loggamma(4)
    end

    @testset "average energy: closed form and quadrature cross-check" begin
        y, m, v = 4, 0.8, 0.6
        marginals = (Marginal(PointMass(y), false, false), Marginal(NormalMeanVariance(m, v), false, false))
        ae = ReactiveMP.score(ReactiveMP.AverageEnergy(), PoissonExp, Val{(:out, :in)}(), marginals, nothing)
        @test ae ≈ exp(m + v / 2) - y * m + loggamma(y + 1)

        zs = range(m - 12 * sqrt(v), m + 12 * sqrt(v); length = 40001)
        dz = step(zs)
        quad = sum(@. (exp(zs) - y * zs + loggamma(y + 1)) * exp(-(zs - m)^2 / (2v)) / sqrt(2pi * v)) * dz
        @test ae ≈ quad rtol = 1e-6
    end
end
