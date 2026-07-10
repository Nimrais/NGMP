import ProbabilisticEnsembling: Exp
import ReactiveMP: @call_marginalrule
import SurrogateModelling:
    NaturalGradientMP,
    SoftplusBackwardMessage,
    SoftplusForwardMessage,
    _inverse_softplus,
    _softplus

@model function softplus_toy(y, deps, damping)
    a ~ Normal(mean = 0.0, variance = 1.0)
    γ ~ GammaShapeRate(2.0, 2.0)
    γ ~ Softplus(a) where {dependencies = deps, meta = damping}
    for index in eachindex(y)
        y[index] ~ NormalMeanPrecision(0.0, γ)
    end
end

@testset "Softplus deterministic NGMP node" begin
    @testset "stable transform and exact log-messages" begin
        for x in (-50.0, -2.0, 0.0, 3.0, 50.0)
            y = _softplus(x)
            @test y >= 0
            @test _inverse_softplus(y) ≈ x atol = 1e-10
        end

        message = SoftplusForwardMessage(0.3, 0.8)
        for y in (1e-8, 0.1, 1.0, 20.0)
            x = _inverse_softplus(y)
            reference = logpdf(Normal(0.3, sqrt(0.8)), x) - log(-expm1(-y))
            @test log(message, y) ≈ reference atol = 1e-12
        end
        @test log(message, 0.0) == -Inf

        backward = SoftplusBackwardMessage(2.5, 1.3)
        for x in (-1000.0, -3.0, 0.0, 4.0)
            expected = (2.5 - 1) * (x < -37 ? x : log(_softplus(x))) - 1.3 * _softplus(x)
            @test log(backward, x) ≈ expected
            @test isfinite(log(backward, x))
        end
    end

    @testset "projection accuracy and rule families" begin
        forward_cases = (
            (-2.0, 1.5, 0.7, 1.2),
            (0.0, 1.0, 2.0, 1.0),
            (2.0, 0.5, 5.0, 2.0),
        )
        for (m, v, a, b) in forward_cases
            target = Logpdf(SoftplusForwardMessage(m, v))
            q = GammaShapeRate(a, b)
            η64 = getnaturalparameters(project(TangentProjection(type = Quadrature(64)), q, target))
            η512 = getnaturalparameters(project(TangentProjection(type = Quadrature(512)), q, target))
            normalized_error = maximum(abs.(η64 .- η512) ./ (1 .+ abs.(η512)))
            @test normalized_error < 1e-3

            state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
            message = @call_rule Softplus(
                :out,
                NaturalGradientMessage(TangentProjection(type = Unscented)),
            ) (m_in = NormalMeanVariance(m, v), q_out = q, meta = state)
            @test message isa GammaShapeRate
            @test all(isfinite, (shape(message), rate(message)))
            @test state.nfired == 1
        end

        backward_cases = (
            (-2.0, 1.5, 0.7, 1.2),
            (0.0, 1.0, 2.0, 1.0),
            (2.0, 0.5, 5.0, 2.0),
        )
        for (m, v, a, b) in backward_cases
            target = Logpdf(SoftplusBackwardMessage(a, b))
            q = NormalMeanVariance(m, v)
            ηut = getnaturalparameters(project(TangentProjection(type = Unscented), q, target))
            η512 = getnaturalparameters(project(TangentProjection(type = Quadrature(512)), q, target))
            normalized_error = maximum(abs.(ηut .- η512) ./ (1 .+ abs.(η512)))
            @test normalized_error < 0.05

            state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
            message = @call_rule Softplus(
                :in,
                NaturalGradientMessage(TangentProjection(type = Unscented)),
            ) (m_out = GammaShapeRate(a, b), q_in = q, meta = state)
            @test message isa NormalWeightedMeanPrecision
            @test all(isfinite, (weightedmean(message), precision(message)))
            @test state.nfired == 1
        end
    end

    @testset "Exp closed-form NGMP rules" begin
        gaussian = NormalMeanVariance(0.2, 0.7)
        gamma = GammaShapeRate(2.5, 1.4)

        out_state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        out_message = @call_rule Exp(:out, NaturalGradientMessage) (
            m_in = gaussian,
            q_out = gamma,
            meta = out_state,
        )
        @test out_message isa GammaShapeRate
        @test all(isfinite, (shape(out_message), rate(out_message)))

        in_state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        in_message = @call_rule Exp(:in, NaturalGradientMessage) (
            m_out = gamma,
            q_in = gaussian,
            meta = in_state,
        )
        @test in_message isa NormalWeightedMeanPrecision
        @test all(isfinite, (weightedmean(in_message), precision(in_message)))
    end

    @testset "input marginal and finite deterministic free energy" begin
        local_marginal = @call_marginalrule Softplus(:in) (
            m_out = GammaShapeRate(2.0, 1.5),
            m_in = NormalMeanVariance(0.1, 0.8),
            meta = DampingMeta(alpha = 0.2, beta = 0.0),
        )
        @test local_marginal isa NormalWeightedMeanPrecision
        @test isfinite(entropy(local_marginal))

        initialization = @initialization begin
            q(a) = NormalMeanVariance(0.0, 1.0)
            q(γ) = GammaShapeRate(2.0, 2.0)
        end
        iterations = 4
        dependencies = NGMPDependencies(
            out = nothing,
            in = nothing,
            projection = TangentProjection(type = Unscented),
        )
        result = infer(
            model = softplus_toy(
                deps = dependencies,
                damping = DampingMeta(alpha = 0.2, beta = 0.0),
            ),
            data = (y = [0.2, -0.1, 0.3],),
            initialization = initialization,
            iterations = iterations,
            free_energy = true,
        )

        @test all(isfinite, result.free_energy)
        @test isfinite(mean(last(result.posteriors[:a])))
        @test all(isfinite, (shape(last(result.posteriors[:γ])), rate(last(result.posteriors[:γ]))))
        @test length(dependencies.states) == 2
        @test all(state -> state.nfired == iterations, dependencies.states)
    end
end
