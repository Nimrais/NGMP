import ProbabilisticEnsembling: Log
import SurrogateModelling: NaturalGradientMP
import SpecialFunctions: trigamma

# Reference implementation of the γ-edge projection, ported from
# src/model_zoo/dynamic_ngmp_mean_field/univariate.jl (project_lognormal_to_gamma).
function ref_project_lognormal_to_gamma(m, v, q_ef)
    c1, c2 = mean(ClosedWilliamsProduct(), Logpdf(LogNormal(m, sqrt(v))), q_ef)
    a = getnaturalparameters(q_ef)[1] + 1
    b = -getnaturalparameters(q_ef)[2]
    f11, f12, f22 = trigamma(a), 1 / b, a / b^2
    detF = f11 * f22 - f12^2
    return (f22 * c1 - f12 * c2) / detF, -(f11 * c2 - f12 * c1) / detF
end

@model function log_toy(y, pred, deps, damping)
    γ ~ GammaShapeRate(1.0, 1.0)
    z ~ Normal(mean = 0.0, variance = 1.0)
    z ~ Log(γ) where { dependencies = deps, meta = damping }
    for k in 1:length(y)
        y[k] ~ NormalMeanPrecision(pred[k], γ)
    end
end

@testset "Log node natural-gradient rules" begin
    @testset "Log(:out): LogGamma projected at q(z), undamped" begin
        for (a, b) in ((1.5, 2.0), (3.0, 0.7)), (m, v) in ((0.3, 0.5), (-1.0, 2.0))
            state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
            msg = @call_rule Log(:out, NaturalGradientMessage) (
                m_in = GammaShapeRate(a, b), q_out = NormalMeanVariance(m, v), meta = state
            )
            Λref = exp(m + v / 2) * b       # e^{m+v/2}/scale, scale = 1/b
            ξref = a + (m - 1) * Λref       # shape + (m−1)Λ
            @test msg isa NormalWeightedMeanPrecision
            @test weightedmean(msg) ≈ ξref
            @test precision(msg) ≈ Λref
            @test state.nfired == 1
        end
    end

    @testset "Log(:in): LogNormal projected at q(γ), undamped, vs reference" begin
        for (m, v) in ((0.3, 0.5), (-0.5, 1.0)), (a, b) in ((2.0, 3.0), (4.0, 1.5))
            state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
            msg = @call_rule Log(:in, NaturalGradientMessage) (
                m_out = NormalMeanVariance(m, v), q_in = GammaShapeRate(a, b), meta = state
            )
            q_ef = convert(ExponentialFamilyDistribution, GammaShapeRate(a, b))
            Δa, Δb = ref_project_lognormal_to_gamma(m, v, q_ef)
            @test msg isa GammaShapeRate
            @test shape(msg) ≈ Δa + 1
            @test rate(msg) ≈ Δb
        end
    end

    @testset "Log(:in): damped recursion matches manual η replay" begin
        α, β = 0.5, 0.2
        state = NGMPEdgeState(DampingMeta(alpha = α, beta = β))
        η = [0.0, 0.0]
        mom = [0.0, 0.0]
        marginals = [GammaShapeRate(2.0, 3.0), GammaShapeRate(2.5, 2.0), GammaShapeRate(3.0, 2.2)]
        for q in marginals
            msg = @call_rule Log(:in, NaturalGradientMessage) (
                m_out = NormalMeanVariance(0.3, 0.5), q_in = q, meta = state
            )
            Δa, Δb = ref_project_lognormal_to_gamma(0.3, 0.5, convert(ExponentialFamilyDistribution, q))
            ηt = [Δa, -Δb]
            @. mom = β * mom + α * (ηt - η)
            @. η += mom
            @test shape(msg) ≈ η[1] + 1
            @test rate(msg) ≈ -η[2]
        end
    end

    @testset "integration: NGMP through a Deterministic Log node" begin
        init = @initialization begin
            q(z) = NormalMeanVariance(0.0, 1.0)
            q(γ) = GammaShapeScale(1.0, 1.0)
        end

        iters = 20
        deps = NGMPDependencies(out = nothing, in = nothing)
        res = infer(
            model = log_toy(pred = [0.1, -0.2, 0.3, 0.0, 0.15], deps = deps, damping = DampingMeta(alpha = 0.2, beta = 0.0)),
            data = (y = [0.3, -0.5, 0.6, -0.1, 0.4],),
            initialization = init,
            iterations = iters
        )
        qz = last(res.posteriors[:z])
        qγ = last(res.posteriors[:γ])
        @test isfinite(mean(qz)) && var(qz) > 0
        @test shape(qγ) > 0 && rate(qγ) > 0
        # one NGMP state per constrained interface, each fired once per iteration
        @test length(deps.states) == 2
        @test all(state -> state.nfired == iters, deps.states)
    end
end
