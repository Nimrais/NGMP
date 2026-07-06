import SurrogateModelling: NaturalGradientMP, NormalPrecisionMessage, StudentTMessage,
    project_to_gamma, project_to_normal
import SpecialFunctions: digamma
import SpecialFunctions: trigamma as sf_trigamma

# Exact posterior on a τ grid: x integrates out in closed form given τ (ground truth).
function exact_normal_posterior(ys; m0, v0, a0, b0, nτ = 4000)
    N, S1, S2 = length(ys), sum(ys), sum(abs2, ys)
    τs = exp.(range(log(1e-5), log(50.0); length = nτ))
    P = @. 1 / v0 + N * τs
    h = @. m0 / v0 + τs * S1
    logp = @. (a0 - 1) * log(τs) - b0 * τs + (N / 2) * log(τs) - 0.5 * τs * S2 + h^2 / (2P) - 0.5 * log(P)
    logp .-= maximum(logp)
    w = exp.(logp)
    dτ = [τs[2] - τs[1]; (τs[3:end] .- τs[1:end-2]) ./ 2; τs[end] - τs[end-1]]
    w = w .* dτ
    w ./= sum(w)
    μx, vx = h ./ P, 1 ./ P
    mx = sum(w .* μx)
    sx = sum(w .* (vx .+ μx .^ 2)) - mx^2
    mτ = sum(w .* τs)
    vτ = sum(w .* τs .^ 2) - mτ^2
    return (mx = mx, vx = sx, mτ = mτ, vτ = vτ)
end

@model function ngbp_normal_toy(y, m0, v0, a0, b0, deps, damping)
    x ~ NormalMeanVariance(m0, v0)
    τ ~ GammaShapeRate(a0, b0)
    for i in 1:length(y)
        y[i] ~ NormalMeanPrecision(x, τ) where { dependencies = deps, meta = damping }
    end
end

@testset "quadrature tangent projection" begin
    @testset "Gaussian edge: Quadrature ≡ ClosedForm for a LogGamma message" begin
        q = NormalMeanVariance(0.4, 0.7)
        f = Logpdf(LogGamma(2.0, 3.0; check_args = false))
        ηc = getnaturalparameters(project(TangentProjection(type = ClosedForm), q, f))
        ηq = getnaturalparameters(project(TangentProjection(type = Quadrature(64)), q, f))
        @test ηq ≈ ηc atol = 1e-10
    end

    @testset "Gamma edge: Quadrature matches a brute-force grid Williams product" begin
        p = NormalPrecisionMessage(9.7, 10.1, 0.3)
        a, b = 1.5, 2.0
        τg = exp.(range(log(1e-10), log(200.0); length = 400_001))
        lw = a .* log.(τg) .- b .* τg
        lw .-= maximum(lw)
        wg = exp.(lw)
        wg ./= sum(wg)
        ℓg = [log(p, τ) for τ in τg]
        Eℓ = sum(wg .* ℓg)
        Es = sum(wg .* log.(τg))
        Eτ = sum(wg .* τg)
        c1 = sum(wg .* (log.(τg) .- Es) .* (ℓg .- Eℓ))
        c2 = sum(wg .* (τg .- Eτ) .* (ℓg .- Eℓ))
        f11, f12, f22 = sf_trigamma(a), 1 / b, a / b^2
        detF = f11 * f22 - f12^2
        Δa_ref = (f22 * c1 - f12 * c2) / detF
        Δb_ref = -((f11 * c2 - f12 * c1) / detF)
        η = getnaturalparameters(project(TangentProjection(type = Quadrature(128)), GammaShapeRate(a, b), Logpdf(p)))
        @test η[1] ≈ Δa_ref rtol = 1e-4
        @test -η[2] ≈ Δb_ref rtol = 1e-4
    end

    @testset "delta-method regimes: agrees when concentrated, differs when wide" begin
        p = NormalPrecisionMessage(9.7, 10.1, 0.3)
        # concentrated Gamma(200, 100): second-order is trustworthy — both agree
        qc = GammaShapeRate(200.0, 100.0)
        Δa2, Δb2 = project_to_gamma(p, convert(ExponentialFamilyDistribution, qc))
        ηq = getnaturalparameters(project(TangentProjection(type = Quadrature(128)), qc, Logpdf(p)))
        @test ηq[1] ≈ Δa2 rtol = 2e-2
        @test -ηq[2] ≈ Δb2 rtol = 2e-2
        # wide Gamma(1.5, 2): the touching quadratic is integrated far from the mean —
        # the second-order rate increment is off by ~3x (the τ fixed-point bias)
        qw = GammaShapeRate(1.5, 2.0)
        Δa2w, Δb2w = project_to_gamma(p, convert(ExponentialFamilyDistribution, qw))
        ηw = getnaturalparameters(project(TangentProjection(type = Quadrature(128)), qw, Logpdf(p)))
        @test abs(Δb2w - (-ηw[2])) / abs(-ηw[2]) > 1.0
    end
end

@testset "NormalMeanPrecision natural-gradient rules" begin
    @testset "τ rule matches the quadrature projection of NormalPrecisionMessage" begin
        for (y, m̃, ṽ) in ((9.7, 10.1, 0.3), (0.5, 0.0, 2.0)), (a, b) in ((2.0, 1.5), (1.0, 1.0))
            state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
            msg = @call_rule NormalMeanPrecision(:τ, NaturalGradientMessage) (
                m_μ = NormalMeanVariance(m̃, ṽ), q_out = PointMass(y),
                q_τ = GammaShapeRate(a, b), meta = state
            )
            η = getnaturalparameters(project(
                TangentProjection(type = Quadrature(128)),
                GammaShapeRate(a, b),
                Logpdf(NormalPrecisionMessage(y, m̃, ṽ))
            ))
            @test msg isa GammaShapeRate
            @test shape(msg) ≈ η[1] + 1
            @test rate(msg) ≈ -η[2]
            @test state.nfired == 1
        end
    end

    @testset "μ rule matches the quadrature projection of StudentTMessage" begin
        for (y, ã, b̃) in ((9.7, 3.0, 0.5), (1.0, 1.5, 2.0)), (m, v) in ((9.5, 0.8), (0.0, 3.0))
            state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
            msg = @call_rule NormalMeanPrecision(:μ, NaturalGradientMessage) (
                m_τ = GammaShapeRate(ã, b̃), q_out = PointMass(y),
                q_μ = NormalMeanVariance(m, v), meta = state
            )
            η = getnaturalparameters(project(
                TangentProjection(type = Quadrature(128)),
                NormalMeanVariance(m, v),
                Logpdf(StudentTMessage(y, ã, b̃))
            ))
            @test msg isa NormalWeightedMeanPrecision
            @test weightedmean(msg) ≈ η[1]
            @test precision(msg) ≈ -2 * η[2]
        end
    end

    @testset "damped recursion on the τ edge matches manual η replay" begin
        α, β = 0.5, 0.2
        state = NGMPEdgeState(DampingMeta(alpha = α, beta = β))
        η = [0.0, 0.0]
        mom = [0.0, 0.0]
        y = 9.7
        for (m̃, ṽ, a, b) in ((10.1, 0.3, 2.0, 1.5), (9.9, 0.2, 2.4, 1.4), (9.8, 0.15, 2.7, 1.3))
            msg = @call_rule NormalMeanPrecision(:τ, NaturalGradientMessage) (
                m_μ = NormalMeanVariance(m̃, ṽ), q_out = PointMass(y),
                q_τ = GammaShapeRate(a, b), meta = state
            )
            ηt = collect(getnaturalparameters(project(
                TangentProjection(type = Quadrature(128)),
                GammaShapeRate(a, b),
                Logpdf(NormalPrecisionMessage(y, m̃, ṽ))
            )))
            @. mom = β * mom + α * (ηt - η)
            @. η += mom
            @test shape(msg) ≈ η[1] + 1
            @test rate(msg) ≈ -η[2]
        end
    end

    @testset "integration: loopy NG-BP on the Normal(mean, precision) model" begin
        ys = [10.244, 9.844, 9.66]   # fixed draw from N(10, 10⁻¹) — deterministic test
        m0, v0, a0, b0 = 0.0, 1e6, 1.0, 1.0
        iters = 50

        deps = NGMPDependencies(μ = nothing, τ = nothing)
        mx0, vx0 = mean(ys), max(var(ys), 0.1)
        # message inits break the loopy-BP deadlock (N ≥ 2: every NGMP message waits
        # on the equality-chain product of the other nodes' messages)
        init = @initialization begin
            q(x) = NormalMeanVariance(mx0, vx0)
            q(τ) = GammaShapeRate(a0, b0)
            μ(x) = NormalMeanVariance(mx0, 10 * vx0)
            μ(τ) = GammaShapeRate(a0, b0)
        end
        res = infer(
            model = ngbp_normal_toy(m0 = m0, v0 = v0, a0 = a0, b0 = b0, deps = deps,
                                    damping = DampingMeta(alpha = 0.2, beta = 0.0)),
            data = (y = ys,), initialization = init, iterations = iters
        )
        qx = last(res.posteriors[:x])
        qτ = last(res.posteriors[:τ])
        ex = exact_normal_posterior(ys; m0 = m0, v0 = v0, a0 = a0, b0 = b0)

        @test length(deps.states) == 2 * length(ys)
        @test all(state -> state.nfired == iters, deps.states)
        @test var(qx) > 0 && shape(qτ) > 0 && rate(qτ) > 0
        # location mean and the τ marginal match the exact grid tightly with the
        # quadrature-exact projection; the x variance is a single-Gaussian moment
        # against a heavy-tailed exact marginal, hence the looser bound
        @test mean(qx) ≈ ex.mx rtol = 1e-2
        @test var(qx) ≈ ex.vx rtol = 0.5
        @test mean(qτ) ≈ ex.mτ rtol = 0.05
        @test var(qτ) ≈ ex.vτ rtol = 0.3
    end
end
