import SurrogateModelling: NaturalGradientMP, ExpGammaSiteMessage, MvExpGammaSiteMessage
import LinearAlgebra: Diagonal, diag, I
import StableRNGs: StableRNG

# Heteroscedastic toy: one shared mean vector μ and one shared log-precision
# vector s explain all observations; each output dimension owns its precision
# e^{sⱼ} — the structure a scalar softdot τ cannot express.
@model function mnep_hetero_toy(y, dim, deps, damping)
    μ ~ MvNormalMeanCovariance(zeros(dim), Matrix(Diagonal(fill(1.0, dim))))
    s ~ MvNormalMeanCovariance(zeros(dim), Matrix(Diagonal(fill(4.0, dim))))
    for k in eachindex(y)
        y[k] ~ MvNormalExpPrecision(μ, s) where { dependencies = deps, meta = damping }
    end
end

@constraints function mnep_hetero_constraints()
    q(μ, s) = q(μ)q(s)
end

@initialization function mnep_hetero_init(dim)
    q(μ) = MvNormalMeanCovariance(zeros(dim), Matrix(Diagonal(fill(1.0, dim))))
    q(s) = MvNormalMeanCovariance(zeros(dim), Matrix(Diagonal(fill(1.0, dim))))
end

# Homoscedastic baseline: the same mean model with ONE scalar precision τ
# shared across dimensions.
@model function mnep_scalar_baseline(y, dim)
    μ ~ MvNormalMeanCovariance(zeros(dim), Matrix(Diagonal(fill(1.0, dim))))
    τ ~ GammaShapeRate(2.0, 2.0)
    for k in eachindex(y)
        y[k] ~ MvNormalMeanScalePrecision(μ, τ)
    end
end

@constraints function mnep_scalar_constraints()
    q(μ, τ) = q(μ)q(τ)
end

@initialization function mnep_scalar_init(dim)
    q(μ) = MvNormalMeanCovariance(zeros(dim), Matrix(Diagonal(fill(1.0, dim))))
    q(τ) = GammaShapeRate(2.0, 2.0)
end

@testset "MvNormalExpPrecision (per-dimension log-precision likelihood)" begin
    ms = [0.2, -0.4, 1.0]
    Vs = [0.5 0.2 0.1; 0.2 0.8 -0.05; 0.1 -0.05 0.3]     # dense: only diag may be read
    mμ = [1.0, 0.0, -0.5]
    Vμ = [0.3 0.1 0.0; 0.1 0.6 0.05; 0.0 0.05 0.2]
    yobs = [1.2, -0.3, 0.4]
    ρ = exp.(ms .+ diag(Vs) ./ 2)

    @testset ":out and :μ rules carry precision diag(E[e^s])" begin
        out_msg = @call_rule MvNormalExpPrecision(:out, Marginalisation) (
            q_μ = MvNormalMeanCovariance(mμ, Vμ),
            q_s = MvNormalMeanCovariance(ms, Vs),
        )
        @test mean(out_msg) ≈ mμ
        @test precision(out_msg) ≈ Matrix(Diagonal(ρ))

        μ_msg = @call_rule MvNormalExpPrecision(:μ, Marginalisation) (
            q_out = PointMass(yobs),
            q_s = MvNormalMeanCovariance(ms, Vs),
        )
        @test mean(μ_msg) ≈ yobs
        @test precision(μ_msg) ≈ Matrix(Diagonal(ρ))

        # DampingMeta reaches the non-NGMP interfaces unwrapped — same result
        μ_msg_meta = @call_rule MvNormalExpPrecision(:μ, Marginalisation) (
            q_out = PointMass(yobs),
            q_s = MvNormalMeanCovariance(ms, Vs),
            meta = DampingMeta(alpha = 0.2, beta = 0.0),
        )
        @test mean(μ_msg_meta) ≈ yobs
        @test precision(μ_msg_meta) ≈ Matrix(Diagonal(ρ))
    end

    @testset "NGMP :s site matches the per-coordinate scalar ExpGamma projection" begin
        E = abs2.(yobs .- mμ) .+ diag(Vμ)
        state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        msg = @call_rule MvNormalExpPrecision(:s, NaturalGradientMessage()) (
            q_out = PointMass(yobs),
            q_μ = MvNormalMeanCovariance(mμ, Vμ),
            q_s = MvNormalMeanCovariance(ms, Vs),
            meta = state,
        )
        Λref = (E ./ 2) .* ρ
        ξref = 0.5 .+ (ms .- 1) .* Λref
        @test weightedmean(msg) ≈ ξref
        @test precision(msg) ≈ Matrix(Diagonal(Λref))
        @test state.nfired == 1

        # each coordinate is exactly the scalar ExpGammaSiteMessage projection
        for j in 1:3
            η = getnaturalparameters(project(
                TangentProjection(type = ClosedForm),
                NormalMeanVariance(ms[j], Vs[j, j]),
                Logpdf(ExpGammaSiteMessage(0.5, E[j] / 2)),
            ))
            @test ξref[j] ≈ η[1]
            @test Λref[j] ≈ -2 * η[2]
        end
    end

    @testset "NGMP :s damped recursion matches manual η replay" begin
        α, β = 0.5, 0.2
        state = NGMPEdgeState(DampingMeta(alpha = α, beta = β))
        d = 3
        η = zeros(d + d^2)
        mom = zeros(d + d^2)
        for (msk, mμk) in ((ms, mμ), (ms .+ 0.1, mμ .- 0.2), (ms .- 0.05, mμ .+ 0.1))
            msg = @call_rule MvNormalExpPrecision(:s, NaturalGradientMessage()) (
                q_out = PointMass(yobs),
                q_μ = MvNormalMeanCovariance(mμk, Vμ),
                q_s = MvNormalMeanCovariance(msk, Vs),
                meta = state,
            )
            Ek = abs2.(yobs .- mμk) .+ diag(Vμ)
            Λt = (Ek ./ 2) .* exp.(msk .+ diag(Vs) ./ 2)
            ξt = 0.5 .+ (msk .- 1) .* Λt
            ηt = vcat(ξt, vec(Matrix(Diagonal(-Λt ./ 2))))
            @. mom = β * mom + α * (ηt - η)
            @. η += mom
            @test weightedmean(msg) ≈ η[1:d]
            @test precision(msg) ≈ -2 .* reshape(η[(d + 1):end], d, d)
        end
        @test state.nfired == 3
    end

    @testset "average energy matches the closed formula (with and without meta)" begin
        E = abs2.(yobs .- mμ) .+ diag(Vμ)
        expected = 3 * log(2π) / 2 - sum(ms) / 2 + sum(ρ .* E) / 2
        marginals = (
            Marginal(PointMass(yobs), false, false),
            Marginal(MvNormalMeanCovariance(mμ, Vμ), false, false),
            Marginal(MvNormalMeanCovariance(ms, Vs), false, false),
        )
        stock = score(AverageEnergy(), MvNormalExpPrecision, Val{(:out, :μ, :s)}(), marginals, nothing)
        damped = score(AverageEnergy(), MvNormalExpPrecision, Val{(:out, :μ, :s)}(), marginals,
                       DampingMeta(alpha = 0.2, beta = 0.0))
        @test stock ≈ expected
        @test damped ≈ expected
    end

    @testset "integration: recovers per-dimension precisions spanning 4 orders" begin
        rng = StableRNG(42)
        dim, n, ntest = 3, 40, 20
        μ_true = [0.5, -0.3, 1.0]
        τ_true = [100.0, 1.0, 0.01]
        draw() = μ_true .+ randn(rng, dim) ./ sqrt.(τ_true)
        ys = [draw() for _ in 1:n]
        ys_test = [draw() for _ in 1:ntest]
        iters = 50

        deps = NGMPDependencies(s = nothing, projection = TangentProjection(type = ClosedForm))
        damping = DampingMeta(alpha = 0.2, beta = 0.0)

        result = infer(
            model = mnep_hetero_toy(dim = dim, deps = deps, damping = damping),
            data = (y = ys,),
            constraints = mnep_hetero_constraints(),
            initialization = mnep_hetero_init(dim),
            iterations = iters,
            free_energy = true,
        )

        @test length(deps.states) == n
        @test all(state -> state.nfired == iters, deps.states)
        @test all(isfinite, result.free_energy)
        @test last(result.free_energy) < first(result.free_energy)

        qs = last(result.posteriors[:s])
        qμ = last(result.posteriors[:μ])
        m̂s, V̂s = mean_cov(qs)
        ρ̂ = exp.(m̂s .+ diag(V̂s) ./ 2)

        # per-dimension recovery in log space, and an ordering no scalar τ can express
        @test all(j -> abs(log(ρ̂[j]) - log(τ_true[j])) < 1.5, 1:dim)
        @test ρ̂[1] > ρ̂[2] > ρ̂[3]
        @test ρ̂[1] / ρ̂[3] > 1e3

        # homoscedastic baseline: same mean model, one shared scalar τ
        baseline = infer(
            model = mnep_scalar_baseline(dim = dim),
            data = (y = ys,),
            constraints = mnep_scalar_constraints(),
            initialization = mnep_scalar_init(dim),
            iterations = iters,
            free_energy = false,
        )
        qμ_b = last(baseline.posteriors[:μ])
        τ̂ = mean(last(baseline.posteriors[:τ]))

        hetero_ll = sum(logpdf(MvNormalMeanPrecision(mean(qμ), Matrix(Diagonal(ρ̂))), y) for y in ys_test)
        scalar_ll = sum(logpdf(MvNormalMeanPrecision(mean(qμ_b), Matrix(τ̂ * I, dim, dim)), y) for y in ys_test)
        @test hetero_ll > scalar_ll
    end
end
