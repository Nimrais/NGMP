import SurrogateModelling: NaturalGradientMP, ExpGammaSiteMessage, MvExpGammaSiteMessage
import LinearAlgebra: Diagonal, Symmetric, diag, I
import StableRNGs: StableRNG
import ReactiveMP: @call_marginalrule
import BayesBase: weightedmean_precision
import ExponentialFamilyProjection: ProjectedTo, ProjectionParameters, ClosedFormStrategy, project_to

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

    @testset "univariate twins match the d = 1 math" begin
        m_s, v_s = 0.3, 0.8
        m_μ, v_μ = 0.9, 0.4
        y1 = 1.4
        ρ1 = exp(m_s + v_s / 2)

        out_msg = @call_rule MvNormalExpPrecision(:out, Marginalisation) (
            q_μ = NormalMeanVariance(m_μ, v_μ), q_s = NormalMeanVariance(m_s, v_s),
        )
        @test mean(out_msg) ≈ m_μ
        @test precision(out_msg) ≈ ρ1

        μ_msg = @call_rule MvNormalExpPrecision(:μ, Marginalisation) (
            q_out = PointMass(y1), q_s = NormalMeanVariance(m_s, v_s),
        )
        @test mean(μ_msg) ≈ y1
        @test precision(μ_msg) ≈ ρ1

        E1 = abs2(y1 - m_μ) + v_μ
        state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        s_msg = @call_rule MvNormalExpPrecision(:s, NaturalGradientMessage()) (
            q_out = PointMass(y1), q_μ = NormalMeanVariance(m_μ, v_μ),
            q_s = NormalMeanVariance(m_s, v_s), meta = state,
        )
        Λ1 = (E1 / 2) * ρ1
        @test weightedmean(s_msg) ≈ 0.5 + (m_s - 1) * Λ1
        @test precision(s_msg) ≈ Λ1
        @test state.nfired == 1

        energy = score(AverageEnergy(), MvNormalExpPrecision, Val{(:out, :μ, :s)}(),
            (Marginal(PointMass(y1), false, false),
             Marginal(NormalMeanVariance(m_μ, v_μ), false, false),
             Marginal(NormalMeanVariance(m_s, v_s), false, false)), nothing)
        @test energy ≈ log(2π) / 2 - m_s / 2 + ρ1 * E1 / 2
    end

    @testset "structured (out, μ) cluster rules" begin
        m_ν_msg = MvNormalMeanCovariance([0.9, -0.2, 0.5], [0.4 0.05 0.0; 0.05 0.3 0.02; 0.0 0.02 0.5])
        m_μ_msg = MvNormalMeanCovariance([1.1, 0.1, -0.4], [0.6 0.1 0.0; 0.1 0.2 0.03; 0.0 0.03 0.4])
        q_s = MvNormalMeanCovariance(ms, Vs)
        P = exp.(ms .+ diag(Vs) ./ 2)

        out_msg = @call_rule MvNormalExpPrecision(:out, Marginalisation) (
            m_μ = m_μ_msg, q_s = q_s,
        )
        @test mean(out_msg) ≈ mean(m_μ_msg)
        @test cov(out_msg) ≈ cov(m_μ_msg) + Matrix(Diagonal(inv.(P)))

        joint = @call_marginalrule MvNormalExpPrecision(:out_μ) (
            m_out = m_ν_msg, m_μ = m_μ_msg, q_s = q_s,
        )
        ξν, Λν = weightedmean_precision(m_ν_msg)
        ξμ, Λμ = weightedmean_precision(m_μ_msg)
        @test weightedmean(joint) ≈ [ξν; ξμ]
        @test precision(joint) ≈ [(Λν + Diagonal(P)) (-Matrix(Diagonal(P)));
                                  (-Matrix(Diagonal(P))) (Λμ + Diagonal(P))]

        # the structured s-site uses the joint second moment incl. cross-covariance
        mj, Vj = mean_cov(joint)
        E = [abs2(mj[j] - mj[3 + j]) + Vj[j, j] + Vj[3 + j, 3 + j] - 2 * Vj[j, 3 + j]
             for j in 1:3]
        state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        s_msg = @call_rule MvNormalExpPrecision(:s, NaturalGradientMessage()) (
            q_out_μ = joint, q_s = q_s, meta = state,
        )
        Λref = (E ./ 2) .* P
        @test weightedmean(s_msg) ≈ 0.5 .+ (ms .- 1) .* Λref
        @test precision(s_msg) ≈ Matrix(Diagonal(Λref))
        @test state.nfired == 1

        energy = score(AverageEnergy(), MvNormalExpPrecision, Val{(:out_μ, :s)}(),
            (Marginal(joint, false, false), Marginal(q_s, false, false)), nothing)
        @test energy ≈ 3 * log(2π) / 2 - sum(ms) / 2 + sum(P .* E) / 2
    end

    @testset "Mv Gaussian diagonal Fisher metric (vector-transport support)" begin
        m1, v1 = 0.7, 0.6
        η1 = [m1 / v1, -1 / (2 * v1), 0.0]                # d = 1 flat layout (ξ, −Λ/2)
        metric = NaturalGradientMP.diagonal_fisher_metric(
            MvNormalMeanCovariance, η1[1:2], 1e-6)
        @test metric ≈ [v1, 2 * v1^2 + 4 * m1^2 * v1]      # univariate reduction

        m2 = [0.4, -0.9]
        V2 = [0.8 0.25; 0.25 0.5]
        Λ2 = inv(V2)
        η2 = vcat(Λ2 * m2, vec(-Λ2 ./ 2))
        metric2 = NaturalGradientMP.diagonal_fisher_metric(
            MvNormalMeanCovariance, η2, 1e-6)
        @test metric2[1:2] ≈ diag(V2)
        wick(i, j) = V2[i, i] * V2[j, j] + V2[i, j]^2 +
            m2[i]^2 * V2[j, j] + m2[j]^2 * V2[i, i] + 2 * m2[i] * m2[j] * V2[i, j]
        @test metric2[3:end] ≈ [wick(1, 1), wick(2, 1), wick(1, 2), wick(2, 2)]
        # improper state falls back to the magnitude metric
        @test NaturalGradientMP.diagonal_fisher_metric(
            MvNormalMeanCovariance, vcat([1.0, 1.0], vec([0.5 0.0; 0.0 0.5])), 1e-6,
        ) == max.(abs.(vcat([1.0, 1.0], vec([0.5 0.0; 0.0 0.5]))), 1e-6)
    end

    @testset "exact dual (m-connection) transport via Amari duality" begin
        # η ↦ μ maps used as ground truth via central finite differences
        function μ_gaussian(η)
            dg = div(isqrt(1 + 4 * length(η)) - 1, 2)
            Λ = -2 .* reshape(η[(dg + 1):end], dg, dg)
            V = inv(Symmetric(Matrix(Λ)))
            m = V * η[1:dg]
            return vcat(m, vec(V .+ m * m'))
        end
        μ_gamma(η) = [SpecialFunctions.digamma(η[1] + 1) - log(-η[2]), (η[1] + 1) / (-η[2])]
        jvp(f, η, u; ε = 1e-6) = (f(η .+ ε .* u) .- f(η .- ε .* u)) ./ (2ε)

        V1 = [0.8 0.25; 0.25 0.5]; m1 = [0.4, -0.9]
        V2 = [0.4 -0.1; -0.1 0.9]; m2 = [-0.2, 0.3]
        Λ1, Λ2 = inv(V1), inv(V2)
        η1 = vcat(Λ1 * m1, vec(-Λ1 ./ 2))
        η2 = vcat(Λ2 * m2, vec(-Λ2 ./ 2))
        vtan = [0.3, -0.2, 0.05, 0.02, 0.02, -0.04]

        @test NaturalGradientMP.transport_natural_vector(
            MvNormalMeanCovariance, η1, η1, vtan) ≈ vtan atol = 1e-12
        transported = NaturalGradientMP.transport_natural_vector(
            MvNormalMeanCovariance, η1, η2, vtan)
        # m-transport is the identity in expectation coordinates:
        # G(η1)·v (pushforward at old) must equal G(η2)·v′ (pushforward at new)
        @test jvp(μ_gaussian, η2, transported) ≈ jvp(μ_gaussian, η1, vtan) rtol = 1e-5

        ηγ1, ηγ2, vγ = [2.0, -1.5], [3.5, -0.7], [0.4, 0.3]
        tγ = NaturalGradientMP.transport_natural_vector(Gamma, ηγ1, ηγ2, vγ)
        @test jvp(μ_gamma, ηγ2, tγ) ≈ jvp(μ_gamma, ηγ1, vγ) rtol = 1e-5

        # improper endpoint falls back to identity (e-transport)
        @test NaturalGradientMP.transport_natural_vector(
            MvNormalMeanCovariance, zero(η1), η2, vtan) == vtan

        # the optimizer accepts the new methods
        @test DampingMeta(alpha = 0.1, beta = 0.2, method = :dual_transport) isa DampingMeta
        @test DampingMeta(alpha = 0.1, beta = 0.2, method = :dual_transport_nesterov) isa DampingMeta
    end

    @testset "β-NLL precision tempering (faithful mean pathway)" begin
        β = 0.5
        temper = SurrogateModelling.PrecisionTempering(β, DampingMeta(alpha = 0.2, beta = 0.0))
        μ_msg = @call_rule MvNormalExpPrecision(:μ, Marginalisation) (
            q_out = PointMass(yobs), q_s = MvNormalMeanCovariance(ms, Vs), meta = temper,
        )
        @test mean(μ_msg) ≈ yobs
        @test precision(μ_msg) ≈ Matrix(Diagonal(ρ .^ (1 - β)))

        μ1 = @call_rule MvNormalExpPrecision(:μ, Marginalisation) (
            q_out = PointMass(1.4), q_s = NormalMeanVariance(0.3, 0.8),
            meta = SurrogateModelling.PrecisionTempering(β, nothing),
        )
        @test precision(μ1) ≈ exp(0.3 + 0.8 / 2)^(1 - β)

        # the s-site is untempered: NGMPEdgeState unwraps the inner damping
        state = NGMPEdgeState(temper)
        s_msg = @call_rule MvNormalExpPrecision(:s, NaturalGradientMessage()) (
            q_out = PointMass(yobs), q_μ = MvNormalMeanCovariance(mμ, Vμ),
            q_s = MvNormalMeanCovariance(ms, Vs), meta = state,
        )
        E = abs2.(yobs .- mμ) .+ diag(Vμ)
        @test precision(s_msg) ≈ 0.2 .* Matrix(Diagonal((E ./ 2) .* ρ))  # α from inner DampingMeta
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

    @testset "VMP ProjectedTo glue: site logpdf/insupport + closed-form expectation" begin
        site = ExpGammaSiteMessage(0.5, 1.3)
        @test BayesBase.insupport(site, -0.7)
        @test BayesBase.logpdf(site, -0.7) ≈ 0.5 * (-0.7) - 1.3 * exp(-0.7)

        mv_site = MvExpGammaSiteMessage([0.5, 0.5], [1.3, 0.4])
        svec = [-0.7, 0.2]
        @test BayesBase.insupport(mv_site, svec)
        @test BayesBase.logpdf(mv_site, svec) ≈ sum(0.5 .* svec .- [1.3, 0.4] .* exp.(svec))

        # E_q[c·s − b·e^s] = c·m − b·e^{m+v/2}
        expectation = ClosedFormExpectations.mean(
            ClosedFormExpectations.ClosedFormExpectation(),
            Logpdf(site),
            Normal(0.3, sqrt(0.8)),
        )
        @test expectation ≈ 0.5 * 0.3 - 1.3 * exp(0.3 + 0.8 / 2)

        # end-to-end VMP marginal path: (Gaussian message × site) product
        # projected onto NormalMeanVariance, checked against quadrature moments
        left = NormalMeanVariance(0.0, 4.0)
        product = BayesBase.prod(BayesBase.GenericProd(), left, site)
        projected = project_to(
            ProjectedTo(
                NormalMeanVariance;
                parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
            ),
            product,
        )
        grid = range(-14.0, 6.0; length = 20001)
        logw = [logpdf(left, s) + BayesBase.logpdf(site, s) for s in grid]
        weights = exp.(logw .- maximum(logw))
        weights ./= sum(weights)
        m_ref = sum(weights .* grid)
        v_ref = sum(weights .* abs2.(grid .- m_ref))
        @test abs(mean(projected) - m_ref) < 0.2
        @test abs(var(projected) - v_ref) / v_ref < 0.5
    end
end
