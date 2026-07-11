import ProbabilisticEnsembling
import ProbabilisticEnsembling: Log
import SurrogateModelling: NaturalGradientMP, MvNormalPrecisionMessage, MvNormalDeviationPrecisionMessage,
    NormalPrecisionMessage, MomentForm
import SpecialFunctions: trigamma as gw_trigamma
import LinearAlgebra: I, Symmetric, eigen, dot, logdet

# Two-layer gated toy: local weights w[i,j] ~ MvN(wbar[i], κ⁻¹I) — the gate node
# IS the local prior, with the input-dependent precision κ[i,j] (from an upper
# softdot → Gamma → Log copy of the lower pipeline) deciding how far the local
# weights may deviate from the shared profile wbar[i]. Stock (plain) softdot
# everywhere — structured rules.
@model function gated_toy(y, features, predictions, dim, log_deps_upper, log_deps_lower, gate_deps, damping)
    local u, τu, βu, s, κ, wbar, w, z, γ, τ, β
    nf = size(predictions, 1)
    for i in 1:nf
        u[i] ~ MvNormalMeanScalePrecision(zeros(dim), 0.1)
        τu[i] ~ GammaShapeRate(2.0, 2.0)
        βu[i] ~ GammaShapeRate(2.0, 2.0)
        wbar[i] ~ MvNormalMeanScalePrecision(zeros(dim), 0.1)
        τ[i] ~ GammaShapeRate(2.0, 2.0)
        β[i] ~ GammaShapeRate(2.0, 2.0)
    end
    for j in 1:length(y), i in 1:nf
        s[i, j] ~ softdot(features[j], u[i], τu[i])
        κ[i, j] ~ GammaShapeRate(1.0, βu[i])
        s[i, j] ~ Log(κ[i, j]) where { dependencies = log_deps_upper, meta = damping }
        w[i, j] ~ MvNormalMeanScalePrecision(wbar[i], κ[i, j]) where { dependencies = gate_deps, meta = damping }
        z[i, j] ~ softdot(features[j], w[i, j], τ[i])
        γ[i, j] ~ GammaShapeRate(1.0, β[i])
        z[i, j] ~ Log(γ[i, j]) where { dependencies = log_deps_lower, meta = damping }
        y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
    end
end

# The full relaxed-style joint cluster now includes wbar (the gate's out edge is
# the latent w, so the message toward wbar must be a BP message too — keeping
# wbar mean-field would require a joint q(w,κ) local marginal that has no rule);
# only τu, βu, τ, β stay mean-field.
@constraints function gated_toy_constraints()
    q(u, τu, βu, s, κ, wbar, w, z, γ, τ, β) = q(τu)q(βu)q(u, w, z, γ, s, κ, wbar)q(τ)q(β)
    q(u)::MomentForm()
    q(w)::MomentForm()
    q(wbar)::MomentForm()
end

@testset "gated weights: MvNormalPrecisionMessage + MvNormalMeanScalePrecision rules" begin
    V3 = [0.8 0.2 -0.1; 0.2 0.6 0.15; -0.1 0.15 0.9]
    y3 = [0.4, -0.7, 1.1]
    m3 = [0.1, 0.3, -0.2]

    @testset "expression matches the dense MvNormal logpdf" begin
        p = MvNormalPrecisionMessage(y3, m3, V3)
        for κ in (0.1, 1.0, 10.0)
            ref = logpdf(MvNormalMeanCovariance(m3, V3 + I / κ), y3)
            @test log(p, κ) ≈ ref
            @test p(κ) ≈ exp(ref)
        end
        # d = 1 collapses to the scalar NormalPrecisionMessage
        p1 = MvNormalPrecisionMessage([9.7], [10.1], reshape([0.3], 1, 1))
        ps = NormalPrecisionMessage(9.7, 10.1, 0.3)
        for κ in (0.5, 2.0, 20.0)
            @test log(p1, κ) ≈ log(ps, κ)
        end
    end

    @testset "Gamma edge: Quadrature matches a brute-force grid Williams product" begin
        p = MvNormalPrecisionMessage([0.4, -0.7], [0.1, 0.3], [0.8 0.2; 0.2 0.6])
        a, b = 1.5, 2.0
        κg = exp.(range(log(1e-10), log(200.0); length = 400_001))
        lw = a .* log.(κg) .- b .* κg
        lw .-= maximum(lw)
        wg = exp.(lw)
        wg ./= sum(wg)
        ℓg = [log(p, κ) for κ in κg]
        Eℓ = sum(wg .* ℓg)
        Es = sum(wg .* log.(κg))
        Eκ = sum(wg .* κg)
        c1 = sum(wg .* (log.(κg) .- Es) .* (ℓg .- Eℓ))
        c2 = sum(wg .* (κg .- Eκ) .* (ℓg .- Eℓ))
        f11, f12, f22 = gw_trigamma(a), 1 / b, a / b^2
        detF = f11 * f22 - f12^2
        Δa_ref = (f22 * c1 - f12 * c2) / detF
        Δb_ref = -((f11 * c2 - f12 * c1) / detF)
        η = getnaturalparameters(project(TangentProjection(type = Quadrature(128)), GammaShapeRate(a, b), Logpdf(p)))
        @test η[1] ≈ Δa_ref rtol = 1e-4
        @test -η[2] ≈ Δb_ref rtol = 1e-4
    end

    @testset "Gamma edge: log-space GenUT tracks the quadrature truth across widths and dims" begin
        B5 = [0.9 0.1 -0.2 0.3 0.0;
              0.1 0.7 0.2 -0.1 0.15;
              -0.2 0.2 0.8 0.05 -0.1;
              0.3 -0.1 0.05 0.6 0.2;
              0.0 0.15 -0.1 0.2 0.75]
        # the 3-point GenUT bias grows with the number of summed eigendirection
        # terms — measured ≤ 0.10 at d = 2 and ≤ 0.23 at d = 5 across widths
        cases = (
            (0.15, [0.4, -0.7], [0.1, 0.3], [0.8 0.2; 0.2 0.6]),
            (0.30, [0.4, -0.7, 1.1, -0.3, 0.6], [0.1, 0.3, -0.2, 0.5, 0.0], Symmetric(B5' * B5 ./ 5 + 0.05 * I) |> Matrix),
        )
        for (bound, yv, mv, Vv) in cases, (a, b) in ((1.5, 2.0), (3.5, 2.0), (10.0, 5.0), (200.0, 100.0))
            p = Logpdf(MvNormalPrecisionMessage(yv, mv, Vv))
            qg = GammaShapeRate(a, b)
            ref = getnaturalparameters(project(TangentProjection(type = Quadrature(4096)), qg, p))
            ηut = getnaturalparameters(project(TangentProjection(type = Unscented), qg, p))
            @test maximum(abs.(ηut .- ref) ./ abs.(ref)) < bound
            @test all(sign.(ηut) .== sign.(ref))
        end
    end

    @testset "ClosedForm and DeltaApproximation error informatively" begin
        p = Logpdf(MvNormalPrecisionMessage(y3, m3, V3))
        @test_throws ErrorException project(TangentProjection(type = ClosedForm), GammaShapeRate(2.0, 2.0), p)
        @test_throws ErrorException project(TangentProjection(type = DeltaApproximation), GammaShapeRate(2.0, 2.0), p)
    end

    @testset "γ rule matches the quadrature projection of MvNormalPrecisionMessage" begin
        for (a, b) in ((2.0, 1.5), (1.0, 1.0))
            state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
            msg = @call_rule MvNormalMeanScalePrecision(:γ, NaturalGradientMessage(TangentProjection(type = Quadrature(128)))) (
                m_μ = MvNormalMeanCovariance(m3, V3), q_out = PointMass(zeros(3)),
                q_γ = GammaShapeRate(a, b), meta = state
            )
            η = getnaturalparameters(project(
                TangentProjection(type = Quadrature(128)),
                GammaShapeRate(a, b),
                Logpdf(MvNormalPrecisionMessage(zeros(3), m3, V3))
            ))
            @test msg isa GammaShapeRate
            @test shape(msg) ≈ η[1] + 1
            @test rate(msg) ≈ -η[2]
            @test state.nfired == 1
        end
    end

    @testset "γ rule: damped recursion matches manual η replay" begin
        α, β = 0.5, 0.2
        state = NGMPEdgeState(DampingMeta(alpha = α, beta = β))
        η = [0.0, 0.0]
        mom = [0.0, 0.0]
        for (mshift, a, b) in ((0.0, 2.0, 1.5), (0.1, 2.4, 1.4), (-0.05, 2.7, 1.3))
            m̃ = m3 .+ mshift
            msg = @call_rule MvNormalMeanScalePrecision(:γ, NaturalGradientMessage(TangentProjection(type = Quadrature(128)))) (
                m_μ = MvNormalMeanCovariance(m̃, V3), q_out = PointMass(zeros(3)),
                q_γ = GammaShapeRate(a, b), meta = state
            )
            ηt = collect(getnaturalparameters(project(
                TangentProjection(type = Quadrature(128)),
                GammaShapeRate(a, b),
                Logpdf(MvNormalPrecisionMessage(zeros(3), m̃, V3))
            )))
            @. mom = β * mom + α * (ηt - η)
            @. η += mom
            @test shape(msg) ≈ η[1] + 1
            @test rate(msg) ≈ -η[2]
        end
    end

    @testset "μ rule: plug-in conjugate message from the Gamma message mean" begin
        msg = @call_rule MvNormalMeanScalePrecision(:μ, Marginalisation) (
            m_γ = GammaShapeRate(3.0, 1.5), q_out = PointMass(zeros(2))
        )
        @test msg isa MvNormalMeanScalePrecision
        @test mean(msg) == zeros(2)
        @test first(invcov(msg)) ≈ 2.0
    end

    @testset "deviation expression: proper info-form cavity reduces to MvNormalPrecisionMessage" begin
        # with a full-rank cavity (ξ = Λm̃, Λ = Ṽ⁻¹) the deviation message equals
        # MvN(m̃ | m_b, Ṽ + V_b + κ⁻¹I) up to a κ-independent constant — so the
        # projected natural-gradient sites must coincide exactly
        m3b = [0.05, 0.25, -0.15]
        V3b = [0.4 0.1 0.0; 0.1 0.5 -0.05; 0.0 -0.05 0.3]
        Λ3 = inv(Symmetric(V3))
        pdev = Logpdf(MvNormalDeviationPrecisionMessage(Λ3 * m3, Matrix(Λ3), m3b, V3b))
        pref = Logpdf(MvNormalPrecisionMessage(m3, m3b, V3 + V3b))
        for (a, b) in ((1.5, 2.0), (5.0, 2.0))
            qg = GammaShapeRate(a, b)
            ηdev = getnaturalparameters(project(TangentProjection(type = Quadrature(256)), qg, pdev))
            ηref = getnaturalparameters(project(TangentProjection(type = Quadrature(256)), qg, pref))
            @test ηdev ≈ ηref rtol = 1e-8
        end
        # the constant offset really is κ-independent
        Δ(κ) = log(pdev.dist, κ) - log(pref.dist, κ)
        @test Δ(0.5) ≈ Δ(5.0) atol = 1e-10
        @test Δ(0.5) ≈ Δ(50.0) atol = 1e-10
    end

    @testset "local-deviation gate: γ rule with a RANK-1 info-form cavity" begin
        # the softdot :x message has precision c·ffᵀ — singular, so the rule must
        # never convert it to moment form; the expression handles it natively
        f3 = [1.0, 0.4, -0.3]
        ξ1 = 0.8 .* f3
        Λ1 = 2.5 .* (f3 * f3')
        m3b = [0.05, 0.25, -0.15]
        V3b = [0.4 0.1 0.0; 0.1 0.5 -0.05; 0.0 -0.05 0.3]
        state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        msg = @call_rule MvNormalMeanScalePrecision(:γ, NaturalGradientMessage(TangentProjection(type = Quadrature(128)))) (
            m_out = MvNormalWeightedMeanPrecision(ξ1, Λ1), m_μ = MvNormalMeanCovariance(m3b, V3b),
            q_γ = GammaShapeRate(2.0, 1.5), meta = state
        )
        η = getnaturalparameters(project(
            TangentProjection(type = Quadrature(128)),
            GammaShapeRate(2.0, 1.5),
            Logpdf(MvNormalDeviationPrecisionMessage(ξ1, Λ1, m3b, V3b))
        ))
        @test msg isa GammaShapeRate
        @test shape(msg) ≈ η[1] + 1
        @test rate(msg) ≈ -η[2]
        @test state.nfired == 1
        @test isfinite(shape(msg)) && isfinite(rate(msg))
    end

    @testset "local-deviation gate: out/μ plug-in messages convolve the cavity" begin
        m3b = [0.05, 0.25, -0.15]
        V3b = [0.4 0.1 0.0; 0.1 0.5 -0.05; 0.0 -0.05 0.3]
        mγ = GammaShapeRate(3.0, 1.5)     # message mean κ̄ = 2.0 → added variance 0.5
        m_out_msg = @call_rule MvNormalMeanScalePrecision(:out, Marginalisation) (
            m_μ = MvNormalMeanCovariance(m3b, V3b), m_γ = mγ
        )
        @test m_out_msg isa MvNormalMeanCovariance
        @test mean(m_out_msg) == m3b
        @test cov(m_out_msg) ≈ V3b + 0.5 * I
        # toward the anchor: info-form convolution; for a full-rank cavity it must
        # equal the moment-form result N(m̃, Ṽ + κ̄⁻¹I)
        m_μ_msg = @call_rule MvNormalMeanScalePrecision(:μ, Marginalisation) (
            m_out = MvNormalMeanCovariance(m3, V3), m_γ = mγ
        )
        @test m_μ_msg isa MvNormalWeightedMeanPrecision
        @test mean(m_μ_msg) ≈ m3
        @test cov(m_μ_msg) ≈ V3 + 0.5 * I
        # a rank-1 cavity comes back as the rank-1 LR container (this is what keeps
        # the wbar equality chain from accumulating dense d×d matrices) and its
        # implied precision matches the dense formula κ̄Λ(Λ + κ̄I)⁻¹
        f3 = [1.0, 0.4, -0.3]
        Λ1 = 2.5 .* (f3 * f3')
        ξ1 = 0.8 .* f3
        m_μ_r1 = @call_rule MvNormalMeanScalePrecision(:μ, Marginalisation) (
            m_out = MvNormalWeightedMeanPrecision(ξ1, Λ1), m_γ = mγ
        )
        @test m_μ_r1 isa ProbabilisticEnsembling.LowRankNormalWeightedMeanPrecision
        κ̄ = 2.0
        @test m_μ_r1.scale .* (m_μ_r1.u * m_μ_r1.u') ≈ κ̄ .* Λ1 * inv(Λ1 + κ̄ * I)
        @test m_μ_r1.xi ≈ κ̄ .* ((Λ1 + κ̄ * I) \ ξ1)
    end

    @testset "deviation expression: rank-2 cavity matches the dense reference algebra" begin
        m3b = [0.05, 0.25, -0.15]
        V3b = [0.4 0.1 0.0; 0.1 0.5 -0.05; 0.0 -0.05 0.3]
        v1, v2 = [1.0, 0.4, -0.3], [0.2, -1.0, 0.5]
        Λ2 = 2.5 .* (v1 * v1') .+ 0.7 .* (v2 * v2')
        ξ2 = 1.5 .* v1 .- 0.4 .* v2
        p = MvNormalDeviationPrecisionMessage(ξ2, Λ2, m3b, V3b)
        dense_ref(κ) = begin
            S = V3b + I / κ
            A = Λ2 + inv(S)
            h = ξ2 + S \ m3b
            -(logdet(S) + logdet(A) + dot(m3b, S \ m3b) - dot(h, A \ h)) / 2
        end
        for κ in (0.1, 1.0, 10.0, 100.0)
            @test log(p, κ) ≈ dense_ref(κ) rtol = 1e-10
        end
    end

    @testset "deviation expression: O(d·r) evaluation allocates O(d), not O(d²)" begin
        d = 65
        f = collect(range(-1.0, 1.0; length = d))
        Vb = Matrix(0.5 * I, d, d) .+ 0.002 .* (f * f')
        p = MvNormalDeviationPrecisionMessage(0.8 .* f, 2.5 .* (f * f'), zeros(d), Vb)
        log(p, 2.0)   # warmup
        @test (@allocated log(p, 2.0)) < 16_384
    end

    @testset "integration: two-layer gated model end-to-end" begin
        nf, n, dim = 2, 10, 2
        ys = [0.62, -0.41, 1.13, 0.25, -0.77, 0.94, -1.22, 0.48, -0.35, 1.01]
        # forecaster 1 is good everywhere; forecaster 2 only in the first half
        preds = vcat((ys .+ 0.05)', (ys .+ vcat(fill(0.05, 5), fill(3.0, 5)))')
        feats = [j <= 5 ? [1.0, 0.1] : [0.1, 1.0] for j in 1:n]
        iters = 20

        log_deps_upper = NGMPDependencies(out = nothing, in = nothing)
        log_deps_lower = NGMPDependencies(out = nothing, in = nothing)
        gate_deps = NGMPDependencies(γ = nothing, projection = TangentProjection(type = Unscented))
        damping = DampingMeta(alpha = 0.2, beta = 0.0)

        init = @initialization begin
            q(u) = MvNormalMeanScalePrecision(zeros(2), 0.1)
            q(τu) = GammaShapeRate(2.0, 2.0)
            q(βu) = GammaShapeRate(2.0, 2.0)
            q(s) = NormalMeanVariance(0.0, 1.0)
            q(κ) = GammaShapeRate(1.0, 1.0)
            q(wbar) = MvNormalMeanScalePrecision(zeros(2), 0.1)
            q(w) = MvNormalMeanScalePrecision(zeros(2), 1.0)
            q(z) = NormalMeanVariance(0.0, 1.0)
            q(γ) = GammaShapeRate(1.0, 1.0)
            q(τ) = GammaShapeRate(2.0, 2.0)
            q(β) = GammaShapeRate(2.0, 2.0)
            # wbar is now inside the joint cluster: BP messages travel its n-node
            # equality chain, which deadlocks without an initial message
            μ(wbar) = MvNormalMeanScalePrecision(zeros(2), 0.1)
        end

        res = infer(
            model = gated_toy(features = feats, predictions = preds, dim = dim,
                              log_deps_upper = log_deps_upper, log_deps_lower = log_deps_lower,
                              gate_deps = gate_deps, damping = damping),
            data = (y = ys,),
            constraints = gated_toy_constraints(),
            initialization = init,
            iterations = iters,
            free_energy = false,
            options = (limit_stack_depth = 500,)
        )

        @test length(gate_deps.states) == nf * n
        @test length(log_deps_upper.states) == 2 * nf * n
        @test length(log_deps_lower.states) == 2 * nf * n
        @test all(state -> state.nfired == iters, gate_deps.states)
        @test all(state -> state.nfired == iters, log_deps_upper.states)

        qκ = last(res.posteriors[:κ])
        qw = last(res.posteriors[:w])
        qz = last(res.posteriors[:z])
        @test all(k -> shape(k) > 0 && rate(k) > 0, qκ)
        @test all(wq -> all(isfinite, cov(wq)), qw)

        # regional response through the local weights: forecaster 2's log-precision
        # must drop in the second half, where its predictions are 3.0 off — the
        # differentiation is only expressible via the per-observation w[2,j]
        mz2_good = mean(mean.(qz[2, 1:5]))
        mz2_bad = mean(mean.(qz[2, 6:10]))
        @test mz2_bad < mz2_good
    end
end
