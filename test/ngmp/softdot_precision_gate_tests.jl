import ProbabilisticEnsembling
import ProbabilisticEnsembling: Log
import SurrogateModelling: NaturalGradientMP, NormalPrecisionMessage, MomentForm
import LinearAlgebra: dot as sp_dot, Symmetric

# Gated-precision toy: κ[i,j] (from the upper softdot → Gamma → Log copy) IS the
# lower softdot's precision — it decides how strongly z[i,j] couples to w[i]ᵀf_j.
# All per-(i,j) latents are scalars; only u[i], w[i] are vectors.
@model function gated_precision_toy(y, features, predictions, dim, log_deps_upper, log_deps_lower, sd_deps, damping)
    local u, τu, βu, s, κ, w, z, γ, β
    nf = size(predictions, 1)
    for i in 1:nf
        u[i] ~ MvNormalMeanScalePrecision(zeros(dim), 0.1)
        τu[i] ~ GammaShapeRate(2.0, 2.0)
        βu[i] ~ GammaShapeRate(2.0, 2.0)
        w[i] ~ MvNormalMeanScalePrecision(zeros(dim), 0.1)
        β[i] ~ GammaShapeRate(2.0, 2.0)
    end
    for j in 1:length(y), i in 1:nf
        s[i, j] ~ softdot(features[j], u[i], τu[i])
        κ[i, j] ~ GammaShapeRate(1.0, βu[i])
        s[i, j] ~ Log(κ[i, j]) where { dependencies = log_deps_upper, meta = damping }
        z[i, j] ~ softdot(features[j], w[i], κ[i, j]) where { dependencies = sd_deps, meta = damping }
        γ[i, j] ~ GammaShapeRate(1.0, β[i])
        z[i, j] ~ Log(γ[i, j]) where { dependencies = log_deps_lower, meta = damping }
        y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
    end
end

@constraints function gated_precision_toy_constraints()
    q(u, τu, βu, s, κ, w, z, γ, β) = q(τu)q(βu)q(u, s, κ, w, z, γ)q(β)
    q(u)::MomentForm()
    q(w)::MomentForm()
end

@testset "softdot precision gate (per-observation κ as the softdot precision)" begin
    f3 = [1.0, 0.4, -0.3]
    m3w = [0.2, -0.5, 0.7]
    V3w = [0.8 0.2 -0.1; 0.2 0.6 0.15; -0.1 0.15 0.9]

    @testset "γ rule matches the quadrature projection of NormalPrecisionMessage" begin
        for (mz, vz) in ((0.6, 0.5), (-1.2, 2.0)), (a, b) in ((2.0, 1.5), (1.0, 1.0))
            state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
            msg = @call_rule softdot(:γ, NaturalGradientMessage(TangentProjection(type = Quadrature(128)))) (
                m_y = NormalMeanVariance(mz, vz), m_x = MvNormalMeanCovariance(m3w, V3w),
                q_θ = PointMass(f3), q_γ = GammaShapeRate(a, b), meta = state
            )
            η = getnaturalparameters(project(
                TangentProjection(type = Quadrature(128)),
                GammaShapeRate(a, b),
                Logpdf(NormalPrecisionMessage(mz, sp_dot(f3, m3w), vz + sp_dot(f3, V3w * f3)))
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
        for (mz, vz, a, b) in ((0.6, 0.5, 2.0, 1.5), (0.5, 0.4, 2.4, 1.4), (0.55, 0.35, 2.7, 1.3))
            msg = @call_rule softdot(:γ, NaturalGradientMessage(TangentProjection(type = Quadrature(128)))) (
                m_y = NormalMeanVariance(mz, vz), m_x = MvNormalMeanCovariance(m3w, V3w),
                q_θ = PointMass(f3), q_γ = GammaShapeRate(a, b), meta = state
            )
            ηt = collect(getnaturalparameters(project(
                TangentProjection(type = Quadrature(128)),
                GammaShapeRate(a, b),
                Logpdf(NormalPrecisionMessage(mz, sp_dot(f3, m3w), vz + sp_dot(f3, V3w * f3)))
            )))
            @. mom = β * mom + α * (ηt - η)
            @. η += mom
            @test shape(msg) ≈ η[1] + 1
            @test rate(msg) ≈ -η[2]
        end
    end

    @testset "hybrid γ rule recovers cavities from joint q(z,w)" begin
        # Build q(z,w) by multiplying independent Gaussian cavities by the
        # current softdot factor. The rule must divide that factor back out
        # before constructing the exact NormalPrecisionMessage.
        mz, vz = 0.6, 0.5
        κbar = 2.0 / 1.5
        Λw = inv(Symmetric(V3w))
        ξjoint = vcat(mz / vz, Λw * m3w)
        Λcavity = zeros(4, 4)
        Λcavity[1, 1] = inv(vz)
        Λcavity[2:4, 2:4] = Λw
        residual_map = vcat(1.0, .-f3)
        qjoint = MvNormalWeightedMeanPrecision(
            ξjoint,
            Λcavity .+ κbar .* (residual_map * residual_map'),
        )
        _, Vjoint = mean_cov(qjoint)
        @test any(x -> !iszero(x), Vjoint[1, 2:4])

        state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        msg = @call_rule softdot(:γ, NaturalGradientMessage(TangentProjection(type = Quadrature(128)))) (
            q_y_x = qjoint, q_θ = PointMass(f3),
            q_γ = GammaShapeRate(2.0, 1.5), meta = state
        )
        η = getnaturalparameters(project(
            TangentProjection(type = Quadrature(128)),
            GammaShapeRate(2.0, 1.5),
            Logpdf(NormalPrecisionMessage(
                mz,
                sp_dot(f3, m3w),
                vz + sp_dot(f3, V3w * f3),
            ))
        ))
        @test msg isa GammaShapeRate
        @test shape(msg) ≈ η[1] + 1
        @test rate(msg) ≈ -η[2]
        @test state.nfired == 1
    end

    @testset "y/x plug-in variants collapse κ to the Gamma-message mean" begin
        mγ = GammaShapeRate(3.0, 1.5)               # κ̄ = 2.0
        # info-form and moment-form :y variants agree with the direct formula
        Λw = inv(Symmetric(V3w))
        ξw = Λw * m3w
        my_info = @call_rule softdot(:y, Marginalisation) (
            m_x = MvNormalWeightedMeanPrecision(ξw, Matrix(Λw)), m_γ = mγ, q_θ = PointMass(f3)
        )
        my_mom = @call_rule softdot(:y, Marginalisation) (
            m_x = MvNormalMeanCovariance(m3w, V3w), m_γ = mγ, q_θ = PointMass(f3)
        )
        @test mean(my_info) ≈ sp_dot(f3, m3w)
        @test var(my_info) ≈ sp_dot(f3, V3w * f3) + 0.5
        @test mean(my_mom) ≈ mean(my_info)
        @test var(my_mom) ≈ var(my_info)
        # :x is the stock structured rank-1 site with κ̄ plugged in
        mx = @call_rule softdot(:x, Marginalisation) (
            m_y = NormalMeanVariance(0.6, 0.5), m_γ = mγ, q_θ = PointMass(f3)
        )
        c = 1 / (0.5 + 0.5)
        @test mx isa MvNormalWeightedMeanPrecision
        @test weightedmean(mx) ≈ (c * 0.6) .* f3
        @test precision(mx) ≈ c .* (f3 * f3')
    end

    @testset "integration: gated-precision model end-to-end" begin
        nf, n, dim = 2, 10, 2
        ys = [0.62, -0.41, 1.13, 0.25, -0.77, 0.94, -1.22, 0.48, -0.35, 1.01]
        # forecaster 1 is good everywhere; forecaster 2 only in the first half
        preds = vcat((ys .+ 0.05)', (ys .+ vcat(fill(0.05, 5), fill(3.0, 5)))')
        feats = [j <= 5 ? [1.0, 0.1] : [0.1, 1.0] for j in 1:n]
        iters = 20

        log_deps_upper = NGMPDependencies(out = nothing, in = nothing)
        log_deps_lower = NGMPDependencies(out = nothing, in = nothing)
        sd_deps = NGMPDependencies(γ = nothing, projection = TangentProjection(type = Unscented))
        damping = DampingMeta(alpha = 0.2, beta = 0.0)

        init = @initialization begin
            q(u) = MvNormalMeanScalePrecision(zeros(2), 0.1)
            q(τu) = GammaShapeRate(2.0, 2.0)
            q(βu) = GammaShapeRate(2.0, 2.0)
            q(s) = NormalMeanVariance(0.0, 1.0)
            q(κ) = GammaShapeRate(1.0, 1.0)
            q(w) = MvNormalMeanScalePrecision(zeros(2), 0.1)
            q(z) = NormalMeanVariance(0.0, 1.0)
            q(γ) = GammaShapeRate(1.0, 1.0)
            q(β) = GammaShapeRate(2.0, 2.0)
            # w's equality chain carries BP messages whose first sweep needs the
            # other nodes' sites (through m_γ) — seed it to break the deadlock
            μ(w) = MvNormalMeanScalePrecision(zeros(2), 0.1)
        end

        res = infer(
            model = gated_precision_toy(features = feats, predictions = preds, dim = dim,
                                        log_deps_upper = log_deps_upper, log_deps_lower = log_deps_lower,
                                        sd_deps = sd_deps, damping = damping),
            data = (y = ys,),
            constraints = gated_precision_toy_constraints(),
            initialization = init,
            iterations = iters,
            free_energy = false,
            options = (limit_stack_depth = 500,)
        )

        @test length(sd_deps.states) == nf * n
        @test length(log_deps_upper.states) == 2 * nf * n
        @test length(log_deps_lower.states) == 2 * nf * n
        @test all(state -> state.nfired == iters, sd_deps.states)
        @test all(state -> state.nfired == iters, log_deps_upper.states)

        qκ = last(res.posteriors[:κ])
        qz = last(res.posteriors[:z])
        qw = last(res.posteriors[:w])
        @test all(k -> shape(k) > 0 && rate(k) > 0, qκ)
        @test all(wq -> all(isfinite, cov(wq)), qw)

        # regional response: forecaster 2's log-precision must drop in the second
        # half, where its predictions are 3.0 off — with static w[2], the
        # differentiation is only expressible through the per-observation gate κ
        mz2_good = mean(mean.(qz[2, 1:5]))
        mz2_bad = mean(mean.(qz[2, 6:10]))
        @test mz2_bad < mz2_good
    end
end
