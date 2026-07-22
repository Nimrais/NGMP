import SurrogateModelling: GaussianStudentTMessage, NaturalGradientMP,
    NormalPrecisionMessage, StudentTMessage, project_to_gamma, project_to_normal
import ProbabilisticEnsembling: Log
import FastGaussQuadrature
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

@model function latent_ngbp_normal_toy(y, deps, damping)
    out ~ NormalMeanVariance(0.5, 1.0)
    μ ~ NormalMeanVariance(0.0, 1.0)
    τ ~ GammaShapeRate(2.0, 2.0)
    out ~ NormalMeanPrecision(μ, τ) where {
        dependencies = deps,
        meta = damping,
    }
    y ~ NormalMeanVariance(out, 0.5)
end

@model function fixed_mean_two_expert_consensus_toy(
    expert_mean_1,
    expert_mean_2,
    y,
    obs_deps,
    obs_damping,
    log_deps,
    log_damping,
)
    out ~ NormalMeanVariance(0.0, 10.0)
    z_1 ~ NormalMeanVariance(0.0, 1.0)
    z_2 ~ NormalMeanVariance(0.0, 1.0)
    beta_1 ~ GammaShapeRate(10.0, 10.0)
    beta_2 ~ GammaShapeRate(10.0, 10.0)
    gamma_1 ~ GammaShapeRate(1.0, beta_1)
    gamma_2 ~ GammaShapeRate(1.0, beta_2)
    z_1 ~ Log(gamma_1) where {
        dependencies = log_deps,
        meta = log_damping,
    }
    z_2 ~ Log(gamma_2) where {
        dependencies = log_deps,
        meta = log_damping,
    }
    out ~ NormalMeanPrecision(expert_mean_1, gamma_1) where {
        dependencies = obs_deps,
        meta = obs_damping,
    }
    out ~ NormalMeanPrecision(expert_mean_2, gamma_2) where {
        dependencies = obs_deps,
        meta = obs_damping,
    }
    y ~ NormalMeanVariance(out, 0.1)
end

@constraints function fixed_mean_two_expert_consensus_constraints()
    q(out, z_1, z_2, gamma_1, gamma_2, beta_1, beta_2) =
        q(out, z_1, z_2, gamma_1, gamma_2)q(beta_1)q(beta_2)
end

function run_fixed_mean_two_expert_consensus(expert_mean_1; iterations = 12)
    observation_dependencies = NGMPDependencies(
        out = nothing,
        τ = nothing,
        projection = TangentProjection(type = DeltaApproximation),
    )
    log_dependencies = NGMPDependencies(out = nothing, in = nothing)
    initialization = @initialization begin
        q(out) = NormalMeanVariance(1.0, 1.0)
        q(z_1) = NormalMeanVariance(0.0, 1.0)
        q(z_2) = NormalMeanVariance(0.0, 1.0)
        q(gamma_1) = GammaShapeRate(2.0, 2.0)
        q(gamma_2) = GammaShapeRate(2.0, 2.0)
        q(beta_1) = GammaShapeRate(10.0, 10.0)
        q(beta_2) = GammaShapeRate(10.0, 10.0)
        μ(out) = NormalMeanVariance(1.0, 1.0)
        μ(gamma_1) = GammaShapeRate(2.0, 2.0)
        μ(gamma_2) = GammaShapeRate(2.0, 2.0)
    end
    result = infer(
        model = fixed_mean_two_expert_consensus_toy(
            expert_mean_1 = expert_mean_1,
            expert_mean_2 = 1.0,
            obs_deps = observation_dependencies,
            obs_damping = DampingMeta(alpha = 0.2, beta = 0.0),
            log_deps = log_dependencies,
            log_damping = DampingMeta(alpha = 0.2, beta = 0.0),
        ),
        data = (y = 1.0,),
        constraints = fixed_mean_two_expert_consensus_constraints(),
        initialization = initialization,
        iterations = iterations,
        free_energy = false,
        returnvars = (out = KeepLast(),),
        disable_inference_error_hint = true,
    )
    return result, observation_dependencies
end

function gaussian_student_t_reference_log(message, x; order = 128)
    nodes, weights = FastGaussQuadrature.gausshermite(order)
    scale = sqrt(2 * message.v)
    component_logs = map(nodes, weights) do node, weight
        residual = x - (message.m + scale * node)
        return log(weight / sqrt(pi)) -
               (2 * message.a + 1) / 2 * log(2 * message.b + residual^2)
    end
    maximum_log = maximum(component_logs)
    return maximum_log + log(sum(exp.(component_logs .- maximum_log)))
end

@testset "GaussianStudentTMessage" begin
    @testset "zero cavity variance reduces to StudentTMessage" begin
        convolution = GaussianStudentTMessage(0.7, 0.0, 2.5, 1.3)
        student = StudentTMessage(0.7, 2.5, 1.3)
        for x in (-2.0, 0.7, 3.0)
            @test log(convolution, x) ≈ log(student, x) atol = 1e-12
        end
    end

    @testset "16-node evaluator tracks dense quadrature" begin
        message = GaussianStudentTMessage(0.7, 0.8, 2.5, 1.3)
        for x in (-1.0, 0.7, 2.0)
            @test log(message, x) ≈ gaussian_student_t_reference_log(message, x) atol = 5e-4
            step = 1e-4
            first_fd = (log(message, x + step) - log(message, x - step)) / (2 * step)
            second_fd =
                (log(message, x + step) - 2 * log(message, x) + log(message, x - step)) /
                step^2
            first, second = SurrogateModelling._gaussian_student_t_logderivatives(message, x)
            @test first ≈ first_fd atol = 1e-7
            @test second ≈ second_fd atol = 1e-5
        end
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
            # the projection strategy now travels inside the NaturalGradientMessage
            # (self-callable instance — the default would be the ClosedForm/delta path)
            msg = @call_rule NormalMeanPrecision(:τ, NaturalGradientMessage(TangentProjection(type = Quadrature(128)))) (
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
            msg = @call_rule NormalMeanPrecision(:μ, NaturalGradientMessage(TangentProjection(type = Quadrature(128)))) (
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

    @testset "three-latent-edge rules match their tangent projections" begin
        projection = TangentProjection(type = Unscented)
        m_out = NormalMeanVariance(0.8, 0.4)
        m_μ = NormalMeanVariance(0.2, 0.7)
        m_τ = GammaShapeRate(2.5, 1.4)
        q_out = NormalMeanVariance(0.6, 0.9)
        q_μ = NormalMeanVariance(0.3, 1.1)
        q_τ = GammaShapeRate(2.0, 1.7)

        out_state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        out_message = @call_rule NormalMeanPrecision(:out, NaturalGradientMessage(projection)) (
            m_μ = m_μ, m_τ = m_τ, q_out = q_out, meta = out_state
        )
        out_target = project(
            projection,
            q_out,
            Logpdf(GaussianStudentTMessage(mean(m_μ), var(m_μ), shape(m_τ), rate(m_τ))),
        )
        @test weightedmean(out_message) ≈ getnaturalparameters(out_target)[1]
        @test precision(out_message) ≈ -2 * getnaturalparameters(out_target)[2]

        mean_state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        mean_message = @call_rule NormalMeanPrecision(:μ, NaturalGradientMessage(projection)) (
            m_out = m_out, m_τ = m_τ, q_μ = q_μ, meta = mean_state
        )
        mean_target = project(
            projection,
            q_μ,
            Logpdf(GaussianStudentTMessage(mean(m_out), var(m_out), shape(m_τ), rate(m_τ))),
        )
        @test weightedmean(mean_message) ≈ getnaturalparameters(mean_target)[1]
        @test precision(mean_message) ≈ -2 * getnaturalparameters(mean_target)[2]

        precision_state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        precision_message = @call_rule NormalMeanPrecision(:τ, NaturalGradientMessage(projection)) (
            m_out = m_out, m_μ = m_μ, q_τ = q_τ, meta = precision_state
        )
        precision_target = project(
            projection,
            q_τ,
            Logpdf(
                NormalPrecisionMessage(
                    mean(m_out),
                    mean(m_μ),
                    var(m_out) + var(m_μ),
                ),
            ),
        )
        @test shape(precision_message) ≈ getnaturalparameters(precision_target)[1] + 1
        @test rate(precision_message) ≈ -getnaturalparameters(precision_target)[2]
    end

    @testset "fixed-mean consensus rules match their tangent projections" begin
        projection = TangentProjection(type = Unscented)
        m_out = NormalMeanVariance(0.8, 0.4)
        m_μ = PointMass(0.2)
        m_τ = GammaShapeRate(2.5, 1.4)
        q_out = NormalMeanVariance(0.6, 0.9)
        q_τ = GammaShapeRate(2.0, 1.7)

        out_state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        out_message = @call_rule NormalMeanPrecision(
            :out,
            NaturalGradientMessage(projection),
        ) (
            m_μ = m_μ,
            m_τ = m_τ,
            q_out = q_out,
            meta = out_state,
        )
        out_target = project(
            projection,
            q_out,
            Logpdf(StudentTMessage(mean(m_μ), shape(m_τ), rate(m_τ))),
        )
        @test weightedmean(out_message) ≈ getnaturalparameters(out_target)[1]
        @test precision(out_message) ≈ -2 * getnaturalparameters(out_target)[2]

        precision_state =
            NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        precision_message = @call_rule NormalMeanPrecision(
            :τ,
            NaturalGradientMessage(projection),
        ) (
            m_out = m_out,
            m_μ = m_μ,
            q_τ = q_τ,
            meta = precision_state,
        )
        precision_target = project(
            projection,
            q_τ,
            Logpdf(
                NormalPrecisionMessage(
                    mean(m_out),
                    mean(m_μ),
                    var(m_out),
                ),
            ),
        )
        @test shape(precision_message) ≈
              getnaturalparameters(precision_target)[1] + 1
        @test rate(precision_message) ≈
              -getnaturalparameters(precision_target)[2]
        @test out_state.nfired == 1
        @test precision_state.nfired == 1

        # Mean-field q(out)q(τ): every interface arrives as a local marginal.
        factorized_out_state =
            NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        factorized_out = @call_rule NormalMeanPrecision(
            :out,
            NaturalGradientMessage(projection),
        ) (
            q_out = q_out,
            q_μ = m_μ,
            q_τ = q_τ,
            meta = factorized_out_state,
        )
        factorized_out_target = project(
            projection,
            q_out,
            Logpdf(StudentTMessage(mean(m_μ), shape(q_τ), rate(q_τ))),
        )
        @test weightedmean(factorized_out) ≈
              getnaturalparameters(factorized_out_target)[1]
        @test precision(factorized_out) ≈
              -2 * getnaturalparameters(factorized_out_target)[2]

        factorized_precision_state =
            NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        factorized_precision = @call_rule NormalMeanPrecision(
            :τ,
            NaturalGradientMessage(projection),
        ) (
            q_out = q_out,
            q_μ = m_μ,
            q_τ = q_τ,
            meta = factorized_precision_state,
        )
        factorized_precision_target = project(
            projection,
            q_τ,
            Logpdf(
                NormalPrecisionMessage(mean(q_out), mean(m_μ), var(q_out)),
            ),
        )
        @test shape(factorized_precision) ≈
              getnaturalparameters(factorized_precision_target)[1] + 1
        @test rate(factorized_precision) ≈
              -getnaturalparameters(factorized_precision_target)[2]

        # Structured q(out, τ): the opposite stochastic interface is a cavity
        # message, while the fixed data mean remains a PointMass marginal.
        structured_out_state =
            NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        structured_out = @call_rule NormalMeanPrecision(
            :out,
            NaturalGradientMessage(projection),
        ) (
            m_τ = m_τ,
            q_out = q_out,
            q_μ = m_μ,
            meta = structured_out_state,
        )
        @test weightedmean(structured_out) ≈
              getnaturalparameters(out_target)[1]
        @test precision(structured_out) ≈
              -2 * getnaturalparameters(out_target)[2]

        structured_precision_state =
            NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        structured_precision = @call_rule NormalMeanPrecision(
            :τ,
            NaturalGradientMessage(projection),
        ) (
            m_out = m_out,
            q_μ = m_μ,
            q_τ = q_τ,
            meta = structured_precision_state,
        )
        @test shape(structured_precision) ≈ shape(precision_message)
        @test rate(structured_precision) ≈ rate(precision_message)
    end

    @testset "damped recursion on the τ edge matches manual η replay" begin
        α, β = 0.5, 0.2
        state = NGMPEdgeState(DampingMeta(alpha = α, beta = β))
        η = [0.0, 0.0]
        mom = [0.0, 0.0]
        y = 9.7
        for (m̃, ṽ, a, b) in ((10.1, 0.3, 2.0, 1.5), (9.9, 0.2, 2.4, 1.4), (9.8, 0.15, 2.7, 1.3))
            msg = @call_rule NormalMeanPrecision(:τ, NaturalGradientMessage(TangentProjection(type = Quadrature(128)))) (
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

        deps = NGMPDependencies(μ = nothing, τ = nothing, projection = TangentProjection(type = Quadrature(128)))
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


    @testset "integration: all NormalMeanPrecision interfaces latent" begin
        for projection in (
            TangentProjection(type = DeltaApproximation),
            TangentProjection(type = Unscented),
            TangentProjection(type = Quadrature(16)),
        )
            deps = NGMPDependencies(
                out = nothing,
                μ = nothing,
                τ = nothing,
                projection = projection,
            )
            init = @initialization begin
                q(out) = NormalMeanVariance(0.5, 1.0)
                q(μ) = NormalMeanVariance(0.0, 1.0)
                q(τ) = GammaShapeRate(2.0, 2.0)
            end
            result = infer(
                model = latent_ngbp_normal_toy(
                    deps = deps,
                    damping = DampingMeta(alpha = 0.2, beta = 0.0),
                ),
                data = (y = 0.7,),
                initialization = init,
                iterations = 10,
                disable_inference_error_hint = true,
            )
            q_out = last(result.posteriors[:out])
            q_μ = last(result.posteriors[:μ])
            q_τ = last(result.posteriors[:τ])
            @test isfinite(mean(q_out)) && var(q_out) > 0
            @test isfinite(mean(q_μ)) && var(q_μ) > 0
            @test shape(q_τ) > 0 && rate(q_τ) > 0
            @test length(deps.states) == 3
            @test all(state -> state.nfired == 10, deps.states)
        end
    end


    @testset "integration: a competing expert changes the other gate through out" begin
        agreeing, agreeing_dependencies =
            run_fixed_mean_two_expert_consensus(1.0)
        conflicting, conflicting_dependencies =
            run_fixed_mean_two_expert_consensus(-3.0)

        agreeing_sites = [
            state.message for state in agreeing_dependencies.states
            if state.message isa GammaDistributionsFamily
        ]
        conflicting_sites = [
            state.message for state in conflicting_dependencies.states
            if state.message isa GammaDistributionsFamily
        ]

        @test length(agreeing_sites) == 2
        @test length(conflicting_sites) == 2
        # Asynchronous equality-chain scheduling leaves a tiny order effect.
        @test mean(agreeing_sites[1]) ≈ mean(agreeing_sites[2]) rtol = 5e-4
        @test mean(conflicting_sites[1]) < mean(conflicting_sites[2])
        @test abs(mean(conflicting_sites[2]) - mean(agreeing_sites[2])) > 1e-4
        @test length(agreeing_dependencies.states) == 4
        @test length(conflicting_dependencies.states) == 4
        @test all(state -> state.nfired >= 1, agreeing_dependencies.states)
        @test all(state -> state.nfired >= 1, conflicting_dependencies.states)
    end
end
