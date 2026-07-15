import ProbabilisticEnsembling: Exp
import ReactiveMP: @call_marginalrule
import SurrogateModelling:
    NaturalGradientMP,
    SoftplusBackwardMessage,
    SoftplusForwardMessage,
    SoftplusGaussianBackwardMessage,
    _inverse_softplus,
    _softplus,
    _softplus_forward_first_derivative,
    _softplus_forward_second_derivative,
    _softplus_gaussian_backward_first_derivative,
    _softplus_gaussian_backward_second_derivative

@model function softplus_toy(y, deps, damping)
    a ~ Normal(mean = 0.0, variance = 1.0)
    γ ~ GammaShapeRate(2.0, 2.0)
    γ ~ Softplus(a) where {dependencies = deps, meta = damping}
    for index in eachindex(y)
        y[index] ~ NormalMeanPrecision(0.0, γ)
    end
end

@model function gaussian_softplus_softdot_toy(
    y,
    deps,
    damping,
    product_deps,
    product_damping,
)
    a ~ NormalMeanVariance(0.541324854612918, 0.025)
    z ~ NormalMeanVariance(1.0, 0.05)
    g ~ Softplus(a) where {dependencies = deps, meta = damping}
    y ~ softdot(g, z, 4.0) where {
        dependencies = product_deps,
        meta = product_damping,
    }
end

@constraints function gaussian_softplus_softdot_constraints()
    q(a, g, z) = q(a)q(g)q(z)
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

    @testset "Gaussian Softplus composition rules" begin
        backward = SoftplusGaussianBackwardMessage(0.8, 0.4)
        for x in (-1000.0, -3.0, 0.0, 4.0)
            expected = logpdf(Normal(0.8, sqrt(0.4)), _softplus(x))
            @test log(backward, x) ≈ expected
            @test isfinite(log(backward, x))
        end

        first_difference(f, x; h = 1e-5) = (f(x + h) - f(x - h)) / (2h)
        second_difference(f, x; h = 1e-4) =
            (f(x + h) - 2f(x) + f(x - h)) / h^2

        forward_derivative_target = Logpdf(SoftplusForwardMessage(0.3, 0.8))
        for y in (0.2, 0.7, 2.0)
            logtarget(value) = log(forward_derivative_target.dist, value)
            @test _softplus_forward_first_derivative(forward_derivative_target, y) ≈
                  first_difference(logtarget, y) rtol = 1e-6
            @test _softplus_forward_second_derivative(forward_derivative_target, y) ≈
                  second_difference(logtarget, y) rtol = 1e-5
        end

        backward_derivative_target = Logpdf(backward)
        for x in (-2.0, 0.0, 2.0)
            logtarget(value) = log(backward_derivative_target.dist, value)
            @test _softplus_gaussian_backward_first_derivative(
                backward_derivative_target,
                x,
            ) ≈ first_difference(logtarget, x) rtol = 1e-6
            @test _softplus_gaussian_backward_second_derivative(
                backward_derivative_target,
                x,
            ) ≈ second_difference(logtarget, x) rtol = 1e-5
        end

        # This is a delta approximation of the exact forward LOG-MESSAGE at
        # q(out), not a delta-method moment transform of softplus(q(in)).
        for (input, output_projection_point) in (
            (NormalMeanVariance(-1.0, 0.8), NormalMeanVariance(0.5, 1e-4)),
            (NormalMeanVariance(0.5, 0.7), NormalMeanVariance(1.0, 4e-4)),
            (NormalMeanVariance(2.0, 0.3), NormalMeanVariance(2.2, 1e-3)),
        )
            exact = Logpdf(SoftplusForwardMessage(mean(input), var(input)))
            expansion_point = mean(output_projection_point)
            d1 = _softplus_forward_first_derivative(exact, expansion_point)
            d2 = _softplus_forward_second_derivative(exact, expansion_point)
            expected_natural = [d1 - expansion_point * d2, d2 / 2]

            delta_site = project(
                TangentProjection(type = DeltaApproximation),
                output_projection_point,
                exact,
            )
            delta_natural = getnaturalparameters(delta_site)
            @test delta_natural ≈ expected_natural atol = 1e-11 rtol = 1e-11

            # With q(out) narrow enough that all selected UT/GH points stay
            # positive, those projections are finite and converge to the same
            # touching quadratic.
            ut_natural = getnaturalparameters(project(
                TangentProjection(type = Unscented), output_projection_point, exact
            ))
            quadrature_natural = getnaturalparameters(project(
                TangentProjection(type = Quadrature(128)), output_projection_point, exact
            ))
            @test maximum(abs.(delta_natural .- ut_natural) ./ (1 .+ abs.(ut_natural))) < 0.02
            @test maximum(abs.(delta_natural .- quadrature_natural) ./ (1 .+ abs.(quadrature_natural))) < 0.02

            state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
            projected = @call_rule Softplus(
                :out,
                NaturalGradientMessage(TangentProjection(type = DeltaApproximation)),
            ) (
                m_in = input,
                q_out = output_projection_point,
                meta = state,
            )
            @test projected isa NormalWeightedMeanPrecision
            @test weightedmean(projected) ≈ expected_natural[1]
            @test precision(projected) ≈ -2expected_natural[2]
            @test state.nfired == 1
        end

        @test_throws DomainError project(
            TangentProjection(type = DeltaApproximation),
            NormalMeanVariance(-0.1, 0.01),
            Logpdf(SoftplusForwardMessage(0.0, 1.0)),
        )
        @test_throws ArgumentError project(
            TangentProjection(type = ClosedForm),
            NormalMeanVariance(1.0, 0.01),
            Logpdf(SoftplusForwardMessage(0.0, 1.0)),
        )

        for (gate_mean, gate_variance, input_mean, input_variance) in (
            (0.8, 0.4, -2.0, 1e-3),
            (1.5, 2.0, 0.0, 2e-3),
            (3.0, 0.2, 2.0, 1e-3),
        )
            target = Logpdf(SoftplusGaussianBackwardMessage(gate_mean, gate_variance))
            q = NormalMeanVariance(input_mean, input_variance)
            d1 = _softplus_gaussian_backward_first_derivative(target, input_mean)
            d2 = _softplus_gaussian_backward_second_derivative(target, input_mean)
            expected_natural = [d1 - input_mean * d2, d2 / 2]
            ηdelta = getnaturalparameters(project(
                TangentProjection(type = DeltaApproximation), q, target
            ))
            ηut = getnaturalparameters(project(TangentProjection(type = Unscented), q, target))
            η256 = getnaturalparameters(project(TangentProjection(type = Quadrature(256)), q, target))
            @test ηdelta ≈ expected_natural atol = 1e-11 rtol = 1e-11
            @test maximum(abs.(ηdelta .- ηut) ./ (1 .+ abs.(ηut))) < 0.01
            @test maximum(abs.(ηdelta .- η256) ./ (1 .+ abs.(η256))) < 0.01

            state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
            message = @call_rule Softplus(
                :in,
                NaturalGradientMessage(TangentProjection(type = DeltaApproximation)),
            ) (m_out = NormalMeanVariance(gate_mean, gate_variance), q_in = q, meta = state)
            @test message isa NormalWeightedMeanPrecision
            @test weightedmean(message) ≈ expected_natural[1]
            @test precision(message) ≈ -2expected_natural[2]
            @test state.nfired == 1
        end
    end

    @testset "integration: Gaussian Softplus gate composes with random-random softdot" begin
        dependencies = NGMPDependencies(
            out = nothing,
            in = nothing,
            projection = TangentProjection(type = DeltaApproximation),
        )
        product_dependencies = NGMPDependencies(θ = nothing)
        initialization = @initialization begin
            q(a) = NormalMeanVariance(0.541324854612918, 0.025)
            q(g) = NormalMeanVariance(1.0, 0.01)
            q(z) = NormalMeanVariance(1.0, 0.05)
        end
        iterations = 10
        result = infer(
            model = gaussian_softplus_softdot_toy(
                deps = dependencies,
                damping = DampingMeta(
                    alpha = 1.0,
                    beta = 0.0,
                    max_step = 1.0,
                ),
                product_deps = product_dependencies,
                product_damping = DampingMeta(
                    alpha = 0.25,
                    beta = 0.0,
                    max_step = 0.01,
                ),
            ),
            data = (y = 1.0,),
            constraints = gaussian_softplus_softdot_constraints(),
            initialization = initialization,
            iterations = iterations,
            free_energy = true,
        )

        qa = last(result.posteriors[:a])
        qg = last(result.posteriors[:g])
        qz = last(result.posteriors[:z])
        @test all(isfinite, (mean(qa), var(qa), mean(qg), var(qg), mean(qz), var(qz)))
        @test var(qa) > 0 && var(qg) > 0 && var(qz) > 0
        @test mean(qg) > 0
        @test all(isfinite, result.free_energy)
        @test length(dependencies.states) == 2
        @test all(state -> state.nfired == iterations, dependencies.states)
        @test length(product_dependencies.states) == 1
        @test all(state -> state.nfired >= iterations, product_dependencies.states)
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
