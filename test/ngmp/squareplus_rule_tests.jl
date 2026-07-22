import ReactiveMP: @call_marginalrule
import ProbabilisticEnsembling: LowRankMeta
import SurrogateModelling:
    SquareplusBackwardMessage,
    SquareplusForwardMessage,
    _inverse_squareplus,
    _log_squareplus,
    _squareplus,
    _squareplus_backward_first_derivative,
    _squareplus_backward_second_derivative,
    _squareplus_forward_first_derivative,
    _squareplus_forward_second_derivative

@model function squareplus_toy(y, dependencies, damping)
    z ~ NormalMeanVariance(0.0, 1.0)
    gamma ~ GammaShapeRate(2.0, 2.0)
    gamma ~ Squareplus(z) where {
        dependencies = dependencies,
        meta = damping,
    }
    for index in eachindex(y)
        y[index] ~ NormalMeanPrecision(0.0, gamma)
    end
end

@model function squareplus_native_toy(
    n_obs,
    y,
    features,
    predictions,
    priors,
    dependencies,
    damping,
)
    local z, gamma
    w ~ priors[:w]
    tau ~ priors[:tau]
    beta ~ priors[:beta]
    for observation in 1:n_obs
        z[observation] ~ softdot(
            features[observation],
            w,
            tau,
        ) where {meta = LowRankMeta()}
        gamma[observation] ~ GammaShapeRate(1.0, beta)
        gamma[observation] ~ Squareplus(z[observation]) where {
            dependencies = dependencies,
            meta = damping,
        }
        y[observation] ~ NormalMeanPrecision(
            predictions[observation],
            gamma[observation],
        )
    end
end

@model function squareplus_structured_out_toy(
    n_experts,
    n_obs,
    y,
    features,
    predictions,
    priors,
    activation_dependencies,
    activation_damping,
    observation_dependencies,
    observation_damping,
)
    local w, z, gamma, tau, beta, out

    obs_noise ~ GammaShapeRate(10.0, 1.0)
    for expert in 1:n_experts
        w[expert] ~ priors[:w][expert]
        tau[expert] ~ priors[:tau][expert]
        beta[expert] ~ priors[:beta][expert]
    end
    for observation in 1:n_obs
        for expert in 1:n_experts
            z[expert, observation] ~ softdot(
                features[observation],
                w[expert],
                tau[expert],
            ) where {meta = LowRankMeta()}
            gamma[expert, observation] ~ GammaShapeRate(
                1.0,
                beta[expert],
            )
            gamma[expert, observation] ~ Squareplus(z[expert, observation]) where {
                dependencies = activation_dependencies,
                meta = activation_damping,
            }
            out[observation] ~ NormalMeanPrecision(
                predictions[expert, observation],
                gamma[expert, observation],
            ) where {
                dependencies = observation_dependencies,
                meta = observation_damping,
            }
        end
        y[observation] ~ NormalMeanPrecision(out[observation], obs_noise)
    end
end

@constraints function squareplus_structured_out_toy_constraints()
    q(w, z, gamma, tau, beta, out, obs_noise) =
        q(w)q(z, gamma, out)q(tau)q(beta)q(obs_noise)
    q(w)::MomentForm()
end

function squareplus_structured_out_toy_initialization(priors, targets)
    output_mean = mean(targets)
    output_variance = max(var(targets), 1e-3)
    return @initialization begin
        q(w) = deepcopy(priors[:w])
        q(z) = NormalMeanVariance(0.0, 1.0)
        q(gamma) = GammaShapeScale(1.0, 1.0)
        q(tau) = deepcopy(priors[:tau])
        q(beta) = deepcopy(priors[:beta])
        q(out) = NormalMeanVariance(output_mean, output_variance)
        q(obs_noise) = GammaShapeRate(10.0, 1.0)
        μ(gamma) = GammaShapeScale(1.0, 1.0)
        μ(out) = NormalMeanVariance(output_mean, output_variance)
    end
end

@constraints function squareplus_native_toy_constraints()
    q(w, z, gamma, tau, beta) = q(w)q(z, gamma)q(tau)q(beta)
    q(w)::MomentForm()
end

function squareplus_native_toy_initialization(priors)
    return @initialization begin
        q(w) = priors[:w]
        q(z) = NormalMeanVariance(0.0, 1.0)
        q(gamma) = GammaShapeScale(1.0, 1.0)
        q(tau) = priors[:tau]
        q(beta) = priors[:beta]
    end
end

@testset "Squareplus deterministic NGMP node" begin
    @testset "transform, inverse, self-reciprocity, and tails" begin
        for x in (-1e6, -50.0, -2.0, 0.0, 2.0, 50.0, 1e6)
            y = _squareplus(x)
            @test isfinite(y) && y > 0
            @test _inverse_squareplus(y) ≈ x rtol = 2e-10 atol = 1e-12
            @test y * _squareplus(-x) ≈ 1.0 rtol = 2e-12
            @test _log_squareplus(x) ≈ log(y) rtol = 2e-12
        end
        tail = 1e6
        @test _squareplus(tail) / tail ≈ 1.0 rtol = 1e-10
        @test tail * _squareplus(-tail) ≈ 1.0 rtol = 1e-10
        @test _inverse_squareplus(0.0) == -Inf
    end

    @testset "exact forward and backward log messages" begin
        forward = SquareplusForwardMessage(0.3, 0.8)
        for y in (1e-6, 0.1, 1.0, 20.0)
            x = _inverse_squareplus(y)
            reference = logpdf(Normal(0.3, sqrt(0.8)), x) +
                        log1p(inv(y^2))
            @test log(forward, y) ≈ reference atol = 1e-11 rtol = 5e-15
        end
        @test log(forward, 0.0) == -Inf
        @test_throws DomainError SquareplusForwardMessage(0.0, 0.0)

        backward = SquareplusBackwardMessage(2.5, 1.3)
        for x in (-1e6, -3.0, 0.0, 4.0, 1e6)
            expected = (2.5 - 1) * log(_squareplus(x)) -
                       1.3 * _squareplus(x)
            @test log(backward, x) ≈ expected
            @test !isnan(log(backward, x))
        end
    end

    @testset "analytic derivatives" begin
        first_difference(f, x; h = 1e-5) = (f(x + h) - f(x - h)) / (2h)
        second_difference(f, x; h = 1e-4) =
            (f(x + h) - 2f(x) + f(x - h)) / h^2

        forward = Logpdf(SquareplusForwardMessage(0.3, 0.8))
        for y in (0.2, 0.7, 2.0)
            target(value) = log(forward.dist, value)
            @test _squareplus_forward_first_derivative(forward, y) ≈
                  first_difference(target, y) rtol = 2e-6
            @test _squareplus_forward_second_derivative(forward, y) ≈
                  second_difference(target, y) rtol = 2e-5
        end

        backward = Logpdf(SquareplusBackwardMessage(2.5, 1.3))
        for x in (-2.0, 0.0, 2.0)
            target(value) = log(backward.dist, value)
            @test _squareplus_backward_first_derivative(backward, x) ≈
                  first_difference(target, x) rtol = 2e-6
            @test _squareplus_backward_second_derivative(backward, x) ≈
                  second_difference(target, x) rtol = 2e-5
        end
    end

    @testset "projection accuracy and rule families" begin
        forward_cases = (
            (NormalMeanVariance(-1.0, 0.8), GammaShapeRate(1.5, 2.0)),
            (NormalMeanVariance(0.3, 0.5), GammaShapeRate(3.0, 1.5)),
            (NormalMeanVariance(2.0, 0.3), GammaShapeRate(5.0, 2.0)),
        )
        for (input, output_projection_point) in forward_cases
            exact = Logpdf(SquareplusForwardMessage(mean(input), var(input)))
            quadrature = getnaturalparameters(project(
                TangentProjection(type = Quadrature(256)),
                output_projection_point,
                exact,
            ))
            reference = getnaturalparameters(project(
                TangentProjection(type = Quadrature(512)),
                output_projection_point,
                exact,
            ))
            normalized_error = maximum(
                abs.(quadrature .- reference) ./ (1 .+ abs.(reference)),
            )
            @test normalized_error < 0.02

            state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
            message = @call_rule Squareplus(
                :out,
                NaturalGradientMessage(TangentProjection(type = Unscented)),
            ) (m_in = input, q_out = output_projection_point, meta = state)
            @test message isa GammaShapeRate
            @test all(isfinite, (shape(message), rate(message)))
            @test state.nfired == 1
        end

        backward_cases = (
            (GammaShapeRate(1.5, 2.0), NormalMeanVariance(-1.0, 0.8)),
            (GammaShapeRate(3.0, 1.5), NormalMeanVariance(0.3, 0.5)),
            (GammaShapeRate(5.0, 2.0), NormalMeanVariance(2.0, 0.3)),
        )
        for (output, input_projection_point) in backward_cases
            exact = Logpdf(SquareplusBackwardMessage(shape(output), rate(output)))
            unscented = getnaturalparameters(project(
                TangentProjection(type = Unscented),
                input_projection_point,
                exact,
            ))
            reference = getnaturalparameters(project(
                TangentProjection(type = Quadrature(512)),
                input_projection_point,
                exact,
            ))
            normalized_error = maximum(
                abs.(unscented .- reference) ./ (1 .+ abs.(reference)),
            )
            @test normalized_error < 0.05

            state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
            message = @call_rule Squareplus(
                :in,
                NaturalGradientMessage(TangentProjection(type = Unscented)),
            ) (
                m_out = output,
                q_in = input_projection_point,
                meta = state,
            )
            @test message isa NormalWeightedMeanPrecision
            @test all(isfinite, (weightedmean(message), precision(message)))
            @test state.nfired == 1
        end

        narrow_output = GammaShapeRate(100.0, 100.0)
        forward = Logpdf(SquareplusForwardMessage(0.0, 0.2))
        delta_forward = getnaturalparameters(project(
            TangentProjection(type = DeltaApproximation),
            narrow_output,
            forward,
        ))
        quadrature_forward = getnaturalparameters(project(
            TangentProjection(type = Quadrature(512)),
            narrow_output,
            forward,
        ))
        @test maximum(
            abs.(delta_forward .- quadrature_forward) ./
            (1 .+ abs.(quadrature_forward)),
        ) < 0.04

        narrow_input = NormalMeanVariance(0.2, 1e-3)
        backward = Logpdf(SquareplusBackwardMessage(2.0, 1.5))
        delta_backward = getnaturalparameters(project(
            TangentProjection(type = DeltaApproximation),
            narrow_input,
            backward,
        ))
        quadrature_backward = getnaturalparameters(project(
            TangentProjection(type = Quadrature(512)),
            narrow_input,
            backward,
        ))
        @test maximum(
            abs.(delta_backward .- quadrature_backward) ./
            (1 .+ abs.(quadrature_backward)),
        ) < 0.01

        @test_throws ArgumentError project(
            TangentProjection(type = ClosedForm),
            GammaShapeRate(2.0, 2.0),
            forward,
        )
        @test_throws ArgumentError project(
            TangentProjection(type = ClosedForm),
            NormalMeanVariance(0.0, 1.0),
            backward,
        )
    end

    @testset "finite marginal and deterministic free energy" begin
        local_marginal = @call_marginalrule Squareplus(:in) (
            m_out = GammaShapeRate(2.0, 1.5),
            m_in = NormalMeanVariance(0.1, 0.8),
            meta = DampingMeta(alpha = 0.2, beta = 0.0),
        )
        @test local_marginal isa NormalWeightedMeanPrecision
        @test isfinite(entropy(local_marginal))

        initialization = @initialization begin
            q(z) = NormalMeanVariance(0.0, 1.0)
            q(gamma) = GammaShapeRate(2.0, 2.0)
        end
        iterations = 4
        dependencies = NGMPDependencies(
            out = nothing,
            in = nothing,
            projection = TangentProjection(type = Unscented),
        )
        result = infer(
            model = squareplus_toy(
                dependencies = dependencies,
                damping = DampingMeta(alpha = 0.2, beta = 0.0),
            ),
            data = (y = [0.2, -0.1, 0.3],),
            initialization = initialization,
            iterations = iterations,
            free_energy = true,
        )
        @test all(isfinite, result.free_energy)
        @test isfinite(mean(last(result.posteriors[:z])))
        qgamma = last(result.posteriors[:gamma])
        @test all(isfinite, (shape(qgamma), rate(qgamma)))
        @test length(dependencies.states) == 2
        @test all(state -> state.nfired == iterations, dependencies.states)
    end

    @testset "native softdot integration" begin
        features = [[1.0, -1.0], [1.0, -0.2], [1.0, 0.4], [1.0, 1.0]]
        predictions = [-0.3, -0.1, 0.2, 0.4]
        observations = [-0.2, 0.0, 0.1, 0.5]
        priors = Dict{Symbol, Any}(
            :w => MvNormalMeanScalePrecision(zeros(2), 0.1),
            :tau => GammaShapeRate(1.0, 1e-3),
            :beta => GammaShapeRate(1.0, 1e3),
        )
        dependencies = NGMPDependencies(
            out = nothing,
            in = nothing,
            projection = TangentProjection(type = Unscented),
        )
        iterations = 3
        result = infer(
            model = squareplus_native_toy(
                n_obs = length(observations),
                priors = priors,
                dependencies = dependencies,
                damping = DampingMeta(alpha = 0.2, beta = 0.0),
            ),
            data = (
                y = observations,
                features = features,
                predictions = predictions,
            ),
            constraints = squareplus_native_toy_constraints(),
            initialization = squareplus_native_toy_initialization(priors),
            iterations = iterations,
            free_energy = false,
            options = (limit_stack_depth = 100,),
        )
        @test all(isfinite, mean(last(result.posteriors[:w])))
        @test all(isfinite, (
            mean(last(result.posteriors[:tau])),
            mean(last(result.posteriors[:beta])),
        ))
        @test length(dependencies.states) == 2length(observations)
        @test all(state -> state.nfired >= iterations, dependencies.states)
    end


    @testset "structured q(z, gamma, out) integration" begin
        observations = [-0.3, 0.2, 0.5]
        features = [[1.0, -0.5], [1.0, 0.0], [1.0, 0.5]]
        predictions = [
            -0.4 0.1 0.6
            -0.1 0.4 0.3
        ]
        priors = Dict{Symbol, Any}(
            :w => [
                MvNormalMeanScalePrecision(zeros(2), 0.1)
                for _ in axes(predictions, 1)
            ],
            :tau => [
                GammaShapeRate(1.0, 1e-3)
                for _ in axes(predictions, 1)
            ],
            :beta => [
                GammaShapeRate(1.0, 1e3)
                for _ in axes(predictions, 1)
            ],
        )
        activation_dependencies = NGMPDependencies(
            out = nothing,
            in = nothing,
            projection = TangentProjection(type = Unscented),
        )
        observation_dependencies = NGMPDependencies(
            out = nothing,
            τ = nothing,
            projection = TangentProjection(type = DeltaApproximation),
        )
        iterations = 3
        result = infer(
            model = squareplus_structured_out_toy(
                n_experts = size(predictions, 1),
                n_obs = length(observations),
                priors = priors,
                activation_dependencies = activation_dependencies,
                activation_damping = DampingMeta(alpha = 0.2, beta = 0.0),
                observation_dependencies = observation_dependencies,
                observation_damping = DampingMeta(
                    alpha = 0.2,
                    beta = 0.0,
                    max_step = 1.0,
                ),
            ),
            data = (
                y = observations,
                features = features,
                predictions = predictions,
            ),
            constraints = squareplus_structured_out_toy_constraints(),
            initialization =
                squareplus_structured_out_toy_initialization(
                    priors,
                    observations,
                ),
            iterations = iterations,
            free_energy = false,
            options = (limit_stack_depth = 100,),
        )

        @test all(
            q -> all(isfinite, (mean(q), var(q))),
            Iterators.flatten(result.posteriors[:out]),
        )
        @test all(isfinite, (
            mean(last(result.posteriors[:obs_noise])),
            var(last(result.posteriors[:obs_noise])),
        ))
        expected_states = 2 * length(predictions)
        @test length(activation_dependencies.states) == expected_states
        @test length(observation_dependencies.states) == expected_states
        @test all(
            state -> state.nfired >= iterations,
            activation_dependencies.states,
        )
        @test all(
            state -> state.nfired >= iterations,
            observation_dependencies.states,
        )
    end
end
