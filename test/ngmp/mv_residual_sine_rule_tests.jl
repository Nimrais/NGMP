import FastGaussQuadrature: gausshermite
import LinearAlgebra
import LinearAlgebra: Diagonal, Symmetric, cholesky, diag
import ReactiveMP: @call_marginalrule, @call_rule
import ExponentialFamily:
    ExponentialFamilyDistribution,
    MvNormalMeanCovariance,
    MvNormalWeightedMeanPrecision,
    getnaturalparameters,
    weightedmean_precision
import SurrogateModelling:
    MvResidualSineForwardMessage,
    MvResidualSineGaussianBackwardMessage,
    NaturalGradientMP,
    _inverse_residual_sine,
    _mv_mean_cov,
    _mv_residual_sine_forward_site,
    _mv_residual_sine_forward_logderivatives,
    _mv_residual_sine_mean_cov,
    _residual_sine,
    _residual_sine_prime,
    _residual_sine_second

@model function mv_residual_sine_ct_toy(
    y,
    features,
    priors,
    feature_covariance,
    map_meta,
    prediction_meta,
    activation,
    activation_dependencies,
)
    a_map ~ priors[:a_map]
    a_prediction ~ priors[:a_prediction]
    theta ~ priors[:theta]
    map_precision ~ priors[:map_precision]
    prediction_precision ~ priors[:prediction_precision]
    observation_precision ~ priors[:observation_precision]
    for index in eachindex(y)
        x_feature[index] ~ MvNormalMeanCovariance(features[index], feature_covariance)
        h1[index] ~ ContinuousTransition(x_feature[index], a_map, map_precision) where {
            meta = map_meta
        }
        s[index] ~ MvResidualSine(h1[index]) where {
            dependencies = activation_dependencies,
            meta = activation
        }
        h2[index] ~ ContinuousTransition(s[index], a_prediction, prediction_precision) where {
            meta = prediction_meta
        }
        y[index] ~ softdot(theta, h2[index], observation_precision)
    end
end

@constraints function mv_residual_sine_ct_toy_constraints()
    q(
        x_feature,
        h1,
        s,
        h2,
        a_map,
        a_prediction,
        theta,
        map_precision,
        prediction_precision,
        observation_precision,
    ) = q(x_feature, h1)q(s, h2)q(a_map)q(a_prediction)q(theta)q(map_precision)q(prediction_precision)q(observation_precision)
end

function residual_sine_tensor_expectation(f, mean, covariance; order = 28)
    nodes, weights = gausshermite(order)
    factor = cholesky(Symmetric(Matrix(covariance))).L
    accumulator = nothing
    for i in eachindex(nodes), j in eachindex(nodes)
        point = mean .+ sqrt(2.0) .* factor * [nodes[i], nodes[j]]
        contribution = (weights[i] * weights[j] / pi) .* f(point)
        accumulator = isnothing(accumulator) ? contribution : accumulator .+ contribution
    end
    return accumulator
end

function finite_gradient(f, point; step = 2e-5)
    d = length(point)
    gradient = zeros(d)
    for i in 1:d
        direction = zeros(d)
        direction[i] = step
        gradient[i] = (f(point .+ direction) - f(point .- direction)) / (2step)
    end
    return gradient
end

function finite_hessian(f, point; step = 8e-5)
    d = length(point)
    hessian = zeros(d, d)
    center = f(point)
    for i in 1:d
        ei = zeros(d)
        ei[i] = step
        hessian[i, i] = (f(point .+ ei) - 2center + f(point .- ei)) / step^2
        for j in (i + 1):d
            ej = zeros(d)
            ej[j] = step
            value = (
                f(point .+ ei .+ ej) - f(point .+ ei .- ej) -
                f(point .- ei .+ ej) + f(point .- ei .- ej)
            ) / (4step^2)
            hessian[i, j] = value
            hessian[j, i] = value
        end
    end
    return hessian
end

@testset "MvResidualSine" begin
    activation = ResidualSineMeta(rho = 0.75, omega = 1.3)
    delta_projection = TangentProjection(type = DeltaApproximation)
    moments_projection = TangentProjection(type = ClosedForm)

    @testset "configuration and inverse" begin
        @test_throws ArgumentError ResidualSineMeta(rho = 1.0)
        @test_throws ArgumentError ResidualSineMeta(rho = 0.0)
        @test_throws ArgumentError ResidualSineMeta(omega = 0.0)

        for x in (-100.0, -8.0, -1.2, 0.0, 0.7, 9.0, 100.0)
            y = _residual_sine(x, activation)
            @test _inverse_residual_sine(y, activation) ≈ x atol = 2e-12 rtol = 2e-12
            @test _residual_sine_prime(x, activation) >= 1 - activation.rho
        end
    end

    @testset "implicit forward derivatives" begin
        input_mean = [0.3, -0.5]
        input_covariance = [0.7 0.16; 0.16 0.9]
        message = MvResidualSineForwardMessage(
            input_mean,
            input_covariance,
            activation,
        )
        expansion_point = [-0.8, 0.55]
        gradient, hessian =
            _mv_residual_sine_forward_logderivatives(message, expansion_point)
        objective = point -> log(message, point)
        @test isfinite(objective([-20.0, 30.0]))
        @test gradient ≈ finite_gradient(objective, expansion_point) atol = 2e-7 rtol = 2e-6
        @test hessian ≈ finite_hessian(objective, expansion_point) atol = 2e-5 rtol = 2e-5

        q_out = MvNormalMeanCovariance(expansion_point, [0.4 0.05; 0.05 0.6])
        site = project(
            delta_projection,
            q_out,
            Logpdf(message),
        )
        eta = getnaturalparameters(site)
        precision = -2 .* reshape(eta[3:end], 2, 2)
        weighted_mean = eta[1:2]
        @test -precision ≈ hessian atol = 1e-11
        @test weighted_mean - precision * expansion_point ≈ gradient atol = 1e-11

        other_q_out = MvNormalMeanCovariance([0.4, -0.2], [0.4 0.05; 0.05 0.6])
        other_site = project(
            delta_projection,
            other_q_out,
            Logpdf(message),
        )
        @test !isapprox(
            getnaturalparameters(site),
            getnaturalparameters(other_site);
            atol = 1e-10,
            rtol = 1e-10,
        )
    end

    @testset "exact correlated forward moments" begin
        input_mean = [0.25, -0.6]
        input_covariance = [0.8 0.31; 0.31 1.1]
        analytic_mean, analytic_covariance =
            _mv_residual_sine_mean_cov(input_mean, input_covariance, activation)
        quadrature_mean = residual_sine_tensor_expectation(
            point -> _residual_sine.(point, Ref(activation)),
            input_mean,
            input_covariance,
        )
        quadrature_second = residual_sine_tensor_expectation(
            point -> begin
                transformed = _residual_sine.(point, Ref(activation))
                transformed * transformed'
            end,
            input_mean,
            input_covariance,
        )
        quadrature_covariance = quadrature_second - quadrature_mean * quadrature_mean'
        @test analytic_mean ≈ quadrature_mean atol = 2e-11 rtol = 2e-11
        @test analytic_covariance ≈ quadrature_covariance atol = 3e-10 rtol = 3e-10
        @test minimum(LinearAlgebra.eigvals(Symmetric(analytic_covariance))) > 0
    end

    @testset "exact analytic backward tangent projection" begin
        q_mean = [0.15, -0.35]
        q_covariance = [0.65 0.18; 0.18 0.85]
        q_in = MvNormalMeanCovariance(q_mean, q_covariance)
        xi = [0.9, -0.4]
        Lambda = [1.7 0.22; 0.22 1.1]
        message = MvResidualSineGaussianBackwardMessage(xi, Lambda, activation)
        site = project(moments_projection, q_in, Logpdf(message))
        eta = getnaturalparameters(site)
        analytic_precision = -2 .* reshape(eta[3:end], 2, 2)
        analytic_gradient = eta[1:2] - analytic_precision * q_mean

        expected_gradient = residual_sine_tensor_expectation(
            q_mean,
            q_covariance,
        ) do point
            transformed = _residual_sine.(point, Ref(activation))
            slope = _residual_sine_prime.(point, Ref(activation))
            slope .* (xi - Lambda * transformed)
        end
        expected_hessian = residual_sine_tensor_expectation(
            q_mean,
            q_covariance,
        ) do point
            transformed = _residual_sine.(point, Ref(activation))
            slope = _residual_sine_prime.(point, Ref(activation))
            curvature = _residual_sine_second.(point, Ref(activation))
            residual = xi - Lambda * transformed
            -Diagonal(slope) * Lambda * Diagonal(slope) +
                Diagonal(curvature .* residual)
        end
        @test analytic_gradient ≈ expected_gradient atol = 3e-10 rtol = 3e-10
        @test -analytic_precision ≈ expected_hessian atol = 5e-10 rtol = 5e-10
    end

    @testset "NGMP rules and scoring marginal" begin
        m_in = MvNormalMeanCovariance([0.2, -0.4], [0.6 0.1; 0.1 0.9])
        q_out = MvNormalMeanCovariance([-0.1, 0.3], Diagonal(fill(0.5, 2)))
        m_out = MvNormalWeightedMeanPrecision([1.0, -0.2], [1.5 0.1; 0.1 1.0])

        for forward_projection in (delta_projection, moments_projection)
            direct_forward = _mv_residual_sine_forward_site(
                activation,
                forward_projection,
                m_in,
                q_out,
            )
            direct_mean, direct_covariance = _mv_mean_cov(direct_forward)
            @test all(isfinite, direct_mean)
            @test all(isfinite, direct_covariance)

            forward_state = NGMPEdgeState(
                activation;
                damping = DampingMeta(alpha = 1.0, beta = 0.0),
            )
            forward = @call_rule MvResidualSine(
                :out,
                NaturalGradientMessage(forward_projection),
            ) (m_in = m_in, q_out = q_out, meta = forward_state)
            @test forward isa MvNormalWeightedMeanPrecision
            @test all(isfinite, weightedmean_precision(forward)[1])
            @test all(isfinite, weightedmean_precision(forward)[2])
            @test forward_state.nfired == 1

            backward_state = NGMPEdgeState(
                activation;
                damping = DampingMeta(alpha = 1.0, beta = 0.0),
            )
            backward = @call_rule MvResidualSine(
                :in,
                NaturalGradientMessage(TangentProjection(type = DeltaApproximation)),
            ) (m_out = m_out, q_in = m_in, meta = backward_state)
            @test backward isa MvNormalWeightedMeanPrecision
            @test all(isfinite, weightedmean_precision(backward)[1])
            @test all(isfinite, weightedmean_precision(backward)[2])
            @test backward_state.nfired == 1

            local_marginal = @call_marginalrule MvResidualSine(:in) (
                m_out = m_out,
                m_in = m_in,
                meta = activation,
            )
            @test local_marginal isa MvNormalWeightedMeanPrecision
            @test isfinite(entropy(local_marginal))
        end

        @test_throws ArgumentError _mv_residual_sine_forward_site(
            activation,
            TangentProjection(type = Quadrature(8)),
            m_in,
            q_out,
        )
    end

    @testset "ContinuousTransition integration" begin
        hidden_dimension = 2
        feature_dimension = 3
        rng = Random.MersenneTwister(17)
        features = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.2, 0.8],
            [1.0, 0.9, 0.1],
        ]
        observations = [
            Float64(xor(feature[2] > 0.5, feature[3] > 0.5)) + 0.03 * randn(rng)
            for feature in features
        ]
        degrees_of_freedom = hidden_dimension + 2.0
        priors = Dict{Symbol, Any}(
            :a_map => MvNormalMeanCovariance(
                0.5 .* randn(rng, hidden_dimension * feature_dimension),
                Diagonal(ones(hidden_dimension * feature_dimension)),
            ),
            :a_prediction => MvNormalMeanCovariance(
                0.5 .* randn(rng, hidden_dimension^2),
                Diagonal(ones(hidden_dimension^2)),
            ),
            :theta => MvNormalMeanCovariance(
                zeros(hidden_dimension),
                Diagonal(ones(hidden_dimension)),
            ),
            :map_precision => ExponentialFamily.WishartFast(
                degrees_of_freedom,
                Matrix(Diagonal(fill(degrees_of_freedom / 10.0, hidden_dimension))),
            ),
            :prediction_precision => ExponentialFamily.WishartFast(
                degrees_of_freedom,
                Matrix(Diagonal(fill(degrees_of_freedom / 10.0, hidden_dimension))),
            ),
            :observation_precision => GammaShapeRate(2.0, 1.0),
        )

        for forward_projection in (delta_projection, moments_projection)
            initial_s_mean, initial_s_covariance = _mv_residual_sine_mean_cov(
                zeros(hidden_dimension),
                Matrix(Diagonal(ones(hidden_dimension))),
                activation,
            )
            initialization = @initialization begin
                q(a_map) = priors[:a_map]
                q(a_prediction) = priors[:a_prediction]
                q(theta) = priors[:theta]
                q(map_precision) = priors[:map_precision]
                q(prediction_precision) = priors[:prediction_precision]
                q(observation_precision) = priors[:observation_precision]
                q(h1) = MvNormalMeanCovariance(
                    zeros(hidden_dimension),
                    Diagonal(ones(hidden_dimension)),
                )
                q(s) = MvNormalMeanCovariance(initial_s_mean, initial_s_covariance)
                q(h2) = MvNormalMeanCovariance(
                    zeros(hidden_dimension),
                    Diagonal(ones(hidden_dimension)),
                )
            end
            activation_dependencies = NGMPDependencies(
                out = nothing,
                in = nothing,
                projection = forward_projection,
                damping = DampingMeta(alpha = 0.2, beta = 0.0, max_step = 1.0),
            )
            result = infer(
                model = mv_residual_sine_ct_toy(
                    priors = priors,
                    feature_covariance = Matrix(Diagonal(fill(1e-4, feature_dimension))),
                    map_meta = LinearReshapeMeta(hidden_dimension, feature_dimension),
                    prediction_meta = LinearReshapeMeta(hidden_dimension, hidden_dimension),
                    activation = activation,
                    activation_dependencies = activation_dependencies,
                ),
                data = (y = observations, features = features),
                constraints = mv_residual_sine_ct_toy_constraints(),
                initialization = initialization,
                iterations = 3,
                free_energy = true,
                showprogress = false,
                options = (limit_stack_depth = 100,),
                disable_inference_error_hint = true,
            )

            @test all(isfinite, result.free_energy)
            @test all(isfinite, mean(last(result.posteriors[:a_map])))
            @test all(isfinite, mean(last(result.posteriors[:a_prediction])))
            @test all(isfinite, mean(last(result.posteriors[:theta])))
            @test length(activation_dependencies.states) == 2 * length(observations)
            @test all(state -> state.nfired == 3, activation_dependencies.states)
        end
    end
end
