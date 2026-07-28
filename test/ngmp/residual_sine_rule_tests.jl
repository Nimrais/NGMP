import ReactiveMP: @call_marginalrule, @call_rule
import ExponentialFamily:
    ExponentialFamilyDistribution,
    MvNormalMeanCovariance,
    NormalMeanVariance,
    NormalWeightedMeanPrecision,
    getnaturalparameters,
    weightedmean_precision
import ClosedFormExpectations: Logpdf
import SurrogateModelling:
    NaturalGradientMP,
    ResidualSineForwardMessage,
    ResidualSineGaussianBackwardMessage,
    _inverse_residual_sine,
    _mv_residual_sine_mean_cov,
    _project_mv_residual_sine_backward,
    _project_residual_sine_backward_1d,
    _residual_sine,
    _residual_sine_forward_logderivatives,
    _residual_sine_mean_var_1d

@testset "ResidualSine 1D node" begin
    meta = ResidualSineMeta(rho = 0.9, omega = 1.3)

    @testset "forward moments pin against MvResidualSine at d = 1" begin
        for (m, v) in ((0.7, 1.8), (-1.2, 0.05), (3.4, 6.0), (0.0, 25.0))
            m1, v1 = _residual_sine_mean_var_1d(m, v, meta)
            mv_mean, mv_cov = _mv_residual_sine_mean_cov([m], fill(v, 1, 1), meta)
            @test m1 ≈ mv_mean[1] atol = 1e-12
            @test v1 ≈ mv_cov[1, 1] atol = 1e-12
            @test v1 > 0
        end
    end

    @testset "backward Fisher projection pins" begin
        for (mq, vq, xi, La) in (
            (0.4, 2.5, 1.1, 0.8),
            (-2.0, 0.3, -0.5, 3.0),
            (1.7, 8.0, 0.2, 0.05),
        )
            q1 = NormalMeanVariance(mq, vq)
            message = ResidualSineGaussianBackwardMessage(xi, La, meta)
            eta = getnaturalparameters(
                _project_residual_sine_backward_1d(q1, message),
            )

            mv_site = _project_mv_residual_sine_backward(
                MvNormalMeanCovariance([mq], fill(vq, 1, 1)),
                SurrogateModelling.MvResidualSineGaussianBackwardMessage(
                    [xi], fill(La, 1, 1), meta,
                ),
            )
            eta_mv = getnaturalparameters(mv_site)
            @test eta[1] ≈ eta_mv[1] atol = 1e-10
            @test eta[2] ≈ eta_mv[2] atol = 1e-10

            eta_quadrature = getnaturalparameters(SurrogateModelling.project(
                TangentProjection(type = Quadrature(129)), q1, Logpdf(message),
            ))
            @test eta[1] ≈ eta_quadrature[1] atol = 1e-7
            @test eta[2] ≈ eta_quadrature[2] atol = 1e-7
        end
    end

    @testset "forward pushforward log-density and derivatives" begin
        m, v = 0.3, 1.1
        forward = ResidualSineForwardMessage(m, v, meta)
        for y in (-1.5, 0.2, 2.7)
            x = _inverse_residual_sine(y, meta)
            @test _residual_sine(x, meta) ≈ y atol = 1e-10
            step = 1e-6
            numeric_first =
                (log(forward, y + step) - log(forward, y - step)) / (2step)
            numeric_second =
                (log(forward, y + step) - 2 * log(forward, y) +
                 log(forward, y - step)) / step^2
            gradient, hessian = _residual_sine_forward_logderivatives(forward, y)
            @test gradient ≈ numeric_first atol = 1e-5
            @test hessian ≈ numeric_second atol = 1e-3
        end
    end

    @testset "NGMP rules emit damped univariate Gaussians" begin
        state = NaturalGradientMP.NGMPEdgeState(
            meta;
            damping = DampingMeta(alpha = 1.0, beta = 0.0, max_step = Inf),
        )
        m_in = NormalMeanVariance(0.4, 0.7)
        q_out = NormalMeanVariance(0.5, 1.0)
        message = @call_rule ResidualSine(:out, NaturalGradientMessage) (
            m_in = m_in, q_out = q_out, meta = state,
        )
        @test message isa NormalWeightedMeanPrecision
        expected_mean, expected_variance =
            _residual_sine_mean_var_1d(0.4, 0.7, meta)
        @test mean(message) ≈ expected_mean atol = 1e-10
        @test var(message) ≈ expected_variance atol = 1e-10

        state_in = NaturalGradientMP.NGMPEdgeState(
            meta;
            damping = DampingMeta(alpha = 1.0, beta = 0.0, max_step = Inf),
        )
        backward = @call_rule ResidualSine(:in, NaturalGradientMessage) (
            m_out = NormalMeanVariance(0.9, 0.5),
            q_in = NormalMeanVariance(0.1, 0.8),
            meta = state_in,
        )
        @test backward isa NormalWeightedMeanPrecision
        expected_site = _project_residual_sine_backward_1d(
            NormalMeanVariance(0.1, 0.8),
            ResidualSineGaussianBackwardMessage(0.9 / 0.5, 1 / 0.5, meta),
        )
        eta_site = getnaturalparameters(expected_site)
        @test weightedmean(backward) ≈ eta_site[1] atol = 1e-10
        @test precision(backward) ≈ -2 * eta_site[2] atol = 1e-10

        improper_state = NaturalGradientMP.NGMPEdgeState(
            meta;
            damping = DampingMeta(alpha = 1.0, beta = 0.0, max_step = Inf),
        )
        recovered = @call_rule ResidualSine(:in, NaturalGradientMessage) (
            m_out = NormalMeanVariance(0.9, 0.5),
            q_in = NormalWeightedMeanPrecision(1.0, -2.0),
            meta = improper_state,
        )
        @test recovered isa NormalWeightedMeanPrecision
        @test weightedmean(recovered) == 0.0
        @test precision(recovered) == 0.0
    end

    @testset "sum-product forward and scoring marginal" begin
        forward = @call_rule ResidualSine(:out, Marginalisation) (
            m_in = NormalMeanVariance(-0.6, 0.9), meta = meta,
        )
        expected_mean, expected_variance =
            _residual_sine_mean_var_1d(-0.6, 0.9, meta)
        @test mean(forward) ≈ expected_mean atol = 1e-12
        @test var(forward) ≈ expected_variance atol = 1e-12

        marginal = @call_marginalrule ResidualSine(:in) (
            m_out = NormalMeanVariance(0.9, 0.5),
            m_in = NormalMeanVariance(0.1, 0.8),
            meta = meta,
        )
        @test marginal isa NormalWeightedMeanPrecision
        @test precision(marginal) > 0
    end
end
