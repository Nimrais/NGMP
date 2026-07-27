# Validation script for the scalar `ResidualSine` node and its composition with
# `ManyPlus` and the scalar (latent x latent) `softdot` product.
#
# Stage A pins the 1D rule math against the multivariate node at d = 1 and
# against dense Gauss-Hermite quadrature. Stage B is the graph handshake: a
# tiny 2-neuron additive model za -> ResidualSine -> softdot(v, h) -> ManyPlus
# with NGMP edges, asserting that every message dispatches and stays proper.
# Stage C recovers a 1D nonlinear regression y = phi(a x + b) + noise.
#
# Run: julia --project=. experiments/check_residual_sine_node.jl

using SurrogateModelling
using RxInfer
using StableRNGs
using LinearAlgebra
using Test

import ClosedFormExpectations: Logpdf
import ExponentialFamily: getnaturalparameters

const SM = SurrogateModelling

println("=== Stage A: rule pins ===")

@testset "1D residual sine pins" begin
    meta = ResidualSineMeta(rho = 0.9, omega = 1.3)

    for (m, v) in ((0.7, 1.8), (-1.2, 0.05), (3.4, 6.0))
        m1, v1 = SM._residual_sine_mean_var_1d(m, v, meta)
        mv_mean, mv_cov = SM._mv_residual_sine_mean_cov([m], fill(v, 1, 1), meta)
        @test m1 ≈ mv_mean[1] atol = 1e-12
        @test v1 ≈ mv_cov[1, 1] atol = 1e-12
        @test v1 > 0
    end

    for (mq, vq, xi, La) in ((0.4, 2.5, 1.1, 0.8), (-2.0, 0.3, -0.5, 3.0))
        q1 = NormalMeanVariance(mq, vq)
        message = SM.ResidualSineGaussianBackwardMessage(xi, La, meta)
        site = SM._project_residual_sine_backward_1d(q1, message)
        eta = getnaturalparameters(site)

        mv_message = SM.MvResidualSineGaussianBackwardMessage(
            [xi], fill(La, 1, 1), meta,
        )
        mv_site = SM._project_mv_residual_sine_backward(
            MvNormalMeanCovariance([mq], fill(vq, 1, 1)), mv_message,
        )
        eta_mv = getnaturalparameters(mv_site)
        @test eta[1] ≈ eta_mv[1] atol = 1e-10
        @test eta[2] ≈ eta_mv[2] atol = 1e-10

        quadrature_site = SM.project(
            TangentProjection(type = Quadrature(129)), q1, Logpdf(message),
        )
        eta_q = getnaturalparameters(quadrature_site)
        @test eta[1] ≈ eta_q[1] atol = 1e-8
        @test eta[2] ≈ eta_q[2] atol = 1e-8

        # The 3-sigma-point UT is an approximation of the same projection.
        unscented_site = SM.project(
            TangentProjection(type = Unscented), q1, Logpdf(message),
        )
        eta_u = getnaturalparameters(unscented_site)
        @test isfinite(eta_u[1]) && isfinite(eta_u[2])
    end
end

println("=== Stage B: ManyPlus + scalar softdot handshake ===")

@model function handshake_model(features, y, priors, activation, activation_deps)
    local w, v, za, h, c, out
    τ ~ priors[:τ]
    τ_c ~ priors[:τ_c]
    obs_noise ~ priors[:obs_noise]
    for k in 1:2
        w[k] ~ priors[:w][k]
        v[k] ~ priors[:v][k]
    end
    for i in eachindex(y)
        for k in 1:2
            za[k, i] ~ softdot(features[i], w[k], τ)
            h[k, i] ~ ResidualSine(za[k, i]) where {
                dependencies = activation_deps,
                meta = activation,
            }
            c[k, i] ~ softdot(v[k], h[k, i], τ_c)
        end
        out[i] ~ ManyPlus(inputs = [c[k, i] for k in 1:2])
        y[i] ~ NormalMeanPrecision(out[i], obs_noise)
    end
end

@constraints function handshake_constraints()
    q(w, v, za, h, c, out, τ, τ_c, obs_noise) =
        q(w, za, h, c, out)q(v)q(τ)q(τ_c)q(obs_noise)
    q(w)::MomentForm()
end

@initialization function handshake_initialization(priors, output_mean)
    q(v) = deepcopy(priors[:v])
    q(za) = NormalMeanVariance(0.0, 1.0)
    q(h) = NormalMeanVariance(0.0, 1.0)
    q(c) = NormalMeanVariance(0.0, 1.0)
    q(out) = NormalMeanVariance(output_mean, 1.0)
    q(τ) = priors[:τ]
    q(τ_c) = priors[:τ_c]
    q(obs_noise) = priors[:obs_noise]
    μ(w) = deepcopy(priors[:w])
end

function handshake_priors()
    precision = Diagonal(fill(1.0, 3))
    w = [
        MvNormalWeightedMeanPrecision(precision * [0.0, 1.2, 0.4], precision),
        MvNormalWeightedMeanPrecision(precision * [0.0, -0.3, 1.1], precision),
    ]
    v = [NormalMeanVariance(1.0, 0.5), NormalMeanVariance(-1.0, 0.5)]
    return Dict{Symbol, Any}(
        :w => w,
        :v => v,
        :τ => GammaShapeRate(1e3, 1.0),
        :τ_c => GammaShapeRate(1e4, 1.0),
        :obs_noise => GammaShapeRate(2.0, 0.1),
    )
end

@testset "handshake" begin
    rng = StableRNG(7)
    n = 12
    features = [[1.0, 4rand(rng) - 2, 4rand(rng) - 2] for _ in 1:n]
    y = [sin(f[2]) * sin(f[3]) / 2 + 0.5 + 0.05 * randn(rng) for f in features]

    priors = handshake_priors()
    activation = ResidualSineMeta(rho = 0.9, omega = 1.0)
    activation_deps = NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = TangentProjection(type = ClosedForm),
        damping = DampingMeta(alpha = 0.1, beta = 0.0, max_step = 1.0),
    )

    result = infer(
        model = handshake_model(
            priors = priors,
            activation = activation,
            activation_deps = activation_deps,
        ),
        data = (y = y, features = features),
        constraints = handshake_constraints(),
        initialization = handshake_initialization(priors, mean(y)),
        iterations = 10,
        free_energy = true,
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )

    @test all(isfinite, result.free_energy)
    for k in 1:2
        posterior_w = result.posteriors[:w][end][k]
        posterior_v = result.posteriors[:v][end][k]
        @test all(isfinite, mean(posterior_w))
        @test isposdef(Matrix(cov(posterior_w)))
        @test isfinite(mean(posterior_v)) && var(posterior_v) > 0
    end
    println("handshake free energy: ", round.(result.free_energy; digits = 3))
end

println("=== Stage C: 1D nonlinear regression recovery ===")

@model function recovery_model(features, y, priors, activation, activation_deps)
    local w, za, h
    τ ~ priors[:τ]
    w ~ priors[:w]
    for i in eachindex(y)
        za[i] ~ softdot(features[i], w, τ)
        h[i] ~ ResidualSine(za[i]) where {
            dependencies = activation_deps,
            meta = activation,
        }
        y[i] ~ NormalMeanPrecision(h[i], 100.0)
    end
end

@constraints function recovery_constraints()
    q(w, za, h, τ) = q(w, za, h)q(τ)
    q(w)::MomentForm()
end

@initialization function recovery_initialization(priors)
    q(za) = NormalMeanVariance(0.0, 1.0)
    q(h) = NormalMeanVariance(0.0, 1.0)
    q(τ) = priors[:τ]
    μ(w) = deepcopy(priors[:w])
end

@testset "recovery" begin
    rng = StableRNG(11)
    activation = ResidualSineMeta(rho = 0.9, omega = 1.5)
    a_true, b_true = 1.3, -0.4
    n = 200
    x = 4 .* rand(rng, n) .- 2
    features = [[1.0, xi] for xi in x]
    y = [
        SM._residual_sine(b_true + a_true * xi, activation) + 0.1 * randn(rng)
        for xi in x
    ]

    precision = Diagonal(fill(0.5, 2))
    priors = Dict{Symbol, Any}(
        :w => MvNormalWeightedMeanPrecision(precision * [0.0, 1.0], precision),
        :τ => GammaShapeRate(1e3, 1.0),
    )
    activation_deps = NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = TangentProjection(type = ClosedForm),
        damping = DampingMeta(alpha = 0.1, beta = 0.0, max_step = 1.0),
    )

    result = infer(
        model = recovery_model(
            priors = priors,
            activation = activation,
            activation_deps = activation_deps,
        ),
        data = (y = y, features = features),
        constraints = recovery_constraints(),
        initialization = recovery_initialization(priors),
        iterations = 50,
        free_energy = true,
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )

    posterior_w = result.posteriors[:w][end]
    w_mean = mean(posterior_w)
    w_std = sqrt.(diag(Matrix(cov(posterior_w))))
    println("recovered (b, a) = ", round.(w_mean; digits = 3),
        " +/- ", round.(w_std; digits = 3),
        " true = ", (b_true, a_true))
    println("final free energy: ", round(result.free_energy[end]; digits = 3))

    @test all(isfinite, result.free_energy)
    # Early-iteration Bethe values are not meaningful under heavy damping (the
    # sites have barely moved off the flat init), so require convergence over
    # the second half of the sweep instead of a global decrease.
    halfway = div(length(result.free_energy), 2)
    @test result.free_energy[end] <= result.free_energy[halfway] + 1e-6
    @test abs(w_mean[1] - b_true) < 0.15
    @test abs(w_mean[2] - a_true) < 0.15
end

println("All stages passed.")
