import ReactiveMP: @call_marginalrule
import LinearAlgebra: dot, I

@model function relaxed_structured_softdot_toy(observation, dependencies, damping)
    θ ~ MvNormalMeanCovariance(zeros(2), Matrix{Float64}(I, 2, 2))
    x ~ MvNormalMeanCovariance(zeros(2), Matrix{Float64}(I, 2, 2))
    γ ~ GammaShapeRate(5.0, 2.0)
    y ~ softdot(θ, x, γ) where {
        dependencies = dependencies,
        meta = damping,
    }
    observation ~ NormalMeanVariance(y, 0.2)
end

@constraints function relaxed_structured_softdot_constraints()
    q(y, x, θ, γ) = q(y, x)q(θ)q(γ)
end

@initialization function relaxed_structured_softdot_initialization()
    q(y) = NormalMeanVariance(0.0, 1.0)
    q(x) = MvNormalMeanCovariance([0.1, 0.3], Matrix{Float64}(I, 2, 2))
    q(θ) = MvNormalMeanCovariance([0.4, -0.2], Matrix{Float64}(I, 2, 2))
    q(γ) = GammaShapeRate(5.0, 2.0)
end

@testset "relaxed structured softdot with random theta" begin
    m_y = NormalMeanVariance(0.7, 0.4)
    m_x = MvNormalWeightedMeanPrecision(
        [0.6, -0.2],
        [2.0 0.3; 0.3 1.5],
    )
    q_θ = MvNormalMeanCovariance(
        [0.8, -0.5],
        [0.25 0.04; 0.04 0.16],
    )
    q_γ = GammaShapeRate(5.0, 2.0)
    damping = DampingMeta(alpha = 1.0, beta = 0.0)

    @testset "metadata adapters equal stock ReactiveMP rules" begin
        y_stock = @call_rule softdot(:y, Marginalisation) (
            m_x = m_x,
            q_θ = q_θ,
            q_γ = q_γ,
        )
        y_relaxed = @call_rule softdot(:y, Marginalisation) (
            m_x = m_x,
            q_θ = q_θ,
            q_γ = q_γ,
            meta = damping,
        )
        @test mean(y_relaxed) ≈ mean(y_stock)
        @test var(y_relaxed) ≈ var(y_stock)

        x_stock = @call_rule softdot(:x, Marginalisation) (
            m_y = m_y,
            q_θ = q_θ,
            q_γ = q_γ,
        )
        x_relaxed = @call_rule softdot(:x, Marginalisation) (
            m_y = m_y,
            q_θ = q_θ,
            q_γ = q_γ,
            meta = damping,
        )
        @test weightedmean(x_relaxed) ≈ weightedmean(x_stock)
        @test precision(x_relaxed) ≈ precision(x_stock)

        joint_stock = @call_marginalrule SoftDot(:y_x) (
            m_y = m_y,
            m_x = m_x,
            q_θ = q_θ,
            q_γ = q_γ,
        )
        joint_relaxed = @call_marginalrule SoftDot(:y_x) (
            m_y = m_y,
            m_x = m_x,
            q_θ = q_θ,
            q_γ = q_γ,
            meta = damping,
        )
        @test weightedmean(joint_relaxed) ≈ weightedmean(joint_stock)
        @test precision(joint_relaxed) ≈ precision(joint_stock)

        theta_stock = @call_rule softdot(:θ, Marginalisation) (
            q_y_x = joint_stock,
            q_γ = q_γ,
        )
        theta_relaxed = @call_rule softdot(:θ, Marginalisation) (
            q_y_x = joint_stock,
            q_γ = q_γ,
            meta = damping,
        )
        @test weightedmean(theta_relaxed) ≈ weightedmean(theta_stock)
        @test precision(theta_relaxed) ≈ precision(theta_stock)

        gamma_stock = @call_rule softdot(:γ, Marginalisation) (
            q_y_x = joint_stock,
            q_θ = q_θ,
        )
        state = NGMPEdgeState(damping)
        gamma_relaxed = @call_rule softdot(:γ, NaturalGradientMessage()) (
            q_y_x = joint_stock,
            q_θ = q_θ,
            q_γ = q_γ,
            meta = state,
        )
        @test shape(gamma_relaxed) ≈ shape(gamma_stock)
        @test rate(gamma_relaxed) ≈ rate(gamma_stock)
        @test state.nfired == 1

        stock_energy = score(
            AverageEnergy(),
            SoftDot,
            Val{(:y_x, :θ, :γ)}(),
            (
                Marginal(joint_stock, false, false),
                Marginal(q_θ, false, false),
                Marginal(q_γ, false, false),
            ),
            nothing,
        )
        relaxed_energy = score(
            AverageEnergy(),
            SoftDot,
            Val{(:y_x, :θ, :γ)}(),
            (
                Marginal(joint_stock, false, false),
                Marginal(q_θ, false, false),
                Marginal(q_γ, false, false),
            ),
            damping,
        )
        @test relaxed_energy ≈ stock_energy
    end

    @testset "theta variance is retained" begin
        mθ, Vθ = mean_cov(q_θ)
        ξx = weightedmean(m_x)
        Λx = precision(m_x)
        γbar = mean(q_γ)
        D = Λx + γbar * Vθ

        y_message = @call_rule softdot(:y, Marginalisation) (
            m_x = m_x,
            q_θ = q_θ,
            q_γ = q_γ,
            meta = damping,
        )
        @test mean(y_message) ≈ dot(mθ, D \ ξx)
        @test var(y_message) ≈ inv(γbar) + dot(mθ, D \ mθ)

        my, vy = mean_var(m_y)
        c = inv(vy + inv(γbar))
        x_message = @call_rule softdot(:x, Marginalisation) (
            m_y = m_y,
            q_θ = q_θ,
            q_γ = q_γ,
            meta = damping,
        )
        @test weightedmean(x_message) ≈ (c * my) .* mθ
        @test precision(x_message) ≈ c .* (mθ * mθ') + γbar .* Vθ
    end

    @testset "theta and x VMP targets are damped in natural coordinates" begin
        α = 0.25

        q_y_x = @call_marginalrule SoftDot(:y_x) (
            m_y = m_y,
            m_x = m_x,
            q_θ = q_θ,
            q_γ = q_γ,
        )
        theta_target = @call_rule softdot(:θ, Marginalisation) (
            q_y_x = q_y_x,
            q_γ = q_γ,
        )
        theta_state = NGMPEdgeState(DampingMeta(alpha = α, beta = 0.0))
        theta_first = @call_rule softdot(:θ, NaturalGradientMessage()) (
            q_y_x = q_y_x,
            q_θ = q_θ,
            q_γ = q_γ,
            meta = theta_state,
        )
        @test weightedmean(theta_first) ≈ α .* weightedmean(theta_target)
        @test precision(theta_first) ≈ α .* precision(theta_target)
        @test theta_state.nfired == 1

        theta_second = @call_rule softdot(:θ, NaturalGradientMessage()) (
            q_y_x = q_y_x,
            q_θ = q_θ,
            q_γ = q_γ,
            meta = theta_state,
        )
        second_scale = α * (2 - α)
        @test weightedmean(theta_second) ≈ second_scale .* weightedmean(theta_target)
        @test precision(theta_second) ≈ second_scale .* precision(theta_target)
        @test theta_state.nfired == 2

        x_target = @call_rule softdot(:x, Marginalisation) (
            m_y = m_y,
            q_θ = q_θ,
            q_γ = q_γ,
        )
        q_x = MvNormalMeanCovariance([0.1, -0.3], [0.7 0.05; 0.05 0.9])
        x_state = NGMPEdgeState(DampingMeta(alpha = α, beta = 0.0))
        x_first = @call_rule softdot(:x, NaturalGradientMessage()) (
            m_y = m_y,
            q_θ = q_θ,
            q_x = q_x,
            q_γ = q_γ,
            meta = x_state,
        )
        @test weightedmean(x_first) ≈ α .* weightedmean(x_target)
        @test precision(x_first) ≈ α .* precision(x_target)
        @test x_state.nfired == 1
        @test theta_state !== x_state
    end

    @testset "NGMPDependencies activates independent theta and x edge states" begin
        dependencies = NGMPDependencies(
            θ = nothing,
            x = nothing,
        )
        iterations = 4
        result = infer(
            model = relaxed_structured_softdot_toy(
                dependencies = dependencies,
                damping = DampingMeta(alpha = 0.25, beta = 0.0),
            ),
            data = (observation = 0.8,),
            constraints = relaxed_structured_softdot_constraints(),
            initialization = relaxed_structured_softdot_initialization(),
            iterations = iterations,
            free_energy = true,
        )

        @test length(dependencies.states) == 2
        @test all(state -> state.nfired >= iterations, dependencies.states)
        @test all(isfinite, result.free_energy)
        @test all(isfinite, mean(last(result.posteriors[:θ])))
        @test all(isfinite, mean(last(result.posteriors[:x])))
    end
end
