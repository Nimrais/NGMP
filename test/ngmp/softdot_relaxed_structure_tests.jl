import ReactiveMP: @call_marginalrule
import LinearAlgebra: dot

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
end
