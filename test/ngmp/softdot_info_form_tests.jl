import ReactiveMP: @call_marginalrule
import LinearAlgebra: Symmetric, I
import Random: MersenneTwister

# The specializations dispatch on `m_x::MvNormalWeightedMeanPrecision`; converting
# `m_x` to moment form forces the stock `::Any`/`::NormalDistributionsFamily`
# methods, giving a reference on identical inputs.
@testset "info-form structured softdot rules" begin
    rng = MersenneTwister(7)
    d = 6
    f = randn(rng, d)
    A = randn(rng, d, d)
    Λx = Symmetric(A * A' + d * I)
    ξx = randn(rng, d)
    m_x_info = MvNormalWeightedMeanPrecision(ξx, Matrix(Λx))
    m_x_moment = convert(MvNormalMeanCovariance, m_x_info)
    m_y = NormalWeightedMeanPrecision(0.7, 2.3)
    q_θ = PointMass(f)
    q_γ = GammaShapeRate(3.0, 2.0)

    @testset "SoftDot(:y_x) marginal rule matches stock" begin
        fast = @call_marginalrule SoftDot(:y_x) (m_y = m_y, m_x = m_x_info, q_θ = q_θ, q_γ = q_γ)
        slow = @call_marginalrule SoftDot(:y_x) (m_y = m_y, m_x = m_x_moment, q_θ = q_θ, q_γ = q_γ)
        @test weightedmean(fast) ≈ weightedmean(slow)
        @test precision(fast) ≈ precision(slow)
    end

    @testset "softdot(:y) rule matches stock" begin
        fast = @call_rule softdot(:y, Marginalisation) (q_θ = q_θ, m_x = m_x_info, q_γ = q_γ)
        slow = @call_rule softdot(:y, Marginalisation) (q_θ = q_θ, m_x = m_x_moment, q_γ = q_γ)
        @test mean(fast) ≈ mean(slow)
        @test var(fast) ≈ var(slow)
    end

    @testset "moment-form m_y also accepted" begin
        m_y_moment = convert(NormalMeanVariance, m_y)
        fast = @call_marginalrule SoftDot(:y_x) (m_y = m_y_moment, m_x = m_x_info, q_θ = q_θ, q_γ = q_γ)
        slow = @call_marginalrule SoftDot(:y_x) (m_y = m_y_moment, m_x = m_x_moment, q_θ = q_θ, q_γ = q_γ)
        @test weightedmean(fast) ≈ weightedmean(slow)
        @test precision(fast) ≈ precision(slow)
    end

    # Dispatch guard: a regression here is silent (Julia falls back to the stock
    # `::Any` methods with identical results, only the speed is lost). Following
    # the `which(f, argtypes)` idiom: resolve the method selected for an info-form
    # `m_x` and for a moment-form `m_x` with otherwise identical argument types —
    # they must be DIFFERENT methods. If the specialization is ever deleted,
    # shadowed, or stops matching, both calls resolve to the same stock method and
    # this fails loudly.
    @testset "specialized methods are the selected dispatch" begin
        msg(d) = typeof(ReactiveMP.Message(d, false, false))
        mar(d) = typeof(ReactiveMP.Marginal(d, false, false))

        rule_argtypes(m_x) = (
            Type{SoftDot}, Val{:y}, Marginalisation, Val{(:x,)},
            Tuple{msg(m_x)},
            Val{(:θ, :γ)}, Tuple{mar(q_θ), mar(q_γ)},
            Nothing, Nothing, Nothing,
        )
        rule_info = Base.which(ReactiveMP.rule, rule_argtypes(m_x_info))
        rule_moment = Base.which(ReactiveMP.rule, rule_argtypes(m_x_moment))
        @test rule_info !== rule_moment

        marginalrule_argtypes(m_x) = (
            Type{SoftDot}, Val{:y_x}, Val{(:y, :x)},
            Tuple{msg(m_y), msg(m_x)},
            Val{(:θ, :γ)}, Tuple{mar(q_θ), mar(q_γ)},
            Nothing, Nothing,
        )
        marginal_info = Base.which(ReactiveMP.marginalrule, marginalrule_argtypes(m_x_info))
        marginal_moment = Base.which(ReactiveMP.marginalrule, marginalrule_argtypes(m_x_moment))
        @test marginal_info !== marginal_moment
    end
end
