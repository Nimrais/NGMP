# Exact-BP cavity messages toward the Gaussian softdot interfaces. With
# q_θ::PointMass, softdot(y | θᵀx, γ⁻¹) is NormalMeanPrecision(y | fᵀx, γ⁻¹) in
# the scalar s = fᵀx, so each rule must agree with the corresponding
# NormalMeanPrecision NGMP rule evaluated at the projected scalar quantities —
# and toward x, the vector site is that scalar site lifted rank-one along f.

@testset "softdot cavity rules (GaussianStudentT toward y and x)" begin
    projection = TangentProjection(type = Quadrature(128))
    f = [0.6, -0.2, 1.1]
    mx = [0.4, 0.1, -0.3]
    Vx = let A = [1.0 0.2 0.0; 0.2 0.8 -0.1; 0.0 -0.1 0.5]
        (A + A') / 2
    end
    m_x = MvNormalMeanCovariance(mx, Vx)
    m_γ = GammaShapeRate(3.0, 2.0)

    @testset "softdot(:y) equals NormalMeanPrecision(:out) on the projected scalar" begin
        q_y = NormalMeanVariance(0.2, 0.9)
        state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        message = @call_rule softdot(:y, NaturalGradientMessage(projection)) (
            m_x = m_x, m_γ = m_γ, q_y = q_y, q_θ = PointMass(f), meta = state,
        )

        reference_state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        reference = @call_rule NormalMeanPrecision(:out, NaturalGradientMessage(projection)) (
            m_μ = NormalMeanVariance(dot(f, mx), dot(f, Vx * f)),
            m_τ = m_γ,
            q_out = q_y,
            meta = reference_state,
        )
        @test weightedmean(message) ≈ weightedmean(reference)
        @test precision(message) ≈ precision(reference)
    end

    @testset "softdot(:x) is the rank-one lift of the scalar site" begin
        m_y = NormalMeanVariance(0.7, 0.3)
        q_x = MvNormalMeanCovariance(mx, Vx)
        q_residual = NormalMeanVariance(dot(f, mx), dot(f, Vx * f))

        state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        message = @call_rule softdot(:x, NaturalGradientMessage(projection)) (
            m_y = m_y, m_γ = m_γ, q_θ = PointMass(f), q_x = q_x, meta = state,
        )

        reference_state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        scalar = @call_rule NormalMeanPrecision(:μ, NaturalGradientMessage(projection)) (
            m_out = m_y, m_τ = m_γ, q_μ = q_residual, meta = reference_state,
        )
        ξ, Λ = weightedmean(scalar), precision(scalar)
        @test weightedmean(message) ≈ ξ .* f
        @test precision(message) ≈ Λ .* (f * f')
    end

    @testset "softdot(:x) with observed y is the lifted Student-t site" begin
        q_x = MvNormalMeanCovariance(mx, Vx)
        q_residual = NormalMeanVariance(dot(f, mx), dot(f, Vx * f))

        state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        message = @call_rule softdot(:x, NaturalGradientMessage(projection)) (
            m_y = PointMass(0.7), m_γ = m_γ, q_θ = PointMass(f), q_x = q_x, meta = state,
        )

        reference_state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        scalar = @call_rule NormalMeanPrecision(:μ, NaturalGradientMessage(projection)) (
            m_τ = m_γ, q_out = PointMass(0.7), q_μ = q_residual, meta = reference_state,
        )
        ξ, Λ = weightedmean(scalar), precision(scalar)
        @test weightedmean(message) ≈ ξ .* f
        @test precision(message) ≈ Λ .* (f * f')
        @test state.nfired == 1

        # The data-edge variant (y as PointMass MARGINAL, the layout GraphPPL
        # produces for observed y) must agree exactly.
        marginal_state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        marginal_form = @call_rule softdot(:x, NaturalGradientMessage(projection)) (
            m_γ = m_γ, q_y = PointMass(0.7), q_θ = PointMass(f), q_x = q_x, meta = marginal_state,
        )
        @test weightedmean(marginal_form) ≈ weightedmean(message)
        @test precision(marginal_form) ≈ precision(message)
    end

    @testset "softdot(:γ) cavity equals NormalMeanPrecision(:τ) on the projected scalar" begin
        q_γ = GammaShapeRate(2.5, 1.5)
        state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        message = @call_rule softdot(:γ, NaturalGradientMessage(projection)) (
            m_x = m_x, q_y = PointMass(0.7), q_θ = PointMass(f), q_γ = q_γ, meta = state,
        )

        reference_state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        reference = @call_rule NormalMeanPrecision(:τ, NaturalGradientMessage(projection)) (
            m_μ = NormalMeanVariance(dot(f, mx), dot(f, Vx * f)),
            q_out = PointMass(0.7),
            q_τ = q_γ,
            meta = reference_state,
        )
        @test shape(message) ≈ shape(reference)
        @test rate(message) ≈ rate(reference)
    end
end
