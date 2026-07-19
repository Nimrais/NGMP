import SurrogateModelling: NaturalGradientMP

@testset "generic exponential-family damping" begin
    @testset "dependency override precedence and constructor compatibility" begin
        node_damping = DampingMeta(alpha = 0.8, beta = 0.1, max_step = 4.0)
        override = DampingMeta(alpha = 0.3, beta = 0.6, max_step = 1.5)
        usermeta = LinearReshapeMeta(2, 3)

        state = NGMPEdgeState(usermeta; damping = override)
        @test state.usermeta === usermeta
        @test state.damping === override
        @test NaturalGradientMP.damping_parameters(state) == (0.3, 0.6)
        @test NaturalGradientMP.damping_max_step(state) == 1.5

        overridden_node_state = NGMPEdgeState(node_damping; damping = override)
        @test overridden_node_state.usermeta === node_damping
        @test NaturalGradientMP.damping_parameters(overridden_node_state) == (0.3, 0.6)
        @test NaturalGradientMP.damping_max_step(overridden_node_state) == 1.5

        # Existing one-argument and full-state constructors keep node-level and
        # fallback behavior unchanged.
        @test NaturalGradientMP.damping_parameters(NGMPEdgeState(node_damping)) == (0.8, 0.1)
        legacy_state = NGMPEdgeState(nothing, nothing, Float64[], Float64[], 0)
        @test legacy_state.damping === nothing
        @test NaturalGradientMP.damping_parameters(legacy_state) == (0.5, 0.2)

        legacy_dependencies = NGMPDependencies(a = nothing)
        @test legacy_dependencies.damping === nothing
        states = NGMPEdgeState[]
        positional_dependencies = NGMPDependencies(
            (a = nothing,), NaturalGradientMP.ClosedFormDefault(), states
        )
        @test positional_dependencies.damping === nothing
        @test positional_dependencies.states === states

        dependencies = NGMPDependencies(a = nothing, damping = override)
        @test dependencies.specification == (a = nothing,)
        @test dependencies.damping === override
        @test_throws ArgumentError NGMPDependencies(a = nothing, damping = :invalid)
    end

    @testset "optional natural-step bound" begin
        state = NGMPEdgeState(
            DampingMeta(alpha = 1.0, beta = 0.0, max_step = 0.25),
        )
        message = NaturalGradientMP.apply_damping!(
            state,
            NormalWeightedMeanPrecision(100.0, 40.0),
        )
        @test sqrt(sum(abs2, state.η)) ≈ 0.25
        @test message isa NormalWeightedMeanPrecision
        @test_throws ArgumentError DampingMeta(max_step = 0.0)
    end

    @testset "Gaussian wrapper matches the manual scalar recursion" begin
        α, β = 0.5, 0.2
        state = NGMPEdgeState(DampingMeta(alpha = α, beta = β))
        ξ, Λ, vξ, vΛ = 0.0, 0.0, 0.0, 0.0
        targets = [(1.2, 0.8), (2.0, 1.5), (1.7, 1.2), (1.9, 1.4)]
        for (t, (ξt, Λt)) in enumerate(targets)
            msg = NaturalGradientMP.apply_damping!(state, ξt, Λt)
            vξ = β * vξ + α * (ξt - ξ); ξ += vξ
            vΛ = β * vΛ + α * (Λt - Λ); Λ += vΛ
            @test msg isa NormalWeightedMeanPrecision
            @test weightedmean(msg) ≈ ξ
            @test precision(msg) ≈ Λ
            @test state.nfired == t
            @test state.message === msg
        end
    end

    @testset "first firing = α · target (implicit flat previous message)" begin
        state = NGMPEdgeState(DampingMeta(alpha = 0.25, beta = 0.7))
        msg = NaturalGradientMP.apply_damping!(state, 4.0, 8.0)
        @test weightedmean(msg) ≈ 0.25 * 4.0
        @test precision(msg) ≈ 0.25 * 8.0
    end

    @testset "Gamma family: recursion in natural parameters η = (a−1, −b)" begin
        α, β = 0.5, 0.2
        state = NGMPEdgeState(DampingMeta(alpha = α, beta = β))
        η = [0.0, 0.0]
        mom = [0.0, 0.0]
        targets = [GammaShapeRate(2.5, 4.0), GammaShapeRate(3.0, 2.0), GammaShapeRate(2.8, 2.5)]
        for (t, target) in enumerate(targets)
            msg = NaturalGradientMP.apply_damping!(state, target)
            ηt = [shape(target) - 1, -rate(target)]
            @. mom = β * mom + α * (ηt - η)
            @. η += mom
            @test msg isa GammaShapeRate
            @test shape(msg) ≈ η[1] + 1
            @test rate(msg) ≈ -η[2]
            @test state.nfired == t
        end
    end

    @testset "Gamma flat init: first message = Gamma(α(a−1)+1, α·b)" begin
        α = 0.2
        state = NGMPEdgeState(DampingMeta(alpha = α, beta = 0.0))
        msg = NaturalGradientMP.apply_damping!(state, GammaShapeRate(2.5, 4.0))
        @test shape(msg) ≈ α * 1.5 + 1
        @test rate(msg) ≈ α * 4.0
    end

    @testset "α = 1, β = 0 passes the target through exactly" begin
        state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        msg = NaturalGradientMP.apply_damping!(state, GammaShapeRate(2.5, 4.0))
        @test shape(msg) ≈ 2.5
        @test rate(msg) ≈ 4.0
    end

    @testset "improper targets never throw" begin
        state = NGMPEdgeState(DampingMeta(alpha = 0.5, beta = 0.0))
        msg = NaturalGradientMP.apply_damping!(state, GammaShapeRate(1.4, -0.3))
        @test shape(msg) ≈ 0.5 * 0.4 + 1
        @test rate(msg) ≈ -0.15
        state2 = NGMPEdgeState(nothing)
        msg2 = NaturalGradientMP.apply_damping!(state2, NormalWeightedMeanPrecision(0.5, -2.0))
        @test precision(msg2) < 0
    end

    @testset "ExponentialFamilyDistribution sites are accepted directly" begin
        state = NGMPEdgeState(DampingMeta(alpha = 1.0, beta = 0.0))
        site = ExponentialFamilyDistribution(Gamma, [1.5, -4.0], nothing, nothing)  # Δa = 1.5, Δb = 4
        msg = NaturalGradientMP.apply_damping!(state, site)
        @test msg isa GammaShapeRate
        @test shape(msg) ≈ 2.5
        @test rate(msg) ≈ 4.0
    end
end
