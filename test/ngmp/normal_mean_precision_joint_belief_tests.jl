import ReactiveMP: @call_marginalrule

function normal_mean_precision_joint_reference(
    m_out,
    v_out,
    m_μ,
    v_μ,
    a,
    b;
    order = 32_768,
)
    V = v_out + v_μ
    residual = m_out - m_μ
    logτ_mean = digamma(a) - log(b)
    logτ_std = sqrt(trigamma(a))
    logτ = range(
        logτ_mean - 16 * logτ_std,
        logτ_mean + 16 * logτ_std;
        length = order,
    )
    logτ_step = step(logτ)
    τ = exp.(logτ)

    log_gamma_density =
        a * log(b) - loggamma(a) .+ (a - 1) .* logτ .- b .* τ
    conditional_variance = V .+ inv.(τ)
    log_likelihood = -0.5 .* (
        log(2π) .+ log.(conditional_variance) .+
        abs2(residual) ./ conditional_variance
    )
    logweight = log_gamma_density .+ logτ .+ log_likelihood
    maximum_logweight = maximum(logweight)
    weights = exp.(logweight .- maximum_logweight)
    weights[firstindex(weights)] *= 0.5
    weights[lastindex(weights)] *= 0.5
    weight_sum = sum(weights)
    weights ./= weight_sum

    log_normalizer =
        maximum_logweight + log(weight_sum) + log(logτ_step)
    denominator = 1 .+ V .* τ
    average_energy = sum(weights .* (
        0.5 .* (
            log(2π) .- logτ .+
            τ .* (V ./ denominator .+ abs2(residual) ./ denominator .^ 2)
        )
    ))

    τ_entropy =
        log_normalizer -
        sum(weights .* log_gamma_density) -
        sum(weights .* log_likelihood)
    center_entropy = 0.5 * log(2π * exp(1) * v_out * v_μ / V)
    conditional_difference_entropy = sum(
        weights .* (0.5 .* log.(2π * exp(1) * V ./ denominator)),
    )
    return (
        log_normalizer = log_normalizer,
        average_energy = average_energy,
        entropy = center_entropy + τ_entropy + conditional_difference_entropy,
    )
end

function make_normal_mean_precision_joint_belief(
    m_out,
    v_out,
    m_μ,
    v_μ,
    a,
    b,
)
    return @call_marginalrule NormalMeanPrecision(:out_μ_τ) (
        m_out = NormalWeightedMeanPrecision(m_out / v_out, inv(v_out)),
        m_μ = NormalMeanVariance(m_μ, v_μ),
        m_τ = GammaShapeRate(a, b),
        meta = DampingMeta(alpha = 0.2, beta = 0.0),
    )
end

function fixed_mean_normal_precision_joint_reference(
    m_out,
    v_out,
    fixed_mean,
    a,
    b;
    order = 32_768,
)
    logτ_mean = digamma(a) - log(b)
    logτ_std = sqrt(trigamma(a))
    logτ = range(
        logτ_mean - 16 * logτ_std,
        logτ_mean + 16 * logτ_std;
        length = order,
    )
    logτ_step = step(logτ)
    τ = exp.(logτ)
    residual = m_out - fixed_mean

    log_gamma_density =
        a * log(b) - loggamma(a) .+ (a - 1) .* logτ .- b .* τ
    overlap_variance = v_out .+ inv.(τ)
    log_overlap = -0.5 .* (
        log(2π) .+ log.(overlap_variance) .+
        abs2(residual) ./ overlap_variance
    )
    logweight = log_gamma_density .+ log_overlap .+ logτ
    maximum_logweight = maximum(logweight)
    weights = exp.(logweight .- maximum_logweight)
    weights[firstindex(weights)] *= 0.5
    weights[lastindex(weights)] *= 0.5
    weight_sum = sum(weights)
    weights ./= weight_sum

    log_normalizer =
        maximum_logweight + log(weight_sum) + log(logτ_step)
    conditional_variance = inv.(inv(v_out) .+ τ)
    conditional_mean = conditional_variance .* (
        m_out / v_out .+ τ .* fixed_mean
    )
    conditional_squared_error =
        conditional_variance .+ abs2.(conditional_mean .- fixed_mean)
    average_energy = sum(weights .* (
        0.5 .* (
            log(2π) .- logτ .+ τ .* conditional_squared_error
        )
    ))
    τ_entropy = log_normalizer - sum(
        weights .* (log_gamma_density .+ log_overlap),
    )
    conditional_entropy = sum(weights .* (
        0.5 .* log.(2π * exp(1) .* conditional_variance)
    ))
    return (;
        log_normalizer,
        average_energy,
        entropy = τ_entropy + conditional_entropy,
    )
end

function make_fixed_mean_normal_precision_joint_belief(
    m_out,
    v_out,
    fixed_mean,
    a,
    b,
)
    return @call_marginalrule NormalMeanPrecision(:out_τ) (
        m_out = NormalMeanVariance(m_out, v_out),
        m_μ = PointMass(fixed_mean),
        m_τ = GammaShapeRate(a, b),
        meta = DampingMeta(alpha = 0.2, beta = 0.0),
    )
end

function make_observed_normal_mean_precision_joint_belief(
    observed_out,
    m_μ,
    v_μ,
    a,
    b,
)
    return @call_marginalrule NormalMeanPrecision(:μ_τ) (
        m_μ = NormalMeanVariance(m_μ, v_μ),
        m_τ = GammaShapeRate(a, b),
        q_out = PointMass(observed_out),
        meta = DampingMeta(alpha = 0.2, beta = 0.0),
    )
end

@model function scored_latent_ngbp_normal_toy(y, deps, damping)
    out ~ NormalMeanVariance(0.5, 1.0)
    μ ~ NormalMeanVariance(0.0, 1.0)
    τ ~ GammaShapeRate(2.0, 2.0)
    out ~ NormalMeanPrecision(μ, τ) where {
        dependencies = deps,
        meta = damping,
    }
    y ~ NormalMeanVariance(out, 0.5)
end

function run_scored_latent_ngbp_normal_toy(free_energy; iterations = 8)
    dependencies = NGMPDependencies(
        out = nothing,
        μ = nothing,
        τ = nothing,
        projection = TangentProjection(type = Unscented),
    )
    initialization = @initialization begin
        q(out) = NormalMeanVariance(0.5, 1.0)
        q(μ) = NormalMeanVariance(0.0, 1.0)
        q(τ) = GammaShapeRate(2.0, 2.0)
    end
    result = infer(
        model = scored_latent_ngbp_normal_toy(
            deps = dependencies,
            damping = DampingMeta(alpha = 0.2, beta = 0.0),
        ),
        data = (y = 0.7,),
        initialization = initialization,
        iterations = iterations,
        free_energy = free_energy,
        disable_inference_error_hint = true,
    )
    return result, dependencies
end

@testset "NormalMeanPrecision joint belief scoring" begin
    @testset "fixed-mean out/precision belief matches dense quadrature" begin
        cases = (
            (0.8, 0.4, 0.2, 2.5, 1.4),
            (4.0, 0.01, 0.0, 0.7, 0.5),
            (0.2, 3.0, 0.2, 100.0, 0.5),
            (-1.5, 0.05, 1.0, 10.0, 2.0),
        )
        for parameters in cases
            belief = make_fixed_mean_normal_precision_joint_belief(parameters...)
            @test belief isa
                  SurrogateModelling.FixedMeanNormalPrecisionJointBelief
            computed =
                SurrogateModelling._fixed_mean_normal_precision_joint_statistics!(belief)
            reference =
                fixed_mean_normal_precision_joint_reference(parameters...)
            @test computed.log_normalizer ≈ reference.log_normalizer atol = 2e-4
            @test computed.average_energy ≈ reference.average_energy atol = 2e-4
            @test computed.entropy ≈ reference.entropy atol = 2e-4

            average_energy = score(
                AverageEnergy(),
                NormalMeanPrecision,
                Val((:out_τ, :μ)),
                (
                    Marginal(belief, false, false),
                    Marginal(PointMass(parameters[3]), false, false),
                ),
                DampingMeta(alpha = 0.2, beta = 0.0),
            )
            @test average_energy === computed.average_energy
            @test entropy(belief) === computed.entropy
        end
    end

    @testset "observed out with structured mean/precision is symmetric" begin
        parameters = (0.2, 0.8, 0.4, 2.5, 1.4)
        observed_out, m_μ, v_μ, a, b = parameters
        belief = make_observed_normal_mean_precision_joint_belief(parameters...)
        @test belief isa
              SurrogateModelling.FixedMeanNormalPrecisionJointBelief
        computed =
            SurrogateModelling._fixed_mean_normal_precision_joint_statistics!(belief)
        reference = fixed_mean_normal_precision_joint_reference(
            m_μ, v_μ, observed_out, a, b,
        )
        @test computed.log_normalizer ≈ reference.log_normalizer atol = 2e-4
        @test computed.average_energy ≈ reference.average_energy atol = 2e-4
        @test computed.entropy ≈ reference.entropy atol = 2e-4

        average_energy = score(
            AverageEnergy(),
            NormalMeanPrecision,
            Val((:out, :μ_τ)),
            (
                Marginal(PointMass(observed_out), false, false),
                Marginal(belief, false, false),
            ),
            DampingMeta(alpha = 0.2, beta = 0.0),
        )
        @test average_energy === computed.average_energy
    end

    @testset "exact dispatch and shared lazy cache" begin
        belief = make_normal_mean_precision_joint_belief(
            0.8,
            0.4,
            0.2,
            0.7,
            2.5,
            1.4,
        )
        @test belief isa SurrogateModelling.NormalMeanPrecisionJointBelief
        @test isnothing(belief.statistics)

        marginal = Marginal(belief, false, false)
        average_energy = score(
            AverageEnergy(),
            NormalMeanPrecision,
            Val((:out_μ_τ,)),
            (marginal,),
            DampingMeta(alpha = 0.2, beta = 0.0),
        )
        cached_statistics = belief.statistics
        @test !isnothing(cached_statistics)
        @test average_energy === cached_statistics.average_energy
        @test entropy(belief) === cached_statistics.entropy
        @test belief.statistics === cached_statistics

        entropy_first_belief = make_normal_mean_precision_joint_belief(
            -0.3,
            1.2,
            0.4,
            0.6,
            1.2,
            0.9,
        )
        entropy_first = entropy(entropy_first_belief)
        entropy_first_cache = entropy_first_belief.statistics
        energy_second = score(
            AverageEnergy(),
            NormalMeanPrecision,
            Val((:out_μ_τ,)),
            (Marginal(entropy_first_belief, false, false),),
            nothing,
        )
        @test entropy_first === entropy_first_cache.entropy
        @test energy_second === entropy_first_cache.average_energy
        @test entropy_first_belief.statistics === entropy_first_cache
    end

    @testset "64-point statistics match a dense reference" begin
        cases = (
            (0.8, 0.4, 0.2, 0.7, 2.5, 1.4),
            (4.0, 0.01, 0.0, 3.0, 0.7, 0.5),
            (0.0, 3.0, 4 * sqrt(3.01), 0.01, 1.0, 2.0),
            (0.2, 0.3, 0.2, 0.3, 100.0, 0.5),
            (-1.5, 0.05, 1.0, 0.2, 10.0, 2.0),
        )
        for parameters in cases
            belief = make_normal_mean_precision_joint_belief(parameters...)
            computed =
                SurrogateModelling._normal_mean_precision_joint_statistics!(belief)
            reference = normal_mean_precision_joint_reference(parameters...)
            @test computed.log_normalizer ≈ reference.log_normalizer atol = 2e-4
            @test computed.average_energy ≈ reference.average_energy atol = 2e-4
            @test computed.entropy ≈ reference.entropy atol = 2e-4
        end
    end

    @testset "free-energy scoring does not alter NGMP inference" begin
        iterations = 8
        unscored, unscored_dependencies =
            run_scored_latent_ngbp_normal_toy(false; iterations = iterations)
        scored, scored_dependencies =
            run_scored_latent_ngbp_normal_toy(true; iterations = iterations)

        @test all(isfinite, scored.free_energy)
        @test length(unscored_dependencies.states) == 3
        @test length(scored_dependencies.states) == 3
        @test all(state -> state.nfired == iterations, unscored_dependencies.states)
        @test all(state -> state.nfired == iterations, scored_dependencies.states)

        for variable in (:out, :μ)
            for (unscored_q, scored_q) in zip(
                unscored.posteriors[variable],
                scored.posteriors[variable],
            )
                @test mean(scored_q) ≈ mean(unscored_q) atol = 1e-12
                @test var(scored_q) ≈ var(unscored_q) atol = 1e-12
            end
        end
        for (unscored_q, scored_q) in zip(
            unscored.posteriors[:τ],
            scored.posteriors[:τ],
        )
            @test shape(scored_q) ≈ shape(unscored_q) atol = 1e-12
            @test rate(scored_q) ≈ rate(unscored_q) atol = 1e-12
        end
    end
end
