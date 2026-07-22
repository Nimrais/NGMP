import BayesBase: mean_cov

@testset "Probit two-class Categorical message bridge" begin
    @testset "matches Bernoulli moment matching" begin
        categorical_messages = (
            Categorical([0.9, 0.1]),
            Categorical([0.5, 0.5]),
            Categorical([0.2, 0.8]),
        )
        cavities = (
            NormalMeanVariance(-2.0, 0.2),
            NormalMeanVariance(0.0, 1.0),
            NormalMeanVariance(1.5, 3.0),
        )

        for categorical in categorical_messages, cavity in cavities
            bridged = @call_rule Probit(:in, Marginalisation) (
                m_out = categorical,
                m_in = cavity,
            )
            reference = @call_rule Probit(:in, Marginalisation) (
                m_out = Bernoulli(probvec(categorical)[2]),
                m_in = cavity,
            )
            bridged_mean, bridged_covariance = mean_cov(bridged)
            reference_mean, reference_covariance = mean_cov(reference)
            @test bridged_mean ≈ reference_mean
            @test bridged_covariance ≈ reference_covariance

            variational = @call_rule Probit(:in, Marginalisation) (
                q_out = categorical,
                m_in = cavity,
            )
            variational_mean, variational_covariance = mean_cov(variational)
            @test variational_mean ≈ reference_mean
            @test variational_covariance ≈ reference_covariance

            bernoulli_variational = @call_rule Probit(:in, Marginalisation) (
                q_out = Bernoulli(probvec(categorical)[2]),
                m_in = cavity,
            )
            bernoulli_mean, bernoulli_covariance = mean_cov(bernoulli_variational)
            @test bernoulli_mean ≈ reference_mean
            @test bernoulli_covariance ≈ reference_covariance
        end
    end

    @testset "component 2 evidence gives a positive score" begin
        switch_evidence = @call_rule NormalMixture{2}(
            :switch,
            Marginalisation,
        ) (
            q_out = PointMass(2.0),
            q_m = ManyOf(PointMass(-2.0), PointMass(2.0)),
            q_p = ManyOf(PointMass(10.0), PointMass(10.0)),
        )
        @test probvec(switch_evidence)[2] > 0.5

        cavity = NormalMeanVariance(0.0, 1.0)
        score_message = @call_rule Probit(:in, Marginalisation) (
            m_out = switch_evidence,
            m_in = cavity,
        )
        cavity_mean, cavity_variance = mean_cov(cavity)
        posterior_precision = inv(cavity_variance) + precision(score_message)
        score_variance = inv(posterior_precision)
        score_mean = score_variance * (
            cavity_mean / cavity_variance + weightedmean(score_message)
        )
        p6 = cdf(Normal(), score_mean / sqrt(1 + score_variance))

        @test score_mean > 0
        @test p6 > 0.5
    end

    @testset "Gaussian marginal forward adapter matches the stock message rule" begin
        for gaussian in (
            NormalMeanVariance(-1.2, 0.3),
            NormalMeanVariance(0.0, 1.0),
            NormalMeanVariance(2.4, 4.0),
        )
            adapted = @call_rule Probit(:out, Marginalisation) (q_in = gaussian,)
            reference = @call_rule Probit(:out, Marginalisation) (m_in = gaussian,)
            @test collect(probvec(adapted)) ≈ collect(probvec(reference))
        end
    end

    @testset "rejects non-binary categorical messages" begin
        invalid = Categorical([0.2, 0.3, 0.5])
        error = try
            @call_rule Probit(:in, Marginalisation) (
                m_out = invalid,
                m_in = NormalMeanVariance(0.0, 1.0),
            )
            nothing
        catch exception
            exception
        end
        @test error isa DimensionMismatch
        @test occursin("exactly two probabilities", sprint(showerror, error))
    end
end
