# `MvStack` must be exact belief propagation, since the whole point of the node is
# that it does NOT throw away covariance the way `ManyPlus` does.
#
# The reference is an explicit equivalent model: instead of stacking scalars, give
# the vector edge a joint prior with the same (diagonal) moments and check the
# posteriors agree. The decisive test is the third one -- correlations learned
# downstream must travel back through the node to the individual scalars, which is
# precisely what a variance-summing aggregation cannot do.

using LinearAlgebra

# `@call_rule` needs a literal interface index, and the variadic `ManyOf` slot has
# to hold `Message` objects, so both rules are invoked through `ReactiveMP.rule`
# directly -- the same pattern as `manyplus_rule_input` in test/manyplus_tests.jl.
mv_stack_message(distribution) = ReactiveMP.Message(distribution, false, false)

function mv_stack_rule_output(inputs)
    messages = (ReactiveMP.ManyOf(map(mv_stack_message, Tuple(inputs))),)
    return ReactiveMP.rule(
        MvStack,
        Val(:out),
        Marginalisation(),
        Val((:inputs,)),
        messages,
        nothing,
        nothing,
        nothing,
        ReactiveMP.AnnotationDict(),
        nothing,
    )
end

function mv_stack_rule_input(output, other_inputs, target_index)
    messages = (
        mv_stack_message(output),
        ReactiveMP.ManyOf(map(mv_stack_message, Tuple(other_inputs))),
    )
    return ReactiveMP.rule(
        MvStack,
        (Val(:inputs), target_index),
        Marginalisation(),
        Val((:out, :inputs)),
        messages,
        nothing,
        nothing,
        nothing,
        ReactiveMP.AnnotationDict(),
        nothing,
    )
end

@model function mv_stack_dense_readout(y, n_units, unit_means, unit_variances, weights, noise_precision)
    local units
    for unit in 1:n_units
        units[unit] ~ NormalMeanVariance(unit_means[unit], unit_variances[unit])
    end
    stacked ~ MvStack(inputs = [units[unit] for unit in 1:n_units])
    y ~ softdot(weights, stacked, noise_precision)
end

@model function mv_stack_reference_readout(y, unit_means, unit_variances, weights, noise_precision)
    stacked ~ MvNormalMeanCovariance(unit_means, Diagonal(unit_variances))
    y ~ softdot(weights, stacked, noise_precision)
end

@testset "MvStack" begin
    n_units = 4
    unit_means = [0.3, -0.7, 1.1, 0.05]
    unit_variances = [0.5, 1.25, 0.2, 2.0]
    weights = [1.0, -2.0, 0.5, 3.0]
    noise_precision = 4.0
    observation = 1.7

    @testset "forward message is the exact joint" begin
        forward = mv_stack_rule_output(
            [NormalMeanVariance(m, v) for (m, v) in zip(unit_means, unit_variances)],
        )
        @test mean(forward) ≈ unit_means
        @test cov(forward) ≈ Diagonal(unit_variances)
    end

    @testset "uninformative half-edges remain neutral" begin
        forward = mv_stack_rule_output(
            fill(Uninformative(), n_units),
        )
        @test forward isa MvNormalWeightedMeanPrecision
        @test norm(weightedmean(forward)) == 0
        @test maximum(abs, Matrix(precision(forward))) <= 1e-10

        out_mean = [0.4, 0.9, -0.2, 1.3]
        out_cov = [
            1.0  0.4 -0.3  0.1;
            0.4  1.5  0.2 -0.2;
           -0.3  0.2  0.8  0.3;
            0.1 -0.2  0.3  1.1
        ]
        for target in 1:n_units
            message = mv_stack_rule_input(
                MvNormalMeanCovariance(out_mean, out_cov),
                fill(Uninformative(), n_units - 1),
                target,
            )
            @test mean(message) ≈ out_mean[target] atol = 1e-10
            @test var(message) ≈ out_cov[target, target] atol = 1e-10
        end
    end

    @testset "backward message equals explicit cavity marginalisation" begin
        # An output cavity with genuine off-diagonal structure, so the test can
        # tell a correct marginalisation from a diagonal shortcut.
        out_mean = [0.4, 0.9, -0.2, 1.3]
        out_cov = [
            1.0  0.4 -0.3  0.1;
            0.4  1.5  0.2 -0.2;
           -0.3  0.2  0.8  0.3;
            0.1 -0.2  0.3  1.1
        ]
        m_out = MvNormalMeanCovariance(out_mean, out_cov)

        for target in 1:n_units
            others = [index for index in 1:n_units if index != target]
            message = mv_stack_rule_input(
                m_out,
                [NormalMeanVariance(unit_means[i], unit_variances[i]) for i in others],
                target,
            )

            # Reference: build the H-dimensional joint by hand, leaving slot
            # `target` empty, then read off that coordinate's marginal.
            precision_matrix = Matrix(inv(out_cov))
            weighted_mean = inv(out_cov) * out_mean
            for index in others
                site = inv(unit_variances[index])
                precision_matrix[index, index] += site
                weighted_mean[index] += site * unit_means[index]
            end
            joint_cov = inv(Symmetric(precision_matrix))
            joint_mean = joint_cov * weighted_mean

            @test mean(message) ≈ joint_mean[target] atol = 1e-10
            @test var(message) ≈ joint_cov[target, target] atol = 1e-10
        end
    end

    @testset "backward message supports improper Gaussian sites" begin
        out_mean = [0.4, 0.9, -0.2, 1.3]
        out_cov = [
            1.0  0.4 -0.3  0.1;
            0.4  1.5  0.2 -0.2;
           -0.3  0.2  0.8  0.3;
            0.1 -0.2  0.3  1.1
        ]
        output = MvNormalMeanCovariance(out_mean, out_cov)
        target = 1
        site_precisions = [-3.0, 0.2, -1.5]
        sites = [
            NormalWeightedMeanPrecision(0.1 * index, site_precision)
            for (index, site_precision) in enumerate(site_precisions)
        ]
        message = mv_stack_rule_input(output, sites, target)

        joint_precision = inv(out_cov)
        joint_weighted_mean = joint_precision * out_mean
        for (offset, slot) in enumerate(2:n_units)
            joint_precision[slot, slot] += site_precisions[offset]
            joint_weighted_mean[slot] += 0.1 * offset
        end
        rest = 2:n_units
        solved = joint_precision[rest, rest] \ hcat(
            joint_precision[rest, target],
            joint_weighted_mean[rest],
        )
        expected_precision =
            joint_precision[target, target] -
            dot(joint_precision[rest, target], solved[:, 1])
        expected_weighted_mean =
            joint_weighted_mean[target] -
            dot(joint_precision[rest, target], solved[:, 2])

        if expected_precision > sqrt(eps(Float64))
            @test precision(message) ≈ expected_precision atol = 1e-10
            @test weightedmean(message) ≈ expected_weighted_mean atol = 1e-10
        else
            @test precision(message) > 0
            @test precision(message) <= 1e-10
            @test weightedmean(message) == 0
        end
    end

    @testset "inference matches an explicit joint prior" begin
        stacked_result = infer(
            model = mv_stack_dense_readout(
                n_units = n_units,
                unit_means = unit_means,
                unit_variances = unit_variances,
                weights = weights,
                noise_precision = noise_precision,
            ),
            data = (y = observation,),
            returnvars = (stacked = KeepLast(),),
            iterations = 10,
            free_energy = false,
            showprogress = false,
        )
        reference_result = infer(
            model = mv_stack_reference_readout(
                unit_means = unit_means,
                unit_variances = unit_variances,
                weights = weights,
                noise_precision = noise_precision,
            ),
            data = (y = observation,),
            returnvars = (stacked = KeepLast(),),
            iterations = 10,
            free_energy = false,
            showprogress = false,
        )

        stacked_posterior = stacked_result.posteriors[:stacked]
        reference_posterior = reference_result.posteriors[:stacked]

        @test mean(stacked_posterior) ≈ mean(reference_posterior) atol = 1e-8
        @test cov(stacked_posterior) ≈ cov(reference_posterior) atol = 1e-8

        # The readout genuinely correlates the units: without off-diagonal mass
        # the previous assertions would be vacuous.
        posterior_covariance = cov(reference_posterior)
        off_diagonal = posterior_covariance - Diagonal(diag(posterior_covariance))
        @test maximum(abs, off_diagonal) > 1e-3
    end

    @testset "downstream correlation reaches the individual units" begin
        # A single observation of a weighted sum must shift every unit, in
        # proportion to its weight and prior variance. `ManyPlus` can also shift
        # them, but it cannot represent the resulting cross-unit covariance --
        # here we check the per-unit marginals against the exact joint model.
        result = infer(
            model = mv_stack_dense_readout(
                n_units = n_units,
                unit_means = unit_means,
                unit_variances = unit_variances,
                weights = weights,
                noise_precision = noise_precision,
            ),
            data = (y = observation,),
            returnvars = (units = KeepLast(), stacked = KeepLast()),
            iterations = 10,
            free_energy = false,
            showprogress = false,
        )
        unit_posteriors = collect(vec(result.posteriors[:units]))
        stacked_posterior = result.posteriors[:stacked]

        for unit in 1:n_units
            @test mean(unit_posteriors[unit]) ≈ mean(stacked_posterior)[unit] atol = 1e-8
            @test var(unit_posteriors[unit]) ≈ cov(stacked_posterior)[unit, unit] atol = 1e-8
            # Each unit's variance must have contracted relative to its prior.
            @test var(unit_posteriors[unit]) < unit_variances[unit]
        end
    end
end
