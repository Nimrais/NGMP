function predictive_samples(
    params,
    features::AbstractMatrix,
    likelihood::String,
    config::BBBConfig;
    seed::Int,
    n_samples::Int = config.eval_samples,
)
    n_samples >= 1 || throw(ArgumentError("n_samples must be positive"))
    rng = StableRNG(seed)
    n = size(features, 1)
    means = Matrix{Float64}(undef, n_samples, n)
    variances = Matrix{Float64}(undef, n_samples, n)
    for sample in 1:n_samples
        epsilon = sample_epsilon(params, rng)
        sample_means, sample_scales = forward_sample(
            params, epsilon, features, likelihood, config.noise_floor,
        )
        means[sample, :] .= sample_means
        variances[sample, :] .= sample_scales .^ 2
    end
    return (means = means, variances = variances)
end

function row_logmeanexp(values::AbstractMatrix)
    n_rows, n_columns = size(values)
    result = Vector{Float64}(undef, n_columns)
    for column in 1:n_columns
        result[column] = logsumexp(@view values[:, column]) - log(n_rows)
    end
    return result
end

function mixture_logdensities(
    samples,
    targets_standardized::AbstractVector,
)
    n_samples, n = size(samples.means)
    n == length(targets_standardized) ||
        throw(DimensionMismatch("samples and targets disagree"))
    components = Matrix{Float64}(undef, n_samples, n)
    for sample in 1:n_samples
        variance = @view samples.variances[sample, :]
        residual_squared = (
            targets_standardized .- @view(samples.means[sample, :])
        ) .^ 2
        components[sample, :] .= -0.5 .* (
            log(2pi) .+ log.(variance) .+ residual_squared ./ variance
        )
    end
    return (
        log_predictive_density = row_logmeanexp(components),
        expected_log_likelihood = vec(mean(components; dims = 1)),
    )
end

function predictive_moments(samples)
    predictive_mean = vec(mean(samples.means; dims = 1))
    epistemic_variance = vec(mean(
        (samples.means .- transpose(predictive_mean)) .^ 2;
        dims = 1,
    ))
    aleatoric_variance = vec(mean(samples.variances; dims = 1))
    total_variance = epistemic_variance + aleatoric_variance
    return (
        mean = predictive_mean,
        epistemic_variance = epistemic_variance,
        aleatoric_variance = aleatoric_variance,
        total_variance = total_variance,
    )
end

coverage(targets, means, standard_deviations, z) = mean(
    abs.(targets .- means) .<= z .* standard_deviations,
)

"""
    predictive_metrics(samples, y_standardized, y_original, standardizer)

Compute the finite-mixture log predictive density using `logmeanexp`, not the
expected component log likelihood. The original-unit density applies the
target scaling Jacobian exactly.
"""
function predictive_metrics(
    samples,
    targets_standardized::AbstractVector,
    targets_original::AbstractVector,
    standardizer,
)
    densities = mixture_logdensities(samples, targets_standardized)
    moments = predictive_moments(samples)
    y_scale = standardizer.y_scale
    y_center = standardizer.y_center

    mean_original = y_center .+ y_scale .* moments.mean
    epistemic_original = y_scale^2 .* moments.epistemic_variance
    aleatoric_original = y_scale^2 .* moments.aleatoric_variance
    total_original = epistemic_original + aleatoric_original
    standard_deviation_original = sqrt.(total_original)

    lpd_standardized = mean(densities.log_predictive_density)
    ell_standardized = mean(densities.expected_log_likelihood)
    lpd_original = lpd_standardized - log(y_scale)
    ell_original = ell_standardized - log(y_scale)

    return (
        lpd_standardized = lpd_standardized,
        nll_standardized = -lpd_standardized,
        expected_log_likelihood_standardized = ell_standardized,
        lpd_original = lpd_original,
        nll_original = -lpd_original,
        expected_log_likelihood_original = ell_original,
        rmse_original = sqrt(mean((targets_original .- mean_original) .^ 2)),
        coverage_50 = coverage(
            targets_original, mean_original, standard_deviation_original,
            0.6744897501960817,
        ),
        coverage_80 = coverage(
            targets_original, mean_original, standard_deviation_original,
            1.2815515655446004,
        ),
        coverage_95 = coverage(
            targets_original, mean_original, standard_deviation_original,
            1.959963984540054,
        ),
        mean_total_variance_original = mean(total_original),
        mean_epistemic_variance_original = mean(epistemic_original),
        mean_aleatoric_variance_original = mean(aleatoric_original),
        predictive_mean_original = mean_original,
        total_variance_original = total_original,
    )
end
