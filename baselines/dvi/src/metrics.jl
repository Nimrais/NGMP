const DVI_SCALAR_METRIC_NAMES = (
    :lpd_standardized,
    :nll_standardized,
    :expected_log_likelihood_standardized,
    :lpd_original,
    :nll_original,
    :expected_log_likelihood_original,
    :rmse_original,
    :coverage_50,
    :coverage_80,
    :coverage_95,
    :mean_total_variance_original,
    :mean_epistemic_variance_original,
    :mean_aleatoric_variance_original,
)

function predictive_distribution(
    params,
    features::AbstractMatrix,
    likelihood::String,
    config::DVIConfig,
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    output = propagate_dvi(params, features, config; tracker = tracker)
    predictive_mean = vec(output.mean[:, 1])
    epistemic_variance = max.(
        output_covariance_entry(output, 1, 1, config),
        0f0,
    )
    aleatoric_variance = if likelihood == "heteroscedastic"
        log_variance_mean = vec(output.mean[:, 2])
        log_variance_variance = max.(
            output_covariance_entry(output, 2, 2, config),
            0f0,
        )
        safe_exp(
            Float64.(
                log_variance_mean .+
                0.5f0 .* log_variance_variance,
            ),
            config,
            tracker,
        )
    elseif likelihood == "homoscedastic"
        fill(
            safe_exp(config.homo_log_variance, config, tracker),
            length(predictive_mean),
        )
    else
        throw(ArgumentError("unknown likelihood '$likelihood'"))
    end
    total_variance = max.(
        epistemic_variance .+ aleatoric_variance,
        DVI_EPSILON,
    )
    return (
        mean = predictive_mean,
        epistemic_variance = epistemic_variance,
        aleatoric_variance = aleatoric_variance,
        total_variance = total_variance,
        output = output,
    )
end

coverage(targets, means, standard_deviations, z) = mean(
    abs.(targets .- means) .<= z .* standard_deviations,
)

function predictive_metrics(
    params,
    features::AbstractMatrix,
    targets_standardized::AbstractVector,
    targets_original::AbstractVector,
    standardizer,
    likelihood::String,
    config::DVIConfig,
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    prediction = predictive_distribution(
        params, features, likelihood, config, tracker,
    )
    residual_squared =
        (targets_standardized .- prediction.mean) .^ 2
    pointwise_lpd = -0.5 .* (
        log(2pi) .+
        log.(prediction.total_variance) .+
        residual_squared ./ prediction.total_variance
    )
    expected_ll = expected_log_likelihood(
        prediction.output,
        targets_standardized,
        likelihood,
        config,
        tracker,
    )

    y_scale = standardizer.y_scale
    y_center = standardizer.y_center
    mean_original = y_center .+ y_scale .* prediction.mean
    epistemic_original =
        y_scale^2 .* prediction.epistemic_variance
    aleatoric_original =
        y_scale^2 .* prediction.aleatoric_variance
    total_original = epistemic_original .+ aleatoric_original
    standard_deviation_original = sqrt.(total_original)

    lpd_standardized = mean(pointwise_lpd)
    ell_standardized = mean(expected_ll)
    lpd_original = lpd_standardized - log(y_scale)
    ell_original = ell_standardized - log(y_scale)
    return (
        lpd_standardized = lpd_standardized,
        nll_standardized = -lpd_standardized,
        expected_log_likelihood_standardized = ell_standardized,
        lpd_original = lpd_original,
        nll_original = -lpd_original,
        expected_log_likelihood_original = ell_original,
        rmse_original = sqrt(mean(
            (targets_original .- mean_original) .^ 2,
        )),
        coverage_50 = coverage(
            targets_original,
            mean_original,
            standard_deviation_original,
            0.6744897501960817,
        ),
        coverage_80 = coverage(
            targets_original,
            mean_original,
            standard_deviation_original,
            1.2815515655446004,
        ),
        coverage_95 = coverage(
            targets_original,
            mean_original,
            standard_deviation_original,
            1.959963984540054,
        ),
        mean_total_variance_original = mean(total_original),
        mean_epistemic_variance_original = mean(epistemic_original),
        mean_aleatoric_variance_original = mean(aleatoric_original),
        predictive_mean_original = mean_original,
        total_variance_original = total_original,
    )
end
