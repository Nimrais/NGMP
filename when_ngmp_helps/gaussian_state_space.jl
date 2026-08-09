#!/usr/bin/env julia

include(joinpath(@__DIR__, "common.jl"))
using .WhenNGMPHelpsCommon

using CSV
using DataFrames
using Distributions
using FastGaussQuadrature
using LinearAlgebra
using Plots
using Random
using RxInfer
using SpecialFunctions
using StableRNGs
using Statistics

function rts_smoother(observations, observation_variances, initial_mean,
                      initial_variance, process_variance)
    n = length(observations)
    filtered_mean = zeros(n)
    filtered_variance = zeros(n)
    log_likelihood = 0.0
    predicted_mean = initial_mean
    predicted_variance = initial_variance
    for index in 1:n
        innovation_variance = predicted_variance + observation_variances[index]
        log_likelihood += logpdf(
            Normal(predicted_mean, sqrt(innovation_variance)),
            observations[index],
        )
        gain = predicted_variance / innovation_variance
        filtered_mean[index] = predicted_mean +
                               gain * (observations[index] - predicted_mean)
        filtered_variance[index] = (1 - gain) * predicted_variance
        predicted_mean = filtered_mean[index]
        predicted_variance = filtered_variance[index] + process_variance
    end

    smoothed_mean = copy(filtered_mean)
    smoothed_variance = copy(filtered_variance)
    for index in (n - 1):-1:1
        gain = filtered_variance[index] /
               (filtered_variance[index] + process_variance)
        smoothed_mean[index] = filtered_mean[index] +
            gain * (smoothed_mean[index + 1] - filtered_mean[index])
        smoothed_variance[index] = filtered_variance[index] + gain^2 *
            (smoothed_variance[index + 1] -
             (filtered_variance[index] + process_variance))
    end
    return (mean = smoothed_mean, variance = smoothed_variance, log_likelihood)
end

function fit_structured_vmp(observations, parameters; iterations = 200, tolerance = 1e-12)
    n = length(observations)
    shape = parameters.prior_shape + n / 2
    rate = parameters.prior_rate
    expected_precision = parameters.prior_shape / parameters.prior_rate
    local smoother
    completed_iterations = iterations
    elapsed = @elapsed for iteration in 1:iterations
        smoother = rts_smoother(
            observations,
            fill(inv(expected_precision), n),
            parameters.initial_mean,
            parameters.initial_variance,
            parameters.process_variance,
        )
        rate = parameters.prior_rate + 0.5 * sum(
            @. (observations - smoother.mean)^2 + smoother.variance
        )
        updated_precision = shape / rate
        difference = abs(updated_precision - expected_precision) / updated_precision
        expected_precision = updated_precision
        if difference < tolerance
            completed_iterations = iteration
            break
        end
    end
    return (;
        mean = smoother.mean,
        variance = smoother.variance,
        shape,
        rate,
        iterations = completed_iterations,
        elapsed,
    )
end

function project_student_t_to_state(
    observation,
    cavity_shape,
    cavity_rate,
    marginal_mean,
    marginal_variance,
    hermite_nodes,
    hermite_weights,
)
    points = marginal_mean .+ sqrt(2marginal_variance) .* hermite_nodes
    weights = hermite_weights ./ sqrt(π)
    residual = points .- observation
    first_derivative = @. (
        -(2cavity_shape + 1) * residual /
        (2cavity_rate + residual^2)
    )
    second_derivative = @. (
        -(2cavity_shape + 1) * (2cavity_rate - residual^2) /
        (2cavity_rate + residual^2)^2
    )
    precision = -sum(weights .* second_derivative)
    weighted_mean = sum(weights .* first_derivative) + marginal_mean * precision
    return weighted_mean, precision
end

function project_normal_precision_message(
    observation,
    cavity_mean,
    cavity_variance,
    marginal_shape,
    marginal_rate,
    legendre_nodes,
    legendre_weights,
)
    marginal = Gamma(marginal_shape, inv(marginal_rate))
    lower = quantile(marginal, 1e-9)
    upper = quantile(marginal, 1 - 1e-9)
    precisions = @. (upper + lower) / 2 + (upper - lower) / 2 * legendre_nodes
    weights = @. (upper - lower) / 2 * legendre_weights * pdf(marginal, precisions)
    normalization = sum(weights)
    log_message = @. (
        -0.5 * log(cavity_variance + inv(precisions)) -
        (observation - cavity_mean)^2 /
        (2 * (cavity_variance + inv(precisions)))
    )
    log_precision = log.(precisions)
    expected_log_precision = sum(weights .* log_precision) / normalization
    expected_precision = sum(weights .* precisions) / normalization
    expected_message = sum(weights .* log_message) / normalization
    covariance = [
        sum(weights .* log_precision .* log_message) / normalization -
        expected_log_precision * expected_message,
        sum(weights .* precisions .* log_message) / normalization -
        expected_precision * expected_message,
    ]
    fisher = [
        trigamma(marginal_shape) inv(marginal_rate)
        inv(marginal_rate) marginal_shape / marginal_rate^2
    ]
    natural = fisher \ covariance
    return natural[1], -natural[2]
end

function fit_ngmp(
    observations,
    parameters,
    quadrature;
    damping = 0.5,
    iterations = 150,
    tolerance = 1e-9,
    mean_field_iterations = 25,
)
    n = length(observations)
    state_weighted_mean = zeros(n)
    state_precision = zeros(n)
    shape_increment = zeros(n)
    rate_increment = zeros(n)
    state_mean = zeros(n)
    state_variance = fill(parameters.initial_variance, n)
    shape = parameters.prior_shape
    rate = parameters.prior_rate
    previous_global = [0.0, 0.0]
    local smoother
    completed_iterations = iterations

    elapsed = @elapsed for iteration in 1:iterations
        proposed_weighted_mean = similar(state_weighted_mean)
        proposed_precision = similar(state_precision)
        proposed_shape = similar(shape_increment)
        proposed_rate = similar(rate_increment)

        if iteration <= mean_field_iterations
            expected_precision = shape / rate
            @. proposed_weighted_mean = expected_precision * observations
            @. proposed_precision = expected_precision
            @. proposed_shape = 0.5
            @. proposed_rate = 0.5 * ((observations - state_mean)^2 + state_variance)
            step = 1.0
        else
            for index in 1:n
                cavity_shape = shape - shape_increment[index]
                cavity_rate = rate - rate_increment[index]
                proposed_weighted_mean[index], proposed_precision[index] =
                    project_student_t_to_state(
                        observations[index],
                        cavity_shape,
                        cavity_rate,
                        state_mean[index],
                        state_variance[index],
                        quadrature.hermite_nodes,
                        quadrature.hermite_weights,
                    )
                cavity_precision = max(
                    inv(state_variance[index]) - state_precision[index],
                    1e-9,
                )
                cavity_mean = (
                    state_mean[index] / state_variance[index] -
                    state_weighted_mean[index]
                ) / cavity_precision
                proposed_shape[index], proposed_rate[index] =
                    project_normal_precision_message(
                        observations[index],
                        cavity_mean,
                        inv(cavity_precision),
                        shape,
                        rate,
                        quadrature.legendre_nodes,
                        quadrature.legendre_weights,
                    )
            end
            step = damping
        end

        state_weighted_mean = (1 - step) .* state_weighted_mean .+
                              step .* proposed_weighted_mean
        state_precision = (1 - step) .* state_precision .+ step .* proposed_precision
        shape_increment = (1 - step) .* shape_increment .+ step .* proposed_shape
        rate_increment = (1 - step) .* rate_increment .+ step .* proposed_rate

        all(state_precision .> 0) || error("NGMP produced a non-positive state site")
        smoother = rts_smoother(
            state_weighted_mean ./ state_precision,
            inv.(state_precision),
            parameters.initial_mean,
            parameters.initial_variance,
            parameters.process_variance,
        )
        state_mean = smoother.mean
        state_variance = smoother.variance
        shape = parameters.prior_shape + sum(shape_increment)
        rate = parameters.prior_rate + sum(rate_increment)
        shape > 0 && rate > 0 || error("NGMP produced an improper precision marginal")

        current_global = [shape, rate]
        difference = maximum(abs.(current_global .- previous_global) ./
                             (1 .+ abs.(current_global)))
        previous_global = current_global
        if difference < tolerance && iteration > mean_field_iterations + 1
            completed_iterations = iteration
            break
        end
    end
    return (;
        mean = state_mean,
        variance = state_variance,
        shape,
        rate,
        iterations = completed_iterations,
        elapsed,
    )
end

function exact_posterior(observations, parameters; grid_size)
    n = length(observations)
    precisions = exp.(range(log(1e-3), log(1e2); length = grid_size))
    log_density = zeros(grid_size)
    conditional_mean = zeros(grid_size, n)
    conditional_variance = zeros(grid_size, n)
    for (index, precision) in enumerate(precisions)
        smoother = rts_smoother(
            observations,
            fill(inv(precision), n),
            parameters.initial_mean,
            parameters.initial_variance,
            parameters.process_variance,
        )
        log_density[index] = logpdf(
            Gamma(parameters.prior_shape, inv(parameters.prior_rate)),
            precision,
        ) + smoother.log_likelihood
        conditional_mean[index, :] .= smoother.mean
        conditional_variance[index, :] .= smoother.variance
    end
    log_density .-= maximum(log_density)
    density = exp.(log_density)
    spacing = vcat(
        precisions[2] - precisions[1],
        (precisions[3:end] .- precisions[1:end-2]) ./ 2,
        precisions[end] - precisions[end-1],
    )
    weights = density .* spacing
    weights ./= sum(weights)
    mean_precision = sum(weights .* precisions)
    variance_precision = sum(weights .* precisions .^ 2) - mean_precision^2
    row_weights = reshape(weights, :, 1)
    state_mean = vec(sum(row_weights .* conditional_mean; dims = 1))
    state_variance = vec(sum(
        row_weights .* (conditional_variance .+ conditional_mean .^ 2);
        dims = 1,
    )) .- state_mean .^ 2
    return (;
        precisions,
        density = weights ./ spacing,
        weights,
        mean_precision,
        variance_precision,
        state_mean,
        state_variance,
    )
end

@model function rxinfer_structured_vmp_model(
    y,
    initial_mean,
    initial_variance,
    process_variance,
    prior_shape,
    prior_rate,
)
    τ ~ GammaShapeRate(prior_shape, prior_rate)
    z[1] ~ NormalMeanVariance(initial_mean, initial_variance)
    y[1] ~ NormalMeanPrecision(z[1], τ)
    for index in 2:length(y)
        z[index] ~ NormalMeanVariance(z[index - 1], process_variance)
        y[index] ~ NormalMeanPrecision(z[index], τ)
    end
end

@constraints function rxinfer_structured_vmp_constraints()
    q(z, τ) = q(z)q(τ)
end

@initialization function rxinfer_structured_vmp_initialization()
    q(τ) = GammaShapeRate(1.0, 1.0)
end

function crosscheck_vmp(observations, parameters)
    reference = fit_structured_vmp(observations, parameters)
    result = infer(
        model = rxinfer_structured_vmp_model(;
            parameters...,
        ),
        data = (y = observations,),
        constraints = rxinfer_structured_vmp_constraints(),
        initialization = rxinfer_structured_vmp_initialization(),
        iterations = 50,
        options = (limit_stack_depth = 500,),
    )
    rxinfer_precision = mean(last(result.posteriors[:τ]))
    isapprox(rxinfer_precision, reference.shape / reference.rate; rtol = 2e-3) ||
        error("closed-form structured VMP does not match RxInfer")
    return rxinfer_precision
end

function method_row(repetition, n, method, mean_state, variance_state,
                    mean_precision, variance_precision, exact, iterations, elapsed)
    return (;
        repetition,
        n,
        method,
        mean_precision,
        variance_precision,
        precision_variance_ratio = variance_precision / exact.variance_precision,
        mean_local_state_variance = mean(variance_state),
        state_variance_ratio = mean(variance_state ./ exact.state_variance),
        state_mean_rmse = sqrt(mean(abs2, mean_state .- exact.state_mean)),
        iterations,
        elapsed_seconds = elapsed,
    )
end

function run_study(config)
    parameters = (
        initial_mean = 0.0,
        initial_variance = 100.0,
        process_variance = config.process_variance,
        prior_shape = 1.0,
        prior_rate = 1.0,
    )
    hermite_nodes, hermite_weights = gausshermite(config.hermite_nodes)
    legendre_nodes, legendre_weights = gausslegendre(config.legendre_nodes)
    quadrature = (; hermite_nodes, hermite_weights, legendre_nodes, legendre_weights)
    rows = NamedTuple[]
    representative = nothing
    smallest_observations = nothing
    maximum_n = maximum(config.chain_lengths)

    for repetition in 1:config.repetitions
        rng = StableRNG(config.seed + repetition - 1)
        latent = cumsum(sqrt(parameters.process_variance) .* randn(rng, maximum_n)) .+ 1.0
        stream = latent .+ randn(rng, maximum_n) ./ sqrt(config.true_precision)
        for n in config.chain_lengths
            observations = stream[1:n]
            exact_elapsed = @elapsed exact = exact_posterior(
                observations,
                parameters;
                grid_size = config.precision_grid_size,
            )
            vmp = fit_structured_vmp(observations, parameters)
            ngmp = fit_ngmp(
                observations,
                parameters,
                quadrature;
                damping = config.damping,
                iterations = config.ngmp_iterations,
                mean_field_iterations = config.mean_field_iterations,
            )

            push!(rows, method_row(
                repetition,
                n,
                "Exact",
                exact.state_mean,
                exact.state_variance,
                exact.mean_precision,
                exact.variance_precision,
                exact,
                0,
                exact_elapsed,
            ))
            push!(rows, method_row(
                repetition,
                n,
                "VMP",
                vmp.mean,
                vmp.variance,
                vmp.shape / vmp.rate,
                vmp.shape / vmp.rate^2,
                exact,
                vmp.iterations,
                vmp.elapsed,
            ))
            push!(rows, method_row(
                repetition,
                n,
                "NGMP",
                ngmp.mean,
                ngmp.variance,
                ngmp.shape / ngmp.rate,
                ngmp.shape / ngmp.rate^2,
                exact,
                ngmp.iterations,
                ngmp.elapsed,
            ))

            if repetition == 1 && n == config.representative_length
                representative = (; observations, latent = latent[1:n], exact, vmp, ngmp)
            end
            if repetition == 1 && n == minimum(config.chain_lengths)
                smallest_observations = copy(observations)
            end
        end
    end
    crosscheck = crosscheck_vmp(smallest_observations, parameters)
    return DataFrame(rows), representative, crosscheck, parameters
end

function summarize_runs(runs)
    rows = NamedTuple[]
    for n in sort(unique(runs.n)), method in ("Exact", "VMP", "NGMP")
        selected = filter(row -> row.n == n && row.method == method, runs)
        precision = empirical_band(selected.precision_variance_ratio)
        state_ratio = empirical_band(selected.state_variance_ratio)
        local_variance = empirical_band(selected.mean_local_state_variance)
        push!(rows, (;
            n,
            method,
            precision_variance_ratio_median = precision[1],
            precision_variance_ratio_p10 = precision[2],
            precision_variance_ratio_p90 = precision[3],
            state_variance_ratio_median = state_ratio[1],
            state_variance_ratio_p10 = state_ratio[2],
            state_variance_ratio_p90 = state_ratio[3],
            local_state_variance_median = local_variance[1],
            local_state_variance_p10 = local_variance[2],
            local_state_variance_p90 = local_variance[3],
        ))
    end
    return DataFrame(rows)
end

function calibration_panel(summary)
    panel = plot(
        xlabel = "chain length N",
        ylabel = "Var(τ) / exact Var(τ)",
        xscale = :log2,
        xticks = let ticks = sort(unique(summary.n)); (ticks, string.(ticks)) end,
    )
    for (method, color, linestyle) in (
        ("VMP", COLORS.vmp, :dash),
        ("NGMP", COLORS.ngmp, :solid),
    )
        selected = sort(filter(row -> row.method == method, summary), :n)
        center = selected.precision_variance_ratio_median
        plot!(
            panel,
            selected.n,
            center;
            ribbon = (
                center .- selected.precision_variance_ratio_p10,
                selected.precision_variance_ratio_p90 .- center,
            ),
            color,
            linestyle,
            linewidth = 2.2,
            marker = :circle,
            fillalpha = 0.12,
            label = method,
        )
    end
    hline!(panel, [1.0];
        color = COLORS.exact, linestyle = :dot, label = "exact reference (= 1)")
    return panel
end

function make_panels(representative, summary, true_precision)
    exact = representative.exact
    n = length(representative.observations)
    mask = exact.density .> 1e-7 * maximum(exact.density)
    precision_panel = plot(
        exact.precisions[mask],
        exact.density[mask];
        color = COLORS.exact,
        linewidth = 2.7,
        label = "exact (quadrature + RTS)",
        xlabel = "observation precision τ",
        ylabel = "density",
    )
    plot!(precision_panel, exact.precisions[mask],
        pdf.(Gamma(representative.vmp.shape, inv(representative.vmp.rate)), exact.precisions[mask]);
        color = COLORS.vmp, linestyle = :dash, linewidth = 2.1, label = "VMP")
    plot!(precision_panel, exact.precisions[mask],
        pdf.(Gamma(representative.ngmp.shape, inv(representative.ngmp.rate)), exact.precisions[mask]);
        color = COLORS.ngmp, linewidth = 2.1, label = "NGMP")
    vline!(precision_panel, [true_precision];
        color = COLORS.truth, linestyle = :dot, label = "true τ")

    midpoint = cld(n, 2)
    trajectory_indices = max(1, midpoint - 35):min(n, midpoint + 35)
    trajectory_panel = plot(
        trajectory_indices,
        representative.latent[trajectory_indices];
        color = COLORS.truth,
        linewidth = 2.0,
        label = "true latent state",
        xlabel = "time index",
        ylabel = "latent state zₖ",
    )
    for (label, mean_state, variance_state, color, linestyle, fillalpha) in (
        (
            "exact mean ± 95% CI (quadrature + RTS)",
            representative.exact.state_mean,
            representative.exact.state_variance,
            COLORS.exact,
            :dot,
            0.10,
        ),
        (
            "NGMP mean ± 95% CI",
            representative.ngmp.mean,
            representative.ngmp.variance,
            COLORS.ngmp,
            :solid,
            0.10,
        ),
        (
            "VMP mean ± 95% CI",
            representative.vmp.mean,
            representative.vmp.variance,
            COLORS.vmp,
            :dash,
            0.08,
        ),
    )
        plot!(
            trajectory_panel,
            trajectory_indices,
            mean_state[trajectory_indices];
            ribbon = 1.96 .* sqrt.(variance_state[trajectory_indices]),
            color,
            linestyle,
            linewidth = 2.0,
            fillalpha,
            label,
        )
    end

    precision_calibration = calibration_panel(summary)

    exact_rows = sort(filter(row -> row.method == "Exact", summary), :n)
    local_center = exact_rows.local_state_variance_median
    persistence_panel = plot(
        exact_rows.n,
        local_center;
        ribbon = (
            local_center .- exact_rows.local_state_variance_p10,
            exact_rows.local_state_variance_p90 .- local_center,
        ),
        color = COLORS.exact,
        linewidth = 2.2,
        marker = :circle,
        fillalpha = 0.14,
        label = "exact posterior (quadrature + RTS)",
        xlabel = "chain length N",
        ylabel = "mean local Var(zₖ | y)",
        xscale = :log2,
        xticks = (exact_rows.n, string.(exact_rows.n)),
        ylims = (0, maximum(exact_rows.local_state_variance_p90) * 1.15),
    )

    return (;
        precision_panel,
        trajectory_panel,
        precision_calibration,
        persistence_panel,
    )
end

function make_preview(panels)
    return plot(
        panels.precision_panel,
        panels.trajectory_panel,
        panels.precision_calibration,
        panels.persistence_panel;
        layout = (2, 2),
        size = (1160, 790),
    )
end

function main()
    ensure_outputs()
    smoke = smoke_mode()
    config = (
        smoke,
        seed = 42,
        repetitions = smoke ? 1 : 20,
        chain_lengths = smoke ? [12, 24] : [25, 50, 100, 200, 400],
        representative_length = smoke ? 24 : 25,
        process_variance = 5.0,
        true_precision = 0.02,
        precision_grid_size = smoke ? 160 : 1200,
        hermite_nodes = smoke ? 16 : 64,
        legendre_nodes = smoke ? 64 : 400,
        damping = 0.5,
        ngmp_iterations = smoke ? 30 : 150,
        mean_field_iterations = smoke ? 15 : 25,
    )
    runs, representative, crosscheck, parameters = run_study(config)
    summary = summarize_runs(runs)
    CSV.write(joinpath(RESULT_DIR, "gaussian_state_space_runs.csv"), runs)
    CSV.write(joinpath(RESULT_DIR, "gaussian_state_space_aggregate.csv"), summary)
    write_config("gaussian_state_space", merge(config, parameters))
    write_summary("gaussian_state_space", [
        "# Gaussian state-space persistence study",
        "",
        "Closed-form VMP/RxInfer precision-mean cross-check: $(round(crosscheck, digits = 8))",
        "The exact mean local state variance is reported across chain lengths in `gaussian_state_space_aggregate.csv`.",
        "A nonzero plateau indicates that increasing sequence length does not eliminate local state uncertainty.",
        "The state panel reports posterior means and marginal 95% credible intervals for `zₖ | y`; these are latent-state intervals, not observation-predictive intervals.",
        "Every black curve is the exact reference: numerical quadrature integrates the unknown observation precision, while an RTS smoother computes the conditional linear-Gaussian state posterior at each quadrature point.",
        "For the illustrated first repetition, max |μ_VMP - μ_NGMP| = $(maximum(abs.(representative.vmp.mean .- representative.ngmp.mean))) and max |SD_VMP - SD_NGMP| = $(maximum(abs.(sqrt.(representative.vmp.variance) .- sqrt.(representative.ngmp.variance)))); the visible difference is uncertainty calibration rather than the posterior mean.",
    ])
    panels = make_panels(representative, summary, config.true_precision)
    save_pdf(panels.precision_panel, "gaussian_precision_posterior")
    save_pdf(panels.trajectory_panel, "gaussian_state_posterior")
    save_pdf(panels.precision_calibration, "gaussian_precision_calibration")
    save_pdf(panels.persistence_panel, "gaussian_local_uncertainty")
    save_figure(make_preview(panels), "gaussian_state_space")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
