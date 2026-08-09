#!/usr/bin/env julia

include(joinpath(@__DIR__, "common.jl"))
using .WhenNGMPHelpsCommon

using CSV
using DataFrames
using Distributions
using ExponentialFamily
using ExponentialFamilyProjection
using Plots
using Random
using RxInfer
using StableRNGs
using Statistics
using SurrogateModelling

@model function normal_vmp_model(y, m0, v0, a0, b0)
    x ~ NormalMeanVariance(m0, v0)
    τ ~ GammaShapeRate(a0, b0)
    for index in 1:length(y)
        y[index] ~ NormalMeanPrecision(x, τ)
    end
end

@initialization function normal_vmp_initialization(initial_mean, initial_variance)
    q(x) = NormalMeanVariance(initial_mean, initial_variance)
end

@model function normal_ngmp_model(y, m0, v0, a0, b0, dependencies, damping)
    x ~ NormalMeanVariance(m0, v0)
    τ ~ GammaShapeRate(a0, b0)
    for index in 1:length(y)
        y[index] ~ NormalMeanPrecision(x, τ) where {
            dependencies = dependencies,
            meta = damping,
        }
    end
end

@initialization function normal_ngmp_initialization(
    initial_mean,
    initial_variance,
    a0,
    b0,
)
    q(x) = NormalMeanVariance(initial_mean, initial_variance)
    q(τ) = GammaShapeRate(a0, b0)
    μ(x) = NormalMeanVariance(initial_mean, 10 * initial_variance)
    μ(τ) = GammaShapeRate(a0, b0)
end

function exact_posterior(observations; m0, v0, a0, b0, grid_size)
    n = length(observations)
    sum_y = sum(observations)
    sum_y2 = sum(abs2, observations)
    precisions = exp.(range(log(1e-5), log(50.0); length = grid_size))
    conditional_precision = @. inv(v0) + n * precisions
    conditional_information = @. m0 / v0 + precisions * sum_y
    log_density = @. (
        (a0 - 1) * log(precisions) - b0 * precisions +
        (n / 2) * log(precisions) - 0.5 * precisions * sum_y2 +
        conditional_information^2 / (2 * conditional_precision) -
        0.5 * log(conditional_precision)
    )
    log_density .-= maximum(log_density)
    density = exp.(log_density)
    spacing = vcat(
        precisions[2] - precisions[1],
        (precisions[3:end] .- precisions[1:end-2]) ./ 2,
        precisions[end] - precisions[end-1],
    )
    weights = density .* spacing
    weights ./= sum(weights)
    conditional_mean = conditional_information ./ conditional_precision
    conditional_variance = inv.(conditional_precision)
    mean_x = sum(weights .* conditional_mean)
    variance_x = sum(weights .* (conditional_variance .+ conditional_mean .^ 2)) - mean_x^2
    mean_precision = sum(weights .* precisions)
    variance_precision = sum(weights .* precisions .^ 2) - mean_precision^2
    return (;
        mean_x,
        variance_x,
        mean_precision,
        variance_precision,
        precisions,
        density = weights ./ spacing,
        weights,
        conditional_mean,
        conditional_variance,
    )
end

function stable_initial_variance(observations)
    length(observations) == 1 && return 1.0
    return max(var(observations), 0.1)
end

function fit_vmp(observations, parameters; iterations)
    initial_mean = mean(observations)
    initial_variance = stable_initial_variance(observations)
    elapsed = @elapsed result = infer(
        model = normal_vmp_model(; parameters...),
        data = (y = observations,),
        constraints = MeanField(),
        initialization = normal_vmp_initialization(initial_mean, initial_variance),
        iterations = iterations,
    )
    return (
        qx = last(result.posteriors[:x]),
        qτ = last(result.posteriors[:τ]),
        elapsed,
    )
end

function fit_ngmp(observations, parameters; iterations)
    initial_mean = mean(observations)
    initial_variance = stable_initial_variance(observations)
    dependencies = NGMPDependencies(
        μ = nothing,
        τ = nothing,
        projection = TangentProjection(type = Unscented),
    )
    elapsed = @elapsed result = infer(
        model = normal_ngmp_model(
            ; parameters...,
            dependencies,
            damping = DampingMeta(alpha = 0.2, beta = 0.0),
        ),
        data = (y = observations,),
        initialization = normal_ngmp_initialization(
            initial_mean,
            initial_variance,
            parameters.a0,
            parameters.b0,
        ),
        iterations = iterations,
        options = (limit_stack_depth = 500,),
    )
    return (
        qx = last(result.posteriors[:x]),
        qτ = last(result.posteriors[:τ]),
        elapsed,
    )
end

function exact_x_density(exact, grid)
    return [
        sum(exact.weights .* pdf.(
            Normal.(exact.conditional_mean, sqrt.(exact.conditional_variance)),
            point,
        ))
        for point in grid
    ]
end

function grid_spacing(grid)
    return vcat(
        grid[2] - grid[1],
        (grid[3:end] .- grid[1:end-2]) ./ 2,
        grid[end] - grid[end - 1],
    )
end

function marginal_kls(exact, qx, qτ; x_grid_size)
    x_grid = range(
        exact.mean_x - 9sqrt(exact.variance_x),
        exact.mean_x + 9sqrt(exact.variance_x);
        length = x_grid_size,
    )
    exact_x = exact_x_density(exact, x_grid)
    x_spacing = grid_spacing(x_grid)
    x_weights = exact_x .* x_spacing
    x_weights ./= sum(x_weights)
    qx_weights = pdf.(Ref(qx), x_grid) .* x_spacing
    qx_weights ./= sum(qx_weights)
    positive_x = x_weights .> 0
    kl_x = sum(x_weights[positive_x] .* (
        log.(x_weights[positive_x]) .-
        log.(max.(qx_weights[positive_x], floatmin(Float64)))
    ))

    precision_spacing = grid_spacing(exact.precisions)
    qτ_weights = pdf.(Ref(qτ), exact.precisions) .* precision_spacing
    qτ_weights ./= sum(qτ_weights)
    positive_precision = exact.weights .> 0
    kl_precision = sum(exact.weights[positive_precision] .* (
        log.(exact.weights[positive_precision]) .-
        log.(max.(qτ_weights[positive_precision], floatmin(Float64)))
    ))
    return (; kl_x = max(kl_x, 0.0), kl_precision = max(kl_precision, 0.0))
end

function result_row(repetition, n, method, qx, qτ, exact, elapsed, kls,
                    true_mean, true_precision)
    return (;
        repetition,
        n,
        method,
        true_mean,
        true_precision,
        mean_x = mean(qx),
        variance_x = var(qx),
        mean_precision = mean(qτ),
        variance_precision = var(qτ),
        kl_exact_to_method_x = kls.kl_x,
        kl_exact_to_method_precision = kls.kl_precision,
        elapsed_seconds = elapsed,
    )
end

function run_study(config)
    rows = NamedTuple[]
    parameters = (
        m0 = config.prior_mean,
        v0 = config.prior_variance,
        a0 = config.prior_shape,
        b0 = config.prior_rate,
    )
    maximum_n = maximum(config.sample_sizes)

    for repetition in 1:config.repetitions
        rng = StableRNG(config.seed + repetition - 1)
        true_mean = rand(rng, Normal(config.prior_mean, sqrt(config.prior_variance)))
        true_precision = rand(
            rng,
            Gamma(config.prior_shape, inv(config.prior_rate)),
        )
        stream = rand(
            rng,
            Normal(true_mean, inv(sqrt(true_precision))),
            maximum_n,
        )
        for n in config.sample_sizes
            observations = stream[1:n]
            exact = exact_posterior(
                observations;
                parameters...,
                grid_size = config.grid_size,
            )
            vmp = fit_vmp(observations, parameters; iterations = config.vmp_iterations)
            ngmp = fit_ngmp(observations, parameters; iterations = config.ngmp_iterations)

            exact_qx = NormalMeanVariance(exact.mean_x, exact.variance_x)
            exact_qτ = GammaShapeRate(
                exact.mean_precision^2 / exact.variance_precision,
                exact.mean_precision / exact.variance_precision,
            )
            zero_kl = (; kl_x = 0.0, kl_precision = 0.0)
            vmp_kls = marginal_kls(
                exact,
                vmp.qx,
                vmp.qτ;
                x_grid_size = config.kl_grid_size,
            )
            ngmp_kls = marginal_kls(
                exact,
                ngmp.qx,
                ngmp.qτ;
                x_grid_size = config.kl_grid_size,
            )
            push!(rows, result_row(repetition, n, "Exact moments", exact_qx,
                exact_qτ, exact, 0.0, zero_kl, true_mean, true_precision))
            push!(rows, result_row(repetition, n, "VMP", vmp.qx, vmp.qτ,
                exact, vmp.elapsed, vmp_kls, true_mean, true_precision))
            push!(rows, result_row(repetition, n, "NGMP", ngmp.qx, ngmp.qτ,
                exact, ngmp.elapsed, ngmp_kls, true_mean, true_precision))

        end
    end
    return DataFrame(rows)
end

function summarize_runs(runs)
    rows = NamedTuple[]
    for n in sort(unique(runs.n)), method in ("VMP", "NGMP")
        selected = filter(row -> row.n == n && row.method == method, runs)
        push!(rows, (;
            n,
            method,
            n_seeds = nrow(selected),
            mean_kl_x = mean(selected.kl_exact_to_method_x),
            kl_x_ci95 = ci95(selected.kl_exact_to_method_x),
            mean_kl_precision = mean(selected.kl_exact_to_method_precision),
            kl_precision_ci95 = ci95(selected.kl_exact_to_method_precision),
        ))
    end
    return DataFrame(rows)
end

function summary_lines(config)
    return [
        "# Joint Normal mean–precision study",
        "",
        "Each of the $(config.repetitions) seeds draws a new true mean and precision from the inference priors, then reuses prefixes of one observation stream across sample sizes.",
        "The two figures report mean marginal KL ± 95% confidence intervals across seeds $(config.seed)–$(config.seed + config.repetitions - 1).",
        "KL is oriented as `KL(p_exact || q_method)` for the shared mean/state and precision marginals separately.",
        "The machine-readable method-wise values are stored in `normal_mean_precision_aggregate.csv`.",
    ]
end

function kl_performance_panel(summary, value_field, ci_field)
    panel = plot(
        xlabel = "number of observations N",
        ylabel = "mean KL(exact ∥ method)",
        title = "KL",
        xscale = :log2,
        yscale = :log10,
        xticks = let ticks = sort(unique(summary.n)); (ticks, string.(ticks)) end,
        legend = :best,
    )
    for (method, color, linestyle) in (
        ("VMP", COLORS.vmp, :dash),
        ("NGMP", COLORS.ngmp, :solid),
    )
        selected = sort(filter(row -> row.method == method, summary), :n)
        center = selected[!, value_field]
        interval = selected[!, ci_field]
        lower_ribbon = min.(interval, 0.999 .* center)
        plot!(
            panel,
            selected.n,
            center;
            ribbon = (lower_ribbon, interval),
            color,
            linestyle,
            linewidth = 2.3,
            marker = :circle,
            markersize = 4,
            fillalpha = 0.12,
            label = "$method mean ± 95% CI",
        )
    end
    return panel
end

function main()
    ensure_outputs()
    smoke = smoke_mode()
    config = (
        smoke,
        seed = 42,
        repetitions = smoke ? 2 : 20,
        sample_sizes = smoke ? [4, 8, 16] : [4, 8, 16, 32, 64, 128],
        grid_size = smoke ? 600 : 4000,
        kl_grid_size = smoke ? 240 : 800,
        vmp_iterations = smoke ? 5 : 15,
        ngmp_iterations = smoke ? 12 : 50,
        prior_mean = 0.0,
        prior_variance = 25.0,
        prior_shape = 2.0,
        prior_rate = 1.0,
        ngmp_projection = "Unscented",
    )
    runs = run_study(config)
    summary = summarize_runs(runs)
    CSV.write(joinpath(RESULT_DIR, "normal_mean_precision_runs.csv"), runs)
    CSV.write(joinpath(RESULT_DIR, "normal_mean_precision_aggregate.csv"), summary)
    write_config("normal_mean_precision", config)
    write_summary("normal_mean_precision", summary_lines(config))
    save_figure(
        kl_performance_panel(summary, :mean_kl_x, :kl_x_ci95),
        "normal_kl_state",
    )
    save_figure(
        kl_performance_panel(summary, :mean_kl_precision, :kl_precision_ci95),
        "normal_kl_precision",
    )
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
