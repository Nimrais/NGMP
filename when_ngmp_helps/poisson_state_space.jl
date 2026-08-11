#!/usr/bin/env julia

include(joinpath(@__DIR__, "common.jl"))
using .WhenNGMPHelpsCommon

using CSV
using DataFrames
using Distributions
using ExponentialFamily
using ExponentialFamilyProjection
using Plots
using Printf
using Random
using RxInfer
using StableRNGs
using Statistics
using SurrogateModelling

import ExponentialFamilyProjection: BoundedNormUpdateRule

# A removed Poisson leaf contributes an uninformative message. These two
# propagation rules let that neutral message pass through the Gaussian chain.
@rule NormalMeanVariance(:μ, Marginalisation) (
    m_out::Uninformative,
    q_v::PointMass,
) = Uninformative()
@rule NormalMeanVariance(:out, Marginalisation) (
    m_μ::Uninformative,
    q_v::PointMass,
) = Uninformative()
@rule NormalMeanVariance(:μ, Marginalisation) (
    m_out::Uninformative,
    m_v::PointMass,
) = Uninformative()
@rule NormalMeanVariance(:out, Marginalisation) (
    m_μ::Uninformative,
    m_v::PointMass,
) = Uninformative()
@marginalrule NormalMeanVariance(:out_μ) (
    m_out::Uninformative,
    m_μ::UnivariateNormalDistributionsFamily,
    q_v::Any,
) = begin
    ξ_μ, W_μ = weightedmean_precision(m_μ)
    W = inv(mean(q_v))
    return MvNormalWeightedMeanPrecision([zero(ξ_μ); ξ_μ], [W -W; -W W_μ + W])
end
@marginalrule NormalMeanVariance(:out_μ) (
    m_out::UnivariateNormalDistributionsFamily,
    m_μ::Uninformative,
    q_v::Any,
) = begin
    ξ_out, W_out = weightedmean_precision(m_out)
    W = inv(mean(q_v))
    return MvNormalWeightedMeanPrecision([ξ_out; zero(ξ_out)], [W_out + W -W; -W W])
end

@model function poisson_vmp_model(
    y,
    n_states,
    observed,
    process_variance,
    initial_mean,
    initial_variance,
)
    z[1] ~ NormalMeanVariance(initial_mean, initial_variance)
    for index in 2:n_states
        z[index] ~ NormalMeanVariance(z[index - 1], process_variance)
    end
    position = 1
    for index in 1:n_states
        if observed[index]
            y[position] ~ PoissonExp(z[index])
            position += 1
        else
            z[index] ~ Uninformative()
        end
    end
end

# The default projection budget (100 iterations x stepsize 0.1 x gradient
# norm bound 1.0 = at most ~10 natural-parameter units per call) silently
# TRAPS marginals that drift far into the tail: the projection returns its
# own input, the trap is a fixed point of the damped map as well, and rare
# masks blow up catastrophically. Widening the norm bound to 100 frees the
# trap and makes the projected-VMP baseline fair (and faster: the inner
# optimizer converges instead of stalling).
@constraints function poisson_vmp_constraints()
    q(z) = MeanField()
    q(z) :: ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(),
            direction = BoundedNormUpdateRule(100.0),
        ),
    )
end

@initialization function poisson_vmp_initialization()
    q(z) = NormalMeanVariance(0.0, 100.0)
end

@model function poisson_ngmp_model(
    y,
    n_states,
    observed,
    process_variance,
    initial_mean,
    initial_variance,
    dependencies,
    damping,
)
    z[1] ~ NormalMeanVariance(initial_mean, initial_variance)
    for index in 2:n_states
        z[index] ~ NormalMeanVariance(z[index - 1], process_variance)
    end
    position = 1
    for index in 1:n_states
        if observed[index]
            y[position] ~ PoissonExp(z[index]) where {
                dependencies = dependencies,
                meta = damping,
            }
            position += 1
        else
            z[index] ~ Uninformative()
        end
    end
end

@initialization function poisson_ngmp_initialization(initial_means)
    q(z) = NormalMeanVariance.(initial_means, 1.0)
end

function fit_projected_vmp(counts, observed_indices, config; free_energy = true)
    observed_counts = counts[observed_indices]
    observed = falses(length(counts))
    observed[observed_indices] .= true
    elapsed = @elapsed result = infer(
        model = poisson_vmp_model(
            n_states = length(counts),
            observed = observed,
            process_variance = config.process_variance,
            initial_mean = config.initial_mean,
            initial_variance = config.initial_variance,
        ),
        constraints = poisson_vmp_constraints(),
        data = (y = observed_counts,),
        initialization = poisson_vmp_initialization(),
        iterations = config.iterations,
        free_energy = free_energy,
        options = (limit_stack_depth = 500,),
    )
    posterior = last(result.posteriors[:z])
    return (
        mean = mean.(posterior),
        variance = var.(posterior),
        free_energy = free_energy ? Float64.(result.free_energy) : Float64[],
        elapsed,
    )
end

function fit_ngmp(counts, observed_indices, config; free_energy = true)
    dependencies = NGMPDependencies(in = nothing)
    observed_counts = counts[observed_indices]
    observed = falses(length(counts))
    observed[observed_indices] .= true
    initial_means = fill(log(2.0), length(counts))
    initial_means[observed_indices] .= log.(observed_counts .+ 1.0)
    elapsed = @elapsed result = infer(
        model = poisson_ngmp_model(
            n_states = length(counts),
            observed = observed,
            process_variance = config.process_variance,
            initial_mean = config.initial_mean,
            initial_variance = config.initial_variance,
            dependencies = dependencies,
            damping = DampingMeta(
                alpha = config.damping_alpha,
                beta = config.damping_beta,
            ),
        ),
        data = (y = observed_counts,),
        initialization = poisson_ngmp_initialization(initial_means),
        iterations = config.iterations,
        free_energy = free_energy,
        options = (limit_stack_depth = 500,),
    )
    posterior = last(result.posteriors[:z])
    return (
        mean = mean.(posterior),
        variance = var.(posterior),
        free_energy = free_energy ? Float64.(result.free_energy) : Float64[],
        elapsed,
    )
end

function synthetic_counts(config)
    rng = StableRNG(config.seed)
    n = config.smoke_length
    time = collect(1:n)
    latent = @. 1.4 + 0.9 * sin(2π * time / 48) + 0.25 * sin(2π * time / 17)
    counts = rand.(Ref(rng), Poisson.(exp.(latent)))
    return Float64.(time), Int.(counts), "synthetic smoke series"
end

function load_counts(config)
    config.smoke && return synthetic_counts(config)
    dataset = Sunspots()
    years = Float64.(dataset.features[!, :year])
    counts = round.(Int, dataset.targets.average)
    return years, counts, "Sunspots"
end

function holdout_indices(n, fraction, seed)
    count = max(1, round(Int, fraction * n))
    rng = StableRNG(seed)
    return sort!(randperm(rng, n)[1:count])
end

function predictive_metrics(means, variances, counts, held_out)
    rates = [
        exp(clamp(means[index] + variances[index] / 2, -20.0, 20.0))
        for index in held_out
    ]
    nll = [
        -logpdf(Poisson(rate), counts[index])
        for (index, rate) in zip(held_out, rates)
    ]
    squared_error = [
        abs2(rate - counts[index])
        for (index, rate) in zip(held_out, rates)
    ]
    return (; rates, nll, squared_error)
end

function append_posterior_rows!(
    rows,
    repetition,
    fraction,
    method,
    fit,
    years,
    counts,
    held_out,
)
    held_out_set = Set(held_out)
    for index in eachindex(counts)
        push!(rows, (;
            repetition,
            holdout_fraction = fraction,
            holdout_pct = round(Int, 100fraction),
            method,
            index,
            time = years[index],
            count = counts[index],
            held_out = index in held_out_set,
            posterior_mean = fit.mean[index],
            posterior_variance = fit.variance[index],
            posterior_sd = sqrt(fit.variance[index]),
            elapsed_seconds = fit.elapsed,
        ))
    end
    return rows
end

function largest_prediction_improvement_window(
    counts,
    held_out,
    vmp,
    ngmp,
    requested_length,
)
    n = length(counts)
    window_length = min(requested_length, n)
    window_length == n && return 1:n
    vmp_rates = exp.(clamp.(vmp.mean .+ vmp.variance ./ 2, -20.0, 20.0))
    ngmp_rates = exp.(clamp.(ngmp.mean .+ ngmp.variance ./ 2, -20.0, 20.0))
    improvement = zeros(n)
    improvement[held_out] .= (
        abs2.(vmp_rates[held_out] .- counts[held_out]) .-
        abs2.(ngmp_rates[held_out] .- counts[held_out])
    )
    cumulative = vcat(0.0, cumsum(improvement))
    scores = [
        cumulative[start + window_length] - cumulative[start]
        for start in 1:(n - window_length + 1)
    ]
    start = argmax(scores)
    return start:(start + window_length - 1)
end

function trajectory_figure(
    years,
    counts,
    held_out,
    vmp,
    ngmp,
    fraction,
    window_length,
)
    indices = largest_prediction_improvement_window(
        counts,
        held_out,
        vmp,
        ngmp,
        window_length,
    )
    held_out_set = Set(held_out)
    held_out_in_window = filter(index -> index in held_out_set, indices)
    ngmp_sd = sqrt.(ngmp.variance[indices])
    ngmp_rate = exp.(clamp.(
        ngmp.mean[indices] .+ ngmp.variance[indices] ./ 2,
        -20.0,
        20.0,
    ))
    ngmp_lower = exp.(clamp.(ngmp.mean[indices] .- 1.96ngmp_sd, -20.0, 20.0))
    ngmp_upper = exp.(clamp.(ngmp.mean[indices] .+ 1.96ngmp_sd, -20.0, 20.0))
    vmp_sd = sqrt.(vmp.variance[indices])
    vmp_rate = exp.(clamp.(
        vmp.mean[indices] .+ vmp.variance[indices] ./ 2,
        -20.0,
        20.0,
    ))
    vmp_lower = exp.(clamp.(vmp.mean[indices] .- 1.96vmp_sd, -20.0, 20.0))
    vmp_upper = exp.(clamp.(vmp.mean[indices] .+ 1.96vmp_sd, -20.0, 20.0))
    transform_rate(values) = log10.(values .+ 1)
    count_ticks = [0, 1, 3, 10, 30, 100, 300]
    ngmp_center_plot = transform_rate(ngmp_rate)
    vmp_center_plot = transform_rate(vmp_rate)
    panel = plot(
        years[indices],
        ngmp_center_plot;
        ribbon = (
            ngmp_center_plot .- transform_rate(ngmp_lower),
            transform_rate(ngmp_upper) .- ngmp_center_plot,
        ),
        color = COLORS.ngmp,
        fillalpha = 0.15,
        linewidth = 2.4,
        label = "NGMP predictive rate ± latent 95% CI",
        xlabel = "time",
        ylabel = "sunspot count / predictive rate",
        legend = :topright,
        legendfontsize = 8,
        yticks = (transform_rate(count_ticks), string.(count_ticks)),
        ylims = (-0.05, log10(401.0)),
        left_margin = 5Plots.mm,
        bottom_margin = 5Plots.mm,
    )
    plot!(
        panel,
        years[indices],
        vmp_center_plot;
        ribbon = (
            vmp_center_plot .- transform_rate(vmp_lower),
            transform_rate(vmp_upper) .- vmp_center_plot,
        ),
        color = COLORS.vmp,
        fillalpha = 0.12,
        linewidth = 2.4,
        linestyle = :dash,
        label = "Projected VMP rate ± latent 95% CI",
    )
    scatter!(
        panel,
        years[held_out_in_window],
        transform_rate(counts[held_out_in_window]);
        markersize = 6.0,
        marker = :diamond,
        markerstrokewidth = 1.2,
        alpha = 0.95,
        color = :firebrick,
        label = "held-out count",
    )
    return plot(panel; size = (1160, 540))
end

# One method per panel around the unluckiest held-out month: overlaying the
# two chains hides the difference, and separate single-hue panels stay
# readable for color-blind readers.
function outlier_panel(
    years,
    counts,
    held_out,
    fit,
    outlier_index,
    half_width,
    method_label,
    color,
    linestyle;
    display_stride = 1,
    mark_center = true,
)
    n = length(counts)
    lo = max(1, outlier_index - half_width)
    hi = min(n, outlier_index + half_width)
    # the model stays monthly; the stride only thins the plotted line/band
    # vertices (e.g. 3 = quarterly display). Held-out markers keep their
    # exact months, and the outlier month is always included.
    indices = sort!(unique!(vcat(collect(lo:display_stride:hi), [outlier_index])))
    # The Sunspots year column is integer-valued (all twelve months of a year
    # share one x value), which draws the chain as vertical stacks. Rebuild a
    # fractional monthly axis when duplicates are present.
    axis = length(years) > 1 && years[2] == years[1] ?
        years .+ ((0:(n - 1)) .% 12) ./ 12 : years
    month_label(index) = string(
        floor(Int, years[index]), ".", ((index - 1) % 12) + 1,
    )
    tick_indices = collect(lo:6:hi)
    sd = sqrt.(fit.variance[indices])
    center = exp.(clamp.(
        fit.mean[indices] .+ fit.variance[indices] ./ 2,
        -20.0,
        20.0,
    ))
    lower = exp.(clamp.(fit.mean[indices] .- 1.96 .* sd, -20.0, 20.0))
    upper = exp.(clamp.(fit.mean[indices] .+ 1.96 .* sd, -20.0, 20.0))
    transform_rate(values) = log10.(values .+ 1)
    count_ticks = [0, 10, 30, 100, 300, 1000, 10000]
    held_out_set = Set(held_out)
    held_out_in_window = filter(index -> index in held_out_set, collect(lo:hi))
    center_plot = transform_rate(center)
    panel = plot(
        axis[indices],
        center_plot;
        ribbon = (
            center_plot .- transform_rate(lower),
            transform_rate(upper) .- center_plot,
        ),
        color,
        linestyle,
        fillalpha = 0.18,
        linewidth = 2.4,
        label = "$(method_label) rate ± latent 95% CI",
        xlabel = "time (year.month)",
        ylabel = "sunspot count / predictive rate",
        legend = :topleft,
        legendfontsize = 8,
        xticks = (axis[tick_indices], month_label.(tick_indices)),
        xrotation = 25,
        yticks = (transform_rate(count_ticks), string.(count_ticks)),
        ylims = (-0.05, log10(3.0e4)),
        left_margin = 5Plots.mm,
        bottom_margin = 7Plots.mm,
    )
    scatter!(
        panel,
        axis[held_out_in_window],
        transform_rate(counts[held_out_in_window]);
        markersize = 6.0,
        marker = :diamond,
        markerstrokewidth = 1.4,
        markerstrokecolor = :black,
        color = :firebrick,
        label = "held-out count",
    )
    if mark_center
        # ring drawn as an explicit line loop: marker-stroke-only circles are
        # dropped by the GR backend when the fill is fully transparent
        angles = range(0, 2π; length = 61)
        ring_center = transform_rate([Float64(counts[outlier_index])])[1]
        plot!(
            panel,
            axis[outlier_index] .+ 0.16 .* cos.(angles),
            ring_center .+ 0.16 .* sin.(angles);
            color = :black,
            linewidth = 2.2,
            label = "unlucky month (see text)",
        )
    end
    return panel
end

# depth of each index inside its contiguous held-out run (0 for observed):
# distance to the nearest observed month.
function _held_out_depth(held_out, n)
    held = falses(n)
    held[held_out] .= true
    depth = zeros(Int, n)
    run_start = 0
    for i in 1:n
        if held[i]
            run_start == 0 && (run_start = i)
        elseif run_start != 0
            for j in run_start:(i - 1)
                depth[j] = min(j - run_start + 1, i - j)
            end
            run_start = 0
        end
    end
    if run_start != 0
        for j in run_start:n
            depth[j] = j - run_start + 1
        end
    end
    return depth
end

# Two single-method panels around the deepest held-out stretch of the
# representative 50% mask: this is where mean-field VMP's fixed-variance
# tilted messages pin the held-out uncertainty while NGMP's preserved chain
# spreads it like a smoother.
function gap_figure_artifacts(years, counts, representative, config)
    held_out = representative.held_out
    depth = _held_out_depth(held_out, length(counts))
    center = argmax(depth)
    vmp_panel = outlier_panel(
        years, counts, held_out, representative.vmp, center,
        config.outlier_half_width, "Projected VMP", COLORS.vmp, :dash;
        mark_center = false,
    )
    ngmp_panel = outlier_panel(
        years, counts, held_out, representative.ngmp, center,
        config.outlier_half_width, "NGMP", COLORS.ngmp, :solid;
        mark_center = false,
    )
    save_pdf(vmp_panel, "poisson_gap50_vmp")
    save_pdf(ngmp_panel, "poisson_gap50_ngmp")
    save_figure(
        plot(
            plot(vmp_panel; title = "Projected VMP"),
            plot(ngmp_panel; title = "NGMP");
            layout = (1, 2),
            size = (1220, 440),
        ),
        "poisson_gap50",
    )
end

const _DEPTH_BUCKETS = (
    (1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 8, "5–8"),
)

_depth_bucket(depth) = findfirst(
    bucket -> bucket[1] <= depth <= bucket[2], _DEPTH_BUCKETS,
)

# Held-out NLL by gap depth (all masks) and posterior variance by gap depth
# (repetition 1 chains) at the deepest holdout fraction.
function depth_profile_artifacts(holdout_frame, posterior_frame)
    fraction_pct = 50
    labels = [bucket[3] for bucket in _DEPTH_BUCKETS]
    n = maximum(posterior_frame.index)

    nll_sums = Dict{Tuple{String, Int}, Vector{Float64}}()
    held50 = filter(row -> row.holdout_pct == fraction_pct, holdout_frame)
    isempty(held50) && return
    for sub in groupby(held50, :repetition)
        depth = _held_out_depth(sort(unique(sub.index)), n)
        for row in eachrow(sub)
            bucket = _depth_bucket(depth[row.index])
            bucket === nothing && continue
            push!(get!(nll_sums, (row.method, bucket), Float64[]), row.nll)
        end
    end

    nll_panel = plot(;
        xlabel = "months to nearest observation",
        ylabel = "held-out NLL",
        yscale = :log10,
        xticks = (1:length(labels), labels),
        legend = :topleft,
        left_margin = 5Plots.mm,
    )
    variance_panel = plot(;
        xlabel = "months to nearest observation",
        ylabel = "posterior variance of held-out z",
        xticks = (1:length(labels), labels),
        legend = :topleft,
        left_margin = 5Plots.mm,
    )
    rep1 = filter(
        row -> row.holdout_pct == fraction_pct &&
            row.repetition == 1 && row.held_out,
        posterior_frame,
    )
    depth1 = _held_out_depth(sort(unique(rep1.index)), n)
    for (method, label, color, linestyle) in (
        ("Projected VMP (mean-field)", "Projected VMP", COLORS.vmp, :dash),
        ("NGMP", "NGMP", COLORS.ngmp, :solid),
    )
        nll = [
            mean(get(nll_sums, (method, bucket), [NaN]))
            for bucket in eachindex(labels)
        ]
        plot!(
            nll_panel, eachindex(labels), nll;
            color, linestyle, linewidth = 2.4, marker = :circle,
            markersize = 5, label,
        )
        rows = filter(row -> row.method == method, rep1)
        variances = [
            begin
                bucket_values = [
                    row.posterior_variance for row in eachrow(rows)
                    if _depth_bucket(depth1[row.index]) == bucket
                ]
                isempty(bucket_values) ? NaN : mean(bucket_values)
            end
            for bucket in eachindex(labels)
        ]
        plot!(
            variance_panel, eachindex(labels), variances;
            color, linestyle, linewidth = 2.4, marker = :circle,
            markersize = 5, label,
        )
    end
    save_pdf(nll_panel, "poisson_depth_nll")
    save_pdf(variance_panel, "poisson_depth_variance")
    save_figure(
        plot(
            plot(nll_panel; title = "held-out NLL by gap depth"),
            plot(variance_panel; title = "held-out marginal variance");
            layout = (1, 2),
            size = (1100, 420),
        ),
        "poisson_depth_profile",
    )
end

# Refit both methods on the 5% mask whose worst held-out point dominates the
# VMP mask average, and save the side-by-side single-method panels.
function outlier_figure_artifacts(years, counts, holdout_frame, config)
    fraction = config.outlier_fraction
    vmp_rows = filter(
        row -> row.holdout_fraction == fraction &&
            row.method == "Projected VMP (mean-field)",
        holdout_frame,
    )
    isempty(vmp_rows) && return
    worst = vmp_rows[argmax(vmp_rows.nll), :]
    held_out = holdout_indices(length(counts), fraction, worst.mask_seed)
    observed = trues(length(counts))
    observed[held_out] .= false
    observed_indices = findall(observed)
    vmp = fit_projected_vmp(counts, observed_indices, config; free_energy = false)
    ngmp = fit_ngmp(counts, observed_indices, config; free_energy = false)
    vmp_panel = outlier_panel(
        years, counts, held_out, vmp, worst.index,
        config.outlier_half_width, "Projected VMP", COLORS.vmp, :dash;
        display_stride = config.outlier_display_stride,
    )
    ngmp_panel = outlier_panel(
        years, counts, held_out, ngmp, worst.index,
        config.outlier_half_width, "NGMP", COLORS.ngmp, :solid;
        display_stride = config.outlier_display_stride,
    )
    save_pdf(vmp_panel, "poisson_outlier_vmp")
    save_pdf(ngmp_panel, "poisson_outlier_ngmp")
    save_figure(
        plot(
            plot(vmp_panel; title = "Projected VMP"),
            plot(ngmp_panel; title = "NGMP");
            layout = (1, 2),
            size = (1220, 440),
        ),
        "poisson_outlier_5pct",
    )
end

function free_energy_panel(frame, fraction)
    summarize = function (method)
        selected = filter(
            row -> row.holdout_fraction == fraction && row.method == method,
            frame,
        )
        return sort(combine(
            groupby(selected, :iteration),
            :per_observed_count => mean => :center,
            :per_observed_count => ci95 => :ci95,
        ), :iteration)
    end
    vmp = summarize("Projected VMP (mean-field)")
    ngmp = summarize("NGMP surrogate")
    all(vmp.center .> 0) && all(ngmp.center .> 0) ||
        error("log-scale Bethe free-energy plot requires positive values")
    panel = plot(
        vmp.iteration,
        vmp.center;
        ribbon = min.(vmp.ci95, vmp.center .* 0.999),
        color = COLORS.vmp,
        fillalpha = 0.12,
        linewidth = 2.2,
        linestyle = :dash,
        marker = :circle,
        markersize = 3,
        label = "Projected VMP (mean-field)",
        xlabel = "iteration",
        ylabel = "Bethe free energy / observed count",
        yscale = :log10,
    )
    plot!(
        panel,
        ngmp.iteration,
        ngmp.center;
        ribbon = min.(ngmp.ci95, ngmp.center .* 0.999),
        color = COLORS.ngmp,
        fillalpha = 0.12,
        linewidth = 2.2,
        marker = :circle,
        markersize = 3,
        label = "NGMP surrogate",
    )
    return panel
end

function free_energy_figure(panels)
    return plot(
        panels...;
        layout = (2, 2),
        size = (1060, 760),
    )
end

function summarize_seed_metrics(metrics)
    rows = NamedTuple[]
    for repetition in sort(unique(metrics.repetition))
        for fraction in sort(unique(metrics.holdout_fraction))
            for method in (
                "Projected VMP (mean-field)", "Projected VMP, 1 sweep", "NGMP",
            )
                selected = filter(
                    row -> row.repetition == repetition &&
                           row.holdout_fraction == fraction &&
                           row.method == method,
                    metrics,
                )
                push!(rows, (;
                    repetition,
                    holdout_fraction = fraction,
                    holdout_pct = round(Int, 100fraction),
                    method,
                    n_held_out = nrow(selected),
                    mean_nll = mean(selected.nll),
                    rmse = sqrt(mean(selected.squared_error)),
                ))
            end
        end
    end
    return DataFrame(rows)
end

function summarize_metrics(seed_metrics)
    rows = NamedTuple[]
    for fraction in sort(unique(seed_metrics.holdout_fraction))
        for method in (
            "Projected VMP (mean-field)", "Projected VMP, 1 sweep", "NGMP",
        )
            selected = filter(
                row -> row.holdout_fraction == fraction && row.method == method,
                seed_metrics,
            )
            push!(rows, (;
                holdout_fraction = fraction,
                holdout_pct = round(Int, 100fraction),
                method,
                n_seeds = nrow(selected),
                mean_nll = mean(selected.mean_nll),
                nll_ci95 = ci95(selected.mean_nll),
                median_nll = median(selected.mean_nll),
                mean_rmse = mean(selected.rmse),
                rmse_ci95 = ci95(selected.rmse),
            ))
        end
    end
    return DataFrame(rows)
end

function wide_metrics(summary)
    rows = NamedTuple[]
    for fraction in sort(unique(summary.holdout_fraction))
        vmp = filter(
            row -> row.holdout_fraction == fraction &&
                   row.method == "Projected VMP (mean-field)",
            summary,
        )[1, :]
        vmp1 = filter(
            row -> row.holdout_fraction == fraction &&
                   row.method == "Projected VMP, 1 sweep",
            summary,
        )[1, :]
        ngmp = filter(
            row -> row.holdout_fraction == fraction && row.method == "NGMP",
            summary,
        )[1, :]
        push!(rows, (;
            holdout_pct = round(Int, 100fraction),
            vmp_nll = vmp.mean_nll,
            vmp_nll_ci95 = vmp.nll_ci95,
            vmp1_nll = vmp1.mean_nll,
            vmp1_nll_ci95 = vmp1.nll_ci95,
            ngmp_nll = ngmp.mean_nll,
            ngmp_nll_ci95 = ngmp.nll_ci95,
            vmp_rmse = vmp.mean_rmse,
            vmp_rmse_ci95 = vmp.rmse_ci95,
            vmp1_rmse = vmp1.mean_rmse,
            vmp1_rmse_ci95 = vmp1.rmse_ci95,
            ngmp_rmse = ngmp.mean_rmse,
            ngmp_rmse_ci95 = ngmp.rmse_ci95,
        ))
    end
    return DataFrame(rows)
end

function write_latex_metrics_table(table)
    path = joinpath(RESULT_DIR, "poisson_state_space_metrics_table.tex")
    open(path, "w") do io
        println(io, raw"\begin{tabular}{rcccccc}")
        println(io, raw"\toprule")
        println(io, " & \\multicolumn{3}{c}{NLL} & \\multicolumn{3}{c}{RMSE} \\\\")
        println(io, raw"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
        println(io, "Held out & VMP & VMP, 1 sweep & NGMP & VMP & VMP, 1 sweep & NGMP \\\\")
        println(io, raw"\midrule")
        for row in eachrow(table)
            println(io, @sprintf(
                "%d\\%% & %.3f \$\\pm\$ %.3f & %.3f \$\\pm\$ %.3f & %.3f \$\\pm\$ %.3f & %.3f \$\\pm\$ %.3f & %.3f \$\\pm\$ %.3f & %.3f \$\\pm\$ %.3f \\\\",
                row.holdout_pct,
                row.vmp_nll,
                row.vmp_nll_ci95,
                row.vmp1_nll,
                row.vmp1_nll_ci95,
                row.ngmp_nll,
                row.ngmp_nll_ci95,
                row.vmp_rmse,
                row.vmp_rmse_ci95,
                row.vmp1_rmse,
                row.vmp1_rmse_ci95,
                row.ngmp_rmse,
                row.ngmp_rmse_ci95,
            ))
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
    end
    return path
end

function caption_lines(summary, table, data_source, figure_fraction)
    n_seeds = minimum(summary.n_seeds)
    comparisons = String[]
    for fraction in sort(unique(summary.holdout_fraction))
        vmp = filter(
            row -> row.holdout_fraction == fraction &&
                   row.method == "Projected VMP (mean-field)",
            summary,
        )[1, :]
        ngmp = filter(
            row -> row.holdout_fraction == fraction && row.method == "NGMP",
            summary,
        )[1, :]
        push!(comparisons, @sprintf(
            "%d%%: NLL %.3f/%.3f and RMSE %.3f/%.3f",
            round(Int, 100fraction),
            vmp.mean_nll,
            ngmp.mean_nll,
            vmp.mean_rmse,
            ngmp.mean_rmse,
        ))
    end
    caption_metrics = join(comparisons, "; ")
    lines = [
        "# Poisson state-space study",
        "",
        "Data source: $data_source",
        "The trajectory figure shows $(round(Int, 100figure_fraction))% held out.",
        "The trajectory is plotted in count space; bands transform the marginal 95% credible intervals for `zₖ | y_observed` through the exponential link.",
        "NLL uses the notebook plug-in predictive rate `exp(m + v/2)`; RMSE is computed from that rate and the held-out count within each mask.",
        "NLL and RMSE are shown as means ± 95% confidence intervals across $n_seeds mask seeds.",
        "",
        "| Held out | Projected VMP NLL | VMP 1-sweep NLL | NGMP NLL | Projected VMP RMSE | VMP 1-sweep RMSE | NGMP RMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in eachrow(table)
        push!(lines, @sprintf(
            "| %d%% | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f |",
            row.holdout_pct,
            row.vmp_nll,
            row.vmp_nll_ci95,
            row.vmp1_nll,
            row.vmp1_nll_ci95,
            row.ngmp_nll,
            row.ngmp_nll_ci95,
            row.vmp_rmse,
            row.vmp_rmse_ci95,
            row.vmp1_rmse,
            row.vmp1_rmse_ci95,
            row.ngmp_rmse,
            row.ngmp_rmse_ci95,
        ))
    end
    append!(lines, [
        "",
        "**Suggested trajectory caption.** Held-out count prediction for projected mean-field VMP and NGMP after removing $(round(Int, 100figure_fraction))% of observations. The displayed window maximizes cumulative held-out squared-error improvement of NGMP over projected VMP for the illustrative first mask; the aggregate table uses all windows and all seeds. Lines are predictive rates `exp(E[zₖ] + Var(zₖ)/2)`, shading transforms pointwise marginal 95% credible intervals for `zₖ` through the exponential link, and red diamonds are held-out counts not used for inference. Across $n_seeds nested-mask seeds, projected VMP/NGMP respectively give $caption_metrics.",
        "",
        "**Bethe free-energy caption.** Mean per-observed-count Bethe free-energy diagnostics with 95% intervals across $n_seeds masks after removing nested 5%, 10%, 20%, and 50% subsets of likelihood factors. Each projected-VMP trace evaluates its holdout graph's variational objective; each NGMP trace is a surrogate Bethe diagnostic because its local Gaussian surrogates change between outer iterations.",
    ])
    return lines
end

function run_mask_repetition(counts, years, config, repetition)
    posterior_rows = NamedTuple[]
    holdout_rows = NamedTuple[]
    free_energy_rows = NamedTuple[]
    representative = nothing
    mask_seed = config.holdout_seed + repetition - 1

    for fraction in config.holdout_fractions
        held_out = holdout_indices(length(counts), fraction, mask_seed)
        observed = trues(length(counts))
        observed[held_out] .= false
        observed_indices = findall(observed)
        vmp = fit_projected_vmp(counts, observed_indices, config)
        # budget-matched ablation: a single sweep roughly matches NGMP's
        # wall-clock (the standard arm spends its time on 20 sweeps of
        # per-edge manifold projections); metrics table only, no figures
        vmp1 = fit_projected_vmp(
            counts, observed_indices, merge(config, (iterations = 1,));
            free_energy = false,
        )
        ngmp = fit_ngmp(counts, observed_indices, config)

        if repetition == 1
            append_posterior_rows!(posterior_rows, repetition, fraction,
                "Projected VMP (mean-field)", vmp, years, counts, held_out)
            append_posterior_rows!(posterior_rows, repetition, fraction,
                "NGMP", ngmp, years, counts, held_out)
        end

        for (method, fit) in (
            ("Projected VMP (mean-field)", vmp),
            ("Projected VMP, 1 sweep", vmp1),
            ("NGMP", ngmp),
        )
            scores = predictive_metrics(fit.mean, fit.variance, counts, held_out)
            for (position, index) in enumerate(held_out)
                push!(holdout_rows, (;
                    repetition,
                    mask_seed,
                    holdout_fraction = fraction,
                    holdout_pct = round(Int, 100fraction),
                    method,
                    index,
                    time = years[index],
                    count = counts[index],
                    predictive_rate = scores.rates[position],
                    nll = scores.nll[position],
                    squared_error = scores.squared_error[position],
                ))
            end
        end

        for (method, values) in (
            ("Projected VMP (mean-field)", vmp.free_energy),
            ("NGMP surrogate", ngmp.free_energy),
        )
            for (iteration, value) in enumerate(values)
                push!(free_energy_rows, (;
                    repetition,
                    mask_seed,
                    holdout_fraction = fraction,
                    holdout_pct = round(Int, 100fraction),
                    n_observed = length(observed_indices),
                    method,
                    iteration,
                    bethe_free_energy = value,
                    per_observed_count = value / length(observed_indices),
                ))
            end
        end

        if repetition == 1 && fraction == config.figure_holdout_fraction
            representative = (; fraction, held_out, vmp, ngmp)
        end
    end
    @printf("completed Poisson mask seed %d/%d\n", repetition, config.repetitions)
    return (; posterior_rows, holdout_rows, free_energy_rows, representative)
end

function main()
    ensure_outputs()
    smoke = smoke_mode()
    config = (
        smoke,
        seed = 42,
        smoke_length = 144,
        process_variance = 0.1,
        initial_mean = 0.0,
        initial_variance = 10.0,
        iterations = smoke ? 4 : 20,
        damping_alpha = 0.5,
        damping_beta = 0.2,
        repetitions = smoke ? 2 : parse(
            Int,
            get(ENV, "WHEN_NGMP_POISSON_REPETITIONS", "20"),
        ),
        holdout_fractions = [0.05, 0.10, 0.20, 0.50],
        figure_holdout_fraction = 0.50,
        trajectory_window_length = smoke ? 144 : 300,
        outlier_fraction = 0.05,
        outlier_half_width = smoke ? 24 : 30,
        outlier_display_stride = 1,
        holdout_seed = 42,
    )
    years, counts, data_source = load_counts(config)

    posterior_rows = NamedTuple[]
    holdout_rows = NamedTuple[]
    free_energy_rows = NamedTuple[]
    representative = nothing
    outputs = Vector{Any}(undef, config.repetitions)
    outputs[1] = run_mask_repetition(counts, years, config, 1)
    Threads.@threads for repetition in 2:config.repetitions
        outputs[repetition] = run_mask_repetition(
            counts,
            years,
            config,
            repetition,
        )
    end
    for output in outputs
        append!(posterior_rows, output.posterior_rows)
        append!(holdout_rows, output.holdout_rows)
        append!(free_energy_rows, output.free_energy_rows)
        isnothing(output.representative) || (representative = output.representative)
    end
    isnothing(representative) && error("figure holdout fraction was not evaluated")

    posterior_frame = DataFrame(posterior_rows)
    holdout_frame = DataFrame(holdout_rows)
    seed_metrics = summarize_seed_metrics(holdout_frame)
    metrics_summary = summarize_metrics(seed_metrics)
    metrics_table = wide_metrics(metrics_summary)
    free_energy_frame = DataFrame(free_energy_rows)

    CSV.write(joinpath(RESULT_DIR, "poisson_state_space_runs.csv"), posterior_frame)
    CSV.write(joinpath(RESULT_DIR, "poisson_state_space_holdout.csv"), holdout_frame)
    CSV.write(joinpath(RESULT_DIR, "poisson_state_space_seed_metrics.csv"), seed_metrics)
    CSV.write(joinpath(RESULT_DIR, "poisson_state_space_metrics.csv"), metrics_summary)
    CSV.write(joinpath(RESULT_DIR, "poisson_state_space_metrics_table.csv"), metrics_table)
    CSV.write(joinpath(RESULT_DIR, "poisson_state_space_free_energy.csv"), free_energy_frame)
    write_latex_metrics_table(metrics_table)
    write_config("poisson_state_space", merge(config, (;
        data_source,
        n_observations = length(counts),
    )))
    write_summary("poisson_state_space",
        caption_lines(
            metrics_summary,
            metrics_table,
            data_source,
            config.figure_holdout_fraction,
        ))
    save_figure(
        trajectory_figure(
            years,
            counts,
            representative.held_out,
            representative.vmp,
            representative.ngmp,
            representative.fraction,
            config.trajectory_window_length,
        ),
        "poisson_state_space",
    )
    outlier_figure_artifacts(years, counts, holdout_frame, config)
    gap_figure_artifacts(years, counts, representative, config)
    depth_profile_artifacts(holdout_frame, posterior_frame)
    free_energy_panels = [
        free_energy_panel(free_energy_frame, fraction)
        for fraction in sort(unique(free_energy_frame.holdout_fraction))
    ]
    for (fraction, panel) in zip(
        sort(unique(free_energy_frame.holdout_fraction)),
        free_energy_panels,
    )
        save_pdf(panel, "poisson_bethe_heldout_$(round(Int, 100fraction))")
    end
    save_figure(
        free_energy_figure(free_energy_panels),
        "poisson_state_space_free_energy",
    )
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
