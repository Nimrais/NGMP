# One figure, three arms, both benchmarks, fit and contraction: 12 panels.
#
#     columns  GP reference | 2-level deep kernel | 3-level deep kernel
#     row 1    cubic gap    — fit
#     row 2    cubic gap    — contraction
#     row 3    sine         — fit
#     row 4    sine         — contraction
#
# The two deep-kernel columns default to `dk3_L2.jls` and `dk3_L3.jls`, which differ
# ONLY in the number of levels -- same mean features, same separate noise features,
# same seeds. So the 2 -> 3 column comparison isolates what the third level buys,
# rather than confounding it with the feature-map change.
#
# (`dk_fixed.jls` is a third variant -- two levels but with the noise head SHARING
# the mean head's features. It is the one whose extrapolated aleatoric was
# asymmetric by 2300x. Pass it via PLOT_ARMS to include it.)
#
# Axes are shared per row: the comparison is about relative band width and relative
# variance, which independent axes would hide.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 julia --project=. experiments/plot_arm_comparison.jl
#
# Env: PLOT_ARMS  semicolon-separated `label=path` list (order = column order)
#      PLOT_OUTPUT

using Printf
using Serialization
using Statistics

using Plots
using RxInfer
using SurrogateModelling

include(joinpath(@__DIR__, "uq_benchmarks.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "..", "results", "uncertainty_diagnosis")

# Entries are separated by ";" (not ","), because arm labels may contain commas.
default_arms = join([
    "GP reference=$(joinpath(RESULTS_DIR, "parametric_gp_reference.jls"))",
    "deep kernel 2-level=$(joinpath(RESULTS_DIR, "dk3_L2.jls"))",
    "deep kernel 3-level=$(joinpath(RESULTS_DIR, "dk3_L3.jls"))",
], ";")

const ARM_SPECS = map(split(get(ENV, "PLOT_ARMS", default_arms), ";")) do entry
    label, path = split(entry, "="; limit = 2)
    isfile(path) || error("missing artifact for '$label': $path")
    (; label = String(strip(label)), path = String(strip(path)))
end

const COLOURS = [:seagreen, :darkorange, :firebrick, :purple, :steelblue]
const OUTPUT_PATH = get(ENV, "PLOT_OUTPUT",
    joinpath(RESULTS_DIR, "three_arm_comparison.png"))

artifacts = map(spec -> (; spec.label, artifact = deserialize(spec.path)), ARM_SPECS)

"""Masks are not stored by every arm, so rebuild them from the shared definitions."""
function masks_for(grid, name)
    return name == "cubic gap" ?
        (observed = (abs.(grid) .>= 3.0) .& (abs.(grid) .<= 5.0),
         gap = abs.(grid) .<= 2.5, outer = abs.(grid) .>= 5.5) :
        (observed = abs.(grid) .<= 1.5,
         gap = falses(length(grid)), outer = abs.(grid) .>= 3.5)
end

"""
    view_of(entry, benchmark_name, colour)

Normalise the artifact shapes to one view. The GP reference keeps its
noise-treatment variants in an `arms` Dict and has a single global precision, so its
aleatoric curve is constant; the deep-kernel arms store an input-dependent
`E[1/lambda]` curve at the top level.
"""
function view_of(entry, benchmark_name::AbstractString, colour)
    results = entry.artifact.results
    index = findfirst(r -> r.benchmark_name == benchmark_name, results)
    isnothing(index) && error("$(entry.label) has no result for $benchmark_name")
    result = results[index]

    if hasproperty(result, :arms)
        arm = result.arms[:learned_noise]
        return (;
            label = entry.label, colour,
            result.grid, result.x_train, result.y_train,
            result.true_mean, result.true_variance,
            predicted_mean = arm.predicted_mean,
            function_variance = arm.function_variance,
            total_variance = arm.total_variance,
            aleatoric_variance = fill(
                abs2(result.y_scale) / arm.noise_precision, length(result.grid),
            ),
            masks = masks_for(result.grid, benchmark_name),
        )
    end
    return (;
        label = entry.label, colour,
        result.grid, result.x_train, result.y_train,
        result.true_mean, result.true_variance,
        result.predicted_mean, result.function_variance,
        result.total_variance, result.aleatoric_variance,
        masks = masks_for(result.grid, benchmark_name),
    )
end

metrics_of(view) = contraction_metrics(
    (; view.true_mean, view.true_variance, view.masks),
    view.predicted_mean, view.function_variance, view.total_variance,
)

function fit_panel(view, ylims, shade, show_legend)
    interval = 1.96 .* sqrt.(max.(view.total_variance, 0.0))
    panel = plot(
        view.grid, view.predicted_mean;
        ribbon = interval, fillalpha = 0.22, color = view.colour, linewidth = 2,
        label = "mean ± 1.96 SD", xlabel = "x", ylabel = "y",
        title = view.label, titlefontsize = 9,
        legend = show_legend ? :topleft : false, legendfontsize = 6, ylims = ylims,
    )
    isnothing(shade) || vspan!(panel, [shade[1], shade[2]];
        color = :gray80, alpha = 0.35, label = "no training data")
    plot!(panel, view.grid, view.true_mean;
        color = :black, linewidth = 2, label = "truth")
    scatter!(panel, view.x_train, view.y_train;
        color = :gray45, markersize = 1.8, markeralpha = 0.45,
        markerstrokewidth = 0, label = "observations")
    metrics = metrics_of(view)
    annotate!(panel,
        maximum(view.grid) - 0.03 * (maximum(view.grid) - minimum(view.grid)),
        ylims[1] + 0.09 * (ylims[2] - ylims[1]),
        text(@sprintf("RMSE %.3f", metrics.observed_rmse), :right, 7, :gray25))
    return panel
end

function contraction_panel(view, ylims, shade, show_legend)
    metrics = metrics_of(view)
    ratio_text = isnan(metrics.gap_over_observed) ?
        @sprintf("out/obs %.0f", metrics.outer_over_observed) :
        @sprintf("gap/obs %.0f", metrics.gap_over_observed)

    panel = plot(
        view.grid, view.function_variance;
        color = :royalblue, linewidth = 2, label = "Var(q(f)) epistemic",
        xlabel = "x", ylabel = "variance", yscale = :log10, ylims = ylims,
        title = "$(view.label): contraction", titlefontsize = 9,
        legend = show_legend ? :topleft : false, legendfontsize = 6,
    )
    isnothing(shade) || vspan!(panel, [shade[1], shade[2]];
        color = :gray80, alpha = 0.35, label = "no training data")
    plot!(panel, view.grid, view.total_variance;
        color = view.colour, linewidth = 2, label = "Var(q(y*)) total")
    plot!(panel, view.grid, max.(view.aleatoric_variance, 1e-8);
        color = :purple, linestyle = :dash, linewidth = 2, label = "learned aleatoric")
    plot!(panel, view.grid, max.(view.true_variance, 1e-8);
        color = :black, linestyle = :dashdot, linewidth = 2, label = "true aleatoric")
    annotate!(panel,
        maximum(view.grid) - 0.03 * (maximum(view.grid) - minimum(view.grid)),
        10 ^ (log10(ylims[1]) + 0.11 * (log10(ylims[2]) - log10(ylims[1]))),
        text(@sprintf("Var obs %.3g\n%s", metrics.observed_function_variance, ratio_text),
             :right, 7, :gray25))
    return panel
end

rows = [
    (name = "cubic gap", shade = (-3.0, 3.0)),
    (name = "heteroscedastic sine", shade = nothing),
]

panels = []
for row in rows
    views = [view_of(entry, row.name, COLOURS[i]) for (i, entry) in enumerate(artifacts)]

    lower = minimum(vcat(
        [v.predicted_mean .- 1.96 .* sqrt.(max.(v.total_variance, 0.0)) for v in views]...,
        first(views).true_mean, first(views).y_train,
    ))
    upper = maximum(vcat(
        [v.predicted_mean .+ 1.96 .* sqrt.(max.(v.total_variance, 0.0)) for v in views]...,
        first(views).true_mean, first(views).y_train,
    ))
    padding = 0.05 * (upper - lower)
    fit_limits = (lower - padding, upper + padding)

    positive(values) = filter(v -> v > 0 && isfinite(v), values)
    all_variances = positive(vcat(
        [v.function_variance for v in views]..., [v.total_variance for v in views]...,
        [v.aleatoric_variance for v in views]..., first(views).true_variance,
    ))
    variance_limits = (minimum(all_variances) / 3, maximum(all_variances) * 3)

    for (i, view) in enumerate(views)
        push!(panels, fit_panel(view, fit_limits, row.shade, i == 1))
    end
    for (i, view) in enumerate(views)
        push!(panels, contraction_panel(view, variance_limits, row.shade, i == 1))
    end
end

columns = length(artifacts)
plot(
    panels...;
    layout = (4, columns), size = (430 * columns, 1450),
    plot_title = "Same data, shared axes per row",
    plot_titlefontsize = 11,
    left_margin = 5Plots.mm, bottom_margin = 3Plots.mm,
)
savefig(OUTPUT_PATH)
@info "saved" OUTPUT_PATH

println("\nNumbers behind the panels")
@printf("%-22s %-24s %9s %10s %10s %10s %12s %10s\n",
    "benchmark", "arm", "obs RMSE", "Var obs", "gap/obs", "out/obs",
    "aleatoric", "truth")
for row in rows
    for (i, entry) in enumerate(artifacts)
        view = view_of(entry, row.name, COLOURS[i])
        metrics = metrics_of(view)
        @printf("%-22s %-24s %9.3f %10.4g %10s %10.1f %12.4f %10.4f\n",
            row.name, view.label, metrics.observed_rmse,
            metrics.observed_function_variance,
            isnan(metrics.gap_over_observed) ? "-" :
                @sprintf("%.0f", metrics.gap_over_observed),
            metrics.outer_over_observed,
            mean(view.aleatoric_variance[view.masks.observed]),
            mean(view.true_variance[view.masks.observed]))
    end
end
