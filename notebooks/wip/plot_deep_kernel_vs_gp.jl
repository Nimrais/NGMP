# Eight panels: the verified GP reference against the deep-kernel heteroscedastic
# arm, fit and contraction, on both benchmarks.
#
#     row 1   cubic gap     — fit          GP | deep kernel
#     row 2   cubic gap     — contraction  GP | deep kernel
#     row 3   sine          — fit          GP | deep kernel
#     row 4   sine          — contraction  GP | deep kernel
#
# Arms are the COLUMNS so the comparison is a straight left-right read, and every
# row shares its y-limits: the whole point is relative band width and relative
# variance, which independent axes would hide.
#
# The contraction rows carry the substance. Watch two things:
#   * `Var(q(f))` (blue) dipping over the observed wings and rising in the gap --
#     epistemic behaviour;
#   * the aleatoric curve (purple) against the true noise (black dash-dot) -- the GP
#     reference's is FLAT by construction, since it has a single global precision,
#     while the deep-kernel arm's noise head is itself a GP on log-precision and can
#     track input-dependent noise. That is the capability a plain GP does not have.
#
# Reads saved artifacts, so the picture cannot drift from the reported numbers.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 julia --project=. experiments/plot_deep_kernel_vs_gp.jl
#
# Env: PLOT_GP, PLOT_DK (artifact paths), PLOT_OUTPUT

using Printf
using Serialization
using Statistics

using Plots
using RxInfer
using SurrogateModelling

include(joinpath(@__DIR__, "uq_benchmarks.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "..", "results", "uncertainty_diagnosis")
const GP_PATH = get(ENV, "PLOT_GP", joinpath(RESULTS_DIR, "parametric_gp_reference.jls"))
const DK_PATH = get(ENV, "PLOT_DK", "/tmp/dk_fixed.jls")
const OUTPUT_PATH = get(ENV, "PLOT_OUTPUT",
    joinpath(RESULTS_DIR, "deep_kernel_vs_gp_reference.png"))

for path in (GP_PATH, DK_PATH)
    isfile(path) || error("missing artifact $path")
end

gp_artifact = deserialize(GP_PATH)
dk_artifact = deserialize(DK_PATH)

find_result(results, name) = results[findfirst(r -> r.benchmark_name == name, results)]

"""
    view_of(source, benchmark_name)

Normalise the two artifact shapes to one view. The GP reference stores its
noise-treatment variants in a `Dict` and has a single global precision, so its
aleatoric curve is a constant vector; the deep-kernel arm stores an input-dependent
`E[1/lambda]` curve directly.
"""
function view_of(source::Symbol, benchmark_name::AbstractString)
    if source === :gp
        result = find_result(gp_artifact.results, benchmark_name)
        arm = result.arms[:learned_noise]
        return (;
            label = "GP reference (learned noise)",
            colour = :seagreen,
            result.grid, result.x_train, result.y_train,
            result.true_mean, result.true_variance,
            predicted_mean = arm.predicted_mean,
            function_variance = arm.function_variance,
            total_variance = arm.total_variance,
            aleatoric_variance = fill(
                abs2(result.y_scale) / arm.noise_precision, length(result.grid),
            ),
            masks = result.masks,
        )
    end
    result = find_result(dk_artifact.results, benchmark_name)
    return (;
        label = "deep kernel + noise GP",
        colour = :darkorange,
        result.grid, result.x_train, result.y_train,
        result.true_mean, result.true_variance,
        result.predicted_mean, result.function_variance,
        result.total_variance, result.aleatoric_variance,
        masks = cubic_or_sine_masks(result, benchmark_name),
    )
end

"""Masks are not stored by the deep-kernel arm; rebuild them from the shared defs."""
function cubic_or_sine_masks(result, name)
    grid = result.grid
    return name == "cubic gap" ?
        (observed = (abs.(grid) .>= 3.0) .& (abs.(grid) .<= 5.0),
         gap = abs.(grid) .<= 2.5, outer = abs.(grid) .>= 5.5) :
        (observed = abs.(grid) .<= 1.5,
         gap = falses(length(grid)), outer = abs.(grid) .>= 1.8)
end

function fit_panel(view, ylims, shade, show_legend)
    interval = 1.96 .* sqrt.(max.(view.total_variance, 0.0))
    panel = plot(
        view.grid, view.predicted_mean;
        ribbon = interval, fillalpha = 0.22, color = view.colour, linewidth = 2,
        label = "mean ± 1.96 SD", xlabel = "x", ylabel = "y",
        title = view.label, titlefontsize = 9,
        legend = show_legend ? :topleft : false, legendfontsize = 6, ylims = ylims,
    )
    isnothing(shade) || vspan!(
        panel, [shade[1], shade[2]];
        color = :gray80, alpha = 0.35, label = "no training data",
    )
    plot!(panel, view.grid, view.true_mean;
        color = :black, linewidth = 2, label = "truth")
    scatter!(panel, view.x_train, view.y_train;
        color = :gray45, markersize = 2, markeralpha = 0.45,
        markerstrokewidth = 0, label = "observations")
    return panel
end

function contraction_panel(view, ylims, shade, show_legend)
    contraction = contraction_metrics(
        (; view.true_mean, view.true_variance, view.masks),
        view.predicted_mean, view.function_variance, view.total_variance,
    )
    ratio_text = isnan(contraction.gap_over_observed) ?
        @sprintf("out/obs %.1f", contraction.outer_over_observed) :
        @sprintf("gap/obs %.0f", contraction.gap_over_observed)

    panel = plot(
        view.grid, view.function_variance;
        color = :royalblue, linewidth = 2, label = "Var(q(f))  epistemic",
        xlabel = "x", ylabel = "variance", yscale = :log10, ylims = ylims,
        title = "$(view.label): contraction", titlefontsize = 9,
        legend = show_legend ? :topleft : false, legendfontsize = 6,
    )
    isnothing(shade) || vspan!(
        panel, [shade[1], shade[2]];
        color = :gray80, alpha = 0.35, label = "no training data",
    )
    plot!(panel, view.grid, view.total_variance;
        color = view.colour, linewidth = 2, label = "Var(q(y*))  total")
    plot!(panel, view.grid, max.(view.aleatoric_variance, 1e-8);
        color = :purple, linestyle = :dash, linewidth = 2, label = "learned aleatoric")
    plot!(panel, view.grid, max.(view.true_variance, 1e-8);
        color = :black, linestyle = :dashdot, linewidth = 2, label = "true aleatoric")
    annotate!(
        panel,
        maximum(view.grid) - 0.03 * (maximum(view.grid) - minimum(view.grid)),
        10 ^ (log10(ylims[1]) + 0.12 * (log10(ylims[2]) - log10(ylims[1]))),
        text(
            @sprintf("Var obs %.3g\n%s", contraction.observed_function_variance, ratio_text),
            :right, 7, :gray25,
        ),
    )
    return panel
end

rows = [
    (name = "cubic gap", shade = (-3.0, 3.0)),
    (name = "heteroscedastic sine", shade = nothing),
]

panels = []
for (row_index, row) in enumerate(rows)
    gp = view_of(:gp, row.name)
    dk = view_of(:dk, row.name)

    # Shared fit limits, driven by whichever arm needs more room.
    lower = minimum(vcat(
        gp.predicted_mean .- 1.96 .* sqrt.(max.(gp.total_variance, 0.0)),
        dk.predicted_mean .- 1.96 .* sqrt.(max.(dk.total_variance, 0.0)),
        gp.true_mean, gp.y_train,
    ))
    upper = maximum(vcat(
        gp.predicted_mean .+ 1.96 .* sqrt.(max.(gp.total_variance, 0.0)),
        dk.predicted_mean .+ 1.96 .* sqrt.(max.(dk.total_variance, 0.0)),
        gp.true_mean, gp.y_train,
    ))
    padding = 0.05 * (upper - lower)
    fit_limits = (lower - padding, upper + padding)

    # Shared log-scale variance limits across all four curves of both arms.
    positive(values) = filter(v -> v > 0 && isfinite(v), values)
    all_variances = positive(vcat(
        gp.function_variance, gp.total_variance, gp.aleatoric_variance,
        dk.function_variance, dk.total_variance, dk.aleatoric_variance,
        gp.true_variance,
    ))
    variance_limits = (minimum(all_variances) / 3, maximum(all_variances) * 3)

    legend_here = row_index == 1
    push!(panels, fit_panel(gp, fit_limits, row.shade, legend_here))
    push!(panels, fit_panel(dk, fit_limits, row.shade, legend_here))
    push!(panels, contraction_panel(gp, variance_limits, row.shade, legend_here))
    push!(panels, contraction_panel(dk, variance_limits, row.shade, legend_here))
end

# panels currently ordered per benchmark as [gp fit, dk fit, gp contr, dk contr];
# that is already row-major for a (4, 2) grid with arms as columns.
plot(
    panels...;
    layout = (4, 2), size = (1150, 1500),
    plot_title = "GP reference vs deep kernel + noise GP — same data, shared axes per row",
    plot_titlefontsize = 11,
    left_margin = 5Plots.mm, bottom_margin = 3Plots.mm,
)
savefig(OUTPUT_PATH)
@info "saved" OUTPUT_PATH

println("\nNumbers behind the panels")
@printf("%-22s %-30s %9s %10s %10s %12s %10s\n",
    "benchmark", "arm", "obs RMSE", "Var obs", "gap/obs", "aleatoric", "truth")
for row in rows
    for source in (:gp, :dk)
        view = view_of(source, row.name)
        contraction = contraction_metrics(
            (; view.true_mean, view.true_variance, view.masks),
            view.predicted_mean, view.function_variance, view.total_variance,
        )
        @printf("%-22s %-30s %9.3f %10.3f %10s %12.4f %10.4f\n",
            row.name, view.label,
            contraction.observed_rmse, contraction.observed_function_variance,
            isnan(contraction.gap_over_observed) ? "-" :
                @sprintf("%.0f", contraction.gap_over_observed),
            mean(view.aleatoric_variance[view.masks.observed]),
            mean(view.true_variance[view.masks.observed]))
    end
end
