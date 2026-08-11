module WhenNGMPHelpsCommon

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using CSV
using DataFrames
using Plots
using Printf
using Statistics
using TOML

export COLORS, FIGURE_DIR, METHOD_LABELS, METHOD_LABELS_SHORT, RESULT_DIR,
       ci95, empirical_band, ensure_outputs, method_label, read_config,
       render_only, save_figure, save_pdf, smoke_mode, write_config,
       write_summary

const OUTPUT_ROOT = abspath(get(ENV, "WHEN_NGMP_OUTPUT_DIR", @__DIR__))
const FIGURE_DIR = joinpath(OUTPUT_ROOT, "figures")
const RESULT_DIR = joinpath(OUTPUT_ROOT, "results")

const COLORS = (
    exact = :black,
    vmp = :darkorange,
    vmp1 = :peru,
    ngmp = :dodgerblue,
    truth = :gray35,
)

# Display names for the inference arms. CSVs and internal dispatch always
# store the stable keys (vmp, vmp1, ngmp, exact); these labels are applied
# only while rendering figures/tables, so renaming every plot and table is:
# edit this NamedTuple, then `julia when_ngmp_helps/render_all.jl`.
const METHOD_LABELS = (
    vmp = "VMP",              # plain mean-field VMP (exhibit 1)
    pvmp = "VMP",             # ProjectedTo arm (exhibits 2–3; paper calls both VMP)
    pvmp1 = "VMP, 1 sweep",   # budget-matched ablation
    ngmp = "NGMP",
    exact = "Exact moments",
)

# Compact variants for tight table headers; keys absent here fall back to the
# full label.
const METHOD_LABELS_SHORT = (;)

function method_label(key; short = false)
    symbol = Symbol(key)
    short && haskey(METHOD_LABELS_SHORT, symbol) &&
        return getfield(METHOD_LABELS_SHORT, symbol)
    return getfield(METHOD_LABELS, symbol)
end

smoke_mode() = lowercase(get(ENV, "WHEN_NGMP_SMOKE", "false")) in
               ("1", "true", "yes", "on")

# Render stage only: skip inference and redraw figures/tables from the CSVs
# in results/.
render_only() = lowercase(get(ENV, "WHEN_NGMP_RENDER_ONLY", "false")) in
                ("1", "true", "yes", "on") || "--render-only" in ARGS

function ensure_outputs()
    mkpath(FIGURE_DIR)
    mkpath(RESULT_DIR)
    return nothing
end

ci95(values) = length(values) > 1 ? 1.96 * std(values) / sqrt(length(values)) : 0.0

function empirical_band(values; lower = 0.1, upper = 0.9)
    isempty(values) && return (NaN, NaN, NaN)
    return (
        median(values),
        quantile(values, lower),
        quantile(values, upper),
    )
end

function save_figure(figure, stem)
    ensure_outputs()
    pdf_path = joinpath(FIGURE_DIR, "$stem.pdf")
    png_path = joinpath(FIGURE_DIR, "$stem.png")
    savefig(figure, pdf_path)
    savefig(figure, png_path)
    @printf("saved %s and %s\n", pdf_path, png_path)
    return (pdf = pdf_path, png = png_path)
end

function save_pdf(figure, stem)
    ensure_outputs()
    path = joinpath(FIGURE_DIR, "$stem.pdf")
    savefig(figure, path)
    @printf("saved %s\n", path)
    return path
end

function write_config(stem, values)
    ensure_outputs()
    path = joinpath(RESULT_DIR, "$(stem)_config.toml")
    open(path, "w") do io
        TOML.print(io, Dict(string(key) => value for (key, value) in pairs(values)))
    end
    return path
end

read_config(stem) = TOML.parsefile(joinpath(RESULT_DIR, "$(stem)_config.toml"))

function write_summary(stem, lines)
    ensure_outputs()
    path = joinpath(RESULT_DIR, "$(stem)_summary.md")
    open(path, "w") do io
        for line in lines
            println(io, line)
        end
    end
    return path
end

end
