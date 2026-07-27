ENV["GKSwstype"] = "100"

using CSV
using DataFrames
using Plots
using Printf

const DEFAULT_RESULTS_DIR = joinpath(@__DIR__, "results", "cifar100_20260721_115154")
const RESULTS_DIR = abspath(get(ENV, "RESULTS_DIR", DEFAULT_RESULTS_DIR))
const SUMMARY_PATH = joinpath(RESULTS_DIR, "summary.tsv")

function configuration_name(model, optimizer, layers)
    method = get(
        Dict(
            "vector_transport" => "vector transport",
            "vector_transport_nesterov" => "vector-transport Nesterov",
            "projected_nesterov" => "projected Nesterov",
        ),
        optimizer,
        optimizer,
    )
    if model == "rxinfer_1h"
        return "RxInfer 1-layer ($method)"
    elseif model == "rxinfer_2h"
        return "RxInfer 2-layer ($method)"
    end
    return "Neural $(layers)-layer"
end

function load_history(log_path)
    history = NamedTuple{(:epoch, :train_acc, :val_acc), Tuple{Int, Float64, Float64}}[]
    pattern = r"epoch=(\d+).*?train_acc=([0-9.]+).*?val_acc=([0-9.]+)"
    for line in eachline(log_path)
        matched = match(pattern, line)
        isnothing(matched) && continue
        push!(history, (
            epoch=parse(Int, matched.captures[1]),
            train_acc=parse(Float64, matched.captures[2]),
            val_acc=parse(Float64, matched.captures[3]),
        ))
    end
    isempty(history) && error("No epoch metrics found in $log_path")
    return history
end

percent(value) = @sprintf("%.1f%%", 100value)
runtime(value) = value < 60 ? @sprintf("%.0f s", value) :
    value < 3600 ? @sprintf("%.0f min", value / 60) : @sprintf("%.2f h", value / 3600)

function markdown_table(rows)
    lines = [
        "| Model | Best val. | Epoch | Test | Runtime |",
        "|---|---:|---:|---:|---:|",
    ]
    best_test = maximum(rows.test_acc)
    for row in eachrow(rows)
        test = percent(row.test_acc)
        isapprox(row.test_acc, best_test) && (test = "**$test**")
        push!(lines, "| $(row.name) | $(percent(row.best_val_acc)) | " *
            "$(row.best_val_epoch) | $test | $(runtime(row.elapsed_seconds)) |")
    end
    return join(lines, "\n")
end

function ordered(rows)
    rx = sort(filter(:model => !=("neural"), rows), [:model, :optimizer])
    neural = sort(filter(:model => ==("neural"), rows), :hidden_layers)
    return vcat(rx, neural)
end

function accuracy_axis(rows, metric)
    maximum_accuracy = maximum(
        maximum(getproperty.(row.history, metric)) for row in eachrow(rows)
    )
    tick_step = maximum_accuracy <= 0.3 ? 0.025 :
        maximum_accuracy <= 0.6 ? 0.05 : 0.1
    upper_limit = min(1.0, max(0.25, ceil(maximum_accuracy * 1.05 / tick_step) * tick_step))
    return (0.0, upper_limit), 0.0:tick_step:upper_limit
end

function plot_history(rows, metric, hidden_count, ntrain, destination)
    label = metric == :train_acc ? "Training" : "Validation"
    ylimits, yticks = accuracy_axis(rows, metric)
    plt = plot(
        title="CIFAR-100 $label Accuracy — $hidden_count Hidden Units, $(ntrain) Train",
        xlabel="Epoch", ylabel="Accuracy", ylims=ylimits, yticks=yticks,
        xlims=(1, 50), legend=:outerright, grid=true, size=(1400, 700), dpi=170,
        left_margin=5Plots.mm, bottom_margin=5Plots.mm,
    )
    palette = [:dodgerblue3, :deepskyblue3, :darkorange, :goldenrod3,
        :deeppink3, :mediumpurple3, :slateblue3, :teal, :sienna3, :olivedrab3,
        :gray35]
    for (index, row) in enumerate(eachrow(ordered(rows)))
        history = row.history
        marker = row.model == "rxinfer_1h" ? :xcross :
            row.model == "rxinfer_2h" ? :diamond : :none
        plot!(plt, getproperty.(history, :epoch), getproperty.(history, metric);
            label=row.name, color=palette[index], linewidth=2.2,
            marker=marker, markersize=row.model == "neural" ? 0 : 3)
    end
    savefig(plt, destination)
end

function plot_test_heatmaps(summary)
    configurations = unique(ordered(filter(:hidden_count => ==(minimum(summary.hidden_count)), summary)).name)
    panels = Plots.Plot[]
    for ntrain in sort(unique(summary.ntrain))
        rows = filter(:ntrain => ==(ntrain), summary)
        values = [only(filter(row -> row.name == name && row.hidden_count == hidden,
                    eachrow(rows))).test_acc
            for name in configurations, hidden in sort(unique(rows.hidden_count))]
        hidden_counts = sort(unique(rows.hidden_count))
        push!(panels, heatmap(
            eachindex(hidden_counts), eachindex(configurations), 100values;
            title="$(ntrain) Training Images", xlabel="Hidden units",
            color=:viridis, colorbar_title="Test %", clim=(0, 20),
            xticks=(eachindex(hidden_counts), string.(hidden_counts)),
            yticks=(eachindex(configurations), configurations),
            xlims=(0.5, length(hidden_counts) + 0.5),
            ylims=(0.5, length(configurations) + 0.5),
            annotations=vec([(column, row, text(@sprintf("%.1f", values[row, column] * 100), 8,
                values[row, column] >= 0.11 ? :white : :black))
                for row in axes(values, 1), column in axes(values, 2)]),
            xrotation=0,
        ))
    end
    combined = plot(panels...; layout=(1, length(panels)), size=(1500, 720), dpi=170,
        plot_title="CIFAR-100 Test Accuracy by Configuration and Hidden Width",
        left_margin=8Plots.mm, bottom_margin=6Plots.mm)
    savefig(combined, joinpath(RESULTS_DIR, "cifar100_test_accuracy_heatmaps.png"))
end

summary = CSV.read(SUMMARY_PATH, DataFrame; delim='\t')
summary.name = [configuration_name(row.model, row.optimizer, row.hidden_layers)
    for row in eachrow(summary)]
summary.history = [load_history(row.log) for row in eachrow(summary)]

Plots.gr()
plot_test_heatmaps(summary)

hidden_links = String[]
for hidden_count in sort(unique(summary.hidden_count))
    report_name = "cifar100_hidden_$(hidden_count)_results.md"
    push!(hidden_links, "- [$hidden_count hidden units]($report_name)")
    sections = String[]
    for ntrain in sort(unique(summary.ntrain))
        rows = ordered(filter(row -> row.hidden_count == hidden_count && row.ntrain == ntrain, summary))
        prefix = "cifar100_hidden_$(hidden_count)_train_$(ntrain)"
        plot_history(rows, :train_acc, hidden_count, ntrain,
            joinpath(RESULTS_DIR, "$(prefix)_training_accuracy.png"))
        plot_history(rows, :val_acc, hidden_count, ntrain,
            joinpath(RESULTS_DIR, "$(prefix)_validation_accuracy.png"))
        push!(sections, """## $(ntrain) Training Images

$(markdown_table(rows))

![Training accuracy, $hidden_count hidden units, $ntrain images]($(prefix)_training_accuracy.png)

![Validation accuracy, $hidden_count hidden units, $ntrain images]($(prefix)_validation_accuracy.png)
""")
    end
    report = """# CIFAR-100 Results — $hidden_count Hidden Units

All runs use 50 epochs and batch size 32. The test score comes from the checkpoint
with the highest validation accuracy. RxInfer two-layer models use $hidden_count
units in both hidden layers.

$(join(sections, "\n"))
"""
    write(joinpath(RESULTS_DIR, report_name), report)
end

best_rows = combine(groupby(summary, [:ntrain, :hidden_count])) do group
    best = group[argmax(group.test_acc), :]
    (; model=best.name, best_val_acc=best.best_val_acc,
        test_acc=best.test_acc, elapsed_seconds=best.elapsed_seconds)
end

overview_lines = [
    "| Training images | Hidden units | Best model | Best val. | Test | Runtime |",
    "|---:|---:|---|---:|---:|---:|",
]
for row in eachrow(sort(best_rows, [:ntrain, :hidden_count]))
    push!(overview_lines, "| $(row.ntrain) | $(row.hidden_count) | $(row.model) | " *
        "$(percent(row.best_val_acc)) | $(percent(row.test_acc)) | $(runtime(row.elapsed_seconds)) |")
end

overview = """# CIFAR-100 Experiment Summary

This report covers all $(nrow(summary)) completed runs in `$(basename(RESULTS_DIR))`.
Every experiment used 50 epochs, batch size 32, and 16×16 RGB inputs. Test accuracy
is measured from the checkpoint selected by validation accuracy.

## Best Result per Hidden Width

$(join(overview_lines, "\n"))

## All Configurations

![CIFAR-100 test accuracy heatmaps](cifar100_test_accuracy_heatmaps.png)

## Reports by Hidden-Unit Count

$(join(hidden_links, "\n"))

Results are single runs. The 100-image test split used by the 1,000-training-image
experiments changes by one percentage point per correctly classified image.
"""

write(joinpath(RESULTS_DIR, "cifar100_results.md"), overview)
println("Wrote CIFAR-100 Markdown reports and plots to $RESULTS_DIR")
