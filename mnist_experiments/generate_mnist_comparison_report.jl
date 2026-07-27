ENV["GKSwstype"] = "100"

using CSV
using DataFrames
using Plots
using Printf
using Statistics

function latest_mnist_results_dir()
    results_root = joinpath(@__DIR__, "results")
    candidates = filter(readdir(results_root; join=true)) do path
        isdir(path) &&
            startswith(basename(path), "mnist_comparison_") &&
            isfile(joinpath(path, "summary.tsv"))
    end
    isempty(candidates) &&
        error("No mnist_comparison_* result directory with summary.tsv found")
    return last(sort(candidates))
end

const DEFAULT_RESULTS_DIR = latest_mnist_results_dir()
const RESULTS_DIR = abspath(get(ENV, "RESULTS_DIR", DEFAULT_RESULTS_DIR))
const SUMMARY_PATH = joinpath(RESULTS_DIR, "summary.tsv")
const DATASET = "mnist"

const OPTIMIZER_NAMES = Dict(
    "damped" => "Damped (α=0.5)",
    "vector_transport_05" => "Vector transport (β=0.5)",
    "vector_transport_08" => "Vector transport (β=0.8)",
    "vector_transport_nesterov_05" => "VT-Nesterov (β=0.5)",
    "vector_transport_nesterov_08" => "VT-Nesterov (β=0.8)",
    "projected_nesterov_09" => "Projected Nesterov (β=0.9)",
)

const OPTIMIZER_ORDER = Dict(
    "damped" => 1,
    "vector_transport_05" => 2,
    "vector_transport_08" => 3,
    "vector_transport_nesterov_05" => 4,
    "vector_transport_nesterov_08" => 5,
    "projected_nesterov_09" => 6,
)

const SERIES_COLORS = [
    :dodgerblue3,
    :deepskyblue3,
    :teal,
    :darkorange,
    :goldenrod3,
    :deeppink3,
    :slateblue3,
    :mediumpurple3,
    :purple3,
    :sienna3,
    :olivedrab3,
    :gray35,
    :navy,
    :forestgreen,
    :firebrick3,
    :darkcyan,
    :darkmagenta,
    :chocolate3,
    :black,
]

function configuration_name(model, optimizer, layers)
    if model == "direct_1h"
        return "Direct 1-layer — $(get(OPTIMIZER_NAMES, optimizer, optimizer))"
    elseif model == "direct_2h"
        return "Direct 2-layer — $(get(OPTIMIZER_NAMES, optimizer, optimizer))"
    end
    return "Neural $(layers)-layer (Adam)"
end

function configuration_key(row)
    if row.model == "direct_1h"
        return (1, get(OPTIMIZER_ORDER, row.optimizer, 99), 0)
    elseif row.model == "direct_2h"
        return (2, get(OPTIMIZER_ORDER, row.optimizer, 99), 0)
    end
    return (3, 0, row.hidden_layers)
end

function ordered(rows)
    indices = sortperm(1:nrow(rows); by=index -> configuration_key(rows[index, :]))
    return rows[indices, :]
end

function resolved_log_path(path)
    isfile(path) && return path
    fallback = joinpath(RESULTS_DIR, "logs", basename(path))
    isfile(fallback) && return fallback
    error("Log file not found: $path")
end

function load_history(log_path)
    history = NamedTuple{
        (:epoch, :train_acc, :val_acc),
        Tuple{Int, Float64, Float64},
    }[]
    pattern = r"epoch=(\d+).*?train_acc=([0-9.]+).*?val_acc=([0-9.]+)"
    for line in eachline(resolved_log_path(log_path))
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

percent(value) = @sprintf("%.2f%%", 100value)
runtime(value) = value < 60 ? @sprintf("%.0f s", value) :
    value < 3600 ? @sprintf("%.1f min", value / 60) :
    @sprintf("%.2f h", value / 3600)

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

function plot_history(rows, metric, hidden_count, destination)
    label = metric == :train_acc ? "Training" : "Validation"
    plt = plot(
        title="MNIST $label Accuracy — $hidden_count Hidden Units",
        xlabel="Epoch",
        ylabel="Accuracy",
        ylims=(0.0, 1.0),
        yticks=0.0:0.1:1.0,
        xlims=(1, maximum(maximum(getproperty.(row.history, :epoch))
            for row in eachrow(rows))),
        legend=:outerright,
        grid=true,
        size=(1700, 850),
        dpi=170,
        left_margin=6Plots.mm,
        bottom_margin=5Plots.mm,
        right_margin=5Plots.mm,
    )
    for (index, row) in enumerate(eachrow(ordered(rows)))
        history = row.history
        marker = row.model == "direct_1h" ? :xcross :
            row.model == "direct_2h" ? :diamond : :none
        linestyle = row.model == "neural" ? :dash : :solid
        plot!(
            plt,
            getproperty.(history, :epoch),
            getproperty.(history, metric);
            label=row.name,
            color=SERIES_COLORS[index],
            linewidth=row.model == "neural" ? 1.9 : 2.2,
            linestyle,
            marker,
            markersize=row.model == "neural" ? 0 : 2.7,
        )
    end
    savefig(plt, destination)
end

function plot_test_heatmap(summary)
    hidden_counts = sort(unique(summary.hidden_count))
    reference = ordered(filter(:hidden_count => ==(first(hidden_counts)), summary))
    names = reference.name
    values = [
        only(filter(
            row -> row.name == name && row.hidden_count == hidden,
            eachrow(summary),
        )).test_acc
        for name in names, hidden in hidden_counts
    ]
    minimum_percent = floor(minimum(values) * 100)
    maximum_percent = ceil(maximum(values) * 100)
    plt = heatmap(
        eachindex(hidden_counts),
        eachindex(names),
        100values;
        title="MNIST Test Accuracy by Configuration and Hidden Width",
        xlabel="Hidden units",
        color=:viridis,
        colorbar_title="Test %",
        clim=(minimum_percent, maximum_percent),
        xticks=(eachindex(hidden_counts), string.(hidden_counts)),
        yticks=(eachindex(names), names),
        xlims=(0.5, length(hidden_counts) + 0.5),
        ylims=(0.5, length(names) + 0.5),
        yflip=true,
        annotations=vec([
            (
                column,
                row,
                text(
                    @sprintf("%.2f", values[row, column] * 100),
                    8,
                    values[row, column] <=
                        (minimum(values) + maximum(values)) / 2 ?
                        :white : :black,
                ),
            )
            for row in axes(values, 1), column in axes(values, 2)
        ]),
        size=(1450, 1200),
        dpi=170,
        left_margin=12Plots.mm,
        bottom_margin=6Plots.mm,
        right_margin=5Plots.mm,
    )
    savefig(plt, joinpath(RESULTS_DIR, "mnist_test_accuracy_heatmap.png"))
end

function plot_accuracy_runtime(summary)
    direct = filter(:model => !=("neural"), summary)
    neural = filter(:model => ==("neural"), summary)
    plt = plot(
        title="MNIST Accuracy–Runtime Trade-off",
        xlabel="Runtime (seconds, log scale)",
        ylabel="Test accuracy",
        xscale=:log10,
        ylims=(0.90, 0.99),
        yticks=0.90:0.01:0.99,
        legend=:bottomright,
        grid=true,
        size=(1200, 720),
        dpi=170,
        left_margin=6Plots.mm,
        bottom_margin=6Plots.mm,
    )
    scatter!(
        plt,
        direct.elapsed_seconds,
        direct.test_acc;
        label="Direct Gaussian surrogate",
        marker=:diamond,
        markersize=6,
        markerstrokewidth=0.5,
        color=:dodgerblue3,
    )
    scatter!(
        plt,
        neural.elapsed_seconds,
        neural.test_acc;
        label="Neural baseline (Adam)",
        marker=:circle,
        markersize=6,
        markerstrokewidth=0.5,
        color=:darkorange,
    )
    best_direct = direct[argmax(direct.test_acc), :]
    best_neural = neural[argmax(neural.test_acc), :]
    annotate!(
        plt,
        best_direct.elapsed_seconds,
        best_direct.test_acc,
        text("  best direct: $(percent(best_direct.test_acc))", 9, :left),
    )
    annotate!(
        plt,
        best_neural.elapsed_seconds,
        best_neural.test_acc,
        text("  best neural: $(percent(best_neural.test_acc))", 9, :left),
    )
    savefig(plt, joinpath(RESULTS_DIR, "mnist_accuracy_runtime.png"))
end

summary_all = CSV.read(SUMMARY_PATH, DataFrame; delim='\t')
summary = filter(:dataset => ==(DATASET), summary_all)
isempty(summary) && error("No MNIST rows found in $SUMMARY_PATH")
summary.name = [
    configuration_name(row.model, row.optimizer, row.hidden_layers)
    for row in eachrow(summary)
]
summary.history = [load_history(row.log) for row in eachrow(summary)]

hidden_counts = sort(unique(summary.hidden_count))
reference_names = Set(filter(:hidden_count => ==(first(hidden_counts)), summary).name)
for hidden_count in hidden_counts
    names = Set(filter(:hidden_count => ==(hidden_count), summary).name)
    names == reference_names ||
        error("Hidden width $hidden_count has a different configuration set")
end

Plots.gr()
plot_test_heatmap(summary)
plot_accuracy_runtime(summary)

hidden_links = String[]
for hidden_count in hidden_counts
    report_name = "mnist_hidden_$(hidden_count)_results.md"
    push!(hidden_links, "- [$hidden_count hidden units]($report_name)")
    rows = ordered(filter(:hidden_count => ==(hidden_count), summary))
    prefix = "mnist_hidden_$(hidden_count)"
    plot_history(
        rows,
        :train_acc,
        hidden_count,
        joinpath(RESULTS_DIR, "$(prefix)_training_accuracy.png"),
    )
    plot_history(
        rows,
        :val_acc,
        hidden_count,
        joinpath(RESULTS_DIR, "$(prefix)_validation_accuracy.png"),
    )
    report = """# MNIST Results — $hidden_count Hidden Units

All runs use 40,000 training images, 10,000 validation images, the complete
10,000-image official test split, 50 epochs, batch size 32, and 14×14 grayscale
inputs. The test score is measured from the checkpoint with the highest
validation accuracy. Two-layer direct models use $hidden_count units in both
hidden layers.

$(markdown_table(rows))

![Training accuracy, $hidden_count hidden units]($(prefix)_training_accuracy.png)

![Validation accuracy, $hidden_count hidden units]($(prefix)_validation_accuracy.png)
"""
    write(joinpath(RESULTS_DIR, report_name), report)
end

best_rows = combine(groupby(summary, :hidden_count)) do group
    best_overall = group[argmax(group.test_acc), :]
    direct = filter(:model => !=("neural"), group)
    neural = filter(:model => ==("neural"), group)
    best_direct = direct[argmax(direct.test_acc), :]
    best_neural = neural[argmax(neural.test_acc), :]
    (;
        best_overall=best_overall.name,
        overall_test=best_overall.test_acc,
        best_direct=best_direct.name,
        direct_test=best_direct.test_acc,
        best_neural=best_neural.name,
        neural_test=best_neural.test_acc,
    )
end

overview_lines = [
    "| Hidden units | Best overall | Test | Best direct | Test | Best neural | Test |",
    "|---:|---|---:|---|---:|---|---:|",
]
for row in eachrow(sort(best_rows, :hidden_count))
    push!(
        overview_lines,
        "| $(row.hidden_count) | $(row.best_overall) | " *
        "$(percent(row.overall_test)) | $(row.best_direct) | " *
        "$(percent(row.direct_test)) | $(row.best_neural) | " *
        "$(percent(row.neural_test)) |",
    )
end

best_overall = summary[argmax(summary.test_acc), :]
best_direct_rows = filter(:model => !=("neural"), summary)
best_direct = best_direct_rows[argmax(best_direct_rows.test_acc), :]
total_runtime = sum(summary.elapsed_seconds)

overview = """# MNIST Experiment Summary

This report covers all $(nrow(summary)) completed MNIST runs in
`$(basename(RESULTS_DIR))`. Every experiment used 40,000 training images,
10,000 validation images, the official 10,000-image test split, 50 epochs,
batch size 32, seed 1, and 14×14 grayscale inputs. Test accuracy is measured
from the checkpoint selected by validation accuracy.

## Main Results

- Best overall: **$(best_overall.name)** with **$(percent(best_overall.test_acc))** test accuracy.
- Best direct model: **$(best_direct.name)** with **$(percent(best_direct.test_acc))** test accuracy.
- Total recorded runtime: **$(runtime(total_runtime))** across all completed MNIST runs.

## Best Result per Hidden Width

$(join(overview_lines, "\n"))

## Test Accuracy

![MNIST test accuracy heatmap](mnist_test_accuracy_heatmap.png)

## Accuracy–Runtime Trade-off

![MNIST accuracy-runtime trade-off](mnist_accuracy_runtime.png)

## Reports by Hidden-Unit Count

$(join(hidden_links, "\n"))

## Research Qualification

The direct backend is the repository's approximate Gaussian-surrogate engine;
it is not exact RxInfer message passing. Nonlinear Softplus and categorical
factors are represented by refreshed Gaussian sites. Optimizer history is local
to the inner message iterations of each minibatch and resets between
minibatches. Loss backtracking was disabled in these runs. Neural baselines are
deterministic Softplus MLPs trained with Adam.

These are single-seed results, so differences should not be interpreted as
uncertainty-aware statistical comparisons without repeated runs.
"""

write(joinpath(RESULTS_DIR, "mnist_results.md"), overview)
println("Wrote MNIST Markdown reports and plots to $RESULTS_DIR")
