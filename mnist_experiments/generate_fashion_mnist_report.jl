ENV["GKSwstype"] = "100"

using CSV
using DataFrames
using Plots
using Printf

const RESULTS_DIR = joinpath(@__DIR__, "results", "fashion_mnist_20260718_185229")
const SUMMARY_PATH = joinpath(RESULTS_DIR, "summary.tsv")

model_name(model, layers) = model == "rxinfer_1h" ? "RxInfer MLP (1 hidden layer)" :
    model == "rxinfer_2h" ? "RxInfer MLP (2 hidden layers)" :
    "Neural MLP ($(layers) hidden " * (layers == 1 ? "layer)" : "layers)")

function load_history(log_path)
    history = NamedTuple{(:epoch, :train_acc, :val_acc), Tuple{Int, Float64, Float64}}[]
    pattern = r"epoch=(\d+).*?train_acc=([0-9.]+).*?val_acc=([0-9.]+)"
    for line in eachline(log_path)
        matched = match(pattern, line)
        isnothing(matched) && continue
        push!(history, (
            epoch = parse(Int, matched.captures[1]),
            train_acc = parse(Float64, matched.captures[2]),
            val_acc = parse(Float64, matched.captures[3]),
        ))
    end
    isempty(history) && error("No epoch metrics found in $log_path")
    return history
end

function plot_accuracy(histories, metric, ntrain, path)
    metric_label = metric == :train_acc ? "Training" : "Validation"
    plt = plot(
        title = "Fashion-MNIST $(metric_label) Accuracy ($(ntrain) Training Images)",
        xlabel = "Epoch", ylabel = "Accuracy", ylims = (0, 1),
        yticks = 0:0.1:1, xlims = (1, 50), grid = true, minorgrid = true,
        legend = :outerright, size = (1300, 650), dpi = 170,
        left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
    )
    colors = [:dodgerblue3, :deeppink3, :darkorange, :goldenrod3, :sienna3,
        :mediumpurple3, :slateblue3, :deepskyblue3, :teal]
    for (index, row) in enumerate(eachrow(histories))
        history = row.history
        rxinfer_marker = row.model == "rxinfer_1h" ? :xcross :
            row.model == "rxinfer_2h" ? :diamond : :none
        plot!(plt, getproperty.(history, :epoch), getproperty.(history, metric);
            label = row.name, color = colors[index], linewidth = 2.5,
            marker = rxinfer_marker, markersize = row.model == "neural" ? 0 : 4)
    end
    savefig(plt, path)
end

percent(value) = @sprintf("%.1f%%", 100value)
seconds(value) = @sprintf("%.0f s", value)

function markdown_table(rows)
    lines = [
        "| Model | Final train accuracy | Best validation accuracy | Best validation epoch | Test accuracy | Runtime |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    best_test = maximum(rows.test_acc)
    for row in eachrow(rows)
        test = percent(row.test_acc)
        isapprox(row.test_acc, best_test) && (test = "**$test**")
        push!(lines, "| $(row.name) | $(percent(row.final_train_acc)) | $(percent(row.best_val_acc)) | $(row.best_val_epoch) | $test | $(seconds(row.elapsed_seconds)) |")
    end
    return join(lines, "\n")
end

summary = CSV.read(SUMMARY_PATH, DataFrame; delim = '\t')
summary.name = [model_name(row.model, row.hidden_layers) for row in eachrow(summary)]
summary.history = [load_history(row.log) for row in eachrow(summary)]
summary.final_train_acc = [history[end].train_acc for history in summary.history]

Plots.gr()
for ntrain in (1000, 10000)
    subset = sort(filter(:ntrain => ==(ntrain), summary), :hidden_layers)
    # Keep the two RxInfer variants before the neural depth sweep.
    subset = vcat(filter(:model => !=("neural"), subset), filter(:model => ==("neural"), subset))
    plot_accuracy(subset, :train_acc, ntrain,
        joinpath(RESULTS_DIR, "fashion_mnist_$(ntrain)_train_accuracy.png"))
    plot_accuracy(subset, :val_acc, ntrain,
        joinpath(RESULTS_DIR, "fashion_mnist_$(ntrain)_val_accuracy.png"))
end

rows_1k = filter(:ntrain => ==(1000), summary)
rows_10k = filter(:ntrain => ==(10000), summary)
sort!(rows_1k, [:model, :hidden_layers])
sort!(rows_10k, [:model, :hidden_layers])

best_1k = rows_1k[argmax(rows_1k.test_acc), :]
best_10k = rows_10k[argmax(rows_10k.test_acc), :]
report = """# Fashion-MNIST Results

Experiment directory: `fashion_mnist_20260718_185229`. Every run used 50 epochs,
196 input features (14×14 images), and batch size 32. Test accuracy is from the
checkpoint selected by best validation accuracy.

## 1,000 Training Images

| Split | Images |
|---|---:|
| Training | 1,000 |
| Validation | 100 |
| Test | 100 |

$(markdown_table(rows_1k))

The strongest 1k result is $(best_1k.name): $(percent(best_1k.test_acc)) test
accuracy, selected at validation epoch $(best_1k.best_val_epoch). The 100-image
test split means one image changes test accuracy by 1 percentage point.

![Training accuracy, 1k images](fashion_mnist_1000_train_accuracy.png)

![Validation accuracy, 1k images](fashion_mnist_1000_val_accuracy.png)

## 10,000 Training Images

| Split | Images |
|---|---:|
| Training | 10,000 |
| Validation | 1,000 |
| Test | 1,000 |

$(markdown_table(rows_10k))

The strongest 10k result is $(best_10k.name): $(percent(best_10k.test_acc)) test
accuracy, selected at validation epoch $(best_10k.best_val_epoch).

![Training accuracy, 10k images](fashion_mnist_10000_train_accuracy.png)

![Validation accuracy, 10k images](fashion_mnist_10000_val_accuracy.png)

## Comparison of Best Test Accuracy

| Training images | Best model | Best test accuracy |
|---:|---|---:|
| 1,000 | $(best_1k.name) | $(percent(best_1k.test_acc)) |
| 10,000 | $(best_10k.name) | $(percent(best_10k.test_acc)) |

Results are single runs; the smaller validation/test splits make small differences
especially sensitive to sampling variation.
"""

write(joinpath(RESULTS_DIR, "fashion_mnist_results.md"), report)
println("Wrote Fashion-MNIST report and four accuracy plots to $RESULTS_DIR")
