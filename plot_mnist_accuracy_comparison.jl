ENV["GKSwstype"] = "100"
using Plots

const rx_vector_train = [
    0.750, 0.860, 0.875, 0.905, 0.923,
    0.934, 0.923, 0.944, 0.949, 0.947,
]
const rx_vector_val = [
    0.760, 0.810, 0.780, 0.870, 0.840,
    0.870, 0.850, 0.870, 0.870, 0.880,
]

const rx_no_vector_train = [
    0.468, 0.678, 0.789, 0.844, 0.886,
    0.888, 0.874, 0.908, 0.910, 0.913,
]
const rx_no_vector_val = [
    0.440, 0.660, 0.790, 0.840, 0.860,
    0.820, 0.850, 0.830, 0.850, 0.850,
]

const rx_projected_nesterov_train = [
    0.667, 0.820, 0.853, 0.890, 0.911,
    0.917, 0.904, 0.929, 0.933, 0.936,
]
const rx_projected_nesterov_val = [
    0.680, 0.800, 0.780, 0.850, 0.840,
    0.860, 0.850, 0.860, 0.850, 0.860,
]

const rx_two_hidden_vector_train = [
    0.785, 0.833, 0.848, 0.864, 0.888,
    0.893, 0.901, 0.906, 0.905, 0.920,
]
const rx_two_hidden_vector_val = [
    0.770, 0.790, 0.800, 0.810, 0.820,
    0.800, 0.810, 0.820, 0.810, 0.810,
]

const neural_train = [
    0.534, 0.690, 0.716, 0.746, 0.786, 0.818, 0.807, 0.828, 0.850, 0.858,
    0.860, 0.869, 0.875, 0.876, 0.883, 0.885, 0.893, 0.885, 0.899, 0.898,
    0.901, 0.907, 0.909, 0.913, 0.915, 0.913, 0.913, 0.916, 0.917, 0.918,
    0.922, 0.923, 0.922, 0.924, 0.927, 0.931, 0.930, 0.929, 0.935, 0.934,
    0.931, 0.936, 0.939, 0.936, 0.941, 0.942, 0.941, 0.944, 0.942, 0.947,
]
const neural_val = [
    0.510, 0.650, 0.660, 0.690, 0.740, 0.770, 0.790, 0.800, 0.820, 0.830,
    0.820, 0.830, 0.850, 0.860, 0.840, 0.830, 0.860, 0.850, 0.840, 0.830,
    0.840, 0.850, 0.860, 0.860, 0.850, 0.860, 0.850, 0.860, 0.860, 0.860,
    0.860, 0.860, 0.860, 0.850, 0.860, 0.850, 0.850, 0.860, 0.860, 0.860,
    0.860, 0.860, 0.860, 0.860, 0.860, 0.860, 0.860, 0.850, 0.860, 0.860,
]

const neural_cnn_train = [
    0.100, 0.100, 0.100, 0.188, 0.144, 0.175, 0.407, 0.321, 0.591, 0.652,
    0.609, 0.710, 0.741, 0.766, 0.771, 0.771, 0.799, 0.798, 0.819, 0.812,
    0.815, 0.823, 0.831, 0.837, 0.830, 0.846, 0.832, 0.837, 0.844, 0.846,
    0.852, 0.856, 0.845, 0.851, 0.862, 0.857, 0.849, 0.863, 0.865, 0.871,
    0.868, 0.860, 0.872, 0.876, 0.871, 0.869, 0.862, 0.877, 0.859, 0.865,
]
const neural_cnn_val = [
    0.100, 0.100, 0.100, 0.190, 0.120, 0.160, 0.430, 0.330, 0.550, 0.630,
    0.570, 0.740, 0.770, 0.770, 0.770, 0.760, 0.780, 0.780, 0.760, 0.790,
    0.800, 0.790, 0.800, 0.790, 0.780, 0.800, 0.790, 0.800, 0.810, 0.810,
    0.810, 0.820, 0.810, 0.820, 0.820, 0.820, 0.820, 0.820, 0.820, 0.820,
    0.820, 0.830, 0.830, 0.820, 0.820, 0.820, 0.820, 0.820, 0.810, 0.820,
]

const rx_cnn_vector_train = [0.594, 0.703, 0.672]
const rx_cnn_vector_val = [0.688, 0.766, 0.703]

function accuracy_plot(
    rx_vector,
    rx_no_vector,
    rx_projected_nesterov,
    rx_two_hidden_vector,
    neural,
    neural_cnn,
    rx_cnn_vector;
    title,
)
    plot(
        1:length(neural), neural;
        label = "Neural MLP",
        color = :darkorange,
        linewidth = 2.5,
        marker = :none,
        xlabel = "Epoch",
        ylabel = "Accuracy",
        title,
        xlims = (1, 50),
        ylims = (0.0, 1.0),
        yticks = 0.0:0.1:1.0,
        legend = :bottomright,
        grid = true,
        minorgrid = true,
        size = (1000, 600),
        dpi = 160,
        left_margin = 5Plots.mm,
        bottom_margin = 5Plots.mm,
    )
    plot!(
        1:length(neural_cnn), neural_cnn;
        label = "Neural categorical CNN",
        color = :firebrick3,
        linewidth = 2.5,
        linestyle = :dash,
        marker = :none,
    )
    plot!(
        1:length(rx_vector), rx_vector;
        label = "RxInfer MLP (vector transport)",
        color = :dodgerblue3,
        linewidth = 3,
        marker = :circle,
        markersize = 5,
    )
    plot!(
        1:length(rx_no_vector), rx_no_vector;
        label = "RxInfer MLP (no vector transport)",
        color = :seagreen4,
        linewidth = 3,
        marker = :diamond,
        markersize = 5,
    )
    plot!(
        1:length(rx_projected_nesterov), rx_projected_nesterov;
        label = "RxInfer MLP (projected Nesterov)",
        color = :black,
        linewidth = 3,
        linestyle = :dash,
        marker = :utriangle,
        markersize = 5,
    )
    plot!(
        1:length(rx_two_hidden_vector), rx_two_hidden_vector;
        label = "RxInfer 2-hidden MLP (vector transport)",
        color = :deeppink3,
        linewidth = 3,
        marker = :xcross,
        markersize = 6,
    )
    plot!(
        1:length(rx_cnn_vector), rx_cnn_vector;
        label = "RxInfer CNN-like (vector transport)",
        color = :purple3,
        linewidth = 3,
        marker = :star5,
        markersize = 7,
    )
end

train_plot = accuracy_plot(
    rx_vector_train,
    rx_no_vector_train,
    rx_projected_nesterov_train,
    rx_two_hidden_vector_train,
    neural_train,
    neural_cnn_train,
    rx_cnn_vector_train;
    title = "MNIST Training Accuracy by Epoch",
)
savefig(train_plot, "mnist_train_accuracy_comparison.png")
train_plot = nothing
Plots.closeall()
GC.gc()
Plots.gr()

val_plot = accuracy_plot(
    rx_vector_val,
    rx_no_vector_val,
    rx_projected_nesterov_val,
    rx_two_hidden_vector_val,
    neural_val,
    neural_cnn_val,
    rx_cnn_vector_val;
    title = "MNIST Validation Accuracy by Epoch",
)
savefig(val_plot, "mnist_val_accuracy_comparison.png")

println("Saved mnist_train_accuracy_comparison.png")
println("Saved mnist_val_accuracy_comparison.png")
