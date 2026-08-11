#!/usr/bin/env julia

# Streaming (sequential-batch) study on the aleatoric heteroscedastic
# benchmark: projected VMP vs cavity (true) NGMP.
#
# Two regimes on ONE problem:
#   * smoothing (full batch): the mean-field arm's overconfident noise-weight
#     posterior q(w) is benign — held-out NLL/RMSE tie;
#   * batching (posterior of batch t = prior of batch t+1): the same
#     overconfidence is catastrophic — q(w) collapses beyond what all the
#     data could justify after the FIRST small batch, the collapsed prior is
#     immune to correction, and the predictive trajectory flatlines.
#
# Artifacts: full-batch table (NLL/RMSE ± CI), four prediction panels
# (each arm: full vs sequential, same seed and axes), the log det Σ_w
# collapse trajectory with full-batch reference lines, and per-seed CSVs.
#
# compute() runs the inference and persists everything the figures need
# (including the repetition-1 grid predictions); render() redraws all
# figures/tables from results/*.csv alone.

include(joinpath(@__DIR__, "hetero_model.jl"))

using CSV
using DataFrames
using Plots
using Printf

import LinearAlgebra: logdet, Symmetric

# (arm key, ReactiveMP method, sweeps per fit). The one-sweep projected-VMP
# arm roughly matches NGMP's wall-clock budget: the standard VMP arm spends
# ~10x NGMP's time on its per-edge manifold projections.
const STREAMING_ARMS = (
    (key = "pvmp", method = "VMP", iterations = 240),
    (key = "pvmp1", method = "VMP", iterations = 1),
    (key = "ngmp", method = "NGMP-cavity", iterations = 240),
)

const ARM_STYLES = Dict(
    "pvmp" => (color = COLORS.vmp, linestyle = :dash),
    "pvmp1" => (color = COLORS.vmp1, linestyle = :dot),
    "ngmp" => (color = COLORS.ngmp, linestyle = :solid),
)

# prediction panels exist only for the two full-budget arms; file stems keep
# the historical names (cavity = the true-NGMP arm)
const PANEL_STEMS = Dict(
    ("pvmp", "full") => "streaming_vmp_full",
    ("pvmp", "sequential") => "streaming_vmp_sequential",
    ("ngmp", "full") => "streaming_cavity_full",
    ("ngmp", "sequential") => "streaming_cavity_sequential",
)

_logdet_cov(q) = logdet(Symmetric(Matrix(mean_cov(q)[2])))

_arm_config(arm, config) = merge(config, (iterations = arm.iterations,))

function streaming_sequential(arm, data, order, feature_map, config)
    arm_config = _arm_config(arm, config)
    n_train = length(data.y_train)
    bounds = round.(Int, range(0, n_train; length = config.n_batches + 1))
    test_rows = design(feature_map, data.x_test)
    priors = make_priors(config, data.x_train, data.y_train)
    logpdfs = Float64[]
    logdets_w = Float64[]
    free_energies = Vector{Float64}[]
    batch_sizes = Int[]
    local fit
    for batch in 1:config.n_batches
        indices = order[(bounds[batch] + 1):bounds[batch + 1]]
        rows = design(feature_map, data.x_train[indices])
        fit = fit_arm(arm.method, data.y_train[indices], rows, priors, arm_config)
        priors = (; v = fit.qv, w = fit.qw)
        prediction = predict(fit, test_rows, config.top_carrier)
        push!(logpdfs, mean_predictive_logpdf(prediction, data.y_test))
        push!(logdets_w, _logdet_cov(fit.qw))
        push!(free_energies, fit.free_energy)
        push!(batch_sizes, length(indices))
    end
    return (; logpdfs, logdets_w, free_energies, batch_sizes, fit)
end

function streaming_full(arm, data, feature_map, config)
    rows = design(feature_map, data.x_train)
    test_rows = design(feature_map, data.x_test)
    priors = make_priors(config, data.x_train, data.y_train)
    fit = fit_arm(arm.method, data.y_train, rows, priors, _arm_config(arm, config))
    prediction = predict(fit, test_rows, config.top_carrier)
    return (;
        logpdf = mean_predictive_logpdf(prediction, data.y_test),
        rmse = sqrt(mean(abs2.(prediction.mean .- data.y_test))),
        logdet_w = _logdet_cov(fit.qw),
        free_energy = fit.free_energy,
        n_obs = length(data.y_train),
        fit,
    )
end

function compute()
    ensure_outputs()
    smoke = smoke_mode()
    config = (
        smoke,
        seed = 42,
        repetitions = smoke ? 2 : 20,
        holdout_fraction = 1 / 3,
        aleatoric_samples = smoke ? 90 : 600,
        n_basis = smoke ? 6 : 32,
        aleatoric_lengthscale = 0.25,
        feature_seed = 20260726,
        signal_sd = 1.0,
        level_sd = 1.6,
        anchor_sd = 1.0,
        top_carrier = 25.0,
        iterations = smoke ? 10 : 240,
        ngmp_alpha = 0.6,
        ngmp_max_step = 0.5,
        cavity_quadrature = 32,
        n_batches = 10,
    )
    feature_map = make_feature_map(
        config.n_basis, config.aleatoric_lengthscale;
        feature_seed = config.feature_seed,
    )
    grid = collect(range(-3.0, 3.0; length = 241))
    grid_rows = design(feature_map, grid)

    rows_out = NamedTuple[]
    track_out = NamedTuple[]
    panel_rows = NamedTuple[]
    train_rows = NamedTuple[]
    free_energy_rows = NamedTuple[]
    for repetition in 1:config.repetitions
        data = aleatoric_data(config, StableRNG(config.seed + repetition - 1))
        order = randperm(StableRNG(1000 + repetition), length(data.y_train))
        if repetition == 1
            append!(train_rows, [
                (; x = x, y = y)
                for (x, y) in zip(data.x_train, data.y_train)
            ])
        end
        for arm in STREAMING_ARMS
            full = streaming_full(arm, data, feature_map, config)
            sequential = streaming_sequential(arm, data, order, feature_map, config)
            test_rows = design(feature_map, data.x_test)
            seq_prediction = predict(sequential.fit, test_rows, config.top_carrier)
            seq_rmse = sqrt(mean(abs2.(seq_prediction.mean .- data.y_test)))
            push!(rows_out, (;
                repetition, arm = arm.key,
                full_logpdf = full.logpdf,
                full_rmse = full.rmse,
                full_logdet_w = full.logdet_w,
                sequential_logpdf = last(sequential.logpdfs),
                sequential_rmse = seq_rmse,
                streaming_penalty = last(sequential.logpdfs) - full.logpdf,
            ))
            for batch in 1:config.n_batches
                push!(track_out, (;
                    repetition, arm = arm.key, batch,
                    logpdf = sequential.logpdfs[batch],
                    logdet_w = sequential.logdets_w[batch],
                ))
            end
            # Bethe free-energy traces (available for the ProjectedTo arms;
            # the cavity arm runs with free_energy = false — its improper
            # sites have no tractable Bethe evaluation)
            for (iteration, value) in enumerate(full.free_energy)
                push!(free_energy_rows, (;
                    repetition, arm = arm.key, mode = "full", batch = 0,
                    iteration, free_energy = value, n_obs = full.n_obs,
                ))
            end
            for batch in 1:config.n_batches
                for (iteration, value) in enumerate(sequential.free_energies[batch])
                    push!(free_energy_rows, (;
                        repetition, arm = arm.key, mode = "sequential", batch,
                        iteration, free_energy = value,
                        n_obs = sequential.batch_sizes[batch],
                    ))
                end
            end
            if repetition == 1 && haskey(PANEL_STEMS, (arm.key, "full"))
                for (mode, fit) in (("full", full.fit), ("sequential", sequential.fit))
                    prediction = predict(fit, grid_rows, config.top_carrier)
                    append!(panel_rows, [
                        (;
                            arm = arm.key, mode, x = grid[i],
                            pred_mean = prediction.mean[i],
                            pred_total_variance = prediction.total_variance[i],
                        )
                        for i in eachindex(grid)
                    ])
                end
            end
        end
        @printf("completed streaming seed %d/%d\n", repetition, config.repetitions)
        flush(stdout)
    end

    CSV.write(joinpath(RESULT_DIR, "streaming_hetero_runs.csv"), DataFrame(rows_out))
    CSV.write(joinpath(RESULT_DIR, "streaming_hetero_track.csv"), DataFrame(track_out))
    CSV.write(joinpath(RESULT_DIR, "streaming_hetero_panels.csv"), DataFrame(panel_rows))
    CSV.write(joinpath(RESULT_DIR, "streaming_hetero_train.csv"), DataFrame(train_rows))
    CSV.write(
        joinpath(RESULT_DIR, "streaming_hetero_free_energy.csv"),
        DataFrame(free_energy_rows),
    )
    write_config(
        "streaming_hetero",
        merge(config, (methods = [arm.key for arm in STREAMING_ARMS],)),
    )
end

function streaming_prediction_panel(panel_frame, train, label, style)
    band = 1.96 .* sqrt.(panel_frame.pred_total_variance)
    panel = plot(;
        xlabel = "x", ylabel = "y", legend = :topright,
        ylims = (-4.5, 4.5), left_margin = 5Plots.mm,
    )
    scatter!(panel, train.x, train.y;
        color = :gray70, markersize = 1.6, markerstrokewidth = 0, label = "train")
    plot!(panel, panel_frame.x, panel_frame.pred_mean;
        ribbon = band, color = style.color, linestyle = style.linestyle,
        linewidth = 2.2, fillalpha = 0.2, label = label)
    plot!(panel, panel_frame.x, aleatoric_mean.(panel_frame.x);
        color = COLORS.truth, linestyle = :dashdot, linewidth = 1.6,
        label = "true mean")
    return panel
end

# Bethe free-energy convergence of the projected-VMP arm: shows the streaming
# comparison probes fixed points, not truncation artifacts — the one
# full-batch fit and each of the ten sequential fits all flatten well within
# the sweep budget, so no further variational iteration could improve the
# mean-field arm. (The cavity-NGMP arm has no tractable Bethe evaluation.)
function streaming_free_energy_figures(fe_frame)
    vmp = filter(row -> row.arm == "pvmp", fe_frame)
    isempty(vmp) && return
    # the first sweeps start orders of magnitude higher and would flatten the
    # plateau view
    start = min(3, maximum(vmp.iteration))
    per_observation = combine(
        groupby(
            filter(row -> row.iteration >= start, vmp),
            [:mode, :batch, :iteration],
        ),
        [:free_energy, :n_obs] => ((f, n) -> mean(f ./ n)) => :center,
        [:free_energy, :n_obs] => ((f, n) -> ci95(f ./ n)) => :ci95,
    )

    full = sort(filter(row -> row.mode == "full", per_observation), :iteration)
    full_panel = plot(;
        xlabel = "iteration",
        ylabel = "Bethe free energy / observation",
        legend = :topright,
        left_margin = 5Plots.mm,
    )
    plot!(full_panel, full.iteration, full.center;
        ribbon = full.ci95, color = COLORS.vmp, linestyle = :dash,
        linewidth = 2.2, fillalpha = 0.15,
        label = "$(method_label(:pvmp)), full batch")

    sequential = filter(row -> row.mode == "sequential", per_observation)
    n_batches = maximum(sequential.batch)
    shades = cgrad(:Oranges_9)
    sequential_panel = plot(;
        xlabel = "iteration",
        ylabel = "Bethe free energy / observation",
        legend = :topright,
        left_margin = 5Plots.mm,
    )
    for batch in 1:n_batches
        by_iteration = sort(
            filter(row -> row.batch == batch, sequential), :iteration,
        )
        shade = get(shades, 0.35 + 0.65 * (batch - 1) / max(n_batches - 1, 1))
        plot!(sequential_panel, by_iteration.iteration, by_iteration.center;
            color = shade, linewidth = 1.8,
            label = batch in (1, n_batches) ? "batch $batch" : "")
    end
    save_pdf(full_panel, "streaming_bethe_full")
    save_pdf(sequential_panel, "streaming_bethe_sequential")
    save_figure(
        plot(
            plot(full_panel; title = "full batch"),
            plot(sequential_panel; title = "sequential batches");
            layout = (1, 2), size = (1100, 420),
        ),
        "streaming_hetero_free_energy",
    )
end

# Everything below reads only results/*.csv (+ the config TOML), so figures
# and the summary can be regenerated without re-running inference.
function render()
    ensure_outputs()
    config = read_config("streaming_hetero")
    runs = CSV.read(joinpath(RESULT_DIR, "streaming_hetero_runs.csv"), DataFrame)
    track = CSV.read(joinpath(RESULT_DIR, "streaming_hetero_track.csv"), DataFrame)
    panels_frame = CSV.read(joinpath(RESULT_DIR, "streaming_hetero_panels.csv"), DataFrame)
    train = CSV.read(joinpath(RESULT_DIR, "streaming_hetero_train.csv"), DataFrame)
    arm_keys = String.(config["methods"])

    summary_lines_out = String[
        "# Streaming heteroscedastic study (aleatoric benchmark, n_basis = $(config["n_basis"]))",
        "",
        "Full batch vs sequential ($(config["n_batches"]) batches), $(config["repetitions"]) paired seeds.",
        "",
        "| arm | full NLL | full RMSE | sequential NLL | sequential RMSE | streaming penalty |",
        "|---|---|---|---|---|---|",
    ]
    for key in arm_keys
        selected = filter(row -> row.arm == key, runs)
        line = string("| ", method_label(key),
            " | ", round(mean(.-selected.full_logpdf); digits = 3),
            " ± ", round(ci95(selected.full_logpdf); digits = 3),
            " | ", round(mean(selected.full_rmse); digits = 3),
            " ± ", round(ci95(selected.full_rmse); digits = 3),
            " | ", round(mean(.-selected.sequential_logpdf); digits = 3),
            " ± ", round(ci95(selected.sequential_logpdf); digits = 3),
            " | ", round(mean(selected.sequential_rmse); digits = 3),
            " ± ", round(ci95(selected.sequential_rmse); digits = 3),
            " | ", round(mean(selected.streaming_penalty); digits = 3),
            " ± ", round(ci95(selected.streaming_penalty); digits = 3), " |")
        push!(summary_lines_out, line)
    end
    write_summary("streaming_hetero", summary_lines_out)

    # four prediction panels (seed 1): full vs sequential for the two
    # full-budget arms (the 1-sweep ablation lives in the table and the
    # collapse figure)
    panels = Dict{Tuple{String, String}, Any}()
    for ((key, mode), stem) in PANEL_STEMS
        selected = sort(
            filter(row -> row.arm == key && row.mode == mode, panels_frame),
            :x,
        )
        panel = streaming_prediction_panel(
            selected, train, string(method_label(key), " ", mode),
            ARM_STYLES[key],
        )
        panels[(key, mode)] = panel
        save_pdf(panel, stem)
    end
    preview = plot(
        plot(panels[("pvmp", "full")]; title = "$(method_label("pvmp")), full batch"),
        plot(panels[("pvmp", "sequential")]; title = "$(method_label("pvmp")), sequential"),
        plot(panels[("ngmp", "full")]; title = "$(method_label("ngmp")), full batch"),
        plot(panels[("ngmp", "sequential")]; title = "$(method_label("ngmp")), sequential");
        layout = (2, 2), size = (1100, 750),
    )
    save_figure(preview, "streaming_hetero_predictions")

    # collapse trajectory: log det Σ_w over batches + full-batch references
    collapse = plot(;
        xlabel = "batch",
        ylabel = "log det Σ_w",
        legend = :topright,
        left_margin = 5Plots.mm,
    )
    for key in arm_keys
        style = ARM_STYLES[key]
        selected = filter(row -> row.arm == key, track)
        by_batch = sort(combine(
            groupby(selected, :batch), :logdet_w => mean => :center,
            :logdet_w => ci95 => :ci95,
        ), :batch)
        plot!(collapse, by_batch.batch, by_batch.center;
            ribbon = by_batch.ci95, color = style.color,
            linestyle = style.linestyle, linewidth = 2.4, marker = :circle,
            markersize = 4, fillalpha = 0.15, label = method_label(key))
        full_reference = mean(
            filter(row -> row.arm == key, runs).full_logdet_w,
        )
        hline!(collapse, [full_reference]; color = style.color,
            linestyle = :dot, linewidth = 1.6, label = "")
    end
    save_pdf(collapse, "streaming_collapse")
    save_figure(plot(collapse; title = "noise-weight posterior collapse"),
        "streaming_hetero_collapse")

    streaming_free_energy_figures(CSV.read(
        joinpath(RESULT_DIR, "streaming_hetero_free_energy.csv"), DataFrame,
    ))
end

function streaming_main()
    render_only() || compute()
    render()
end

abspath(PROGRAM_FILE) == (@__FILE__) && streaming_main()
