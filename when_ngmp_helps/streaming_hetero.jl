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

include(joinpath(@__DIR__, "hetero_hierarchy.jl"))

using Printf

import LinearAlgebra: logdet, Symmetric

# (label, ReactiveMP arm, sweeps per fit). The one-sweep projected-VMP arm
# roughly matches NGMP's wall-clock budget: the standard VMP arm spends ~10x
# NGMP's time on its per-edge manifold projections.
const STREAMING_ARMS = (
    (label = "Projected VMP", method = "VMP", iterations = 240),
    (label = "Projected VMP, 1 sweep", method = "VMP", iterations = 1),
    (label = "NGMP", method = "NGMP-cavity", iterations = 240),
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
    local fit
    for batch in 1:config.n_batches
        indices = order[(bounds[batch] + 1):bounds[batch + 1]]
        rows = design(feature_map, data.x_train[indices])
        fit = fit_arm(arm.method, data.y_train[indices], rows, priors, arm_config)
        priors = (; v = fit.qv, w = fit.qw)
        prediction = predict(fit, test_rows, config.top_carrier)
        push!(logpdfs, mean_predictive_logpdf(prediction, data.y_test))
        push!(logdets_w, _logdet_cov(fit.qw))
    end
    return (; logpdfs, logdets_w, fit)
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
        fit,
    )
end

function streaming_prediction_panel(fit, data, grid, feature_map, config, label, style)
    grid_rows = design(feature_map, grid)
    prediction = predict(fit, grid_rows, config.top_carrier)
    band = 1.96 .* sqrt.(prediction.total_variance)
    panel = plot(;
        xlabel = "x", ylabel = "y", legend = :topright,
        ylims = (-4.5, 4.5), left_margin = 5Plots.mm,
    )
    scatter!(panel, data.x_train, data.y_train;
        color = :gray70, markersize = 1.6, markerstrokewidth = 0, label = "train")
    plot!(panel, grid, prediction.mean;
        ribbon = band, color = style.color, linestyle = style.linestyle,
        linewidth = 2.2, fillalpha = 0.2, label = label)
    plot!(panel, grid, aleatoric_mean.(grid);
        color = COLORS.truth, linestyle = :dashdot, linewidth = 1.6,
        label = "true mean")
    return panel
end

function streaming_main()
    ensure_outputs()
    smoke = smoke_mode()
    config = (
        smoke,
        seed = 42,
        repetitions = smoke ? 2 : 20,
        holdout_fraction = 1 / 3,
        aleatoric_samples = smoke ? 90 : 600,
        epistemic_samples = 150,
        epistemic_noise_sd = 0.02,
        n_basis = smoke ? 6 : 32,
        aleatoric_lengthscale = 0.25,
        epistemic_lengthscale = 1.0,
        feature_seed = 20260726,
        signal_sd = 1.0,
        level_sd = 1.6,
        anchor_sd = 1.0,
        top_carrier = 25.0,
        iterations = smoke ? 10 : 240,
        ngmp_alpha = 0.6,
        ngmp_max_step = 0.5,
        cavity_quadrature = 32,
        contamination_fraction = 0.0,
        contamination_sd = 3.0,
        n_batches = 10,
        methods = STREAMING_ARMS,
    )
    feature_map = make_feature_map(
        config.n_basis, config.aleatoric_lengthscale;
        feature_seed = config.feature_seed,
    )
    grid = collect(range(-3.0, 3.0; length = 241))

    rows_out = NamedTuple[]
    track_out = NamedTuple[]
    kept = Dict{Tuple{String, String}, Any}()
    kept_data = Ref{Any}(nothing)
    for repetition in 1:config.repetitions
        data = aleatoric_data(config, StableRNG(config.seed + repetition - 1))
        order = randperm(StableRNG(1000 + repetition), length(data.y_train))
        repetition == 1 && (kept_data[] = data)
        for arm in STREAMING_ARMS
            full = streaming_full(arm, data, feature_map, config)
            sequential = streaming_sequential(arm, data, order, feature_map, config)
            test_rows = design(feature_map, data.x_test)
            seq_prediction = predict(sequential.fit, test_rows, config.top_carrier)
            seq_rmse = sqrt(mean(abs2.(seq_prediction.mean .- data.y_test)))
            push!(rows_out, (;
                repetition, arm = arm.label,
                full_logpdf = full.logpdf,
                full_rmse = full.rmse,
                full_logdet_w = full.logdet_w,
                sequential_logpdf = last(sequential.logpdfs),
                sequential_rmse = seq_rmse,
                streaming_penalty = last(sequential.logpdfs) - full.logpdf,
            ))
            for batch in 1:config.n_batches
                push!(track_out, (;
                    repetition, arm = arm.label, batch,
                    logpdf = sequential.logpdfs[batch],
                    logdet_w = sequential.logdets_w[batch],
                ))
            end
            if repetition == 1
                kept[(arm.label, "full")] = full.fit
                kept[(arm.label, "sequential")] = sequential.fit
            end
        end
        @printf("completed streaming seed %d/%d\n", repetition, config.repetitions)
        flush(stdout)
    end

    runs = DataFrame(rows_out)
    track = DataFrame(track_out)
    CSV.write(joinpath(RESULT_DIR, "streaming_hetero_runs.csv"), runs)
    CSV.write(joinpath(RESULT_DIR, "streaming_hetero_track.csv"), track)

    summary_lines_out = String[
        "# Streaming heteroscedastic study (aleatoric benchmark, n_basis = $(config.n_basis))",
        "",
        "Full batch vs sequential ($(config.n_batches) batches), $(config.repetitions) paired seeds.",
        "",
        "| arm | full NLL | full RMSE | sequential NLL | sequential RMSE | streaming penalty |",
        "|---|---|---|---|---|---|",
    ]
    for arm in STREAMING_ARMS
        selected = filter(row -> row.arm == arm.label, runs)
        line = string("| ", arm.label,
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
    write_config(
        "streaming_hetero",
        merge(config, (methods = [arm.label for arm in STREAMING_ARMS],)),
    )
    write_summary("streaming_hetero", summary_lines_out)

    # four prediction panels (seed 1): full vs sequential for the two
    # full-budget arms (the 1-sweep ablation lives in the table and the
    # collapse figure)
    styles = Dict(
        "Projected VMP" => (color = COLORS.vmp, linestyle = :dash),
        "Projected VMP, 1 sweep" => (color = :peru, linestyle = :dot),
        "NGMP" => (color = COLORS.ngmp, linestyle = :solid),
    )
    stems = Dict(
        ("Projected VMP", "full") => "streaming_vmp_full",
        ("Projected VMP", "sequential") => "streaming_vmp_sequential",
        ("NGMP", "full") => "streaming_cavity_full",
        ("NGMP", "sequential") => "streaming_cavity_sequential",
    )
    panels = Dict{Tuple{String, String}, Any}()
    for label in ("Projected VMP", "NGMP"), mode in ("full", "sequential")
        panel = streaming_prediction_panel(
            kept[(label, mode)], kept_data[], grid, feature_map, config,
            string(label, " ", mode), styles[label],
        )
        panels[(label, mode)] = panel
        save_pdf(panel, stems[(label, mode)])
    end
    preview = plot(
        plot(panels[("Projected VMP", "full")]; title = "Projected VMP, full batch"),
        plot(panels[("Projected VMP", "sequential")]; title = "Projected VMP, sequential"),
        plot(panels[("NGMP", "full")]; title = "NGMP, full batch"),
        plot(panels[("NGMP", "sequential")]; title = "NGMP, sequential");
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
    for arm in STREAMING_ARMS
        style = styles[arm.label]
        selected = filter(row -> row.arm == arm.label, track)
        by_batch = sort(combine(
            groupby(selected, :batch), :logdet_w => mean => :center,
            :logdet_w => ci95 => :ci95,
        ), :batch)
        plot!(collapse, by_batch.batch, by_batch.center;
            ribbon = by_batch.ci95, color = style.color,
            linestyle = style.linestyle, linewidth = 2.4, marker = :circle,
            markersize = 4, fillalpha = 0.15, label = arm.label)
        full_reference = mean(
            filter(row -> row.arm == arm.label, runs).full_logdet_w,
        )
        hline!(collapse, [full_reference]; color = style.color,
            linestyle = :dot, linewidth = 1.6, label = "")
    end
    save_pdf(collapse, "streaming_collapse")
    save_figure(plot(collapse; title = "noise-weight posterior collapse"),
        "streaming_hetero_collapse")
end

abspath(PROGRAM_FILE) == (@__FILE__) && streaming_main()
