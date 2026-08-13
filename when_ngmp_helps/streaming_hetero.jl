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
# Artifacts: full-batch table (NLL/RMSE ± CI), four prediction panels and
# four learned-versus-true aleatoric-variance panels (each arm: full vs
# sequential, same seed and axes), the log det Σ_w collapse trajectory with
# full-batch reference lines, and per-seed CSVs.
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

# (arm key, ReactiveMP method, sweeps per fit, inner projection iterations).
# The pvmp1 arm keeps the full 240-sweep schedule but gives each ProjectedTo
# call a SINGLE inner Manopt step (instead of the default 100) — the
# budget-matched control: VMP's cost is dominated by the inner projections.
const STREAMING_ARMS = (
    (key = "pvmp", method = "VMP", iterations = 240, projection_iterations = 100),
    (key = "pvmp1", method = "VMP", iterations = 240, projection_iterations = 1),
    (key = "ngmp", method = "NGMP-cavity", iterations = 240, projection_iterations = 100),
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

const VARIANCE_PANEL_STEMS = Dict(
    ("pvmp", "full") => "streaming_vmp_variance_full",
    ("pvmp", "sequential") => "streaming_vmp_variance_sequential",
    ("ngmp", "full") => "streaming_cavity_variance_full",
    ("ngmp", "sequential") => "streaming_cavity_variance_sequential",
)

_logdet_cov(q) = logdet(Symmetric(Matrix(mean_cov(q)[2])))

_arm_config(arm, config) = merge(config, (
    iterations = config.smoke ? config.iterations : arm.iterations,
    projection_iterations = arm.projection_iterations,
))

function prediction_panel_rows(arm, mode, grid, prediction)
    return [
        (;
            arm,
            mode,
            x = grid[i],
            pred_mean = prediction.mean[i],
            pred_total_variance = prediction.total_variance[i],
            pred_noise_variance = prediction.noise_mean[i],
            pred_noise_lower = prediction.noise_lower[i],
            pred_noise_upper = prediction.noise_upper[i],
        )
        for i in eachindex(grid)
    ]
end

function streaming_config(smoke)
    level_n_basis = smoke ? 6 : 32
    mean_n_basis = smoke ? 24 : 128
    return (
        smoke,
        seed = 42,
        repetitions = smoke ? 2 : 20,
        holdout_fraction = 1 / 3,
        aleatoric_samples = smoke ? 90 : 600,
        # `n_basis` remains as a legacy alias for the log-precision path.
        # The mean path uses the capacity/Matérn setup selected by the
        # NGMP-only diagnostic; the exponentiated variance path stays at its
        # historical RBF capacity and prior scale.
        n_basis = level_n_basis,
        mean_n_basis,
        level_n_basis,
        aleatoric_lengthscale = 0.25,
        feature_seed = 20260726,
        mean_kernel = "matern32",
        level_kernel = "rbf",
        mean_lengthscale = 0.25,
        level_lengthscale = 0.25,
        mean_feature_seed = 20260726,
        level_feature_seed = 20260726,
        signal_sd = 2.0,
        level_sd = 1.6,
        anchor_sd = 1.0,
        top_carrier = 25.0,
        iterations = smoke ? 10 : 240,
        ngmp_alpha = 0.6,
        ngmp_max_step = 0.5,
        cavity_quadrature = 32,
        n_batches = 10,
    )
end

function make_streaming_feature_maps(config)
    return (;
        mean = make_feature_map(
            get(config, :mean_n_basis, config.n_basis),
            get(config, :mean_lengthscale, config.aleatoric_lengthscale);
            feature_seed = get(config, :mean_feature_seed, config.feature_seed),
            kernel = get(config, :mean_kernel, "rbf"),
        ),
        level = make_feature_map(
            get(config, :level_n_basis, config.n_basis),
            get(config, :level_lengthscale, config.aleatoric_lengthscale);
            feature_seed = get(config, :level_feature_seed, config.feature_seed),
            kernel = get(config, :level_kernel, "rbf"),
        ),
    )
end

design(feature_maps::NamedTuple, xs) = (;
    mean = design(feature_maps.mean, xs),
    level = design(feature_maps.level, xs),
)

function streaming_sequential(arm, data, order, feature_maps, config; free_energy = false)
    arm_config = _arm_config(arm, config)
    n_train = length(data.y_train)
    bounds = round.(Int, range(0, n_train; length = config.n_batches + 1))
    test_rows = design(feature_maps, data.x_test)
    priors = make_priors(config, data.x_train, data.y_train)
    logpdfs = Float64[]
    logdets_w = Float64[]
    free_energies = Vector{Float64}[]
    batch_sizes = Int[]
    local fit
    for batch in 1:config.n_batches
        indices = order[(bounds[batch] + 1):bounds[batch + 1]]
        rows = design(feature_maps, data.x_train[indices])
        fit = fit_arm(
            arm.method,
            data.y_train[indices],
            rows.mean,
            rows.level,
            priors,
            arm_config;
            free_energy,
        )
        priors = (; v = fit.qv, w = fit.qw)
        prediction = predict(
            fit,
            test_rows.mean,
            test_rows.level,
            config.top_carrier,
        )
        push!(logpdfs, mean_predictive_logpdf(prediction, data.y_test))
        push!(logdets_w, _logdet_cov(fit.qw))
        push!(free_energies, fit.free_energy)
        push!(batch_sizes, length(indices))
    end
    return (; logpdfs, logdets_w, free_energies, batch_sizes, fit)
end

function streaming_full(arm, data, feature_maps, config; free_energy = false)
    rows = design(feature_maps, data.x_train)
    test_rows = design(feature_maps, data.x_test)
    priors = make_priors(config, data.x_train, data.y_train)
    fit = fit_arm(
        arm.method,
        data.y_train,
        rows.mean,
        rows.level,
        priors,
        _arm_config(arm, config);
        free_energy,
    )
    prediction = predict(
        fit,
        test_rows.mean,
        test_rows.level,
        config.top_carrier,
    )
    return (;
        logpdf = mean_predictive_logpdf(prediction, data.y_test),
        rmse = sqrt(mean(abs2.(prediction.mean .- data.y_test))),
        latent_mean_rmse = sqrt(mean(abs2.(
            prediction.mean .- aleatoric_mean.(data.x_test),
        ))),
        logdet_w = _logdet_cov(fit.qw),
        free_energy = fit.free_energy,
        n_obs = length(data.y_train),
        fit,
    )
end

function compute()
    ensure_outputs()
    config = streaming_config(smoke_mode())
    feature_maps = make_streaming_feature_maps(config)
    grid = collect(range(-3.0, 3.0; length = 241))
    grid_rows = design(feature_maps, grid)

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
            full = streaming_full(arm, data, feature_maps, config; free_energy = true)
            sequential = streaming_sequential(
                arm, data, order, feature_maps, config; free_energy = true,
            )
            test_rows = design(feature_maps, data.x_test)
            seq_prediction = predict(
                sequential.fit,
                test_rows.mean,
                test_rows.level,
                config.top_carrier,
            )
            seq_rmse = sqrt(mean(abs2.(seq_prediction.mean .- data.y_test)))
            seq_latent_mean_rmse = sqrt(mean(abs2.(
                seq_prediction.mean .- aleatoric_mean.(data.x_test),
            )))
            push!(rows_out, (;
                repetition, arm = arm.key,
                full_logpdf = full.logpdf,
                full_rmse = full.rmse,
                full_latent_mean_rmse = full.latent_mean_rmse,
                full_logdet_w = full.logdet_w,
                sequential_logpdf = last(sequential.logpdfs),
                sequential_rmse = seq_rmse,
                sequential_latent_mean_rmse = seq_latent_mean_rmse,
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
                    prediction = predict(
                        fit,
                        grid_rows.mean,
                        grid_rows.level,
                        config.top_carrier,
                    )
                    append!(panel_rows, prediction_panel_rows(
                        arm.key, mode, grid, prediction,
                    ))
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
        xlabel = "x", ylabel = "y", legend = false,
        ylims = (-4.5, 4.5), left_margin = 5Plots.mm,
    )
    scatter!(panel, train.x, train.y;
        color = :gray70, markersize = 2.2, markerstrokewidth = 0, label = "train")
    plot!(panel, panel_frame.x, panel_frame.pred_mean;
        ribbon = band, color = style.color, linestyle = style.linestyle,
        linewidth = 2.2, fillalpha = 0.2, label = label)
    plot!(panel, panel_frame.x, aleatoric_mean.(panel_frame.x);
        color = COLORS.truth, linestyle = :dashdot, linewidth = 1.6,
        label = "true mean")
    return panel
end

function streaming_variance_panel(panel_frame, style)
    variance_floor = 1e-4
    posterior_mean = max.(panel_frame.pred_total_variance, variance_floor)
    latent_variance = max.(
        panel_frame.pred_total_variance .- panel_frame.pred_noise_variance,
        variance_floor,
    )
    posterior_lower = max.(
        latent_variance .+ panel_frame.pred_noise_lower,
        variance_floor,
    )
    posterior_upper = max.(
        latent_variance .+ panel_frame.pred_noise_upper,
        variance_floor,
    )
    panel = plot(;
        xlabel = "x", ylabel = "predictive variance", yscale = :log10,
        legend = false,
        guidefontsize = 13, tickfontsize = 11,
        ylims = (variance_floor, 1e2), left_margin = 5Plots.mm,
        # A compact vector canvas keeps axes readable when LaTeX
        # embeds each PDF at half-column width. `dpi = 250` preserves
        # 1200-by-800 PNGs. Curve meanings are defined once in the caption.
        size = (480, 320), dpi = 250,
    )
    plot!(panel, panel_frame.x, posterior_lower;
        fillrange = posterior_upper, fillcolor = style.color,
        fillalpha = 0.18, color = :transparent, linewidth = 0,
        label = "95% CrI: total")
    plot!(panel, panel_frame.x, posterior_mean;
        color = style.color, linestyle = style.linestyle, linewidth = 2.4,
        label = "total = epistemic + aleatoric")
    plot!(panel, panel_frame.x, latent_variance;
        color = COLORS.mean_uncertainty, linestyle = :dot, linewidth = 1.8,
        label = "epistemic contribution")
    plot!(panel, panel_frame.x, max.(
            aleatoric_noise_variance.(panel_frame.x), variance_floor,
        );
        color = COLORS.truth, linestyle = :dashdot, linewidth = 1.8,
        label = "true aleatoric variance")
    return panel
end

_merge_arm_rows(filename, key, rows) = begin
    path = joinpath(RESULT_DIR, filename)
    existing = isfile(path) ?
        filter(row -> row.arm != key, CSV.read(path, DataFrame)) :
        DataFrame()
    CSV.write(path, vcat(existing, DataFrame(rows)))
end

# One-off refresh of a SINGLE arm (`--refresh-arm <key>`): reruns only that
# arm's fits and merges its rows into the runs/track/free-energy CSVs,
# leaving the other (possibly much slower) arms' recorded results untouched.
# Prediction-panel rows are refreshed too when the arm has panels.
function refresh_arm(key)
    ensure_outputs()
    config = streaming_config(smoke_mode())
    feature_maps = make_streaming_feature_maps(config)
    grid = collect(range(-3.0, 3.0; length = 241))
    grid_rows = design(feature_maps, grid)
    arm = STREAMING_ARMS[findfirst(arm -> arm.key == key, STREAMING_ARMS)]
    runs_rows = NamedTuple[]
    track_rows = NamedTuple[]
    fe_rows = NamedTuple[]
    panel_rows = NamedTuple[]
    for repetition in 1:config.repetitions
        data = aleatoric_data(config, StableRNG(config.seed + repetition - 1))
        order = randperm(StableRNG(1000 + repetition), length(data.y_train))
        full = streaming_full(arm, data, feature_maps, config; free_energy = true)
        sequential = streaming_sequential(
            arm, data, order, feature_maps, config; free_energy = true,
        )
        test_rows = design(feature_maps, data.x_test)
        seq_prediction = predict(
            sequential.fit,
            test_rows.mean,
            test_rows.level,
            config.top_carrier,
        )
        seq_rmse = sqrt(mean(abs2.(seq_prediction.mean .- data.y_test)))
        seq_latent_mean_rmse = sqrt(mean(abs2.(
            seq_prediction.mean .- aleatoric_mean.(data.x_test),
        )))
        push!(runs_rows, (;
            repetition, arm = arm.key,
            full_logpdf = full.logpdf,
            full_rmse = full.rmse,
            full_latent_mean_rmse = full.latent_mean_rmse,
            full_logdet_w = full.logdet_w,
            sequential_logpdf = last(sequential.logpdfs),
            sequential_rmse = seq_rmse,
            sequential_latent_mean_rmse = seq_latent_mean_rmse,
            streaming_penalty = last(sequential.logpdfs) - full.logpdf,
        ))
        for batch in 1:config.n_batches
            push!(track_rows, (;
                repetition, arm = arm.key, batch,
                logpdf = sequential.logpdfs[batch],
                logdet_w = sequential.logdets_w[batch],
            ))
        end
        for (iteration, value) in enumerate(full.free_energy)
            push!(fe_rows, (;
                repetition, arm = arm.key, mode = "full", batch = 0,
                iteration, free_energy = value, n_obs = full.n_obs,
            ))
        end
        for batch in 1:config.n_batches
            for (iteration, value) in enumerate(sequential.free_energies[batch])
                push!(fe_rows, (;
                    repetition, arm = arm.key, mode = "sequential", batch,
                    iteration, free_energy = value,
                    n_obs = sequential.batch_sizes[batch],
                ))
            end
        end
        if repetition == 1 && haskey(PANEL_STEMS, (arm.key, "full"))
            for (mode, fit) in (("full", full.fit), ("sequential", sequential.fit))
                prediction = predict(
                    fit,
                    grid_rows.mean,
                    grid_rows.level,
                    config.top_carrier,
                )
                append!(panel_rows, prediction_panel_rows(
                    arm.key, mode, grid, prediction,
                ))
            end
        end
        @printf("refreshed %s seed %d/%d\n", key, repetition, config.repetitions)
        flush(stdout)
    end
    _merge_arm_rows("streaming_hetero_runs.csv", key, runs_rows)
    _merge_arm_rows("streaming_hetero_track.csv", key, track_rows)
    _merge_arm_rows("streaming_hetero_free_energy.csv", key, fe_rows)
    isempty(panel_rows) ||
        _merge_arm_rows("streaming_hetero_panels.csv", key, panel_rows)
end

# Refresh only the representative-seed panel data. This is enough when a new
# diagnostic needs additional posterior quantities and avoids repeating the
# complete 20-seed metric and convergence study.
function refresh_panels()
    ensure_outputs()
    config = streaming_config(smoke_mode())
    feature_maps = make_streaming_feature_maps(config)
    grid = collect(range(-3.0, 3.0; length = 241))
    grid_rows = design(feature_maps, grid)
    data = aleatoric_data(config, StableRNG(config.seed))
    order = randperm(StableRNG(1001), length(data.y_train))
    panel_rows = NamedTuple[]
    for arm in STREAMING_ARMS
        haskey(PANEL_STEMS, (arm.key, "full")) || continue
        full = streaming_full(arm, data, feature_maps, config)
        sequential = streaming_sequential(arm, data, order, feature_maps, config)
        for (mode, fit) in (("full", full.fit), ("sequential", sequential.fit))
            prediction = predict(
                fit,
                grid_rows.mean,
                grid_rows.level,
                config.top_carrier,
            )
            append!(panel_rows, prediction_panel_rows(
                arm.key, mode, grid, prediction,
            ))
        end
        @printf("refreshed representative panels for %s\n", arm.key)
        flush(stdout)
    end
    CSV.write(
        joinpath(RESULT_DIR, "streaming_hetero_panels.csv"),
        DataFrame(panel_rows),
    )
end

# Bethe free-energy convergence: shows the streaming comparison probes fixed
# points, not truncation artifacts. The VMP trace is that arm's variational
# objective; the NGMP trace is the surrogate Bethe diagnostic evaluated
# through the moment-matched joint (μ, s) cluster marginals (cavity.jl).
function streaming_free_energy_figures(fe_frame)
    full_panel = plot(;
        xlabel = "iteration",
        ylabel = "Bethe free energy / observation",
        legend = :topright,
        left_margin = 5Plots.mm,
    )
    sequential_panel = plot(;
        xlabel = "batch",
        ylabel = "Bethe free energy / observation",
        legend = :topright,
        left_margin = 5Plots.mm,
    )
    drew = false
    for key in ("pvmp", "pvmp1", "ngmp")
        arm_rows = filter(row -> row.arm == key, fe_frame)
        isempty(arm_rows) && continue
        drew = true
        style = ARM_STYLES[key]
        # the first sweeps start orders of magnitude higher and would flatten
        # the plateau view
        start = min(3, maximum(arm_rows.iteration))
        full = sort(
            combine(
                groupby(
                    filter(
                        row -> row.mode == "full" && row.iteration >= start,
                        arm_rows,
                    ),
                    :iteration,
                ),
                [:free_energy, :n_obs] => ((f, n) -> mean(f ./ n)) => :center,
                [:free_energy, :n_obs] => ((f, n) -> ci95(f ./ n)) => :ci95,
            ),
            :iteration,
        )
        plot!(full_panel, full.iteration, full.center;
            ribbon = full.ci95, color = style.color,
            linestyle = style.linestyle, linewidth = 2.2, fillalpha = 0.15,
            label = "$(method_label(key)), full batch")

        # per batch: the CONVERGED (final-sweep) free energy of each
        # sequential fit, mean ± CI over seeds
        sequential = filter(row -> row.mode == "sequential", arm_rows)
        final_iteration = combine(
            groupby(sequential, [:batch]),
            :iteration => maximum => :iteration,
        )
        finals = sort(
            combine(
                groupby(
                    innerjoin(sequential, final_iteration; on = [:batch, :iteration]),
                    :batch,
                ),
                [:free_energy, :n_obs] => ((f, n) -> mean(f ./ n)) => :center,
                [:free_energy, :n_obs] => ((f, n) -> ci95(f ./ n)) => :ci95,
            ),
            :batch,
        )
        plot!(sequential_panel, finals.batch, finals.center;
            ribbon = finals.ci95, color = style.color,
            linestyle = style.linestyle, linewidth = 2.2, marker = :circle,
            markersize = 4, fillalpha = 0.15, xticks = finals.batch,
            label = "$(method_label(key)), converged per batch")
    end
    drew || return
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
    mean_kernel = config["mean_kernel"] == "matern32" ? "Matérn-3/2" : uppercase(config["mean_kernel"])
    mean_n_basis = config["mean_n_basis"]
    level_kernel = uppercase(config["level_kernel"])
    level_n_basis = config["level_n_basis"]
    signal_sd = config["signal_sd"]
    level_sd = config["level_sd"]

    summary_lines_out = String[
        "# Streaming heteroscedastic study (aleatoric benchmark)",
        "",
        "Mean path: $mean_kernel RFF-$mean_n_basis with `signal_sd = $signal_sd`; log-precision path: $level_kernel RFF-$level_n_basis with `level_sd = $level_sd`.",
        "Full batch vs sequential ($(config["n_batches"]) batches), $(config["repetitions"]) paired seeds.",
        "Mean RMSE is measured against the benchmark's known latent mean; observed RMSE uses noisy held-out targets.",
        "",
        "| arm | full NLL | full observed RMSE | full mean RMSE | sequential NLL | sequential observed RMSE | sequential mean RMSE | streaming penalty |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in arm_keys
        selected = filter(row -> row.arm == key, runs)
        line = string("| ", method_label(key),
            " | ", round(mean(.-selected.full_logpdf); digits = 3),
            " ± ", round(ci95(selected.full_logpdf); digits = 3),
            " | ", round(mean(selected.full_rmse); digits = 3),
            " ± ", round(ci95(selected.full_rmse); digits = 3),
            " | ", round(mean(selected.full_latent_mean_rmse); digits = 3),
            " ± ", round(ci95(selected.full_latent_mean_rmse); digits = 3),
            " | ", round(mean(.-selected.sequential_logpdf); digits = 3),
            " ± ", round(ci95(selected.sequential_logpdf); digits = 3),
            " | ", round(mean(selected.sequential_rmse); digits = 3),
            " ± ", round(ci95(selected.sequential_rmse); digits = 3),
            " | ", round(mean(selected.sequential_latent_mean_rmse); digits = 3),
            " ± ", round(ci95(selected.sequential_latent_mean_rmse); digits = 3),
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

    variance_panels = Dict{Tuple{String, String}, Any}()
    for ((key, mode), stem) in VARIANCE_PANEL_STEMS
        selected = sort(
            filter(row -> row.arm == key && row.mode == mode, panels_frame),
            :x,
        )
        panel = streaming_variance_panel(selected, ARM_STYLES[key])
        variance_panels[(key, mode)] = panel
        save_figure(panel, stem)
    end
    variance_preview = plot(
        plot(variance_panels[("pvmp", "full")]; title = "$(method_label("pvmp")), full batch"),
        plot(variance_panels[("pvmp", "sequential")]; title = "$(method_label("pvmp")), sequential"),
        plot(variance_panels[("ngmp", "full")]; title = "$(method_label("ngmp")), full batch"),
        plot(variance_panels[("ngmp", "sequential")]; title = "$(method_label("ngmp")), sequential"),
        layout = (2, 2), size = (1100, 750),
    )
    save_figure(variance_preview, "streaming_hetero_variances")

    # collapse trajectory: log det Σ_w over batches + full-batch references
    collapse = plot(;
        xlabel = "batch",
        ylabel = "\$\\log \\det \\Sigma_w\$",
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
    if "--refresh-panels" in ARGS
        refresh_panels()
        render()
        return
    end
    position = findfirst(==("--refresh-arm"), ARGS)
    if position !== nothing
        refresh_arm(ARGS[position + 1])
        # re-render only when the study's other artifacts are present
        isfile(joinpath(RESULT_DIR, "streaming_hetero_train.csv")) && render()
        return
    end
    render_only() || compute()
    render()
end

abspath(PROGRAM_FILE) == (@__FILE__) && streaming_main()
