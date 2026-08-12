#!/usr/bin/env julia

# NGMP-only diagnostic for the heteroscedastic model's latent mean fit.
#
# It separates three questions: widening only the mean prior, increasing only
# mean-path RFF capacity, and changing that larger bank to Matérn-3/2. The
# log-precision path is intentionally frozen at the historical 32 RBF features
# and prior scales in every preset. No VMP model is built or fitted here.

include(joinpath(@__DIR__, "streaming_hetero.jl"))

using CSV
using DataFrames
using Plots
using Printf
using Statistics

const MEAN_CAPACITY_PRESETS = (
    (
        key = "baseline",
        label = "NGMP, mean RBF-32",
        kernel = "rbf",
        n_basis = 32,
        signal_sd = 1.0,
        color = :gray45,
        linestyle = :dash,
        show_panel = true,
    ),
    (
        key = "wide32",
        label = "NGMP, mean RBF-32, wide prior",
        kernel = "rbf",
        n_basis = 32,
        signal_sd = 2.0,
        color = :darkorange,
        linestyle = :dot,
        show_panel = false,
    ),
    (
        key = "rbf128",
        label = "NGMP, mean RBF-128",
        kernel = "rbf",
        n_basis = 128,
        signal_sd = 2.0,
        color = :mediumorchid,
        linestyle = :solid,
        show_panel = true,
    ),
    (
        key = "matern128",
        label = "NGMP, mean Matérn-3/2 RFF-128",
        kernel = "matern32",
        n_basis = 128,
        signal_sd = 2.0,
        color = COLORS.ngmp,
        linestyle = :solid,
        show_panel = true,
    ),
)

function argument_int(flag, default)
    position = findfirst(==(flag), ARGS)
    return isnothing(position) ? default : parse(Int, ARGS[position + 1])
end

function mean_capacity_config(base, preset)
    mean_n_basis = if base.smoke
        preset.n_basis == 32 ? base.n_basis : min(preset.n_basis, 24)
    else
        preset.n_basis
    end
    return merge(base, (;
        mean_n_basis,
        mean_kernel = preset.kernel,
        signal_sd = preset.signal_sd,
        # Freeze the exponentiated variance pathway.
        level_n_basis = base.n_basis,
        level_kernel = "rbf",
        level_lengthscale = base.aleatoric_lengthscale,
        level_feature_seed = base.feature_seed,
        level_sd = 1.6,
        anchor_sd = 1.0,
    ))
end

latent_mean_rmse(prediction, xs) =
    sqrt(mean(abs2.(prediction.mean .- aleatoric_mean.(xs))))

function evaluate_fit(fit, maps, xs, ys, top_carrier)
    rows = design(maps, xs)
    prediction = predict(fit, rows.mean, rows.level, top_carrier)
    return (;
        prediction,
        observed_rmse = sqrt(mean(abs2.(prediction.mean .- ys))),
        latent_rmse = latent_mean_rmse(prediction, xs),
    )
end

function evaluate_grid(fit, maps, grid, top_carrier)
    rows = design(maps, grid)
    prediction = predict(fit, rows.mean, rows.level, top_carrier)
    core = findall(abs.(grid) .<= 2)
    return (;
        prediction,
        rmse = latent_mean_rmse(prediction, grid),
        core_rmse = latent_mean_rmse(
            (mean = prediction.mean[core],),
            grid[core],
        ),
    )
end

function compute_mean_capacity()
    ensure_outputs()
    base = streaming_config(smoke_mode())
    repetitions = argument_int(
        "--repetitions",
        1,
    )
    iterations = argument_int(
        "--iterations",
        base.smoke ? 10 : 240,
    )
    vary_mean_feature_seed = "--vary-mean-feature-seed" in ARGS
    arm = (
        key = "ngmp",
        method = "NGMP-cavity",
        iterations,
        projection_iterations = 100,
    )
    grid = collect(range(-3.0, 3.0; length = 241))
    runs = NamedTuple[]
    panels = NamedTuple[]
    train = NamedTuple[]

    for repetition in 1:repetitions
        data_seed = vary_mean_feature_seed ? base.seed : base.seed + repetition - 1
        data = aleatoric_data(base, StableRNG(data_seed))
        order = randperm(StableRNG(1000 + repetition), length(data.y_train))
        if repetition == 1
            append!(train, [(; x, y) for (x, y) in zip(data.x_train, data.y_train)])
        end
        for preset in MEAN_CAPACITY_PRESETS
            config = mean_capacity_config(base, preset)
            if vary_mean_feature_seed
                config = merge(config, (;
                    mean_feature_seed = base.mean_feature_seed + repetition - 1,
                ))
            end
            maps = make_streaming_feature_maps(config)
            full = streaming_full(arm, data, maps, config; free_energy = false)
            sequential_elapsed = @elapsed sequential = streaming_sequential(
                arm,
                data,
                order,
                maps,
                config;
                free_energy = false,
            )
            full_test = evaluate_fit(
                full.fit,
                maps,
                data.x_test,
                data.y_test,
                config.top_carrier,
            )
            sequential_test = evaluate_fit(
                sequential.fit,
                maps,
                data.x_test,
                data.y_test,
                config.top_carrier,
            )
            full_grid = evaluate_grid(full.fit, maps, grid, config.top_carrier)
            sequential_grid = evaluate_grid(
                sequential.fit,
                maps,
                grid,
                config.top_carrier,
            )
            push!(runs, (;
                repetition,
                preset = preset.key,
                mean_kernel = preset.kernel,
                mean_n_basis = config.mean_n_basis,
                signal_sd = preset.signal_sd,
                level_kernel = config.level_kernel,
                level_n_basis = config.level_n_basis,
                level_sd = config.level_sd,
                iterations,
                full_nll = -full.logpdf,
                full_observed_rmse = full_test.observed_rmse,
                full_latent_mean_rmse = full_test.latent_rmse,
                full_grid_mean_rmse = full_grid.rmse,
                full_core_grid_mean_rmse = full_grid.core_rmse,
                full_seconds = full.fit.elapsed,
                sequential_nll = -last(sequential.logpdfs),
                sequential_observed_rmse = sequential_test.observed_rmse,
                sequential_latent_mean_rmse = sequential_test.latent_rmse,
                sequential_grid_mean_rmse = sequential_grid.rmse,
                sequential_core_grid_mean_rmse = sequential_grid.core_rmse,
                sequential_seconds = sequential_elapsed,
            ))
            if repetition == 1 && preset.show_panel
                for (mode, prediction) in (
                    ("full", full_grid.prediction),
                    ("sequential", sequential_grid.prediction),
                )
                    append!(panels, [
                        (;
                            preset = preset.key,
                            mode,
                            x = grid[index],
                            pred_mean = prediction.mean[index],
                            pred_total_variance = prediction.total_variance[index],
                        )
                        for index in eachindex(grid)
                    ])
                end
            end
            @printf(
                "mean-capacity rep=%d/%d %-10s full latent=%.4f sequential latent=%.4f\n",
                repetition,
                repetitions,
                preset.key,
                full_test.latent_rmse,
                sequential_test.latent_rmse,
            )
            flush(stdout)
        end
    end

    CSV.write(joinpath(RESULT_DIR, "ngmp_mean_capacity_runs.csv"), DataFrame(runs))
    CSV.write(joinpath(RESULT_DIR, "ngmp_mean_capacity_panels.csv"), DataFrame(panels))
    CSV.write(joinpath(RESULT_DIR, "ngmp_mean_capacity_train.csv"), DataFrame(train))
    write_config("ngmp_mean_capacity", (;
        smoke = base.smoke,
        seed = base.seed,
        repetitions,
        iterations,
        aleatoric_samples = base.aleatoric_samples,
        n_batches = base.n_batches,
        mean_lengthscale = base.mean_lengthscale,
        level_kernel = "rbf",
        level_n_basis = base.n_basis,
        level_sd = base.level_sd,
        presets = [preset.key for preset in MEAN_CAPACITY_PRESETS],
        sweep_axis = vary_mean_feature_seed ? "mean_feature_seed" : "data_seed",
    ))
end

function mean_capacity_panel(frame, train, preset, mode)
    band = 1.96 .* sqrt.(frame.pred_total_variance)
    panel = plot(;
        xlabel = "x",
        ylabel = "y",
        legend = :topright,
        ylims = (-4.5, 4.5),
        left_margin = 5Plots.mm,
    )
    scatter!(
        panel,
        train.x,
        train.y;
        color = :gray75,
        markersize = 1.6,
        markerstrokewidth = 0,
        label = "train",
    )
    plot!(
        panel,
        frame.x,
        frame.pred_mean;
        ribbon = band,
        color = preset.color,
        linestyle = preset.linestyle,
        linewidth = 2.2,
        fillalpha = 0.18,
        label = "$(preset.label), $mode",
    )
    plot!(
        panel,
        frame.x,
        aleatoric_mean.(frame.x);
        color = COLORS.truth,
        linestyle = :dashdot,
        linewidth = 1.6,
        label = "true mean",
    )
    return panel
end

function render_mean_capacity()
    ensure_outputs()
    config = read_config("ngmp_mean_capacity")
    runs = CSV.read(joinpath(RESULT_DIR, "ngmp_mean_capacity_runs.csv"), DataFrame)
    panels = CSV.read(joinpath(RESULT_DIR, "ngmp_mean_capacity_panels.csv"), DataFrame)
    train = CSV.read(joinpath(RESULT_DIR, "ngmp_mean_capacity_train.csv"), DataFrame)
    level_sd = config["level_sd"]
    level_n_basis = config["level_n_basis"]
    iterations = config["iterations"]
    repetitions = config["repetitions"]
    sweep_axis = replace(config["sweep_axis"], "_" => "-")
    lines = String[
        "# NGMP-only heteroscedastic mean-capacity diagnostic",
        "",
        "The log-precision path is frozen at $level_n_basis RBF features with `level_sd = $level_sd`.",
        "Results use $iterations NGMP sweeps over $repetitions $sweep_axis setting(s).",
        "Latent-mean RMSE is measured against the known benchmark mean; the core grid is `|x| <= 2`.",
        "",
        "| preset | full NLL | full latent-mean RMSE | full core-grid RMSE | sequential NLL | sequential latent-mean RMSE | sequential core-grid RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for preset in MEAN_CAPACITY_PRESETS
        selected = filter(row -> row.preset == preset.key, runs)
        push!(lines, string(
            "| ", preset.label,
            " | ", round(mean(selected.full_nll); digits = 3),
            " | ", round(mean(selected.full_latent_mean_rmse); digits = 3),
            " | ", round(mean(selected.full_core_grid_mean_rmse); digits = 3),
            " | ", round(mean(selected.sequential_nll); digits = 3),
            " | ", round(mean(selected.sequential_latent_mean_rmse); digits = 3),
            " | ", round(mean(selected.sequential_core_grid_mean_rmse); digits = 3),
            " |",
        ))
    end
    write_summary("ngmp_mean_capacity", lines)

    figures = Any[]
    panel_presets = filter(preset -> preset.show_panel, MEAN_CAPACITY_PRESETS)
    for preset in panel_presets, mode in ("full", "sequential")
        selected = sort(
            filter(row -> row.preset == preset.key && row.mode == mode, panels),
            :x,
        )
        push!(figures, plot(
            mean_capacity_panel(selected, train, preset, mode);
            title = "$(preset.label), $mode",
        ))
    end
    save_figure(
        plot(
            figures...;
            layout = (length(panel_presets), 2),
            size = (1100, 360 * length(panel_presets)),
        ),
        "ngmp_mean_capacity_predictions",
    )
end

function ngmp_mean_capacity_main()
    render_only() || compute_mean_capacity()
    render_mean_capacity()
end

abspath(PROGRAM_FILE) == (@__FILE__) && ngmp_mean_capacity_main()
