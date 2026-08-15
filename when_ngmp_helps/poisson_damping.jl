#!/usr/bin/env julia

# Damping diagnostic for the Poisson state-space (sunspot) study.
#
# `poisson_state_space.jl` runs NGMP with the damped natural-parameter update
# (alpha = 0.5, beta = 0.2, see its `compute()` config). This script asks what
# happens to the same model, the same 20 masks, and the same four holdout
# fractions when the natural-gradient fixed-point map is iterated undamped
# (alpha = 1) or with pure natural damping (alpha = 0.25, beta = 0), and it
# records RxInfer's Bethe free energy of the model (exact PoissonExp energies
# under the current Gaussian beliefs) at every variational iteration. It
# reuses the model, rules, initialization, data loader, and mask construction of
# `poisson_state_space.jl` by including it (that file only runs `main()` when it
# is the program file).
#
# The paper figure comes from a second, synthetic scenario. On the sunspot masks
# all three settings converge (the undamped map fastest), because the counts
# are moderate and the initialization is close to the fixed point. The outer
# map stops being a contraction only when the latent log-rate makes long
# excursions to extreme values, so the figure uses the random-walk series of
# the earlier synthetic damping figure (`poisson_surrogate_model.jl` at the
# repository root: `Random.seed!(seed)`, z_k = z_{k-1} + N(0, 0.1), all counts
# observed) at four chain lengths; longer chains wander further, and seed 42
# with N = 1000 is exactly the earlier series (log-rates down to -20, 922 zero
# counts). Fits that throw (a momentum step leaving the natural domain) keep
# the free energy of the completed sweeps through `catch_exception = true`.
#
# Artifacts: paper panels `poisson_damping_synthetic_n{100,250,500,1000}.pdf`
# with the preview `poisson_damping_synthetic_free_energy.{pdf,png}`;
# supplementary sunspot panels `poisson_damping_heldout_{5,10,20,50}.pdf` with
# the preview `poisson_damping_free_energy.{pdf,png}`;
# `results/poisson_damping_free_energy.csv` (sunspot traces),
# `results/poisson_damping_synthetic.csv` (synthetic traces),
# `results/poisson_damping_synthetic_fits.csv` (one row per synthetic fit),
# `results/poisson_damping_config.toml`, and `results/poisson_damping_summary.md`.
#
# Env knobs: WHEN_NGMP_POISSON_REPETITIONS (masks and synthetic seeds, default
# 20), WHEN_NGMP_DAMPING_ITERATIONS (sunspot outer sweeps, default 60),
# WHEN_NGMP_DAMPING_SYNTHETIC_ITERATIONS (synthetic outer sweeps, default 200),
# plus the shared
# WHEN_NGMP_SMOKE / WHEN_NGMP_RENDER_ONLY / WHEN_NGMP_OUTPUT_DIR.

include(joinpath(@__DIR__, "poisson_state_space.jl"))

# The three outer-loop settings compared. `damped50m20` must match the setting
# used for every NGMP sunspot result in `poisson_state_space.jl` (asserted
# against its config TOML in `damping_compute`).
const DAMPING_ARMS = (
    (key = "undamped", alpha = 1.0, beta = 0.0),
    (key = "damped25", alpha = 0.25, beta = 0.0),
    (key = "damped50m20", alpha = 0.5, beta = 0.2),
)

const DAMPING_ARM_KEYS = Tuple(arm.key for arm in DAMPING_ARMS)

const DAMPING_ARM_STYLES = (
    undamped = (color = :darkorange, linestyle = :solid),
    damped25 = (color = :dodgerblue, linestyle = :solid),
    damped50m20 = (color = :seagreen, linestyle = :dash),
)

function damping_arm_label(arm)
    if arm.key == "undamped"
        return "undamped (α = 1)"
    elseif arm.beta == 0
        return "damped (α = $(arm.alpha))"
    else
        return "damped (α = $(arm.alpha), β = $(arm.beta))"
    end
end

damping_arm(key) = only(filter(arm -> arm.key == key, collect(DAMPING_ARMS)))

function damping_config()
    smoke = smoke_mode()
    return (
        smoke,
        seed = 42,
        smoke_length = 144,
        process_variance = 0.1,
        initial_mean = 0.0,
        initial_variance = 10.0,
        iterations = smoke ? 6 : parse(
            Int,
            get(ENV, "WHEN_NGMP_DAMPING_ITERATIONS", "60"),
        ),
        repetitions = smoke ? 2 : parse(
            Int,
            get(ENV, "WHEN_NGMP_POISSON_REPETITIONS", "20"),
        ),
        holdout_fractions = [0.05, 0.10, 0.20, 0.50],
        holdout_seed = 42,
        synthetic_seed = 42,
        synthetic_lengths = smoke ? [20, 40] : [100, 250, 500, 1000],
        synthetic_iterations = smoke ? 6 : parse(
            Int,
            get(ENV, "WHEN_NGMP_DAMPING_SYNTHETIC_ITERATIONS", "200"),
        ),
        arms = collect(string(arm.key) for arm in DAMPING_ARMS),
        arm_alphas = collect(arm.alpha for arm in DAMPING_ARMS),
        arm_betas = collect(arm.beta for arm in DAMPING_ARMS),
    )
end

# The main study's config is the reference for the "experiment setting" arm.
function assert_experiment_arm_matches_study()
    path = joinpath(RESULT_DIR, "poisson_state_space_config.toml")
    isfile(path) || return nothing
    study = read_config("poisson_state_space")
    arm = damping_arm("damped50m20")
    (study["damping_alpha"] == arm.alpha && study["damping_beta"] == arm.beta) ||
        error(
            "damped50m20 arm (alpha = $(arm.alpha), beta = $(arm.beta)) does not " *
            "match poisson_state_space config (alpha = $(study["damping_alpha"]), " *
            "beta = $(study["damping_beta"]))",
        )
    return nothing
end

function run_damping_repetition(counts, config, repetition)
    rows = NamedTuple[]
    mask_seed = config.holdout_seed + repetition - 1
    for fraction in config.holdout_fractions
        held_out = holdout_indices(length(counts), fraction, mask_seed)
        observed = trues(length(counts))
        observed[held_out] .= false
        observed_indices = findall(observed)
        for arm in DAMPING_ARMS
            fit = fit_ngmp(
                counts,
                observed_indices,
                merge(config, (damping_alpha = arm.alpha, damping_beta = arm.beta)),
            )
            for (iteration, value) in enumerate(fit.free_energy)
                push!(rows, (;
                    repetition,
                    mask_seed,
                    holdout_fraction = fraction,
                    holdout_pct = round(Int, 100fraction),
                    n_observed = length(observed_indices),
                    arm = arm.key,
                    alpha = arm.alpha,
                    beta = arm.beta,
                    iteration,
                    bethe_free_energy = value,
                    per_observed_count = value / length(observed_indices),
                ))
            end
        end
    end
    @printf("completed Poisson damping mask seed %d/%d\n", repetition, config.repetitions)
    return rows
end

# Synthetic random walk of the earlier synthetic figure, one series per
# (chain length, seed). `Random.seed!(seed)` followed by `randn` and the Poisson
# draws is the draw order of `poisson_surrogate_model.jl`, so seed 42 with
# N = 1000 reproduces that series exactly, and the shorter chains are prefixes
# of the longer ones for the same seed.
function synthetic_random_walk(config, chain_length, seed)
    Random.seed!(seed)
    latent = cumsum(sqrt(config.process_variance) .* randn(chain_length))
    counts = [rand(Poisson(exp(z))) for z in latent]
    return latent, counts
end

# RxInfer stores a caught exception as `(exception, backtrace)`; keep only the
# first line of the exception message, truncated, so the CSV stays readable.
function short_error(error)
    exception = error isa Tuple ? first(error) : error
    message = first(split(sprint(showerror, exception), '\n'))
    return length(message) > 120 ? first(message, 117) * "..." : message
end

# One fit: returns the per-sweep trace rows and one fit-level row. A fit that
# throws (a momentum step leaving the natural domain, for example) is recorded
# with `status = "failed: ..."`; its free energy covers the completed sweeps.
function run_synthetic_fit(config, chain_length, seed, latent, counts, arm)
    fit_config = merge(config, (
        damping_alpha = arm.alpha,
        damping_beta = arm.beta,
        iterations = config.synthetic_iterations,
    ))
    fit = fit_ngmp(counts, collect(1:chain_length), fit_config;
                   catch_exception = true)
    failed = !isnothing(fit.error)
    status = failed ? "failed: " * short_error(fit.error) : "ok"
    free_energy = fit.free_energy
    trace_rows = [
        (;
            chain_length,
            seed,
            arm = arm.key,
            alpha = arm.alpha,
            beta = arm.beta,
            iteration,
            bethe_free_energy = value,
            per_observation = value / chain_length,
        )
        for (iteration, value) in enumerate(free_energy)
    ]
    tail = free_energy[max(1, end - 9):end]
    fit_row = (;
        chain_length,
        seed,
        arm = arm.key,
        alpha = arm.alpha,
        beta = arm.beta,
        zero_counts = count(==(0), counts),
        latent_min = minimum(latent),
        latent_max = maximum(latent),
        iterations = config.synthetic_iterations,
        completed_sweeps = length(free_energy),
        status,
        first_free_energy = isempty(free_energy) ? NaN : first(free_energy),
        peak_free_energy = isempty(free_energy) ? NaN : maximum(free_energy),
        final_free_energy = isempty(free_energy) ? NaN : last(free_energy),
        last10_range = isempty(tail) ? NaN : maximum(tail) - minimum(tail),
        settle_sweep = isempty(free_energy) ? missing :
            something(settle_iteration(free_energy ./ chain_length), missing),
        rmse = failed ? NaN : sqrt(mean((fit.mean .- latent) .^ 2)),
    )
    return trace_rows, fit_row
end

function synthetic_compute(config)
    seeds = collect(config.synthetic_seed:(config.synthetic_seed + config.repetitions - 1))
    # Data first, serially, so the seeded global RNG is never touched by threads.
    series = Dict(
        (chain_length, seed) => synthetic_random_walk(config, chain_length, seed)
        for chain_length in config.synthetic_lengths, seed in seeds
    )
    jobs = [(chain_length, seed) for chain_length in config.synthetic_lengths, seed in seeds]
    outputs = Vector{Any}(undef, length(jobs))
    Threads.@threads for index in eachindex(jobs)
        chain_length, seed = jobs[index]
        latent, counts = series[(chain_length, seed)]
        outputs[index] = [
            run_synthetic_fit(config, chain_length, seed, latent, counts, arm)
            for arm in DAMPING_ARMS
        ]
        @printf("completed synthetic chain N = %d, seed %d\n", chain_length, seed)
    end
    trace_rows = reduce(vcat, [rows for fits in outputs for (rows, _) in fits])
    fit_rows = [row for fits in outputs for (_, row) in fits]
    CSV.write(
        joinpath(RESULT_DIR, "poisson_damping_synthetic.csv"),
        DataFrame(trace_rows),
    )
    CSV.write(
        joinpath(RESULT_DIR, "poisson_damping_synthetic_fits.csv"),
        DataFrame(fit_rows),
    )
end

function damping_compute()
    ensure_outputs()
    config = damping_config()
    assert_experiment_arm_matches_study()
    years, counts, data_source = load_counts(config)

    synthetic_compute(config)

    outputs = Vector{Any}(undef, config.repetitions)
    outputs[1] = run_damping_repetition(counts, config, 1)
    Threads.@threads for repetition in 2:config.repetitions
        outputs[repetition] = run_damping_repetition(counts, config, repetition)
    end
    rows = reduce(vcat, outputs)

    CSV.write(
        joinpath(RESULT_DIR, "poisson_damping_free_energy.csv"),
        DataFrame(rows),
    )
    write_config("poisson_damping", merge(config, (;
        data_source,
        n_observations = length(counts),
    )))
end

function damping_trace_summary(frame, fraction, key)
    selected = filter(
        row -> row.holdout_fraction == fraction && row.arm == key,
        frame,
    )
    return sort(combine(
        groupby(selected, :iteration),
        :per_observed_count => mean => :center,
        :per_observed_count => ci95 => :ci95,
    ), :iteration)
end

# Sweeps shown in the paper panels; the fits run longer (see `iterations`) so
# that the settle sweeps in the summary are measured on the full budget.
const DAMPING_PLOT_SWEEPS = 20

function damping_panel(frame, fraction; legend = :topright, max_sweep = DAMPING_PLOT_SWEEPS)
    traces = [
        (arm, filter(row -> row.iteration <= max_sweep,
                     damping_trace_summary(frame, fraction, arm.key)))
        for arm in DAMPING_ARMS
    ]
    positive = all(all(trace.center .> 0) for (_, trace) in traces)
    # log axis only when the traces span more than a decade; otherwise the
    # log ticks (10^0.40, ...) obscure a narrow range
    spread = positive ?
        maximum(maximum(trace.center) for (_, trace) in traces) /
        minimum(minimum(trace.center) for (_, trace) in traces) : 1.0
    use_log = positive && spread > 10
    panel = plot(;
        xlabel = "variational iteration",
        ylabel = "Bethe free energy / observed count",
        yscale = use_log ? :log10 : :identity,
        title = "$(round(Int, 100fraction))% held out",
        titlefontsize = 10,
        legend,
        legendfontsize = 7,
        size = (480, 320),
        dpi = 250,
        left_margin = 4Plots.mm,
        bottom_margin = 3Plots.mm,
    )
    for (arm, trace) in traces
        style = getfield(DAMPING_ARM_STYLES, Symbol(arm.key))
        ribbon = use_log ? min.(trace.ci95, trace.center .* 0.999) : trace.ci95
        plot!(
            panel,
            trace.iteration,
            trace.center;
            ribbon,
            color = style.color,
            linestyle = style.linestyle,
            fillalpha = 0.12,
            linewidth = 2.0,
            label = damping_arm_label(arm),
        )
    end
    return panel
end

# First outer sweep after which the per-count free energy of a single mask
# changes by less than `tol` relative to its current value for the rest of the
# budget; `nothing` when the trace never settles within the budget.
function settle_iteration(values; tol = 1e-4)
    n = length(values)
    for start in 2:n
        settled = true
        for t in start:n
            reference = max(abs(values[t - 1]), eps())
            if abs(values[t] - values[t - 1]) / reference >= tol
                settled = false
                break
            end
        end
        settled && return start
    end
    return nothing
end

function damping_summary_lines(frame, config)
    tail = min(10, config["iterations"])
    lines = [
        "# Poisson damping diagnostic",
        "",
        "## Sunspot masks (supplementary panels)",
        "",
        "Data source: $(config["data_source"]); $(config["repetitions"]) masks; " *
        "$(config["iterations"]) variational iterations per fit; Bethe free energy per " *
        "observed count.",
        "",
        "Settle sweep: first sweep from which the relative sweep-to-sweep change of a " *
        "mask's per-count free energy stays below 1e-4; reported as the mean over the " *
        "masks that settle, with the number of settling masks. Amplitude: mean " *
        "peak-to-peak range of the per-count free energy over the last $tail sweeps.",
        "",
        "| Held out | Arm | α | β | settle sweep (masks settled) | last-$tail amplitude |",
        "|---:|:---|---:|---:|:---|---:|",
    ]
    for fraction in sort(unique(frame.holdout_fraction)), arm in DAMPING_ARMS
        selected = filter(
            row -> row.holdout_fraction == fraction && row.arm == arm.key,
            frame,
        )
        settles = Int[]
        amplitudes = Float64[]
        n_masks = 0
        for group in groupby(selected, :repetition)
            values = sort(group, :iteration).per_observed_count
            n_masks += 1
            settle = settle_iteration(values)
            isnothing(settle) || push!(settles, settle)
            last_values = values[max(1, end - tail + 1):end]
            push!(amplitudes, maximum(last_values) - minimum(last_values))
        end
        settle_text = isempty(settles) ? "never ($n_masks masks)" :
            @sprintf("%.1f (%d/%d)", mean(settles), length(settles), n_masks)
        push!(lines, @sprintf(
            "| %d%% | %s | %.2f | %.1f | %s | %.3g |",
            round(Int, 100fraction),
            arm.key,
            arm.alpha,
            arm.beta,
            settle_text,
            mean(amplitudes),
        ))
    end
    return lines
end

# Reference level per (chain length, seed): the Bethe free energy per
# observation reached after the full budget by the α = 0.25 setting, which
# converged on every seed. Panels show the excess of each trace over it.
function synthetic_reference(fits)
    reference = Dict{Tuple{Int, Int}, Float64}()
    for row in eachrow(fits)
        row.arm == "damped25" && row.status == "ok" || continue
        reference[(row.chain_length, row.seed)] = row.final_free_energy / row.chain_length
    end
    return reference
end

const EXCESS_FLOOR = 1e-8

# Synthetic panel: per setting, the geometric mean across seeds of the Bethe
# free energy above the converged value of the same seed (nats per
# observation), with a 95% confidence band (mean ± 1.96 SE of log10, symmetric
# on the log axis). Traces of failed fits stop at the failure iteration, so
# later iterations average over the seeds that reached them.
function synthetic_panel(traces, fits, chain_length; legend = :topright)
    reference = synthetic_reference(fits)
    panel = plot(;
        xlabel = "variational iteration",
        ylabel = "BFE",
        yscale = :log10,
        title = "N = $chain_length",
        titlefontsize = 10,
        legend,
        legendfontsize = 7,
        size = (480, 320),
        dpi = 250,
        left_margin = 4Plots.mm,
        bottom_margin = 3Plots.mm,
    )
    for arm in DAMPING_ARMS
        selected = filter(
            row -> row.chain_length == chain_length && row.arm == arm.key &&
                   haskey(reference, (row.chain_length, row.seed)),
            traces,
        )
        isempty(selected) && continue
        excess = [
            max(row.per_observation - reference[(row.chain_length, row.seed)],
                EXCESS_FLOOR)
            for row in eachrow(selected)
        ]
        rows = DataFrame(iteration = selected.iteration, log_excess = log10.(excess))
        summary = sort(combine(
            groupby(rows, :iteration),
            :log_excess => mean => :center,
            :log_excess => ci95 => :half_width,
        ), :iteration)
        center = 10 .^ summary.center
        lower = 10 .^ (summary.center .- summary.half_width)
        upper = 10 .^ (summary.center .+ summary.half_width)
        style = getfield(DAMPING_ARM_STYLES, Symbol(arm.key))
        # failure counts are reported in the summary and the paper caption,
        # not in the legend
        label = damping_arm_label(arm)
        plot!(
            panel,
            summary.iteration,
            center;
            ribbon = (center .- lower, upper .- center),
            color = style.color,
            linestyle = style.linestyle,
            fillalpha = 0.15,
            linewidth = 2.2,
            label,
        )
    end
    return panel
end

# Direct reproduction of the earlier synthetic figure: seed 42, the longest
# chain, absolute free energy per observation, all sweeps.
function synthetic_seed_panel(traces, fits, chain_length, seed)
    panel = plot(;
        xlabel = "variational iteration",
        ylabel = "Bethe free energy / observation",
        yscale = :log10,
        title = "N = $chain_length, seed $seed",
        titlefontsize = 10,
        legend = :topright,
        legendfontsize = 7,
        size = (640, 360),
        dpi = 250,
        left_margin = 4Plots.mm,
        bottom_margin = 3Plots.mm,
    )
    for arm in DAMPING_ARMS
        selected = sort(filter(
            row -> row.chain_length == chain_length && row.seed == seed &&
                   row.arm == arm.key,
            traces,
        ), :iteration)
        isempty(selected) && continue
        status = only(filter(
            row -> row.chain_length == chain_length && row.seed == seed &&
                   row.arm == arm.key,
            fits,
        ).status)
        style = getfield(DAMPING_ARM_STYLES, Symbol(arm.key))
        label = damping_arm_label(arm) *
            (status == "ok" ? "" : " (failed at iteration $(nrow(selected) + 1))")
        plot!(
            panel,
            selected.iteration,
            selected.per_observation;
            color = style.color,
            linestyle = style.linestyle,
            linewidth = 2.0,
            label,
        )
    end
    return panel
end

function synthetic_summary_lines(fits, config)
    reference = synthetic_reference(fits)
    lines = [
        "",
        "## Synthetic random walk (paper figure)",
        "",
        @sprintf(
            "z_k = z_{k-1} + N(0, %.2f), all counts observed, %d outer sweeps, seeds %d-%d (`Random.seed!(seed)`, the draw order of `poisson_surrogate_model.jl`; seed %d with N = 1000 is the earlier synthetic series). Per seed and setting: settled = the free energy per observation changes by less than 1e-4 (relative) from some sweep on; oscillating = never settles within the budget but ends within 10x of the same seed's α = 0.25 final value; diverged = ends more than 10x above it; failed = the fit threw (a message left the natural domain). Medians are over the seeds that finished. RMSE is against the true latent log-rate.",
            config["process_variance"], config["synthetic_iterations"],
            minimum(fits.seed), maximum(fits.seed), config["synthetic_seed"],
        ),
        "",
        "| N | Arm | α | β | settled / oscillating / diverged / failed | median settle sweep | median peak F | median final F | median RMSE | log-rate min (median over seeds) | zero counts (median) |",
        "|---:|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for chain_length in sort(unique(fits.chain_length))
        for arm in DAMPING_ARMS
            selected = filter(
                row -> row.chain_length == chain_length && row.arm == arm.key,
                fits,
            )
            isempty(selected) && continue
            failed = count(!=("ok"), selected.status)
            finished = filter(row -> row.status == "ok", selected)
            ratio(row) = row.final_free_energy / row.chain_length /
                get(reference, (row.chain_length, row.seed), NaN)
            diverged = count(row -> ratio(row) > 10, eachrow(finished))
            settled = count(row -> !ismissing(row.settle_sweep) && ratio(row) <= 10,
                            eachrow(finished))
            oscillating = nrow(finished) - settled - diverged
            settles = collect(skipmissing(finished.settle_sweep))
            push!(lines, @sprintf(
                "| %d | %s | %.2f | %.1f | %d / %d / %d / %d | %s | %.4g | %.4g | %s | %.1f | %d |",
                chain_length, arm.key, arm.alpha, arm.beta,
                settled, oscillating, diverged, failed,
                isempty(settles) ? "never" : @sprintf("%.0f", median(settles)),
                isempty(finished) ? NaN : median(finished.peak_free_energy),
                isempty(finished) ? NaN : median(finished.final_free_energy),
                isempty(finished) ? "-" : @sprintf("%.3f", median(finished.rmse)),
                median(selected.latent_min),
                round(Int, median(selected.zero_counts)),
            ))
        end
    end
    seed = config["synthetic_seed"]
    longest = maximum(fits.chain_length)
    append!(lines, [
        "",
        "Seed $seed, N = $longest (the earlier synthetic series):",
        "",
        "| Arm | status | peak F / obs | final F / obs | RMSE | settle sweep |",
        "|:---|:---|---:|---:|---:|---:|",
    ])
    for row in eachrow(filter(row -> row.chain_length == longest && row.seed == seed, fits))
        push!(lines, @sprintf(
            "| %s | %s | %.4g | %.4g | %s | %s |",
            row.arm, row.status,
            row.peak_free_energy / longest, row.final_free_energy / longest,
            isnan(row.rmse) ? "-" : @sprintf("%.3f", row.rmse),
            ismissing(row.settle_sweep) ? "never" : string(row.settle_sweep),
        ))
    end
    return lines
end

function damping_render()
    ensure_outputs()
    config = read_config("poisson_damping")
    frame = CSV.read(
        joinpath(RESULT_DIR, "poisson_damping_free_energy.csv"), DataFrame,
    )
    fractions = sort(unique(frame.holdout_fraction))
    panels = [
        damping_panel(frame, fraction; legend = index == 1 ? :topright : false)
        for (index, fraction) in enumerate(fractions)
    ]
    for (fraction, panel) in zip(fractions, panels)
        save_pdf(panel, "poisson_damping_heldout_$(round(Int, 100fraction))")
    end
    save_figure(
        plot(panels...; layout = (2, 2), size = (1060, 760)),
        "poisson_damping_free_energy",
    )

    traces = CSV.read(
        joinpath(RESULT_DIR, "poisson_damping_synthetic.csv"), DataFrame,
    )
    fits = CSV.read(
        joinpath(RESULT_DIR, "poisson_damping_synthetic_fits.csv"), DataFrame,
    )
    lengths = sort(unique(fits.chain_length))
    synthetic_panels = [
        synthetic_panel(traces, fits, chain_length; legend = :topright)
        for chain_length in lengths
    ]
    for (chain_length, panel) in zip(lengths, synthetic_panels)
        save_pdf(panel, "poisson_damping_synthetic_n$(chain_length)")
    end
    save_figure(
        plot(synthetic_panels...; layout = (2, 2), size = (1060, 760)),
        "poisson_damping_synthetic_free_energy",
    )
    save_figure(
        synthetic_seed_panel(traces, fits, maximum(lengths), config["synthetic_seed"]),
        "poisson_damping_synthetic_seed$(config["synthetic_seed"])",
    )

    write_summary("poisson_damping", vcat(
        damping_summary_lines(frame, config),
        synthetic_summary_lines(fits, config),
    ))
end

function damping_main()
    render_only() || damping_compute()
    damping_render()
end

abspath(PROGRAM_FILE) == (@__FILE__) && damping_main()
