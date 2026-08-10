#!/usr/bin/env julia

# L = 2 heteroscedastic hierarchy, VMP vs NGMP, on two 1D regression benchmarks.
#
# Model (identical graph and factorization for both arms):
#
#     v ~ N(0, σ_v² I)                       # mean weights
#     w ~ N(anchored mean, diag)             # log-precision weights
#     f[o] := dot(φ(x_o), v)                 # deterministic mean pathway (exact BP)
#     s[o] ~ softdot(ψ(x_o), w, c_top)       # level-2 log-precision
#     y[o] ~ MvNormalExpPrecision(f[o], s[o])   #  y ~ N(f, e^{-s})
#
# The arms differ ONLY in how the non-conjugate message toward each s[o] is
# executed: the VMP arm multiplies the exact mean-field site exp(E_q[log f])
# into the marginal and projects the product onto a Gaussian (ProjectedTo +
# ClosedFormStrategy); the NGMP arm sends the damped natural-gradient tangent
# projection of the same site (NGMPDependencies + DampingMeta).
#
# Benchmarks:
#   * aleatoric — heteroscedastic wave (why_hierarchy_deep_kernel.jl):
#         y = -(x+1/2)sin(3πx) + ε,  ε ~ N(0, [0.45(x+1/2)]²),  x ~ N(0,1)
#     The noise level s(x_o) is a per-observation latent: its uncertainty
#     cannot concentrate no matter how long the dataset grows.
#   * epistemic — sin(x) with a wide input gap (IVONRepro regression_1d):
#     near-noiseless data on two disjoint intervals; predictive bands must
#     widen inside the gap. The mean pathway is conjugate in both arms, so
#     this benchmark acts as the control.

include(joinpath(@__DIR__, "common.jl"))
using .WhenNGMPHelpsCommon

using CSV
using DataFrames
using Distributions
using ExponentialFamily
using ExponentialFamilyProjection
using LinearAlgebra
using Plots
using Random
using RxInfer
using StableRNGs
using Statistics
using SurrogateModelling

import ExponentialFamilyProjection: BoundedNormUpdateRule

# ---------------------------------------------------------------------------
# models: same graph, arms differ only by the `where` clause on the likelihood
# ---------------------------------------------------------------------------

@model function hetero_vmp_model(y, mean_features, level_features, v_prior, w_prior, top_carrier)
    local f, s
    v ~ v_prior
    w ~ w_prior
    for o in eachindex(y)
        f[o] := dot(mean_features[o], v)
        s[o] ~ softdot(level_features[o], w, top_carrier)
        y[o] ~ MvNormalExpPrecision(f[o], s[o])
    end
end

@model function hetero_ngmp_model(y, mean_features, level_features, v_prior, w_prior, top_carrier, dependencies, damping)
    local f, s
    v ~ v_prior
    w ~ w_prior
    for o in eachindex(y)
        f[o] := dot(mean_features[o], v)
        s[o] ~ softdot(level_features[o], w, top_carrier)
        y[o] ~ MvNormalExpPrecision(f[o], s[o]) where {
            dependencies = dependencies,
            meta = damping,
        }
    end
end

# Norm bound 100: the default projection budget can trap marginals far from
# their optimum (see poisson_state_space.jl) — widen it so the VMP arm is a
# fair baseline.
@constraints function hetero_vmp_constraints()
    q(f, w, s) = q(f)q(w)q(s)
    q(s) :: ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(),
            direction = BoundedNormUpdateRule(100.0),
        ),
    )
end

# NGMP's projected site toward s is Gaussian, so it composes with the exact
# conjugate (w, s) cluster — the notebook's structured configuration. The VMP
# arm cannot join this cluster: its raw ExpGamma site is projected at the
# MARGINAL, which forces the mean-field split between w and s.
@constraints function hetero_ngmp_constraints()
    q(f, w, s) = q(f)q(w, s)
end

# ---------------------------------------------------------------------------
# features, priors, initialization (why_hierarchy_deep_kernel.jl choices)
# ---------------------------------------------------------------------------

function make_feature_map(n_basis, lengthscale; feature_seed)
    rng = StableRNG(feature_seed)
    frequencies = randn(rng, n_basis) ./ lengthscale
    phases = 2pi .* rand(rng, n_basis)
    scale = sqrt(2 / n_basis)
    return x -> vcat(scale .* cos.(frequencies .* x .+ phases), [1.0])
end

design(feature_map, xs) = [feature_map(x) for x in xs]

# Rice/Gasser first-difference noise estimate: a basis-free anchor for the
# constant coordinate of the log-precision weights.
noise_anchor(xs, ys) = -log(max(mean(abs2.(diff(ys[sortperm(xs)]))) / 2, 1e-8))

gaussian(means, variances) =
    MvNormalMeanCovariance(collect(means), Matrix(Diagonal(collect(variances))))

function make_priors(config, xs, ys)
    n_basis = config.n_basis
    anchor = noise_anchor(xs, ys)
    return (;
        v = gaussian(
            zeros(n_basis + 1),
            fill(abs2(config.signal_sd), n_basis + 1),
        ),
        w = gaussian(
            vcat(zeros(n_basis), [anchor]),
            vcat(fill(abs2(config.level_sd), n_basis), [abs2(config.anchor_sd)]),
        ),
    )
end

# Push the priors forward so the first damped step is well scaled.
function initial_marginals(priors, rows, top_carrier)
    Φ = reduce(hcat, rows)'
    mv, Vv = mean_cov(priors.v)
    mw, Vw = mean_cov(priors.w)
    f_means = Φ * mv
    f_variances = vec(sum((Φ * Vv) .* Φ; dims = 2))
    s_means = Φ * mw
    s_variances = vec(sum((Φ * Vw) .* Φ; dims = 2)) .+ inv(top_carrier)
    return (;
        f = NormalMeanVariance.(f_means, f_variances),
        s = NormalMeanVariance.(s_means, s_variances),
    )
end

# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

function fit_arm(method, observations, rows, priors, config)
    states = initial_marginals(priors, rows, config.top_carrier)
    initialization = @initialization begin
        q(v) = deepcopy(priors.v)
        q(w) = deepcopy(priors.w)
        q(f) = states.f
        q(s) = states.s
    end
    if method == "NGMP-cavity"
        # True NGMP: no factorization constraints (the fused node keeps its
        # (μ, s) cluster, the softdot keeps (w, s)), both latent interfaces
        # named so the cavity rules receive messages; message inits break the
        # message-marginal cycle.
        dependencies = NGMPDependencies(
            s = nothing,
            μ = nothing,
            projection = TangentProjection(type = Quadrature(config.cavity_quadrature)),
        )
        damping = DampingMeta(
            alpha = config.ngmp_alpha,
            beta = 0.0,
            max_step = config.ngmp_max_step,
            method = :damped,
        )
        cavity_initialization = @initialization begin
            q(v) = deepcopy(priors.v)
            q(w) = deepcopy(priors.w)
            q(f) = states.f
            q(s) = states.s
            μ(f) = states.f
            μ(s) = states.s
        end
        elapsed = @elapsed result = infer(
            model = hetero_ngmp_model(
                mean_features = rows,
                level_features = rows,
                v_prior = priors.v,
                w_prior = priors.w,
                top_carrier = config.top_carrier,
                dependencies = dependencies,
                damping = damping,
            ),
            data = (y = observations,),
            initialization = cavity_initialization,
            returnvars = (v = KeepLast(), w = KeepLast()),
            iterations = config.iterations,
            free_energy = false,
            showprogress = false,
            options = (limit_stack_depth = 100,),
        )
        return (;
            qv = result.posteriors[:v],
            qw = result.posteriors[:w],
            free_energy = Float64[],
            elapsed,
        )
    elseif method == "NGMP"
        dependencies = NGMPDependencies(
            s = nothing,
            projection = TangentProjection(type = ClosedForm),
        )
        damping = DampingMeta(
            alpha = config.ngmp_alpha,
            beta = 0.0,
            max_step = config.ngmp_max_step,
            method = :damped,
        )
        elapsed = @elapsed result = infer(
            model = hetero_ngmp_model(
                mean_features = rows,
                level_features = rows,
                v_prior = priors.v,
                w_prior = priors.w,
                top_carrier = config.top_carrier,
                dependencies = dependencies,
                damping = damping,
            ),
            data = (y = observations,),
            constraints = hetero_ngmp_constraints(),
            initialization = initialization,
            returnvars = (v = KeepLast(), w = KeepLast()),
            iterations = config.iterations,
            free_energy = true,
            showprogress = false,
            options = (limit_stack_depth = 100,),
        )
    else
        elapsed = @elapsed result = infer(
            model = hetero_vmp_model(
                mean_features = rows,
                level_features = rows,
                v_prior = priors.v,
                w_prior = priors.w,
                top_carrier = config.top_carrier,
            ),
            data = (y = observations,),
            constraints = hetero_vmp_constraints(),
            initialization = initialization,
            returnvars = (v = KeepLast(), w = KeepLast()),
            iterations = config.iterations,
            free_energy = true,
            showprogress = false,
            options = (limit_stack_depth = 100,),
        )
    end
    return (;
        qv = result.posteriors[:v],
        qw = result.posteriors[:w],
        free_energy = collect(Float64.(result.free_energy)),
        elapsed,
    )
end

# ---------------------------------------------------------------------------
# closed-form prediction: everything follows from q(v), q(w)
# ---------------------------------------------------------------------------

function predict(fit, rows, top_carrier)
    Φ = reduce(hcat, rows)'
    mv, Vv = mean_cov(fit.qv)
    mw, Vw = mean_cov(fit.qw)
    f_mean = Φ * mv
    f_variance = vec(sum((Φ * Vv) .* Φ; dims = 2))
    s_mean = Φ * mw
    s_variance = vec(sum((Φ * Vw) .* Φ; dims = 2)) .+ inv(top_carrier)
    # lognormal moments of the noise variance e^{-s}
    noise_mean = exp.(-s_mean .+ s_variance ./ 2)
    noise_lower = exp.(-(s_mean .+ 1.96 .* sqrt.(s_variance)))
    noise_upper = exp.(-(s_mean .- 1.96 .* sqrt.(s_variance)))
    return (;
        mean = f_mean,
        latent_variance = f_variance,
        total_variance = f_variance .+ noise_mean,
        noise_mean,
        noise_lower,
        noise_upper,
        s_mean,
        s_variance,
    )
end

# p(y*) = ∫ N(y*; f_mean, f_var + e^{-s}) N(s; m, v) ds, by quadrature over s.
function predictive_logpdf(f_mean, f_variance, s_mean, s_variance, y)
    sd = sqrt(s_variance)
    points = range(s_mean - 8sd, s_mean + 8sd; length = 201)
    weights = pdf.(Normal(s_mean, sd), points)
    weights ./= sum(weights)
    density = sum(
        weights .* pdf.(Normal.(f_mean, sqrt.(f_variance .+ exp.(-points))), y),
    )
    return log(max(density, floatmin(Float64)))
end

mean_predictive_logpdf(prediction, ys) = mean(
    predictive_logpdf(
        prediction.mean[i],
        prediction.latent_variance[i],
        prediction.s_mean[i],
        prediction.s_variance[i],
        ys[i],
    ) for i in eachindex(ys)
)

# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------

aleatoric_mean(x) = -(x + 0.5) * sin(3pi * x)
aleatoric_noise_variance(x) = abs2(0.45 * (x + 0.5))

function aleatoric_data(config, rng)
    n = config.aleatoric_samples
    x = randn(rng, n)
    y = aleatoric_mean.(x) .+ sqrt.(aleatoric_noise_variance.(x)) .* randn(rng, n)
    order = randperm(rng, n)
    n_test = round(Int, config.holdout_fraction * n)
    test, train = order[1:n_test], order[(n_test + 1):end]
    return (;
        x_train = x[train], y_train = y[train],
        x_test = x[test], y_test = y[test],
    )
end

# Huber ε-contamination in the TRAINING noise only; the held-out set is drawn
# from the clean process (standard robust-statistics protocol: both arms fit
# the same honestly-misspecified Gaussian-given-s model, and we measure who
# recovers the underlying relationship despite the contamination).
function contaminated_data(config, rng)
    n = config.aleatoric_samples
    n_test = round(Int, config.holdout_fraction * n)
    n_train = n - n_test
    x_train = randn(rng, n_train)
    noise = sqrt.(aleatoric_noise_variance.(x_train)) .* randn(rng, n_train)
    outlier = rand(rng, n_train) .< config.contamination_fraction
    noise[outlier] .= config.contamination_sd .* randn(rng, count(outlier))
    y_train = aleatoric_mean.(x_train) .+ noise
    x_test = randn(rng, n_test)
    y_test = aleatoric_mean.(x_test) .+
        sqrt.(aleatoric_noise_variance.(x_test)) .* randn(rng, n_test)
    return (; x_train, y_train, x_test, y_test)
end

const EPISTEMIC_INTERVALS = ((-0.40pi, -0.05pi), (1.4pi, 2.15pi))

epistemic_gap() = (-0.05pi, 1.4pi)

function epistemic_data(config, rng)
    half = config.epistemic_samples ÷ 2
    x_train = vcat(
        (collect(range(lo, hi; length = half)) for (lo, hi) in EPISTEMIC_INTERVALS)...,
    )
    y_train = sin.(x_train) .+ config.epistemic_noise_sd .* randn(rng, length(x_train))
    x_test = vcat(
        (lo .+ (hi - lo) .* rand(rng, half) for (lo, hi) in EPISTEMIC_INTERVALS)...,
    )
    y_test = sin.(x_test) .+ config.epistemic_noise_sd .* randn(rng, length(x_test))
    return (; x_train, y_train, x_test, y_test)
end

# ---------------------------------------------------------------------------
# study
# ---------------------------------------------------------------------------

function run_benchmark(benchmark, repetition, config)
    rng = StableRNG(config.seed + repetition - 1)
    if benchmark == "aleatoric" || benchmark == "contaminated"
        data = benchmark == "contaminated" ? contaminated_data(config, rng) :
            aleatoric_data(config, rng)
        feature_map = make_feature_map(
            config.n_basis, config.aleatoric_lengthscale;
            feature_seed = config.feature_seed,
        )
        grid = collect(range(-3.0, 3.0; length = 241))
    else
        data = epistemic_data(config, rng)
        feature_map = make_feature_map(
            config.n_basis, config.epistemic_lengthscale;
            feature_seed = config.feature_seed,
        )
        grid = collect(range(-2.5, 8.0; length = 241))
    end
    rows = design(feature_map, data.x_train)
    test_rows = design(feature_map, data.x_test)
    grid_rows = design(feature_map, grid)
    priors = make_priors(config, data.x_train, data.y_train)

    outputs = NamedTuple[]
    for method in config.methods
        fit = fit_arm(method, data.y_train, rows, priors, config)
        test_prediction = predict(fit, test_rows, config.top_carrier)
        grid_prediction = predict(fit, grid_rows, config.top_carrier)
        logpdf_test = mean_predictive_logpdf(test_prediction, data.y_test)
        rmse = sqrt(mean(abs2.(test_prediction.mean .- data.y_test)))
        if benchmark == "aleatoric" || benchmark == "contaminated"
            noise_corr = cor(
                test_prediction.noise_mean,
                aleatoric_noise_variance.(data.x_test),
            )
            gap_sd_ratio = NaN
            mean_rmse_truth = sqrt(mean(abs2.(
                grid_prediction.mean .- aleatoric_mean.(grid),
            )))
        else
            noise_corr = NaN
            lo, hi = epistemic_gap()
            in_gap = (grid .> lo) .& (grid .< hi)
            in_support = map(
                x -> any(lo <= x <= hi for (lo, hi) in EPISTEMIC_INTERVALS),
                grid,
            )
            gap_sd = mean(sqrt.(grid_prediction.total_variance[in_gap]))
            support_sd = mean(sqrt.(grid_prediction.total_variance[in_support]))
            gap_sd_ratio = gap_sd / support_sd
            mean_rmse_truth = NaN
        end
        push!(outputs, (;
            benchmark,
            repetition,
            method,
            logpdf_test,
            rmse,
            noise_corr,
            gap_sd_ratio,
            mean_rmse_truth,
            elapsed_seconds = fit.elapsed,
            final_free_energy = isempty(fit.free_energy) ? NaN : last(fit.free_energy),
            free_energy = fit.free_energy,
            grid,
            data,
            grid_prediction,
        ))
    end
    return outputs
end

function run_study(config)
    tasks = [
        (benchmark, repetition)
        for benchmark in ("aleatoric", "epistemic")
        for repetition in 1:config.repetitions
    ]
    results = Vector{Vector{NamedTuple}}(undef, length(tasks))
    Threads.@threads for index in eachindex(tasks)
        benchmark, repetition = tasks[index]
        results[index] = run_benchmark(benchmark, repetition, config)
    end
    return reduce(vcat, results)
end

# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

function runs_frame(outputs)
    return DataFrame([
        (;
            output.benchmark,
            output.repetition,
            output.method,
            output.logpdf_test,
            output.rmse,
            output.noise_corr,
            output.gap_sd_ratio,
            output.elapsed_seconds,
            output.final_free_energy,
        )
        for output in outputs
    ])
end

function summarize_runs(runs)
    rows = NamedTuple[]
    for benchmark in ("aleatoric", "epistemic"),
        method in intersect(("VMP", "NGMP", "NGMP-cavity"), unique(runs.method))
        selected = filter(
            row -> row.benchmark == benchmark && row.method == method,
            runs,
        )
        push!(rows, (;
            benchmark,
            method,
            n_seeds = nrow(selected),
            mean_logpdf = mean(selected.logpdf_test),
            logpdf_ci95 = ci95(selected.logpdf_test),
            mean_rmse = mean(selected.rmse),
            rmse_ci95 = ci95(selected.rmse),
            mean_noise_corr = mean(selected.noise_corr),
            noise_corr_ci95 = ci95(selected.noise_corr),
            mean_gap_sd_ratio = mean(selected.gap_sd_ratio),
            gap_sd_ratio_ci95 = ci95(selected.gap_sd_ratio),
            mean_seconds = mean(selected.elapsed_seconds),
        ))
    end
    return DataFrame(rows)
end

function free_energy_frame(outputs)
    rows = NamedTuple[]
    for output in outputs, (iteration, value) in enumerate(output.free_energy)
        push!(rows, (;
            output.benchmark,
            output.method,
            output.repetition,
            iteration,
            free_energy = value,
        ))
    end
    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# figures (representative repetition = 1)
# ---------------------------------------------------------------------------

method_style(method) = method == "VMP" ?
    (color = COLORS.vmp, linestyle = :dash) :
    (color = COLORS.ngmp, linestyle = :solid)

function fit_panel(output; title = "", ylims = nothing)
    style = method_style(output.method)
    prediction = output.grid_prediction
    band = 1.96 .* sqrt.(prediction.total_variance)
    panel = plot(;
        xlabel = "x",
        ylabel = "y",
        title,
        legend = :topright,
        left_margin = 5Plots.mm,
    )
    ylims === nothing || plot!(panel; ylims)
    scatter!(
        panel,
        output.data.x_train,
        output.data.y_train;
        color = :gray70,
        markersize = 1.6,
        markerstrokewidth = 0,
        label = "train",
    )
    plot!(
        panel,
        output.grid,
        prediction.mean;
        ribbon = band,
        color = style.color,
        linestyle = style.linestyle,
        linewidth = 2.2,
        fillalpha = 0.2,
        label = "$(output.method) mean ± 1.96 sd",
    )
    return panel
end

function aleatoric_truth!(panel, grid)
    plot!(
        panel,
        grid,
        aleatoric_mean.(grid);
        color = COLORS.truth,
        linestyle = :dashdot,
        linewidth = 1.6,
        label = "true mean",
    )
    return panel
end

function epistemic_truth!(panel, grid)
    plot!(
        panel,
        grid,
        sin.(grid);
        color = COLORS.truth,
        linestyle = :dashdot,
        linewidth = 1.6,
        label = "sin(x)",
    )
    lo, hi = epistemic_gap()
    vspan!(panel, [lo, hi]; color = :gray, alpha = 0.08, label = "gap")
    return panel
end

# One method per panel: overlaid credible bands are indistinguishable where
# they intersect, so each arm's noise-variance prediction gets its own axes.
function noise_variance_panel(output; title = "")
    style = method_style(output.method)
    prediction = output.grid_prediction
    grid = output.grid
    panel = plot(;
        xlabel = "x",
        ylabel = "noise variance",
        yscale = :log10,
        ylims = (1e-4, 1e2),
        title,
        legend = :topleft,
        left_margin = 5Plots.mm,
    )
    plot!(
        panel,
        grid,
        max.(aleatoric_noise_variance.(grid), 1e-4);
        color = COLORS.truth,
        linestyle = :dashdot,
        linewidth = 2.0,
        label = "true noise variance",
    )
    plot!(
        panel,
        grid,
        max.(prediction.noise_mean, 1e-6);
        ribbon = (
            max.(prediction.noise_mean .- prediction.noise_lower, 0.0),
            max.(prediction.noise_upper .- prediction.noise_mean, 0.0),
        ),
        color = style.color,
        linestyle = style.linestyle,
        linewidth = 2.2,
        fillalpha = 0.18,
        label = "$(output.method) E[e^{-s}] ± 95% CrI",
    )
    return panel
end

function predictive_sd_panel(outputs; title = "")
    panel = plot(;
        xlabel = "x",
        ylabel = "predictive sd",
        yscale = :log10,
        title,
        legend = :topleft,
        left_margin = 5Plots.mm,
    )
    for output in outputs
        style = method_style(output.method)
        plot!(
            panel,
            output.grid,
            sqrt.(output.grid_prediction.total_variance);
            color = style.color,
            linestyle = style.linestyle,
            linewidth = 2.2,
            label = output.method,
        )
    end
    lo, hi = epistemic_gap()
    vspan!(panel, [lo, hi]; color = :gray, alpha = 0.08, label = "gap")
    return panel
end

function free_energy_panel(outputs; title = "")
    panel = plot(;
        xlabel = "iteration",
        ylabel = "Bethe free energy",
        title,
        legend = :topright,
        left_margin = 5Plots.mm,
    )
    for output in outputs
        style = method_style(output.method)
        values = output.free_energy
        # skip the first sweeps: the VMP arm starts orders of magnitude higher
        # and would flatten the plateau comparison
        start = min(3, length(values))
        plot!(
            panel,
            start:length(values),
            values[start:end];
            color = style.color,
            linestyle = style.linestyle,
            linewidth = 2.0,
            label = output.method,
        )
    end
    return panel
end

function benchmark_outputs(outputs, benchmark; repetition = 1)
    selected = [
        output for output in outputs
        if output.benchmark == benchmark && output.repetition == repetition
    ]
    order = Dict("VMP" => 1, "NGMP" => 2, "NGMP-cavity" => 3)
    return sort(selected; by = output -> order[output.method])
end

function make_figures(outputs)
    aleatoric = benchmark_outputs(outputs, "aleatoric")
    epistemic = benchmark_outputs(outputs, "epistemic")

    aleatoric_panels = [
        aleatoric_truth!(fit_panel(aleatoric[1]; ylims = (-4.5, 4.5)), aleatoric[1].grid),
        aleatoric_truth!(fit_panel(aleatoric[2]; ylims = (-4.5, 4.5)), aleatoric[2].grid),
        noise_variance_panel(aleatoric[1]),
        noise_variance_panel(aleatoric[2]),
        free_energy_panel(aleatoric),
    ]
    save_pdf(aleatoric_panels[1], "hetero_aleatoric_vmp_fit")
    save_pdf(aleatoric_panels[2], "hetero_aleatoric_ngmp_fit")
    save_pdf(aleatoric_panels[3], "hetero_aleatoric_vmp_variance")
    save_pdf(aleatoric_panels[4], "hetero_aleatoric_ngmp_variance")
    save_pdf(aleatoric_panels[5], "hetero_aleatoric_free_energy")
    preview = plot(
        (plot(panel; title = t) for (panel, t) in zip(
            aleatoric_panels,
            (
                "VMP fit", "NGMP fit", "VMP noise variance",
                "NGMP noise variance", "free energy",
            ),
        ))...,
        plot(; framestyle = :none);
        layout = (2, 3),
        size = (1500, 750),
    )
    save_figure(preview, "hetero_hierarchy_aleatoric")

    epistemic_panels = [
        epistemic_truth!(fit_panel(epistemic[1]; ylims = (-2.5, 2.5)), epistemic[1].grid),
        epistemic_truth!(fit_panel(epistemic[2]; ylims = (-2.5, 2.5)), epistemic[2].grid),
        predictive_sd_panel(epistemic),
        free_energy_panel(epistemic),
    ]
    save_pdf(epistemic_panels[1], "hetero_epistemic_vmp_fit")
    save_pdf(epistemic_panels[2], "hetero_epistemic_ngmp_fit")
    save_pdf(epistemic_panels[3], "hetero_epistemic_sd")
    save_pdf(epistemic_panels[4], "hetero_epistemic_free_energy")
    preview = plot(
        (plot(panel; title = t) for (panel, t) in zip(
            epistemic_panels,
            ("VMP fit", "NGMP fit", "predictive sd", "free energy"),
        ))...;
        layout = (2, 2),
        size = (1100, 750),
    )
    save_figure(preview, "hetero_hierarchy_epistemic")
end

# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

function summary_lines(config, summary)
    lines = [
        "# L = 2 heteroscedastic hierarchy study",
        "",
        "Both arms share one graph: v, w weight vectors on a fixed random Fourier basis, a deterministic dot-product mean pathway, `s[o] ~ softdot(ψ(x_o), w, $(config.top_carrier))`, and the fused likelihood `y[o] ~ MvNormalExpPrecision(f[o], s[o])`.",
        "They differ only in the non-conjugate message toward each `s[o]`: the VMP arm projects the marginal product (ProjectedTo, ClosedFormStrategy); the NGMP arm sends the damped natural-gradient site (alpha = $(config.ngmp_alpha), max_step = $(config.ngmp_max_step)).",
        "Metrics are means ± 95% CI across $(config.repetitions) data seeds (seed $(config.seed) onward); figures show seed $(config.seed).",
        "",
        "| benchmark | method | held-out logpdf | RMSE | noise corr | gap sd ratio |",
        "|---|---|---|---|---|---|",
    ]
    for row in eachrow(summary)
        push!(lines, string(
            "| ", row.benchmark, " | ", row.method,
            " | ", round(row.mean_logpdf; digits = 3),
            " ± ", round(row.logpdf_ci95; digits = 3),
            " | ", round(row.mean_rmse; digits = 3),
            " ± ", round(row.rmse_ci95; digits = 3),
            " | ", isnan(row.mean_noise_corr) ? "—" : string(
                round(row.mean_noise_corr; digits = 3),
                " ± ", round(row.noise_corr_ci95; digits = 3),
            ),
            " | ", isnan(row.mean_gap_sd_ratio) ? "—" : string(
                round(row.mean_gap_sd_ratio; digits = 3),
                " ± ", round(row.gap_sd_ratio_ci95; digits = 3),
            ),
            " |",
        ))
    end
    return lines
end

function main()
    ensure_outputs()
    smoke = smoke_mode()
    config = (
        smoke,
        seed = 42,
        repetitions = parse(
            Int,
            get(ENV, "WHEN_NGMP_HETERO_REPETITIONS", smoke ? "2" : "10"),
        ),
        holdout_fraction = 1 / 3,
        aleatoric_samples = smoke ? 90 : 600,
        epistemic_samples = smoke ? 60 : 150,
        epistemic_noise_sd = 0.02,
        n_basis = smoke ? 6 : 16,
        aleatoric_lengthscale = 0.25,
        epistemic_lengthscale = 1.0,
        feature_seed = 20260726,
        signal_sd = 1.0,
        level_sd = 0.4,
        anchor_sd = 1.0,
        top_carrier = 25.0,
        iterations = parse(
            Int,
            get(ENV, "WHEN_NGMP_HETERO_ITERATIONS", smoke ? "10" : "240"),
        ),
        ngmp_alpha = 0.6,
        ngmp_max_step = 0.5,
        cavity_quadrature = 32,
        methods = ("VMP", "NGMP"),
    )
    outputs = run_study(config)
    runs = runs_frame(outputs)
    summary = summarize_runs(runs)
    CSV.write(joinpath(RESULT_DIR, "hetero_hierarchy_runs.csv"), runs)
    CSV.write(joinpath(RESULT_DIR, "hetero_hierarchy_aggregate.csv"), summary)
    CSV.write(
        joinpath(RESULT_DIR, "hetero_hierarchy_free_energy.csv"),
        free_energy_frame(outputs),
    )
    write_config(
        "hetero_hierarchy",
        merge(config, (methods = collect(config.methods),)),
    )
    write_summary("hetero_hierarchy", summary_lines(config, summary))
    make_figures(outputs)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
