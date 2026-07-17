# Identifiable K=2 Probit-gate feature ablation on the noisy XOR checkerboard.
#
# This experiment changes the model structure, not the initialization oracle:
#
#     w ~ N(0, P^-1)
#     z_n = phi(x_n)'w
#     c_n ~ Probit(z_n)
#     y_n ~ NormalMixture(c_n; means = (0, 1), precisions = (100, 100))
#
# The two mixture states now have distinct observed meanings. Three fixed,
# population-orthonormal feature maps isolate the structural question:
#
#   linear:      [1, l1, l2]
#   interaction: [1, l1, l2, i12]
#   quadratic:   [1, l1, l2, i12, q1, q2]
#
# where t_j=x_j/2, l_j=sqrt(3)t_j, i12=3t1t2, and
# q_j=sqrt(5)/2*(3t_j^2-1). All basis terms have population second moment one
# for the known uniform input domain [-2, 2]^2. The interaction is therefore a
# structural feature, not an oracle initialization.
#
# Every run uses a fixed zero-mean model prior. Seed zero initializes q(w) at
# exactly zero; nonzero seeds add only small, unscreened jitter to q(w). A
# seed's underlying six-dimensional draw is shared across the nested feature
# arms, giving paired/common-random-number comparisons. Test data is used only
# for reporting, never for choosing a seed or hyperparameter.
#
# Smoke:
#   env XOR_PROBIT_SMOKE=true SAVE_OUTPUTS=false SHOW_PROGRESS=false \
#       julia --project=../.. xor_probit_k2_feature_ablation.jl
#
# Full fixed-seed run:
#   env PROBIT_SEEDS=0:10 N_ITERATIONS=100 SAVE_OUTPUTS=true \
#       SHOW_PROGRESS=false julia --project=../.. \
#       xor_probit_k2_feature_ablation.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using BayesBase
using CSV
using DataFrames
using Dates
using LinearAlgebra: Diagonal, dot
using Plots
using Random
using RxInfer
using StableRNGs
using Statistics
using StatsFuns: normcdf, normlogcdf

import RxInfer: @average_energy, @call_rule, @rule

# NormalMixture emits a two-state Categorical message on its switch, whereas
# native Probit rules use Bernoulli. These adapters only change representation:
# state 2 is the Bernoulli-one event and state 1 is Bernoulli zero.
@rule Probit(:in, Marginalisation) (
    m_in::UnivariateNormalDistributionsFamily,
    q_out::Categorical,
    meta::Union{ProbitMeta, Nothing},
) = @call_rule Probit(:in, Marginalisation) (
    m_out = Bernoulli(probvec(q_out)[2]),
    m_in = m_in,
    meta = meta,
)

@rule Probit(:out, Marginalisation) (
    q_in::UnivariateNormalDistributionsFamily,
    meta::Union{ProbitMeta, Nothing},
) = Bernoulli(normcdf(mean(q_in) / sqrt(1 + var(q_in))))

@average_energy Probit (
    q_out::Categorical,
    q_in::UnivariateNormalDistributionsFamily,
    meta::ProbitMeta,
) = begin
    p = probvec(q_out)[2]
    m, v = mean_var(q_in)
    cubature = ReactiveMP.GaussHermiteCubature(meta.p)
    total = 0.0
    scale = sqrt(2v)
    for k in 1:meta.p
        x = cubature.piter[k] * scale + m
        cross_entropy =
            (p > 0 ? -p * normlogcdf(x) : 0.0) +
            (p < 1 ? -(1 - p) * normlogcdf(-x) : 0.0)
        total += cubature.witer[k] * cross_entropy
    end
    total / sqrt(pi)
end

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
env_bool(name, default = false) =
    lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")

function parse_seed_spec(specification)
    seeds = Int[]
    for raw_token in split(specification, ',')
        token = strip(raw_token)
        isempty(token) && continue
        if occursin(':', token)
            fields = parse.(Int, strip.(split(token, ':')))
            if length(fields) == 2
                append!(seeds, fields[1]:fields[2])
            elseif length(fields) == 3
                append!(seeds, fields[1]:fields[2]:fields[3])
            else
                throw(ArgumentError("invalid seed range: $token"))
            end
        else
            push!(seeds, parse(Int, token))
        end
    end
    seeds = unique(seeds)
    isempty(seeds) && throw(ArgumentError("PROBIT_SEEDS must contain at least one seed"))
    all(seed >= 0 for seed in seeds) ||
        throw(ArgumentError("PROBIT_SEEDS must be nonnegative; seed zero is exact-zero init"))
    return seeds
end

const SMOKE = env_bool("XOR_PROBIT_SMOKE")
const FEATURE_ARMS = (:linear, :interaction, :quadratic)
const FEATURE_NAMES = Dict(
    :linear => ["intercept", "linear_x1", "linear_x2"],
    :interaction => ["intercept", "linear_x1", "linear_x2", "interaction_x1_x2"],
    :quadratic => [
        "intercept",
        "linear_x1",
        "linear_x2",
        "interaction_x1_x2",
        "quadratic_x1",
        "quadratic_x2",
    ],
)

const DEFAULT_RUN_ID = Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS") *
                       "_pid$(getpid())"
const RUN_ID = replace(
    get(ENV, "PROBIT_RUN_ID", DEFAULT_RUN_ID),
    r"[^A-Za-z0-9_.-]" => "_",
)
const CONFIG = (
    n_samples = env_int("N_SAMPLES", SMOKE ? 60 : 1_600),
    iterations = env_int("N_ITERATIONS", SMOKE ? 2 : 100),
    train_fraction = env_float("TRAIN_FRACTION", 0.30),
    noise_std = env_float("NOISE_STD", 0.10),
    data_seed = env_int("DATA_SEED", 2_026),
    split_seed = env_int("SPLIT_SEED", 2_027),
    seeds = parse_seed_spec(get(ENV, "PROBIT_SEEDS", SMOKE ? "0:1" : "0:10")),
    # Per-coefficient precision is base_precision * feature dimension. Since
    # basis terms have unit second moment, induced score variance stays O(1)
    # when the feature dimension changes.
    gate_prior_base_precision = env_float("GATE_PRIOR_BASE_PRECISION", 1.0),
    init_q_base_precision = env_float("INIT_Q_BASE_PRECISION", 1.0),
    init_jitter_scale = env_float("INIT_JITTER_SCALE", 0.10),
    mixture_precision = env_float("MIXTURE_PRECISION", 100.0),
    grid_size = env_int("GRID_SIZE", SMOKE ? 21 : 61),
    free_energy = env_bool("FREE_ENERGY", false),
    showprogress = env_bool("SHOW_PROGRESS", false),
    save_outputs = env_bool("SAVE_OUTPUTS", !SMOKE),
    save_plots = env_bool("SAVE_PLOTS", !SMOKE),
    output_root = get(ENV, "PROBIT_OUTPUT_ROOT", joinpath(@__DIR__, "viz")),
)

CONFIG.n_samples >= 4 || throw(ArgumentError("N_SAMPLES must be at least 4"))
CONFIG.iterations > 0 || throw(ArgumentError("N_ITERATIONS must be positive"))
0 < CONFIG.train_fraction < 1 ||
    throw(ArgumentError("TRAIN_FRACTION must lie strictly between zero and one"))
CONFIG.gate_prior_base_precision > 0 ||
    throw(ArgumentError("GATE_PRIOR_BASE_PRECISION must be positive"))
CONFIG.init_q_base_precision > 0 ||
    throw(ArgumentError("INIT_Q_BASE_PRECISION must be positive"))
CONFIG.init_jitter_scale >= 0 ||
    throw(ArgumentError("INIT_JITTER_SCALE must be nonnegative"))

function unique_output_directory(root, run_id)
    stem = joinpath(root, "xor_probit_k2_feature_ablation_" * run_id)
    !ispath(stem) && return stem
    suffix = 2
    while ispath(stem * "_$(suffix)")
        suffix += 1
    end
    return stem * "_$(suffix)"
end

const OUTPUT_DIRECTORY = unique_output_directory(CONFIG.output_root, RUN_ID)

function checkerboard_label(x1, x2)
    cell_x = clamp(floor(Int, 2 * (x1 + 2) / 4), 0, 1)
    cell_y = clamp(floor(Int, 2 * (x2 + 2) / 4), 0, 1)
    return Float64(isodd(cell_x + cell_y))
end

function make_checkerboard_dataset(; n, noise_std, seed)
    rng = StableRNG(seed)
    x1 = 4 .* rand(rng, n) .- 2
    x2 = 4 .* rand(rng, n) .- 2
    clean = checkerboard_label.(x1, x2)
    target = clamp.(clean .+ noise_std .* randn(rng, n), 0.0, 1.0)
    return DataFrame(x1 = x1, x2 = x2, clean = clean, OT = target)
end

function split_dataset(data; train_fraction, seed)
    rng = StableRNG(seed)
    indices = randperm(rng, nrow(data))
    n_train = round(Int, train_fraction * nrow(data))
    return data[indices[1:n_train], :], data[indices[(n_train + 1):end], :]
end

# Population-orthonormal Legendre-product basis on [-2, 2]^2.
function feature_at(x1, x2, arm::Symbol)
    arm in FEATURE_ARMS || throw(ArgumentError("unknown feature arm: $arm"))
    t1 = x1 / 2
    t2 = x2 / 2
    linear1 = sqrt(3.0) * t1
    linear2 = sqrt(3.0) * t2
    interaction = 3.0 * t1 * t2
    quadratic1 = sqrt(5.0) / 2 * (3t1^2 - 1)
    quadratic2 = sqrt(5.0) / 2 * (3t2^2 - 1)
    arm === :linear && return [1.0, linear1, linear2]
    arm === :interaction && return [1.0, linear1, linear2, interaction]
    return [1.0, linear1, linear2, interaction, quadratic1, quadratic2]
end

build_features(data, arm) = [
    feature_at(data.x1[index], data.x2[index], arm)
    for index in 1:nrow(data)
]

# The actual model prior is zero mean and has no seed argument.
function model_prior(n_features)
    precision_value = CONFIG.gate_prior_base_precision * n_features
    precision = Diagonal(fill(precision_value, n_features))
    return MvNormalMeanPrecision(zeros(n_features), precision)
end

# Common six-dimensional draw gives paired initializations across nested arms.
function initial_mean(seed, n_features)
    seed == 0 && return zeros(n_features)
    draw = randn(StableRNG(seed), 6)
    return CONFIG.init_jitter_scale / sqrt(n_features) .* draw[1:n_features]
end

function initial_belief(seed, n_features)
    precision_value = CONFIG.init_q_base_precision * n_features
    precision = Diagonal(fill(precision_value, n_features))
    location = initial_mean(seed, n_features)
    return MvNormalMeanPrecision(location, precision)
end

@model function xor_probit_gate(features, y, w_prior, mixture_precision)
    w ~ w_prior
    for n in eachindex(y)
        # With fixed features and Gaussian q(w), this deterministic linear node
        # sends an exact Gaussian score message in both directions.
        z[n] ~ dot(features[n], w)
        c[n] ~ Probit(z[n])
        y[n] ~ NormalMixture(
            switch = c[n],
            m = (0.0, 1.0),
            p = (mixture_precision, mixture_precision),
        )
    end
end

@model function xor_probit_gate(features, y, w_prior, mixture_precision)
    w ~ w_prior
    for n in eachindex(y)
        # With fixed features and Gaussian q(w), this deterministic linear node
        # sends an exact Gaussian score message in both directions.
        z[n] ~ dot(features[n], w)
        c[n] ~ Probit(z[n])
        y[n] ~ NormalMixture(
            switch = c[n],
            m = (0.0, 1.0),
            p = (mixture_precision, mixture_precision),
        )
    end
end

@constraints function probit_constraints()
    # Required because NormalMixture represents c as Categorical while the
    # native structured Probit marginal is Bernoulli-specific.
    q(c, z, w) = q(c)q(z)q(w)
end

@initialization function probit_initialization(initial_w)
    q(w) = initial_w
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(c) = Categorical([0.5, 0.5])
end

function predictive_probability(feature, weight_belief)
    weight_mean, weight_covariance = mean_cov(weight_belief)
    score_mean = dot(feature, weight_mean)
    score_variance = dot(feature, weight_covariance, feature)
    # Exact E_q[Phi(phi'w)] for Gaussian q(w).
    return normcdf(score_mean / sqrt(1 + score_variance))
end

plugin_probability(feature, weight_belief) =
    normcdf(dot(feature, mean(weight_belief)))

function predictions(features, weight_belief; plugin = false)
    predictor = plugin ? plugin_probability : predictive_probability
    return [predictor(feature, weight_belief) for feature in features]
end

mse(prediction, target) = mean(abs2, prediction .- target)

function evaluate_belief(weight_belief, train_data, test_data, train_features, test_features)
    train_prediction = predictions(train_features, weight_belief)
    test_prediction = predictions(test_features, weight_belief)
    train_plugin = predictions(train_features, weight_belief; plugin = true)
    test_plugin = predictions(test_features, weight_belief; plugin = true)
    return (
        train_mse = mse(train_prediction, train_data.OT),
        test_mse = mse(test_prediction, test_data.OT),
        test_clean_mse = mse(test_prediction, test_data.clean),
        train_plugin_mse = mse(train_plugin, train_data.OT),
        test_plugin_mse = mse(test_plugin, test_data.OT),
        train_prediction = train_prediction,
        test_prediction = test_prediction,
    )
end

function run_arm(arm, seed, train_data, test_data, train_features, test_features)
    n_features = length(FEATURE_NAMES[arm])
    prior = model_prior(n_features)
    initial_w = initial_belief(seed, n_features)
    initial_metrics = evaluate_belief(
        initial_w,
        train_data,
        test_data,
        train_features,
        test_features,
    )

    timed = @timed infer(
        model = xor_probit_gate(
            w_prior = prior,
            mixture_precision = CONFIG.mixture_precision,
        ),
        data = (features = train_features, y = train_data.OT),
        constraints = probit_constraints(),
        initialization = probit_initialization(initial_w),
        iterations = CONFIG.iterations,
        free_energy = CONFIG.free_energy,
        showprogress = CONFIG.showprogress,
        returnvars = (w = KeepEach(), c = KeepLast(), z = KeepLast()),
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )

    result = timed.value
    weight_history = result.posteriors[:w]
    all_beliefs = Any[initial_w]
    append!(all_beliefs, weight_history)
    history_rows = NamedTuple[]
    for (offset, belief) in enumerate(all_beliefs)
        iteration = offset - 1
        metrics = evaluate_belief(
            belief,
            train_data,
            test_data,
            train_features,
            test_features,
        )
        push!(history_rows, (
            arm = String(arm),
            seed = seed,
            init = seed == 0 ? "exact_zero" : "small_jitter",
            iteration = iteration,
            train_mse = metrics.train_mse,
            test_mse = metrics.test_mse,
            test_clean_mse = metrics.test_clean_mse,
            train_plugin_mse = metrics.train_plugin_mse,
            test_plugin_mse = metrics.test_plugin_mse,
            free_energy = CONFIG.free_energy && iteration > 0 ?
                result.free_energy[iteration] : NaN,
        ))
    end

    final_w = last(weight_history)
    final_metrics = evaluate_belief(
        final_w,
        train_data,
        test_data,
        train_features,
        test_features,
    )
    initial_weight_mean = mean(initial_w)
    final_weight_mean = mean(final_w)
    coefficient_change = sqrt(sum(abs2, final_weight_mean .- initial_weight_mean))
    prediction_change = mean(abs.(
        final_metrics.test_prediction .- initial_metrics.test_prediction,
    ))

    summary = (
        arm = String(arm),
        seed = seed,
        init = seed == 0 ? "exact_zero" : "small_jitter",
        n_features = n_features,
        iterations = CONFIG.iterations,
        train_points = nrow(train_data),
        test_points = nrow(test_data),
        seconds = timed.time,
        allocated_gib = timed.bytes / 2.0^30,
        initial_train_mse = initial_metrics.train_mse,
        final_train_mse = final_metrics.train_mse,
        train_mse_gain = initial_metrics.train_mse - final_metrics.train_mse,
        initial_test_mse = initial_metrics.test_mse,
        final_test_mse = final_metrics.test_mse,
        test_mse_gain = initial_metrics.test_mse - final_metrics.test_mse,
        relative_test_mse_reduction =
            (initial_metrics.test_mse - final_metrics.test_mse) /
            initial_metrics.test_mse,
        initial_test_clean_mse = initial_metrics.test_clean_mse,
        final_test_clean_mse = final_metrics.test_clean_mse,
        initial_test_plugin_mse = initial_metrics.test_plugin_mse,
        final_test_plugin_mse = final_metrics.test_plugin_mse,
        mean_abs_test_prediction_change = prediction_change,
        coefficient_l2_change = coefficient_change,
        finite_weights = all(isfinite, final_weight_mean),
    )

    weight_rows = NamedTuple[]
    initial_covariance = cov(initial_w)
    final_covariance = cov(final_w)
    for index in eachindex(final_weight_mean)
        push!(weight_rows, (
            arm = String(arm),
            seed = seed,
            feature = FEATURE_NAMES[arm][index],
            initial_mean = initial_weight_mean[index],
            final_mean = final_weight_mean[index],
            initial_std = sqrt(initial_covariance[index, index]),
            final_std = sqrt(final_covariance[index, index]),
        ))
    end

    return (
        summary = summary,
        history_rows = history_rows,
        weight_rows = weight_rows,
        initial_w = initial_w,
        final_w = final_w,
    )
end

function save_surface_plot(arm, seed, initial_w, final_w)
    axis = range(-2.0, 2.0; length = CONFIG.grid_size)
    initial_surface = [
        predictive_probability(feature_at(x1, x2, arm), initial_w)
        for x2 in axis, x1 in axis
    ]
    final_surface = [
        predictive_probability(feature_at(x1, x2, arm), final_w)
        for x2 in axis, x1 in axis
    ]
    truth_surface = [checkerboard_label(x1, x2) for x2 in axis, x1 in axis]
    common = (
        color = :RdBu,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        aspect_ratio = :equal,
        colorbar = false,
    )
    p0 = heatmap(axis, axis, initial_surface; title = "iteration 0", common...)
    pf = heatmap(axis, axis, final_surface; title = "iteration $(CONFIG.iterations)", common...)
    pt = heatmap(axis, axis, truth_surface; title = "clean XOR", common...)
    figure = plot(
        p0,
        pf,
        pt;
        layout = (1, 3),
        size = (1_050, 340),
        plot_title = "K=2 Probit gate: $(arm), seed $(seed)",
    )
    savefig(figure, joinpath(OUTPUT_DIRECTORY, "surface_$(arm)_seed$(seed).png"))

    surface_rows = NamedTuple[]
    for (row, x2) in enumerate(axis), (column, x1) in enumerate(axis)
        push!(surface_rows, (
            arm = String(arm),
            seed = seed,
            x1 = x1,
            x2 = x2,
            iteration0 = initial_surface[row, column],
            final = final_surface[row, column],
            clean = truth_surface[row, column],
        ))
    end
    CSV.write(
        joinpath(OUTPUT_DIRECTORY, "surface_$(arm)_seed$(seed).csv"),
        DataFrame(surface_rows),
    )
end

println("K=2 identifiable Probit-gate feature ablation")
println(CONFIG)
println("Feature arms: $(join(string.(FEATURE_ARMS), ", "))")
println("Seeds are fixed before evaluation: $(CONFIG.seeds)")

dataset = make_checkerboard_dataset(
    n = CONFIG.n_samples,
    noise_std = CONFIG.noise_std,
    seed = CONFIG.data_seed,
)
train_data, test_data = split_dataset(
    dataset;
    train_fraction = CONFIG.train_fraction,
    seed = CONFIG.split_seed,
)
constant_prediction = mean(train_data.OT)
constant_test_mse = mse(fill(constant_prediction, nrow(test_data)), test_data.OT)
println(
    "Train/test = $(nrow(train_data))/$(nrow(test_data)); " *
    "constant test MSE = $(round(constant_test_mse; digits = 6))",
)

if CONFIG.save_outputs
    mkpath(OUTPUT_DIRECTORY)
    CSV.write(joinpath(OUTPUT_DIRECTORY, "train_data.csv"), train_data)
    CSV.write(joinpath(OUTPUT_DIRECTORY, "test_data.csv"), test_data)
end

summary_rows = NamedTuple[]
history_rows = NamedTuple[]
weight_rows = NamedTuple[]

# Seed outer loop preserves paired comparisons across feature arms.
for seed in CONFIG.seeds
    for arm in FEATURE_ARMS
        println("Running arm=$(arm), seed=$(seed) ...")
        train_features = build_features(train_data, arm)
        test_features = build_features(test_data, arm)
        run = run_arm(
            arm,
            seed,
            train_data,
            test_data,
            train_features,
            test_features,
        )
        row = merge(run.summary, (constant_test_mse = constant_test_mse,))
        push!(summary_rows, row)
        append!(history_rows, run.history_rows)
        append!(weight_rows, run.weight_rows)
        println(
            "  test MSE $(round(row.initial_test_mse; digits = 6)) -> " *
            "$(round(row.final_test_mse; digits = 6)); " *
            "mean |Δprediction|=$(round(row.mean_abs_test_prediction_change; digits = 6)); " *
            "$(round(row.seconds; digits = 2)) s",
        )
        # One before/after surface per arm is enough to visualize genuine
        # learning; jitter seeds remain fully represented in the CSV metrics.
        if CONFIG.save_outputs && CONFIG.save_plots && seed == 0
            save_surface_plot(arm, seed, run.initial_w, run.final_w)
        end
    end
end

summary = DataFrame(summary_rows)
history = DataFrame(history_rows)
weights = DataFrame(weight_rows)

aggregate = combine(
    groupby(summary, :arm),
    :initial_test_mse => mean => :mean_initial_test_mse,
    :final_test_mse => mean => :mean_final_test_mse,
    :test_mse_gain => mean => :mean_test_mse_gain,
    :final_test_mse => median => :median_final_test_mse,
    :relative_test_mse_reduction => mean => :mean_relative_test_mse_reduction,
    :seconds => mean => :mean_seconds,
    nrow => :runs,
)

println("\nPer-run summary")
show(summary; allrows = true, allcols = true)
println("\n\nAggregate (all pre-specified seeds; no seed selection)")
show(aggregate; allrows = true, allcols = true)
println()

if CONFIG.save_outputs
    CSV.write(joinpath(OUTPUT_DIRECTORY, "summary.csv"), summary)
    CSV.write(joinpath(OUTPUT_DIRECTORY, "history.csv"), history)
    CSV.write(joinpath(OUTPUT_DIRECTORY, "weights.csv"), weights)
    CSV.write(joinpath(OUTPUT_DIRECTORY, "aggregate.csv"), aggregate)
    open(joinpath(OUTPUT_DIRECTORY, "configuration.txt"), "w") do io
        println(io, CONFIG)
        println(io, "feature_arms = ", FEATURE_ARMS)
        println(io, "constant_prediction = ", constant_prediction)
        println(io, "constant_test_mse = ", constant_test_mse)
        println(io, "no_seed_selection = true")
        println(io, "iteration0_is_initial_q = true")
        println(io, "prediction_integrates_qw = true")
    end
    println("Outputs: $OUTPUT_DIRECTORY")
end
