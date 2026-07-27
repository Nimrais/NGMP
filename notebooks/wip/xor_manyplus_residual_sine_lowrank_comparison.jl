# Structured versus mean-field softdot comparison for the ManyPlus
# residual-sine XOR experiment.
#
# The production experiment is loaded only through its definition section; its
# run section is never evaluated and the source file is not modified.
#
# Default (400-sample screen, bounded stabilization sweep, then a full run only
# for a screen survivor):
#   OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_manyplus_residual_sine_lowrank_comparison.jl
#
# Fast wiring check:
#   XOR_MANYPLUS_COMPARISON_SMOKE=true OPENBLAS_NUM_THREADS=1 \
#     julia --project=. \
#     experiments/xor_manyplus_residual_sine_lowrank_comparison.jl

const BASELINE_EXPERIMENT =
    joinpath(@__DIR__, "xor_manyplus_residual_sine_ngmp.jl")
const BASELINE_RUN_MARKER = """
# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
"""

let source = read(BASELINE_EXPERIMENT, String)
    sections = split(source, BASELINE_RUN_MARKER; limit = 2)
    length(sections) == 2 || error(
        "could not find the run-section marker in $(BASELINE_EXPERIMENT)",
    )
    Base.include_string(@__MODULE__, first(sections), BASELINE_EXPERIMENT)
end

using ProbabilisticEnsembling: LowRankMeta
using Printf
using Statistics

const COMPARISON_OUTPUT_DIR = joinpath(
    @__DIR__, "xor_manyplus_residual_sine_lowrank_comparison_output",
)
const COMPARISON_CSV = joinpath(COMPARISON_OUTPUT_DIR, "comparison.csv")

const MAX_TIME_RATIO = 0.85
const MAX_TEST_MSE_RATIO = 1.10
const MAX_CONSTANT_MSE_RATIO = 0.95
const EQUIVALENCE_ATOL = 1e-8
const EQUIVALENCE_RTOL = 1e-7
const CHECKED_POSTERIORS = (:w, :v, :za, :h, :c, :out, :τ, :τ_c, :obs_noise)

# ---------------------------------------------------------------------------
# Mean-field training arms
# ---------------------------------------------------------------------------

@model function xor_manyplus_residual_sine_meanfield_dense(
    n_neurons,
    features,
    y,
    priors,
    activation,
    activation_deps,
)
    local w, v, za, h, c, out

    τ ~ priors[:τ]
    τ_c ~ priors[:τ_c]
    obs_noise ~ priors[:obs_noise]

    for neuron in 1:n_neurons
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
    end

    for observation in eachindex(y)
        for neuron in 1:n_neurons
            za[neuron, observation] ~
                softdot(features[observation], w[neuron], τ)
            h[neuron, observation] ~ ResidualSine(za[neuron, observation]) where {
                dependencies = activation_deps,
                meta = activation,
            }
            c[neuron, observation] ~
                softdot(v[neuron], h[neuron, observation], τ_c)
        end
        out[observation] ~ ManyPlus(
            inputs = [c[neuron, observation] for neuron in 1:n_neurons],
        )
        y[observation] ~ NormalMeanPrecision(out[observation], obs_noise)
    end
end

@model function xor_manyplus_residual_sine_meanfield_lowrank(
    n_neurons,
    features,
    y,
    priors,
    activation,
    activation_deps,
)
    local w, v, za, h, c, out

    τ ~ priors[:τ]
    τ_c ~ priors[:τ_c]
    obs_noise ~ priors[:obs_noise]

    for neuron in 1:n_neurons
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
    end

    for observation in eachindex(y)
        for neuron in 1:n_neurons
            # LowRankMeta changes only the representation of the message to w.
            za[neuron, observation] ~
                softdot(features[observation], w[neuron], τ) where {
                    meta = LowRankMeta(),
                }
            h[neuron, observation] ~ ResidualSine(za[neuron, observation]) where {
                dependencies = activation_deps,
                meta = activation,
            }
            # Deliberately unchanged: this is the scalar v * h soft product.
            c[neuron, observation] ~
                softdot(v[neuron], h[neuron, observation], τ_c)
        end
        out[observation] ~ ManyPlus(
            inputs = [c[neuron, observation] for neuron in 1:n_neurons],
        )
        y[observation] ~ NormalMeanPrecision(out[observation], obs_noise)
    end
end

@constraints function xor_manyplus_meanfield_constraints()
    q(w, za, h, c, out, v, τ, τ_c, obs_noise) =
        q(w)q(za, h, c, out)q(v)q(τ)q(τ_c)q(obs_noise)

    q(w)::MomentForm()
end

@initialization function xor_manyplus_meanfield_initialization(priors, inits)
    # Unlike the structured arm's edge-message initialization, q(w) is a
    # separate variational factor here and is initialized directly.
    q(w) = deepcopy(priors[:w])
    q(v) = deepcopy(priors[:v])
    q(za) = inits.za
    q(h) = inits.h
    q(c) = inits.c
    q(out) = inits.out
    q(τ) = priors[:τ]
    q(τ_c) = priors[:τ_c]
    q(obs_noise) = priors[:obs_noise]
end

function run_meanfield_training(arm, observations, features, run_config;
    showprogress = false,
)
    arm in (:dense_meanfield, :lowrank_meanfield) ||
        throw(ArgumentError("unknown mean-field arm: $arm"))
    priors = make_manyplus_priors(run_config)
    activation = ResidualSineMeta(
        rho = run_config.phi_rho,
        omega = run_config.phi_omega,
    )
    model = if arm === :dense_meanfield
        xor_manyplus_residual_sine_meanfield_dense(
            n_neurons = run_config.n_neurons,
            priors = priors,
            activation = activation,
            activation_deps = make_activation_dependencies(run_config),
        )
    else
        xor_manyplus_residual_sine_meanfield_lowrank(
            n_neurons = run_config.n_neurons,
            priors = priors,
            activation = activation,
            activation_deps = make_activation_dependencies(run_config),
        )
    end
    result = infer(
        model = model,
        data = (y = observations, features = features),
        constraints = xor_manyplus_meanfield_constraints(),
        initialization = xor_manyplus_meanfield_initialization(
            priors, pushforward_inits(priors, features, run_config),
        ),
        iterations = run_config.iterations,
        free_energy = true,
        showprogress = showprogress,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    return (result = result, priors = priors)
end

function run_training_arm(arm, observations, features, run_config)
    if arm === :structured
        return run_manyplus_training(
            observations, features, run_config; showprogress = false,
        )
    end
    return run_meanfield_training(
        arm, observations, features, run_config; showprogress = false,
    )
end

# ---------------------------------------------------------------------------
# Validation and metrics
# ---------------------------------------------------------------------------

arm_name(arm) = String(arm)
representation_name(arm) =
    arm === :structured ? "structured_dense" :
    arm === :dense_meanfield ? "meanfield_dense" :
    "meanfield_lowrank_message"

function comparison_config_id(run_config)
    shape, rate = run_config.tau_prior
    return @sprintf(
        "alpha=%.3g;tau=(%.0f,%.0f)", run_config.ngmp_alpha, shape, rate,
    )
end

function make_comparison_data(run_config)
    dataset = make_checkerboard_dataset(
        n = run_config.n_samples,
        noise_std = run_config.noise_std,
        seed = run_config.data_seed,
        checkerboard_size = run_config.checkboard_size,
    )
    train_data, test_data = split_dataset(
        dataset;
        train_fraction = run_config.train_fraction,
        seed = run_config.split_seed,
    )
    return (
        train_data = train_data,
        test_data = test_data,
        train_features = build_features(train_data),
        test_features = build_features(test_data),
    )
end

as_distributions(value) = value isa AbstractArray ? value : (value,)

function distribution_statistics(distribution)
    distribution_mean = mean(distribution)
    if distribution_mean isa Number
        return (
            mean = [Float64(distribution_mean)],
            covariance = [Float64(var(distribution))],
            variances = [Float64(var(distribution))],
        )
    end
    covariance = Matrix(cov(distribution))
    return (
        mean = Float64.(vec(distribution_mean)),
        covariance = Float64.(vec(covariance)),
        variances = Float64.(diag(covariance)),
    )
end

function validate_final_posteriors(result)
    all(isfinite, result.free_energy) ||
        error("free energy contains a non-finite value")
    for variable in CHECKED_POSTERIORS
        haskey(result.posteriors, variable) ||
            error("training result is missing q($variable)")
        final_value = result.posteriors[variable][end]
        for distribution in as_distributions(final_value)
            statistics = distribution_statistics(distribution)
            all(isfinite, statistics.mean) ||
                error("q($variable) contains a non-finite mean")
            all(isfinite, statistics.covariance) ||
                error("q($variable) contains a non-finite covariance")
            all(value -> value > 0, statistics.variances) ||
                error("q($variable) contains a non-positive variance")
        end
    end
    return true
end

function posterior_difference(first_result, second_result)
    maximum_mean_difference = 0.0
    maximum_covariance_difference = 0.0
    maximum_mean_scale = 0.0
    maximum_covariance_scale = 0.0

    for variable in CHECKED_POSTERIORS
        first_value = first_result.posteriors[variable][end]
        second_value = second_result.posteriors[variable][end]
        first_distributions = collect(as_distributions(first_value))
        second_distributions = collect(as_distributions(second_value))
        length(first_distributions) == length(second_distributions) ||
            error("q($variable) has different sizes between mean-field arms")
        for (first_distribution, second_distribution) in
            zip(first_distributions, second_distributions)
            first_statistics = distribution_statistics(first_distribution)
            second_statistics = distribution_statistics(second_distribution)
            size(first_statistics.mean) == size(second_statistics.mean) ||
                error("q($variable) has incompatible mean dimensions")
            size(first_statistics.covariance) ==
                size(second_statistics.covariance) ||
                error("q($variable) has incompatible covariance dimensions")
            maximum_mean_difference = max(
                maximum_mean_difference,
                maximum(abs, first_statistics.mean .- second_statistics.mean),
            )
            maximum_covariance_difference = max(
                maximum_covariance_difference,
                maximum(abs,
                    first_statistics.covariance .-
                    second_statistics.covariance,
                ),
            )
            maximum_mean_scale = max(
                maximum_mean_scale,
                maximum(abs, first_statistics.mean),
                maximum(abs, second_statistics.mean),
            )
            maximum_covariance_scale = max(
                maximum_covariance_scale,
                maximum(abs, first_statistics.covariance),
                maximum(abs, second_statistics.covariance),
            )
        end
    end

    equivalent =
        maximum_mean_difference <=
            EQUIVALENCE_ATOL + EQUIVALENCE_RTOL * maximum_mean_scale &&
        maximum_covariance_difference <=
            EQUIVALENCE_ATOL + EQUIVALENCE_RTOL * maximum_covariance_scale
    return (
        equivalent = equivalent,
        maximum_mean_difference = maximum_mean_difference,
        maximum_covariance_difference = maximum_covariance_difference,
        maximum_mean_scale = maximum_mean_scale,
        maximum_covariance_scale = maximum_covariance_scale,
    )
end

function final_prediction_metrics(fit, comparison_data, run_config)
    result = fit.result
    final_iteration = length(result.posteriors[:w])
    final_priors = manyplus_prediction_priors(result, final_iteration)
    output_mean = mean(comparison_data.train_data.OT)

    train_timing = @timed begin
        train_statistics = predictive_statistics(predict_manyplus_marginals(
            final_priors,
            comparison_data.train_features,
            run_config;
            output_mean = output_mean,
        ))
        (
            mse = mean(abs2,
                train_statistics.mean .- comparison_data.train_data.OT,
            ),
            variance = train_statistics.variance,
        )
    end
    test_timing = @timed begin
        test_statistics = predictive_statistics(predict_manyplus_marginals(
            final_priors,
            comparison_data.test_features,
            run_config;
            output_mean = output_mean,
        ))
        (
            mse = mean(abs2,
                test_statistics.mean .- comparison_data.test_data.OT,
            ),
            variance = test_statistics.variance,
        )
    end
    constant_mse = mean(abs2, output_mean .- comparison_data.test_data.OT)
    return (
        train_mse = train_timing.value.mse,
        test_mse = test_timing.value.mse,
        constant_mse = constant_mse,
        train_prediction_seconds = train_timing.time,
        test_prediction_seconds = test_timing.time,
        prediction_allocated_bytes = train_timing.bytes + test_timing.bytes,
        minimum_predictive_variance = minimum(test_timing.value.variance),
        mean_predictive_variance = mean(test_timing.value.variance),
        maximum_predictive_variance = maximum(test_timing.value.variance),
    )
end

timing_compile_seconds(timing) =
    hasproperty(timing, :compile_time) ? timing.compile_time : 0.0

function comparison_row(stage, arm, run_config, comparison_data)
    tau_shape, tau_rate = run_config.tau_prior
    return (
        stage = String(stage),
        arm = arm_name(arm),
        representation = representation_name(arm),
        config_id = comparison_config_id(run_config),
        selected_configuration = false,
        decision = "not_evaluated",
        status = "failed",
        error = "",
        n_samples = run_config.n_samples,
        iterations = run_config.iterations,
        train_points = nrow(comparison_data.train_data),
        test_points = nrow(comparison_data.test_data),
        ngmp_alpha = Float64(run_config.ngmp_alpha),
        tau_prior_shape = Float64(tau_shape),
        tau_prior_rate = Float64(tau_rate),
        training_seconds = missing,
        training_allocated_bytes = missing,
        training_gc_seconds = missing,
        training_compile_seconds = missing,
        final_free_energy = missing,
        train_prediction_seconds = missing,
        test_prediction_seconds = missing,
        prediction_allocated_bytes = missing,
        train_mse = missing,
        test_mse = missing,
        constant_mse = missing,
        time_ratio_to_baseline = missing,
        test_mse_ratio_to_baseline = missing,
        beats_constant = false,
        time_gate = false,
        mse_gate = false,
        posterior_valid = false,
        predictive_valid = false,
        equivalent_to_dense = missing,
        max_posterior_mean_difference = missing,
        max_posterior_covariance_difference = missing,
        minimum_predictive_variance = missing,
        mean_predictive_variance = missing,
        maximum_predictive_variance = missing,
        softdot_precision = missing,
        product_precision = missing,
        observation_precision = missing,
        passes_gate = false,
        accepted = false,
    )
end

function run_evaluated_arm(stage, arm, run_config, comparison_data)
    row = comparison_row(stage, arm, run_config, comparison_data)
    fit = nothing
    try
        GC.gc()
        training_timing = @timed run_training_arm(
            arm,
            comparison_data.train_data.OT,
            comparison_data.train_features,
            run_config,
        )
        fit = training_timing.value
        validate_final_posteriors(fit.result)
        predictions = final_prediction_metrics(fit, comparison_data, run_config)
        final_iteration = length(fit.result.posteriors[:w])
        final_priors = manyplus_prediction_priors(fit.result, final_iteration)
        row = merge(row, (
            decision = "measured",
            status = "ok",
            training_seconds = training_timing.time,
            training_allocated_bytes = training_timing.bytes,
            training_gc_seconds = training_timing.gctime,
            training_compile_seconds = timing_compile_seconds(training_timing),
            final_free_energy = Float64(fit.result.free_energy[end]),
            train_prediction_seconds = predictions.train_prediction_seconds,
            test_prediction_seconds = predictions.test_prediction_seconds,
            prediction_allocated_bytes = predictions.prediction_allocated_bytes,
            train_mse = predictions.train_mse,
            test_mse = predictions.test_mse,
            constant_mse = predictions.constant_mse,
            posterior_valid = true,
            predictive_valid = true,
            minimum_predictive_variance =
                predictions.minimum_predictive_variance,
            mean_predictive_variance = predictions.mean_predictive_variance,
            maximum_predictive_variance =
                predictions.maximum_predictive_variance,
            softdot_precision = mean(final_priors[:τ]),
            product_precision = mean(final_priors[:τ_c]),
            observation_precision = mean(final_priors[:obs_noise]),
        ))
    catch exception
        row = merge(row, (
            error = sprint(showerror, exception),
            decision = "runtime_failure",
        ))
        fit = nothing
    end
    return (row = row, fit = fit)
end

function annotate_equivalence(dense_record, lowrank_record)
    if dense_record.fit === nothing || lowrank_record.fit === nothing
        return dense_record, lowrank_record
    end
    difference = posterior_difference(
        dense_record.fit.result, lowrank_record.fit.result,
    )
    annotation = (
        equivalent_to_dense = difference.equivalent,
        max_posterior_mean_difference = difference.maximum_mean_difference,
        max_posterior_covariance_difference =
            difference.maximum_covariance_difference,
    )
    return (
        (row = merge(dense_record.row, annotation), fit = dense_record.fit),
        (row = merge(lowrank_record.row, annotation), fit = lowrank_record.fit),
    )
end

function apply_candidate_gate(row, baseline_row; require_equivalence = false)
    row.status == "ok" || return row
    baseline_row.status == "ok" || return merge(row, (
        decision = "baseline_unavailable",
    ))
    time_ratio = row.training_seconds / baseline_row.training_seconds
    mse_ratio = row.test_mse / baseline_row.test_mse
    beats_constant = row.test_mse <= MAX_CONSTANT_MSE_RATIO * row.constant_mse
    time_gate = time_ratio <= MAX_TIME_RATIO
    mse_gate = mse_ratio <= MAX_TEST_MSE_RATIO
    equivalence_gate = !require_equivalence || row.equivalent_to_dense === true
    passes_gate =
        row.posterior_valid && row.predictive_valid && beats_constant &&
        time_gate && mse_gate && equivalence_gate
    return merge(row, (
        decision = passes_gate ? "screen_survivor" : "screen_rejected",
        time_ratio_to_baseline = time_ratio,
        test_mse_ratio_to_baseline = mse_ratio,
        beats_constant = beats_constant,
        time_gate = time_gate,
        mse_gate = mse_gate,
        passes_gate = passes_gate,
    ))
end

function write_comparison_rows(rows; output_path = COMPARISON_CSV)
    mkpath(dirname(output_path))
    CSV.write(output_path, DataFrame(rows))
    return output_path
end

function print_record(row)
    if row.status != "ok"
        println("  $(row.arm): FAILED: $(row.error)")
        return
    end
    println(
        "  $(row.arm): train = ", round(row.training_seconds; digits = 2),
        " s, allocated = ", round(row.training_allocated_bytes / 2.0^30; digits = 2),
        " GiB, train MSE = ", round(row.train_mse; digits = 4),
        ", test MSE = ", round(row.test_mse; digits = 4),
        ", FE = ", round(row.final_free_energy; digits = 3),
    )
end

function warm_training_specializations(screen_config, comparison_data)
    warm_count = min(8, nrow(comparison_data.train_data))
    warm_count > 0 || return
    warm_indices = 1:warm_count
    warm_config = merge(screen_config, (
        iterations = 1,
        prediction_iterations = 1,
    ))
    last_fit = nothing
    println("Warming structured, dense mean-field, and low-rank specializations ...")
    for arm in (:structured, :dense_meanfield, :lowrank_meanfield)
        try
            last_fit = run_training_arm(
                arm,
                comparison_data.train_data.OT[warm_indices],
                comparison_data.train_features[warm_indices],
                warm_config,
            )
        catch exception
            println("  warm-up $(arm) failed: ", sprint(showerror, exception))
        end
    end
    if last_fit !== nothing
        try
            priors = manyplus_prediction_priors(last_fit.result, 1)
            predictive_statistics(predict_manyplus_marginals(
                priors,
                comparison_data.test_features[1:min(2, length(comparison_data.test_features))],
                warm_config;
                output_mean = mean(comparison_data.train_data.OT),
            ))
        catch exception
            println("  prediction warm-up failed: ", sprint(showerror, exception))
        end
    end
    last_fit = nothing
    GC.gc()
end

function run_meanfield_pair!(rows, stage, run_config, comparison_data, baseline_row;
    output_path = COMPARISON_CSV,
)
    println("\n$(uppercasefirst(String(stage))) configuration ",
        comparison_config_id(run_config))
    dense_record = run_evaluated_arm(
        stage, :dense_meanfield, run_config, comparison_data,
    )
    print_record(dense_record.row)
    lowrank_record = run_evaluated_arm(
        stage, :lowrank_meanfield, run_config, comparison_data,
    )
    print_record(lowrank_record.row)
    dense_record, lowrank_record =
        annotate_equivalence(dense_record, lowrank_record)
    dense_record = (
        row = apply_candidate_gate(dense_record.row, baseline_row),
        fit = dense_record.fit,
    )
    lowrank_record = (
        row = apply_candidate_gate(
            lowrank_record.row, baseline_row; require_equivalence = true,
        ),
        fit = lowrank_record.fit,
    )
    if dense_record.row.equivalent_to_dense !== missing
        println(
            "  dense/low-rank posterior max differences: mean = ",
            @sprintf("%.3e", dense_record.row.max_posterior_mean_difference),
            ", covariance = ",
            @sprintf("%.3e", dense_record.row.max_posterior_covariance_difference),
            ", equivalent = ", dense_record.row.equivalent_to_dense,
        )
    end
    push!(rows, dense_record.row, lowrank_record.row)
    write_comparison_rows(rows; output_path = output_path)
    dense_record = nothing
    lowrank_record = nothing
    GC.gc()
    return nothing
end

function stabilization_configs(screen_config)
    configurations = Any[]
    for alpha in (0.05, 0.02, 0.01)
        for tau_prior in ((1e3, 1.0), (1e3, 3.0), (1e3, 10.0))
            candidate = merge(screen_config, (
                ngmp_alpha = alpha,
                tau_prior = tau_prior,
            ))
            comparison_config_id(candidate) == comparison_config_id(screen_config) &&
                continue
            push!(configurations, candidate)
        end
    end
    return configurations
end

function select_screen_configuration!(rows)
    survivor_indices = findall(row ->
        row.stage == "screen" && row.arm != "structured" && row.passes_gate,
        rows,
    )
    isempty(survivor_indices) && return nothing
    survivor_ids = unique(rows[index].config_id for index in survivor_indices)
    ordered_ids = sort(collect(survivor_ids); by = config_id -> begin
        indices = filter(index -> rows[index].config_id == config_id, survivor_indices)
        (-length(indices), minimum(rows[index].training_seconds for index in indices))
    end)
    selected_id = first(ordered_ids)
    for index in eachindex(rows)
        row = rows[index]
        if row.stage == "screen" && row.arm != "structured" &&
            row.config_id == selected_id
            selected = row.passes_gate
            rows[index] = merge(row, (
                selected_configuration = selected,
                decision = selected ? "selected_for_full" : row.decision,
            ))
        end
    end
    return selected_id
end

function env_bool(name, default)
    value = lowercase(get(ENV, name, string(default)))
    value in ("true", "1", "yes", "on") && return true
    value in ("false", "0", "no", "off") && return false
    throw(ArgumentError("$name must be true or false, got $(repr(value))"))
end

env_int(name, default) = parse(Int, get(ENV, name, string(default)))

function run_comparison()
    smoke = env_bool("XOR_MANYPLUS_COMPARISON_SMOKE", false)
    screen_samples = env_int(
        "XOR_MANYPLUS_SCREEN_SAMPLES", smoke ? 40 : 400,
    )
    screen_iterations = env_int(
        "XOR_MANYPLUS_SCREEN_ITERATIONS", smoke ? 2 : 100,
    )
    screen_prediction_iterations = env_int(
        "XOR_MANYPLUS_SCREEN_PREDICTION_ITERATIONS",
        smoke ? 1 : config.prediction_iterations,
    )
    run_sweep = env_bool("XOR_MANYPLUS_RUN_SWEEP", !smoke)
    run_full = env_bool("XOR_MANYPLUS_RUN_FULL", !smoke)
    output_path = get(ENV, "XOR_MANYPLUS_COMPARISON_CSV", COMPARISON_CSV)

    screen_config = merge(config, (
        n_samples = screen_samples,
        iterations = screen_iterations,
        prediction_iterations = screen_prediction_iterations,
    ))
    screen_data = make_comparison_data(screen_config)
    rows = NamedTuple[]

    println("Mean-field/LowRankMeta screen: $(screen_config.n_samples) samples, ",
        "$(nrow(screen_data.train_data)) train, $(screen_config.iterations) iterations")
    warm_training_specializations(screen_config, screen_data)

    println("\nScreen structured baseline")
    baseline_record = run_evaluated_arm(
        :screen, :structured, screen_config, screen_data,
    )
    baseline_row = merge(baseline_record.row, (
        decision = baseline_record.row.status == "ok" ? "reference" :
            baseline_record.row.decision,
        time_ratio_to_baseline = baseline_record.row.status == "ok" ? 1.0 : missing,
        test_mse_ratio_to_baseline =
            baseline_record.row.status == "ok" ? 1.0 : missing,
    ))
    print_record(baseline_row)
    push!(rows, baseline_row)
    write_comparison_rows(rows; output_path = output_path)
    baseline_record = nothing
    GC.gc()
    baseline_row.status == "ok" || error(
        "structured screen baseline failed; see $output_path",
    )

    run_meanfield_pair!(
        rows, :screen, screen_config, screen_data, baseline_row;
        output_path = output_path,
    )
    default_survived = any(row ->
        row.stage == "screen" && row.arm != "structured" &&
        row.config_id == comparison_config_id(screen_config) && row.passes_gate,
        rows,
    )

    if !default_survived && run_sweep
        println("\nThe unchanged mean-field configuration failed the screen gate; ",
            "running the bounded alpha/tau stabilization sweep.")
        for sweep_config in stabilization_configs(screen_config)
            run_meanfield_pair!(
                rows, :screen, sweep_config, screen_data, baseline_row;
                output_path = output_path,
            )
        end
    end

    selected_id = select_screen_configuration!(rows)
    write_comparison_rows(rows; output_path = output_path)
    if selected_id === nothing
        println("\nNo mean-field candidate passed the seeded screen. ",
            "The structured method is retained; no full run or repository suite was run.")
        println("Comparison CSV: $output_path")
        return (rows = rows, accepted = false, output_path = output_path)
    end

    println("\nSelected screen configuration: $selected_id")
    if !run_full
        println("Full comparison disabled; screen results are in $output_path")
        return (rows = rows, accepted = false, output_path = output_path)
    end

    selected_screen_rows = filter(row ->
        row.stage == "screen" && row.config_id == selected_id &&
        row.selected_configuration,
        rows,
    )
    selected_reference = first(selected_screen_rows)
    full_baseline_config = config
    full_candidate_config = merge(config, (
        ngmp_alpha = selected_reference.ngmp_alpha,
        tau_prior = (
            selected_reference.tau_prior_shape,
            selected_reference.tau_prior_rate,
        ),
    ))
    full_data = make_comparison_data(full_baseline_config)

    println("\nFull structured baseline: $(full_baseline_config.n_samples) samples, ",
        "$(nrow(full_data.train_data)) train, ",
        "$(full_baseline_config.iterations) iterations")
    full_baseline_record = run_evaluated_arm(
        :full, :structured, full_baseline_config, full_data,
    )
    full_baseline_row = merge(full_baseline_record.row, (
        decision = full_baseline_record.row.status == "ok" ? "reference" :
            full_baseline_record.row.decision,
        time_ratio_to_baseline =
            full_baseline_record.row.status == "ok" ? 1.0 : missing,
        test_mse_ratio_to_baseline =
            full_baseline_record.row.status == "ok" ? 1.0 : missing,
    ))
    print_record(full_baseline_row)
    push!(rows, full_baseline_row)
    write_comparison_rows(rows; output_path = output_path)
    full_baseline_record = nothing
    GC.gc()
    full_baseline_row.status == "ok" || error(
        "structured full baseline failed; see $output_path",
    )

    full_records = Dict{Symbol, Any}()
    for screen_row in selected_screen_rows
        arm = Symbol(screen_row.arm)
        record = run_evaluated_arm(:full, arm, full_candidate_config, full_data)
        record = (
            row = merge(record.row, (
                selected_configuration = true,
                # If only the low-rank arm cleared the timing screen, retain
                # the dense/low-rank algebra check already made at this exact
                # configuration. A paired full run below supersedes it.
                equivalent_to_dense = screen_row.equivalent_to_dense,
                max_posterior_mean_difference =
                    screen_row.max_posterior_mean_difference,
                max_posterior_covariance_difference =
                    screen_row.max_posterior_covariance_difference,
            )),
            fit = record.fit,
        )
        full_records[arm] = record
    end
    if haskey(full_records, :dense_meanfield) &&
        haskey(full_records, :lowrank_meanfield)
        dense_record, lowrank_record = annotate_equivalence(
            full_records[:dense_meanfield], full_records[:lowrank_meanfield],
        )
        full_records[:dense_meanfield] = dense_record
        full_records[:lowrank_meanfield] = lowrank_record
    end

    accepted = false
    for arm in (:dense_meanfield, :lowrank_meanfield)
        haskey(full_records, arm) || continue
        record = full_records[arm]
        require_equivalence = arm === :lowrank_meanfield
        row = apply_candidate_gate(
            record.row,
            full_baseline_row;
            require_equivalence = require_equivalence,
        )
        row = merge(row, (
            accepted = row.passes_gate,
            decision = row.passes_gate ? "accepted" : "full_rejected",
        ))
        accepted |= row.accepted
        print_record(row)
        push!(rows, row)
    end
    write_comparison_rows(rows; output_path = output_path)
    empty!(full_records)
    GC.gc()

    if accepted
        println("\nAt least one candidate passed the full acceptance gate. ",
            "Run the full repository test suite once before promotion.")
    else
        println("\nNo candidate passed the full acceptance gate. ",
            "The structured production method is retained.")
    end
    println("Comparison CSV: $output_path")
    return (rows = rows, accepted = accepted, output_path = output_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_comparison()
end
