ENV["PERF_BATCH_SIZE"] = get(ENV, "PERF_BATCH_SIZE", "64")
ENV["PERF_SAMPLES"] = get(ENV, "PERF_SAMPLES", "5")

include(joinpath(@__DIR__, "rxinfer_hotpaths.jl"))
include(joinpath(@__DIR__, "static_prediction.jl"))

const STATIC_ITERATIONS = ITERATION_CONFIG.prediction_iterations
const STATIC_LANES = length(ITERATION_FEATURES)
const STATIC_SAMPLES =
    parse(Int, get(ENV, "PERF_SAMPLES", "5"))
const STATIC_TOLERANCE =
    parse(Float64, get(ENV, "STATIC_TOLERANCE", "1e-10"))
const STATIC_THREADED =
    get(ENV, "STATIC_THREADED", "false") == "true"
const STATIC_PROGRAM = StaticPredictionProgram(
    ITERATION_PRIORS,
    ITERATION_DEPENDENCIES.projection,
    ITERATION_CONFIG.n_neurons,
    STATIC_LANES,
)

function run_static_bootstrap_wave!(; threaded::Bool)
    bootstrap = RxInfer.__run_batch_inference!(
        ITERATION_MODEL;
        rematerialize = false,
        initialization = iteration_initialization(),
        constraints = ITERATION_STATE.constraints,
        data = ITERATION_DATA,
        returnvars = (y = KeepLast(),),
        predictvars = ITERATION_PREDICTVARS,
        iterations = 1,
        free_energy = false,
        free_energy_diagnostics =
            RxInfer.DefaultObjectiveDiagnosticChecks,
        showprogress = false,
        callbacks = nothing,
        postprocess = RxInfer.__default_batch_postprocess(
            ITERATION_STATE.options,
        ),
        warn = false,
        catch_exception = false,
        disable_inference_error_hint = true,
    )
    vardict =
        GraphPPL.variables(RxInfer.getvardict(ITERATION_MODEL))
    executor = compile_static_prediction(
        STATIC_PROGRAM,
        vardict,
        ITERATION_DEPENDENCIES.states,
        lanes = STATIC_LANES,
        nneurons = ITERATION_CONFIG.n_neurons,
        iterations = STATIC_ITERATIONS,
    )
    return execute_static_prediction!(executor; threaded)
end

function run_reactive_reference!(returnvars)
    return RxInfer.__run_batch_inference!(
        ITERATION_MODEL;
        rematerialize = false,
        initialization = iteration_initialization(),
        constraints = ITERATION_STATE.constraints,
        data = ITERATION_DATA,
        returnvars,
        predictvars = ITERATION_PREDICTVARS,
        iterations = STATIC_ITERATIONS,
        free_energy = false,
        free_energy_diagnostics =
            RxInfer.DefaultObjectiveDiagnosticChecks,
        showprogress = false,
        callbacks = nothing,
        postprocess = RxInfer.__default_batch_postprocess(
            ITERATION_STATE.options,
        ),
        warn = false,
        catch_exception = false,
        disable_inference_error_hint = true,
    )
end

function verify_static_history!()
    rematerialize_for_iteration!()
    reference = run_reactive_reference!((y = KeepEach(),))
    rematerialize_for_iteration!()
    candidate =
        run_static_bootstrap_wave!(; threaded = STATIC_THREADED)

    max_mean_error = 0.0
    max_variance_error = 0.0
    max_mean_location = (0, 0)
    max_variance_location = (0, 0)
    correctness_passed = true
    for iteration in 1:STATIC_ITERATIONS
        for lane in 1:STATIC_LANES
            expected =
                reference.posteriors[:y][iteration][lane]
            actual = candidate[iteration, lane]
            mean_error = abs(mean(actual) - mean(expected))
            variance_error = abs(var(actual) - var(expected))
            if mean_error > max_mean_error
                max_mean_error = mean_error
                max_mean_location = (iteration, lane)
            end
            if variance_error > max_variance_error
                max_variance_error = variance_error
                max_variance_location = (iteration, lane)
            end
            correctness_passed &= isapprox(
                mean(actual),
                mean(expected);
                atol = STATIC_TOLERANCE,
                rtol = STATIC_TOLERANCE,
            )
            correctness_passed &= isapprox(
                var(actual),
                var(expected);
                atol = STATIC_TOLERANCE,
                rtol = STATIC_TOLERANCE,
            )
        end
    end
    println(
        "STATIC_CORRECTNESS ",
        (
            passed = correctness_passed,
            max_mean_error,
            max_mean_location,
            max_variance_error,
            max_variance_location,
        ),
    )
    correctness_passed ||
        error("static marginal history exceeds tolerance")
    if STATIC_THREADED && Threads.nthreads() > 1
        rematerialize_for_iteration!()
        serial_candidate =
            run_static_bootstrap_wave!(; threaded = false)
        max_thread_mean_error = 0.0
        max_thread_variance_error = 0.0
        for index in eachindex(candidate, serial_candidate)
            threaded_value = candidate[index]
            serial_value = serial_candidate[index]
            max_thread_mean_error = max(
                max_thread_mean_error,
                abs(mean(threaded_value) - mean(serial_value)),
            )
            max_thread_variance_error = max(
                max_thread_variance_error,
                abs(var(threaded_value) - var(serial_value)),
            )
        end
        max_thread_mean_error <= STATIC_TOLERANCE ||
            error("threaded and serial marginal means disagree")
        max_thread_variance_error <= STATIC_TOLERANCE ||
            error("threaded and serial marginal variances disagree")
        println(
            "STATIC_THREAD_AGREEMENT ",
            (
                max_mean_error = max_thread_mean_error,
                max_variance_error =
                    max_thread_variance_error,
            ),
        )
    end
    return (;
        max_mean_error,
        max_variance_error,
        candidate,
    )
end

function collect_timings!(runner, samples::Int)
    timings = Float64[]
    allocations = Int[]
    sizehint!(timings, samples)
    sizehint!(allocations, samples)
    for _ in 1:samples
        rematerialize_for_iteration!()
        GC.gc()
        measurement = @timed runner()
        push!(timings, measurement.time)
        push!(allocations, measurement.bytes)
    end
    return timings, allocations
end

correctness = verify_static_history!()

# Compile both complete waves before collecting samples.
rematerialize_for_iteration!()
run_reactive_reference!((y = KeepLast(),))
rematerialize_for_iteration!()
run_static_bootstrap_wave!(; threaded = STATIC_THREADED)

reactive_times, reactive_bytes = collect_timings!(
    () -> run_reactive_reference!((y = KeepLast(),)),
    STATIC_SAMPLES,
)
static_times, static_bytes = collect_timings!(
    () -> run_static_bootstrap_wave!(; threaded = STATIC_THREADED),
    STATIC_SAMPLES,
)

median_reactive_time = median(reactive_times)
median_static_time = median(static_times)
median_reactive_bytes = median(reactive_bytes)
median_static_bytes = median(static_bytes)
time_ratio = median_static_time / median_reactive_time
allocation_ratio = median_static_bytes / median_reactive_bytes

println(
    "STATIC_TILE_SUMMARY ",
    (
        lanes = STATIC_LANES,
        iterations = STATIC_ITERATIONS,
        samples = STATIC_SAMPLES,
        threads = Threads.nthreads(),
        threaded = STATIC_THREADED,
        max_mean_error = correctness.max_mean_error,
        max_variance_error = correctness.max_variance_error,
        reactive_seconds = median_reactive_time,
        static_seconds = median_static_time,
        time_ratio,
        speedup = inv(time_ratio),
        reactive_bytes = median_reactive_bytes,
        static_bytes = median_static_bytes,
        allocation_ratio,
    ),
)

if STATIC_LANES == 64
    time_ratio <= 0.80 ||
        error(
            "count-64 static kill gate failed: time ratio $time_ratio exceeds 0.80",
        )
    allocation_ratio <= 1.0 ||
        error(
            "count-64 static kill gate failed: allocation ratio $allocation_ratio exceeds 1.0",
        )
end
