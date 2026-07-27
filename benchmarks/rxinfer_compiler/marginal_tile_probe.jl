ENV["PERF_BATCH_SIZE"] = get(ENV, "PERF_BATCH_SIZE", "2")
ENV["PERF_SAMPLES"] = get(ENV, "PERF_SAMPLES", "1")

include(joinpath(@__DIR__, "rxinfer_hotpaths.jl"))
include(joinpath(@__DIR__, "static_prediction.jl"))

const TILE_ITERATIONS =
    parse(Int, get(ENV, "TILE_ITERATIONS", "8"))
const TILE_TOLERANCE =
    parse(Float64, get(ENV, "TILE_TOLERANCE", "1e-10"))

function run_probe_wave!(iterations, returnvars)
    return RxInfer.__run_batch_inference!(
        ITERATION_MODEL;
        rematerialize = false,
        initialization = iteration_initialization(),
        constraints = ITERATION_STATE.constraints,
        data = ITERATION_DATA,
        returnvars = returnvars,
        predictvars = ITERATION_PREDICTVARS,
        iterations = iterations,
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

rematerialize_for_iteration!()
const TILE_REFERENCE =
    run_probe_wave!(TILE_ITERATIONS, (y = KeepEach(),))

rematerialize_for_iteration!()
const TILE_BOOTSTRAP = run_probe_wave!(1, (y = KeepLast(),))
const TILE_VARDICT =
    GraphPPL.variables(RxInfer.getvardict(ITERATION_MODEL))
const TILE_PROGRAM = StaticTileProgram(
    ITERATION_PRIORS,
    ITERATION_DEPENDENCIES.projection,
    ITERATION_CONFIG.n_neurons,
    length(ITERATION_FEATURES),
)
const TILE_STATES = extract_static_tiles(
    TILE_VARDICT,
    ITERATION_DEPENDENCIES.states,
    length(ITERATION_FEATURES),
    ITERATION_CONFIG.n_neurons,
    TILE_PROGRAM,
)
const TILE_HISTORY = Matrix{
    NormalWeightedMeanPrecision{Float64}
}(
    undef,
    TILE_ITERATIONS,
    length(TILE_STATES),
)

run_static_sweeps!(
    TILE_HISTORY,
    TILE_STATES,
    TILE_PROGRAM,
    TILE_ITERATIONS;
    threaded = false,
)

max_mean_error = 0.0
max_variance_error = 0.0
for iteration in 1:TILE_ITERATIONS
    for lane in eachindex(TILE_STATES)
        reference = TILE_REFERENCE.posteriors[:y][iteration][lane]
        candidate = TILE_HISTORY[iteration, lane]
        mean_error = abs(mean(candidate) - mean(reference))
        variance_error = abs(var(candidate) - var(reference))
        global max_mean_error = max(max_mean_error, mean_error)
        global max_variance_error =
            max(max_variance_error, variance_error)
        isapprox(
            mean(candidate),
            mean(reference);
            atol = TILE_TOLERANCE,
            rtol = TILE_TOLERANCE,
        ) || error(
            "mean mismatch at iteration=$iteration lane=$lane: $mean_error",
        )
        isapprox(
            var(candidate),
            var(reference);
            atol = TILE_TOLERANCE,
            rtol = TILE_TOLERANCE,
        ) || error(
            "variance mismatch at iteration=$iteration lane=$lane: $variance_error",
        )
    end
end

println(
    "static tile correctness passed",
    " lanes=", length(TILE_STATES),
    " iterations=", TILE_ITERATIONS,
    " max_mean_error=", max_mean_error,
    " max_variance_error=", max_variance_error,
    " state_type=", eltype(TILE_STATES),
)
