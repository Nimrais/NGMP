module RxInferBackendHarness

using Distributions
using LinearAlgebra
using Printf
using Random
using RxInfer
using Statistics

import ReactiveMP

export BackendAdapter,
    BackendRun,
    HarnessCase,
    HarnessConfig,
    IterationSnapshot,
    default_cases,
    make_rxinfer_adapter,
    register_backend!,
    run_harness,
    run_rxinfer_backend

const SCHEMA_VERSION = "1"

"""
    BackendAdapter(name, execute)

Register a backend through a callable with the signature
`execute(case, sample, config)::BackendRun`.  Most RxInfer backends can use
[`make_rxinfer_adapter`](@ref), which only requires keyword transformations.
"""
struct BackendAdapter{F}
    name::String
    execute::F
end

BackendAdapter(name::Symbol, execute) = BackendAdapter(String(name), execute)
const BACKENDS = BackendAdapter[]

"""
One deterministic model workload. `build(sample, config)` returns
`(prepare = <keywords>, run = <keywords>)`.
"""
struct HarnessCase{F}
    name::String
    build::F
    output_variables::Vector{Symbol}
    requested_iterations::Int
    stop_after::Int
end

struct IterationSnapshot
    iteration::Int
    free_energy::Float64
    metrics::Dict{String, Vector{Float64}}
end

struct BackendRun
    case_name::String
    backend_name::String
    sample::Int
    status::Symbol
    error::String
    preparation_seconds::Float64
    preparation_bytes::Int
    preparation_gc_seconds::Float64
    preparation_compile_seconds::Float64
    inference_seconds::Float64
    inference_bytes::Int
    inference_gc_seconds::Float64
    inference_compile_seconds::Float64
    requested_iterations::Int
    executed_iterations::Int
    stopped_early::Bool
    callback_iterations::Vector{Int}
    callback_fe_matches_result::Bool
    trace::Vector{IterationSnapshot}
end

Base.@kwdef struct HarnessConfig
    samples::Int = 3
    warmup::Bool = true
    chain_states::Int = 24
    high_degree::Int = 128
    atol::Float64 = 1e-8
    rtol::Float64 = 1e-7
    reference::String = "reactive"
    output_dir::String =
        joinpath(tempdir(), "rxinfer_backend_differential")
    fail_on_mismatch::Bool = true
    run_hotpath::Bool = false
    hotpath_samples::Int = 1
end

mutable struct IterationRecorder
    iterations::Vector{Int}
    free_energy::Vector{Float64}
    stop_after::Int
end

IterationRecorder(stop_after) =
    IterationRecorder(Int[], Float64[], stop_after)

function (recorder::IterationRecorder)(event)
    isnothing(event.free_energy) &&
        error("after_iteration did not contain free energy")
    push!(recorder.iterations, event.iteration)
    push!(recorder.free_energy, Float64(event.free_energy))
    if event.iteration >= recorder.stop_after
        event.stop_iteration = true
    end
    return nothing
end

function clear!(recorder::IterationRecorder)
    empty!(recorder.iterations)
    empty!(recorder.free_energy)
    return recorder
end

@model function differential_beta_bernoulli(y)
    theta ~ Beta(2.0, 2.0)
    y .~ Bernoulli(theta)
end

@model function differential_gaussian_chain(
    y,
    process_variance,
    observation_variance,
)
    x_previous ~ Normal(mean = 0.0, var = 1.0)
    for index in eachindex(y)
        x[index] ~
            Normal(mean = x_previous, var = process_variance)
        y[index] ~
            Normal(mean = x[index], var = observation_variance)
        x_previous = x[index]
    end
end

@model function differential_high_degree_gaussian(
    y,
    prior_variance,
    observation_variance,
)
    mu ~ Normal(mean = 0.0, var = prior_variance)
    for index in eachindex(y)
        y[index] ~
            Normal(mean = mu, var = observation_variance)
    end
end

function beta_request(sample, config, iterations)
    rng = MersenneTwister(10_000 + sample)
    initial = Float64.(rand(rng, Bernoulli(0.58), 20))
    observed = Float64.(rand(rng, Bernoulli(0.72), 20))
    return (
        prepare = (
            model = differential_beta_bernoulli(),
            data = (y = initial,),
            free_energy = Float64,
            warn = false,
        ),
        run = (
            data = (y = observed,),
            iterations = iterations,
            returnvars = (theta = KeepEach(),),
            free_energy = Float64,
            showprogress = false,
            warn = false,
            disable_inference_error_hint = true,
        ),
    )
end

function gaussian_chain_data(n, seed)
    rng = MersenneTwister(seed)
    hidden = Vector{Float64}(undef, n)
    observed = Vector{Float64}(undef, n)
    previous = 0.0
    for index in eachindex(hidden)
        hidden[index] = previous + sqrt(0.2) * randn(rng)
        observed[index] = hidden[index] + sqrt(0.5) * randn(rng)
        previous = hidden[index]
    end
    return observed
end

function gaussian_chain_request(sample, config, iterations)
    n = config.chain_states
    initial = gaussian_chain_data(n, 20_000 + 2sample)
    observed = gaussian_chain_data(n, 20_001 + 2sample)
    return (
        prepare = (
            model = differential_gaussian_chain(
                process_variance = 0.2,
                observation_variance = 0.5,
            ),
            data = (y = initial,),
            free_energy = Float64,
            warn = false,
        ),
        run = (
            data = (y = observed,),
            iterations = iterations,
            returnvars = (x = KeepEach(),),
            free_energy = Float64,
            showprogress = false,
            warn = false,
            disable_inference_error_hint = true,
        ),
    )
end

function high_degree_request(sample, config, iterations)
    rng = MersenneTwister(30_000 + sample)
    initial = 1.25 .+ sqrt(0.7) .* randn(rng, config.high_degree)
    observed = -0.4 .+ sqrt(0.7) .* randn(rng, config.high_degree)
    return (
        prepare = (
            model = differential_high_degree_gaussian(
                prior_variance = 4.0,
                observation_variance = 0.7,
            ),
            data = (y = initial,),
            free_energy = Float64,
            warn = false,
        ),
        run = (
            data = (y = observed,),
            iterations = iterations,
            returnvars = (mu = KeepEach(),),
            free_energy = Float64,
            showprogress = false,
            warn = false,
            disable_inference_error_hint = true,
        ),
    )
end

function default_cases()
    iterations = 5
    stop_after = 3
    return [
        HarnessCase(
            "beta_bernoulli",
            (sample, config) ->
                beta_request(sample, config, iterations),
            [:theta],
            iterations,
            stop_after,
        ),
        HarnessCase(
            "small_gaussian_chain",
            (sample, config) ->
                gaussian_chain_request(sample, config, iterations),
            [:x],
            iterations,
            stop_after,
        ),
        HarnessCase(
            "high_degree_gaussian",
            (sample, config) ->
                high_degree_request(sample, config, iterations),
            [:mu],
            iterations,
            stop_after,
        ),
    ]
end

function transform_keywords(transform, keywords, case, sample)
    transformed = if applicable(transform, keywords, case, sample)
        transform(keywords, case, sample)
    elseif applicable(transform, keywords, case)
        transform(keywords, case)
    else
        transform(keywords)
    end
    transformed isa NamedTuple ||
        throw(
            ArgumentError(
                "backend keyword transforms must return a NamedTuple",
            ),
        )
    return transformed
end

timed_field(measurement, name, default) =
    hasproperty(measurement, name) ? getproperty(measurement, name) :
    default

function append_numeric!(destination, value)
    if value isa Number
        push!(destination, Float64(value))
    elseif value isa AbstractArray || value isa Tuple
        for element in value
            append_numeric!(destination, element)
        end
    else
        throw(
            ArgumentError(
                "cannot flatten moment value of type $(typeof(value))",
            ),
        )
    end
    return destination
end

function append_moments!(means, variances, value)
    if value isa AbstractArray || value isa Tuple
        for element in value
            append_moments!(means, variances, element)
        end
    else
        append_numeric!(means, mean(value))
        append_numeric!(variances, var(value))
    end
    return nothing
end

function result_trace(result, case, recorder)
    executed = length(recorder.iterations)
    expected_iterations = collect(1:executed)
    recorder.iterations == expected_iterations ||
        error(
            "callback iterations $(recorder.iterations) did not equal $expected_iterations",
        )

    result_free_energy = Float64.(result.free_energy)
    length(result_free_energy) == executed ||
        error(
            "free-energy history has $(length(result_free_energy)) values, expected $executed",
        )
    callback_matches =
        length(recorder.free_energy) == length(result_free_energy) &&
        all(isequal.(recorder.free_energy, result_free_energy))

    histories = Dict{Symbol, Any}()
    for variable in case.output_variables
        haskey(result.posteriors, variable) ||
            error("result has no posterior history for :$variable")
        history = result.posteriors[variable]
        length(history) == executed ||
            error(
                "posterior :$variable has $(length(history)) iterations, expected $executed",
            )
        histories[variable] = history
    end

    trace = IterationSnapshot[]
    for iteration in 1:executed
        metrics = Dict{String, Vector{Float64}}()
        for variable in sort(case.output_variables; by = String)
            means = Float64[]
            variances = Float64[]
            append_moments!(
                means,
                variances,
                histories[variable][iteration],
            )
            metrics["$(variable).mean"] = means
            metrics["$(variable).variance"] = variances
        end
        push!(
            trace,
            IterationSnapshot(
                iteration,
                result_free_energy[iteration],
                metrics,
            ),
        )
    end
    return trace, callback_matches
end

"""
    run_rxinfer_backend(case, sample, config; backend_name,
                        prepare_transform=identity, run_transform=identity)

Execute an RxInfer-compatible backend. Transform functions receive the
keyword `NamedTuple` and may optionally also accept `(case, sample)`. The
measured inference is the second `infer!` call on the prepared model, so the
measurement covers prepared-topology runtime rematerialization rather than
first-use setup.
"""
function run_rxinfer_backend(
    case,
    sample,
    config;
    backend_name,
    prepare_transform = identity,
    run_transform = identity,
)
    request = case.build(sample, config)
    recorder = IterationRecorder(case.stop_after)
    prepare_keywords = merge(
        request.prepare,
        (callbacks = (after_iteration = recorder,),),
    )
    prepare_keywords = transform_keywords(
        prepare_transform,
        prepare_keywords,
        case,
        sample,
    )
    run_keywords =
        transform_keywords(run_transform, request.run, case, sample)

    GC.gc()
    preparation = @timed prepare_inference(; prepare_keywords...)
    prepared = preparation.value

    # Prime the prepared model and Julia specializations. The following call
    # is the reusable-model path and is the one measured below.
    infer!(prepared; run_keywords...)
    clear!(recorder)

    GC.gc()
    inference = @timed infer!(prepared; run_keywords...)
    result = inference.value
    trace, callback_matches = result_trace(result, case, recorder)
    executed = length(recorder.iterations)

    return BackendRun(
        case.name,
        String(backend_name),
        sample,
        :ok,
        "",
        Float64(preparation.time),
        Int(preparation.bytes),
        Float64(preparation.gctime),
        Float64(timed_field(preparation, :compile_time, 0.0)),
        Float64(inference.time),
        Int(inference.bytes),
        Float64(inference.gctime),
        Float64(timed_field(inference, :compile_time, 0.0)),
        case.requested_iterations,
        executed,
        executed < case.requested_iterations,
        copy(recorder.iterations),
        callback_matches,
        trace,
    )
end

"""
    make_rxinfer_adapter(name; prepare_transform=identity,
                         run_transform=identity)

Build an adapter for a backend exposed through RxInfer keyword arguments.
This deliberately does not assume the name or shape of a future compiled
backend API.
"""
function make_rxinfer_adapter(
    name;
    prepare_transform = identity,
    run_transform = identity,
)
    backend_name = String(name)
    execute = (case, sample, config) -> run_rxinfer_backend(
        case,
        sample,
        config;
        backend_name = backend_name,
        prepare_transform = prepare_transform,
        run_transform = run_transform,
    )
    return BackendAdapter(backend_name, execute)
end

function register_backend!(adapter::BackendAdapter)
    any(existing -> existing.name == adapter.name, BACKENDS) &&
        throw(
            ArgumentError(
                "a backend named $(repr(adapter.name)) is already registered",
            ),
        )
    push!(BACKENDS, adapter)
    return adapter
end

function failed_run(case, adapter, sample, exception, backtrace)
    message = sprint(showerror, exception, backtrace)
    return BackendRun(
        case.name,
        adapter.name,
        sample,
        :error,
        message,
        NaN,
        0,
        NaN,
        NaN,
        NaN,
        0,
        NaN,
        NaN,
        case.requested_iterations,
        0,
        false,
        Int[],
        false,
        IterationSnapshot[],
    )
end

function validate_backend_run(run, case, adapter, sample)
    run isa BackendRun ||
        throw(
            ArgumentError(
                "backend $(repr(adapter.name)) returned $(typeof(run)); expected BackendRun",
            ),
        )
    run.case_name == case.name ||
        throw(
            ArgumentError(
                "backend returned case $(repr(run.case_name)); expected $(repr(case.name))",
            ),
        )
    run.backend_name == adapter.name ||
        throw(
            ArgumentError(
                "backend returned name $(repr(run.backend_name)); expected $(repr(adapter.name))",
            ),
        )
    run.sample == sample ||
        throw(
            ArgumentError(
                "backend returned sample $(run.sample); expected $sample",
            ),
        )
    return run
end

struct Comparison
    ok::Bool
    iteration_match::Bool
    stopping_match::Bool
    max_abs_output_error::Float64
    max_rel_output_error::Float64
    max_abs_free_energy_error::Float64
    max_rel_free_energy_error::Float64
end

function scalar_error(reference, candidate)
    if !isfinite(reference) || !isfinite(candidate)
        return Inf, Inf, false
    elseif isequal(reference, candidate)
        return 0.0, 0.0, true
    end
    absolute = abs(candidate - reference)
    relative = absolute / max(abs(reference), eps(Float64))
    return absolute, relative, true
end

function compare_runs(reference, candidate, case, config)
    both_ok = reference.status === :ok && candidate.status === :ok
    both_ok ||
        return Comparison(false, false, false, Inf, Inf, Inf, Inf)

    expected_executed =
        min(case.stop_after, case.requested_iterations)
    expected_iterations = collect(1:expected_executed)
    expected_stopped_early =
        expected_executed < case.requested_iterations
    iteration_match =
        reference.requested_iterations == case.requested_iterations &&
        candidate.requested_iterations == case.requested_iterations &&
        reference.executed_iterations == expected_executed &&
        candidate.executed_iterations == expected_executed &&
        reference.callback_iterations == expected_iterations &&
        candidate.callback_iterations == expected_iterations &&
        reference.executed_iterations == candidate.executed_iterations &&
        reference.callback_iterations == candidate.callback_iterations
    stopping_match =
        reference.stopped_early == expected_stopped_early &&
        candidate.stopped_early == expected_stopped_early &&
        reference.stopped_early == candidate.stopped_early

    max_abs_output = 0.0
    max_rel_output = 0.0
    max_abs_fe = 0.0
    max_rel_fe = 0.0
    outputs_match = true
    fe_match = true

    if length(reference.trace) != length(candidate.trace)
        outputs_match = false
        fe_match = false
        max_abs_output = Inf
        max_rel_output = Inf
        max_abs_fe = Inf
        max_rel_fe = Inf
    else
        for (left, right) in zip(reference.trace, candidate.trace)
            if left.iteration != right.iteration ||
               keys(left.metrics) != keys(right.metrics)
                outputs_match = false
                max_abs_output = Inf
                max_rel_output = Inf
            else
                for metric in sort!(collect(keys(left.metrics)))
                    left_values = left.metrics[metric]
                    right_values = right.metrics[metric]
                    if length(left_values) != length(right_values)
                        outputs_match = false
                        max_abs_output = Inf
                        max_rel_output = Inf
                        continue
                    end
                    for (left_value, right_value) in
                        zip(left_values, right_values)
                        absolute, relative, finite =
                            scalar_error(left_value, right_value)
                        max_abs_output =
                            max(max_abs_output, absolute)
                        max_rel_output =
                            max(max_rel_output, relative)
                        outputs_match &=
                            finite &&
                            isapprox(
                                left_value,
                                right_value;
                                atol = config.atol,
                                rtol = config.rtol,
                            )
                    end
                end
            end
            absolute, relative, finite =
                scalar_error(left.free_energy, right.free_energy)
            max_abs_fe = max(max_abs_fe, absolute)
            max_rel_fe = max(max_rel_fe, relative)
            fe_match &=
                finite &&
                isapprox(
                    left.free_energy,
                    right.free_energy;
                    atol = config.atol,
                    rtol = config.rtol,
                )
        end
    end

    internal_match =
        reference.callback_fe_matches_result &&
        candidate.callback_fe_matches_result
    ok =
        iteration_match &&
        stopping_match &&
        outputs_match &&
        fe_match &&
        internal_match
    return Comparison(
        ok,
        iteration_match,
        stopping_match,
        max_abs_output,
        max_rel_output,
        max_abs_fe,
        max_rel_fe,
    )
end

csv_value(value::Bool) = value ? "true" : "false"
csv_value(value::Symbol) = String(value)
csv_value(value::AbstractFloat) = @sprintf("%.17g", value)
csv_value(value) = string(value)

function csv_escape(value)
    text = csv_value(value)
    if occursin(',', text) ||
       occursin('"', text) ||
       occursin('\n', text) ||
       occursin('\r', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(csv_escape.(row), ','))
        end
    end
    return path
end

function run_rows(runs)
    rows = Vector{Any}[]
    for run in sort(
        runs;
        by = item -> (item.case_name, item.backend_name, item.sample),
    )
        total_seconds =
            run.preparation_seconds + run.inference_seconds
        total_bytes = run.preparation_bytes + run.inference_bytes
        push!(
            rows,
            Any[
                SCHEMA_VERSION,
                run.case_name,
                run.backend_name,
                run.sample,
                run.status,
                run.error,
                run.preparation_seconds,
                run.preparation_bytes,
                run.preparation_gc_seconds,
                run.preparation_compile_seconds,
                run.inference_seconds,
                run.inference_bytes,
                run.inference_gc_seconds,
                run.inference_compile_seconds,
                total_seconds,
                total_bytes,
                run.requested_iterations,
                run.executed_iterations,
                run.stopped_early,
                join(run.callback_iterations, ';'),
                run.callback_fe_matches_result,
            ],
        )
    end
    return rows
end

const RUN_HEADER = [
    "schema_version",
    "case",
    "backend",
    "sample",
    "status",
    "error",
    "preparation_seconds",
    "preparation_bytes",
    "preparation_gc_seconds",
    "preparation_compile_seconds",
    "inference_seconds",
    "inference_bytes",
    "inference_gc_seconds",
    "inference_compile_seconds",
    "total_seconds",
    "total_bytes",
    "requested_iterations",
    "executed_iterations",
    "stopped_early",
    "callback_iterations",
    "callback_fe_matches_result",
]

function trace_rows(runs)
    rows = Vector{Any}[]
    for run in sort(
        runs;
        by = item -> (item.case_name, item.backend_name, item.sample),
    )
        for snapshot in run.trace
            push!(
                rows,
                Any[
                    SCHEMA_VERSION,
                    run.case_name,
                    run.backend_name,
                    run.sample,
                    snapshot.iteration,
                    "free_energy",
                    1,
                    snapshot.free_energy,
                ],
            )
            for metric in sort!(collect(keys(snapshot.metrics)))
                for (index, value) in
                    enumerate(snapshot.metrics[metric])
                    push!(
                        rows,
                        Any[
                            SCHEMA_VERSION,
                            run.case_name,
                            run.backend_name,
                            run.sample,
                            snapshot.iteration,
                            metric,
                            index,
                            value,
                        ],
                    )
                end
            end
        end
    end
    return rows
end

const TRACE_HEADER = [
    "schema_version",
    "case",
    "backend",
    "sample",
    "iteration",
    "metric",
    "index",
    "value",
]

safe_median(values) = isempty(values) ? NaN : median(values)

function summary_rows(cases, adapters, runs, comparisons, config)
    rows = Vector{Any}[]
    for case in sort(cases; by = item -> item.name)
        reference_runs = [
            run for run in runs if
            run.case_name == case.name &&
            run.backend_name == config.reference &&
            run.status === :ok
        ]
        reference_inference_seconds =
            safe_median(getfield.(reference_runs, :inference_seconds))
        reference_inference_bytes =
            safe_median(Float64.(getfield.(reference_runs, :inference_bytes)))

        for adapter in sort(adapters; by = item -> item.name)
            group = [
                run for run in runs if
                run.case_name == case.name &&
                run.backend_name == adapter.name
            ]
            successful = [run for run in group if run.status === :ok]
            group_comparisons = [
                comparisons[(case.name, adapter.name, run.sample)] for
                run in group
            ]
            max_abs_output =
                maximum(
                    getfield.(group_comparisons, :max_abs_output_error);
                    init = 0.0,
                )
            max_rel_output =
                maximum(
                    getfield.(group_comparisons, :max_rel_output_error);
                    init = 0.0,
                )
            max_abs_fe =
                maximum(
                    getfield.(
                        group_comparisons,
                        :max_abs_free_energy_error,
                    );
                    init = 0.0,
                )
            max_rel_fe =
                maximum(
                    getfield.(
                        group_comparisons,
                        :max_rel_free_energy_error,
                    );
                    init = 0.0,
                )
            preparation_seconds =
                safe_median(getfield.(successful, :preparation_seconds))
            inference_seconds =
                safe_median(getfield.(successful, :inference_seconds))
            total_seconds = safe_median([
                run.preparation_seconds + run.inference_seconds for
                run in successful
            ])
            preparation_bytes =
                safe_median(Float64.(getfield.(successful, :preparation_bytes)))
            inference_bytes =
                safe_median(Float64.(getfield.(successful, :inference_bytes)))
            total_bytes = safe_median([
                Float64(run.preparation_bytes + run.inference_bytes) for
                run in successful
            ])
            speedup =
                reference_inference_seconds / inference_seconds
            allocation_ratio =
                inference_bytes / reference_inference_bytes

            push!(
                rows,
                Any[
                    SCHEMA_VERSION,
                    case.name,
                    adapter.name,
                    length(group),
                    length(successful),
                    all(getfield.(group_comparisons, :ok)),
                    all(getfield.(group_comparisons, :iteration_match)),
                    all(getfield.(group_comparisons, :stopping_match)),
                    max_abs_output,
                    max_rel_output,
                    max_abs_fe,
                    max_rel_fe,
                    preparation_seconds,
                    inference_seconds,
                    total_seconds,
                    preparation_bytes,
                    inference_bytes,
                    total_bytes,
                    speedup,
                    allocation_ratio,
                ],
            )
        end
    end
    return rows
end

const SUMMARY_HEADER = [
    "schema_version",
    "case",
    "backend",
    "samples",
    "successful_samples",
    "all_correct",
    "iteration_match",
    "stopping_match",
    "max_abs_output_error",
    "max_rel_output_error",
    "max_abs_free_energy_error",
    "max_rel_free_energy_error",
    "median_preparation_seconds",
    "median_inference_seconds",
    "median_total_seconds",
    "median_preparation_bytes",
    "median_inference_bytes",
    "median_total_bytes",
    "inference_speedup_vs_reference",
    "inference_allocation_ratio_vs_reference",
]

function metadata_rows(config, hotpath_path)
    pairs = [
        "schema_version" => SCHEMA_VERSION,
        "julia_version" => string(VERSION),
        "rxinfer_version" => string(Base.pkgversion(RxInfer)),
        "reactivemp_version" => string(Base.pkgversion(ReactiveMP)),
        "reference_backend" => config.reference,
        "samples" => config.samples,
        "warmup" => config.warmup,
        "chain_states" => config.chain_states,
        "high_degree" => config.high_degree,
        "atol" => config.atol,
        "rtol" => config.rtol,
        "measurement" =>
            "second infer! call on a newly prepared model",
        "hotpath_available" => isfile(hotpath_path),
        "hotpath_path" => hotpath_path,
    ]
    return [Any[SCHEMA_VERSION, key, value] for (key, value) in pairs]
end

function run_hotpath_hook(config, hotpath_path)
    available = isfile(hotpath_path)
    requested = config.run_hotpath
    status = !available ? "unavailable" :
             !requested ? "not_requested" : "pending"
    output = joinpath(config.output_dir, "hotpath.csv")
    error_message = ""
    if available && requested
        posterior_output =
            joinpath(config.output_dir, "hotpath_posterior.csv")
        environment = copy(ENV)
        environment["PERF_PROJECT"] =
            normpath(joinpath(@__DIR__, "..", ".."))
        environment["PERF_SAMPLES"] = string(config.hotpath_samples)
        environment["PERF_LABEL"] = "differential-hook"
        environment["PERF_OUTPUT"] = output
        environment["PERF_POSTERIOR_OUTPUT"] = posterior_output
        command = `$(Base.julia_cmd()) --startup-file=no $hotpath_path`
        try
            run(setenv(command, environment))
            status = "ok"
        catch exception
            status = "error"
            error_message =
                sprint(showerror, exception, catch_backtrace())
        end
    end
    return Any[
        SCHEMA_VERSION,
        "rxinfer_hotpaths",
        available,
        requested,
        status,
        output,
        error_message,
    ]
end

function validate_config(config)
    config.samples > 0 ||
        throw(ArgumentError("samples must be positive"))
    config.chain_states > 0 ||
        throw(ArgumentError("chain_states must be positive"))
    config.high_degree > 0 ||
        throw(ArgumentError("high_degree must be positive"))
    config.hotpath_samples > 0 ||
        throw(ArgumentError("hotpath_samples must be positive"))
    config.atol >= 0 ||
        throw(ArgumentError("atol must be non-negative"))
    config.rtol >= 0 ||
        throw(ArgumentError("rtol must be non-negative"))
    return config
end

"""
    run_harness(config=HarnessConfig(); adapters, cases=default_cases())

Run correctness and performance samples, write stable long-form CSV files,
and return a named tuple containing runs, comparisons, and output paths.
"""
function run_harness(
    config = HarnessConfig();
    adapters = copy(BACKENDS),
    cases = default_cases(),
)
    validate_config(config)
    isempty(adapters) &&
        push!(adapters, make_rxinfer_adapter("reactive"))
    any(adapter -> adapter.name == config.reference, adapters) ||
        throw(
            ArgumentError(
                "reference backend $(repr(config.reference)) is not registered",
            ),
        )
    length(unique(getfield.(adapters, :name))) == length(adapters) ||
        throw(ArgumentError("backend names must be unique"))
    cases_by_name = Dict(case.name => case for case in cases)
    length(cases_by_name) == length(cases) ||
        throw(ArgumentError("case names must be unique"))
    for case in cases
        case.requested_iterations > 0 ||
            throw(
                ArgumentError(
                    "case $(repr(case.name)) must request at least one iteration",
                ),
            )
        case.stop_after > 0 ||
            throw(
                ArgumentError(
                    "case $(repr(case.name)) must stop after at least one iteration",
                ),
            )
    end

    mkpath(config.output_dir)

    if config.warmup
        for case in cases
            for adapter in adapters
                try
                    validate_backend_run(
                        adapter.execute(case, 0, config),
                        case,
                        adapter,
                        0,
                    )
                catch
                    # The measured call below records a stable error row.
                end
            end
        end
    end

    runs = BackendRun[]
    for sample in 1:config.samples
        order = isodd(sample) ? adapters : reverse(adapters)
        for case in cases
            for adapter in order
                run = try
                    validate_backend_run(
                        adapter.execute(case, sample, config),
                        case,
                        adapter,
                        sample,
                    )
                catch exception
                    failed_run(
                        case,
                        adapter,
                        sample,
                        exception,
                        catch_backtrace(),
                    )
                end
                push!(runs, run)
            end
        end
    end

    by_key = Dict(
        (run.case_name, run.backend_name, run.sample) => run for
        run in runs
    )
    comparisons = Dict{Tuple{String, String, Int}, Comparison}()
    for run in runs
        reference = by_key[
            (run.case_name, config.reference, run.sample)
        ]
        case = cases_by_name[run.case_name]
        comparisons[
            (run.case_name, run.backend_name, run.sample)
        ] = compare_runs(reference, run, case, config)
    end

    run_path = write_csv(
        joinpath(config.output_dir, "runs.csv"),
        RUN_HEADER,
        run_rows(runs),
    )
    trace_path = write_csv(
        joinpath(config.output_dir, "trace.csv"),
        TRACE_HEADER,
        trace_rows(runs),
    )
    summaries = summary_rows(
        cases,
        adapters,
        runs,
        comparisons,
        config,
    )
    summary_path = write_csv(
        joinpath(config.output_dir, "summary.csv"),
        SUMMARY_HEADER,
        summaries,
    )
    hotpath_path = joinpath(@__DIR__, "rxinfer_hotpaths.jl")
    metadata_path = write_csv(
        joinpath(config.output_dir, "metadata.csv"),
        ["schema_version", "key", "value"],
        metadata_rows(config, hotpath_path),
    )
    hook_row = run_hotpath_hook(config, hotpath_path)
    hook_path = write_csv(
        joinpath(config.output_dir, "hooks.csv"),
        [
            "schema_version",
            "hook",
            "available",
            "requested",
            "status",
            "output",
            "error",
        ],
        [hook_row],
    )

    differential_correct =
        all(comparison.ok for comparison in values(comparisons))
    requested_hook_ok =
        !config.run_hotpath || hook_row[5] == "ok"
    all_correct = differential_correct && requested_hook_ok
    for row in summaries
        println(
            "DIFF_SUMMARY ",
            (
                case = row[2],
                backend = row[3],
                samples = row[4],
                successful_samples = row[5],
                all_correct = row[6],
                median_inference_seconds = row[14],
                median_inference_bytes = row[17],
                inference_speedup_vs_reference = row[19],
            ),
        )
    end
    println(
        "DIFF_OUTPUT ",
        (
            directory = config.output_dir,
            all_correct = all_correct,
        ),
    )

    if config.fail_on_mismatch && !all_correct
        error(
            "backend differential mismatch; inspect $summary_path and $run_path",
        )
    end

    return (
        all_correct = all_correct,
        runs = runs,
        comparisons = comparisons,
        paths = (
            runs = run_path,
            trace = trace_path,
            summary = summary_path,
            metadata = metadata_path,
            hooks = hook_path,
        ),
    )
end

end
