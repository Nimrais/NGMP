using Pkg

Pkg.activate(
    get(ENV, "PERF_PROJECT", joinpath(@__DIR__, "..", ".."))
)

using Distributions
using LinearAlgebra
using Random
using Rocket
using RxInfer
using Statistics

LinearAlgebra.BLAS.set_num_threads(1)

const PERF_SAMPLES = parse(Int, get(ENV, "PERF_SAMPLES", "8"))
const PERF_ITERATIONS = parse(Int, get(ENV, "PERF_ITERATIONS", "20"))
const PERF_STATES = parse(Int, get(ENV, "PERF_STATES", "200"))

@model function perf_gaussian_state_space(y, x0, process_var, obs_var)
    x_prev ~ Normal(mean = mean(x0), var = var(x0))
    for index in eachindex(y)
        x[index] ~ Normal(mean = x_prev, var = process_var)
        y[index] ~ Normal(mean = x[index], var = obs_var)
        x_prev = x[index]
    end
end

function make_data(n)
    rng = MersenneTwister(123)
    hidden = Vector{Float64}(undef, n)
    observations = Vector{Float64}(undef, n)
    hidden[1] = rand(rng, Normal(0.0, sqrt(0.2)))
    observations[1] = rand(rng, Normal(hidden[1], sqrt(0.5)))
    for index in 2:n
        hidden[index] = rand(rng, Normal(hidden[index - 1], sqrt(0.2)))
        observations[index] =
            rand(rng, Normal(hidden[index], sqrt(0.5)))
    end
    return observations
end

const INITIALIZATION = @initialization begin
    q(x_prev) = NormalMeanVariance(0.0, 1.0)
    q(x) = NormalMeanVariance(0.0, 1.0)
end

function legacy_recorder(values)
    return function (event)
        current = Ref(0.0)
        subscribe!(
            score(
                event.model,
                RxInfer.BetheFreeEnergy(Real),
                RxInfer.DefaultObjectiveDiagnosticChecks,
            ) |> take(1),
            value -> current[] = value,
        )
        push!(values, current[])
        return nothing
    end
end

function reused_recorder(values)
    return function (event)
        isnothing(event.free_energy) &&
            error("batch inference did not attach its collected free energy")
        push!(values, Float64(event.free_energy))
        return nothing
    end
end

function prepare(callback, observations)
    return prepare_inference(
        perf_gaussian_state_space(
            x0 = NormalMeanVariance(0.0, 10.0),
            process_var = 0.2,
            obs_var = 0.5,
        );
        data = (y = observations,),
        constraints = MeanField(),
        initialization = INITIALIZATION,
        free_energy = true,
        callbacks = (after_iteration = callback,),
    )
end

function run!(model, observations)
    return infer!(
        model;
        data = (y = observations,),
        returnvars = (x = KeepLast(),),
        iterations = PERF_ITERATIONS,
        free_energy = true,
        showprogress = false,
        disable_inference_error_hint = true,
    )
end

function sample!(model, observations, values)
    empty!(values)
    GC.gc()
    measurement = @timed run!(model, observations)
    length(values) == PERF_ITERATIONS ||
        error("callback did not observe every iteration")
    values == measurement.value.free_energy ||
        error("callback values and returned free energy differ")
    return measurement
end

observations = make_data(PERF_STATES)
legacy_values = Float64[]
reused_values = Float64[]
legacy_model = prepare(legacy_recorder(legacy_values), observations)
reused_model = prepare(reused_recorder(reused_values), observations)

# Compile and force both prepared models onto their reusable-model path.
run!(legacy_model, observations)
run!(reused_model, observations)

legacy_samples = Any[]
reused_samples = Any[]
for sample in 1:PERF_SAMPLES
    order = isodd(sample) ? (:legacy, :reused) : (:reused, :legacy)
    for arm in order
        if arm === :legacy
            push!(
                legacy_samples,
                sample!(legacy_model, observations, legacy_values),
            )
        else
            push!(
                reused_samples,
                sample!(reused_model, observations, reused_values),
            )
        end
    end
end

legacy_result = last(legacy_samples).value
reused_result = last(reused_samples).value
legacy_result.free_energy == reused_result.free_energy ||
    error("free-energy history changed")
all(
    isapprox.(
        mean.(legacy_result.posteriors[:x]),
        mean.(reused_result.posteriors[:x]);
        atol = 1e-8,
        rtol = 1e-7,
    ),
) || error("posterior means changed")
all(
    isapprox.(
        var.(legacy_result.posteriors[:x]),
        var.(reused_result.posteriors[:x]);
        atol = 1e-8,
        rtol = 1e-7,
    ),
) || error("posterior variances changed")

legacy_seconds = median(getfield.(legacy_samples, :time))
reused_seconds = median(getfield.(reused_samples, :time))
legacy_bytes = median(getfield.(legacy_samples, :bytes))
reused_bytes = median(getfield.(reused_samples, :bytes))

println(
    "FREE_ENERGY_CALLBACK_SUMMARY ",
    (
        samples = PERF_SAMPLES,
        iterations = PERF_ITERATIONS,
        states = PERF_STATES,
        legacy_seconds = legacy_seconds,
        reused_seconds = reused_seconds,
        speedup = legacy_seconds / reused_seconds,
        legacy_bytes = legacy_bytes,
        reused_bytes = reused_bytes,
        allocation_reduction = 1 - reused_bytes / legacy_bytes,
    ),
)
