# Archived microbenchmark for the unintegrated bucket-fusion prototype.
using BenchmarkTools
using Rocket

include(joinpath(@__DIR__, "bucket_fusion.jl"))

import .BucketFusion:
    FusionEdge,
    HomogeneousLaneBucket,
    FusionExecutor,
    fuse_lane!,
    notify_next!,
    on_lane_next!

struct BucketFusionSumKernel end

@inline fuse_lane!(
    ::BucketFusionSumKernel,
    ::Nothing,
    inputs::Tuple{Float64,Float64},
    lane,
) = inputs[1] + inputs[2]

mutable struct BucketFusionChecksum
    count::Int
    sum::Float64
end

@inline function on_lane_next!(
    checksum::BucketFusionChecksum,
    executor,
    lane::Int,
    value::Float64,
)
    checksum.count += 1
    checksum.sum += value
    return nothing
end

struct BucketFusionCase{E}
    executor::E
    lanes::Int
end

function build_bucket_fusion_case(lanes::Int)
    bucket = HomogeneousLaneBucket(
        Float64,
        Val(2),
        fill(BucketFusionSumKernel(), lanes),
        fill(nothing, lanes),
    )
    edges = FusionEdge[]
    append!(edges, (FusionEdge(1, lane, 1) for lane = 1:lanes))
    append!(
        edges,
        (FusionEdge(lane + 1, lane, 2) for lane = 1:lanes),
    )
    checksum = BucketFusionChecksum(0, 0.0)
    executor = FusionExecutor(bucket, lanes + 1, edges; handler = checksum)
    return BucketFusionCase(executor, lanes)
end

@inline function bucket_fusion_wave!(case::BucketFusionCase, value::Float64)
    bucket_fusion_prime_private!(case, value)
    bucket_fusion_shared!(case, value)
    return nothing
end

@inline function bucket_fusion_prime_private!(
    case::BucketFusionCase,
    value::Float64,
)
    for lane = 1:case.lanes
        notify_next!(case.executor, lane + 1, value + lane)
    end
    return nothing
end

@inline function bucket_fusion_shared!(
    case::BucketFusionCase,
    value::Float64,
)
    notify_next!(case.executor, 1, value)
    return nothing
end

mutable struct BucketFusionRocketActor <: Actor{Tuple{Float64,Float64}}
    count::Int
    sum::Float64
end

@inline function Rocket.on_next!(
    actor::BucketFusionRocketActor,
    inputs::Tuple{Float64,Float64},
)
    actor.count += 1
    actor.sum += inputs[1] + inputs[2]
    return nothing
end

Rocket.on_error!(actor::BucketFusionRocketActor, error) = nothing
Rocket.on_complete!(actor::BucketFusionRocketActor) = nothing

struct BucketFusionRocketCase{S,P,A,U}
    shared::S
    private_sources::P
    actors::A
    subscriptions::U
end

function build_bucket_fusion_rocket_case(lanes::Int)
    shared = Subject(Float64)
    private_sources = [Subject(Float64) for _ = 1:lanes]
    actors = [BucketFusionRocketActor(0, 0.0) for _ = 1:lanes]
    subscriptions = map(1:lanes) do lane
        subscribe!(
            combineLatest(
                (shared, private_sources[lane]), PushNew()
            ),
            actors[lane],
        )
    end
    return BucketFusionRocketCase(
        shared, private_sources, actors, subscriptions
    )
end

@inline function bucket_fusion_rocket_wave!(
    case::BucketFusionRocketCase,
    value::Float64,
)
    bucket_fusion_rocket_prime_private!(case, value)
    bucket_fusion_rocket_shared!(case, value)
    return nothing
end

@inline function bucket_fusion_rocket_prime_private!(
    case::BucketFusionRocketCase,
    value::Float64,
)
    for lane in eachindex(case.private_sources)
        next!(case.private_sources[lane], value + lane)
    end
    return nothing
end

@inline function bucket_fusion_rocket_shared!(
    case::BucketFusionRocketCase,
    value::Float64,
)
    next!(case.shared, value)
    return nothing
end

const BUCKET_FUSION_STEADY_REPEATS = 64

function build_bucket_fusion_batch(lanes::Int, repeats::Int)
    return [build_bucket_fusion_case(lanes) for _ = 1:repeats]
end

function build_bucket_fusion_rocket_batch(lanes::Int, repeats::Int)
    return [build_bucket_fusion_rocket_case(lanes) for _ = 1:repeats]
end

@inline function bucket_fusion_prime_batch!(cases, value::Float64)
    for case in cases
        bucket_fusion_prime_private!(case, value)
    end
    return nothing
end

@inline function bucket_fusion_shared_batch!(cases, value::Float64)
    for case in cases
        bucket_fusion_shared!(case, value)
    end
    return nothing
end

@inline function bucket_fusion_wave_batch!(cases, value::Float64)
    for case in cases
        bucket_fusion_wave!(case, value)
    end
    return nothing
end

@inline function bucket_fusion_rocket_prime_batch!(cases, value::Float64)
    for case in cases
        bucket_fusion_rocket_prime_private!(case, value)
    end
    return nothing
end

@inline function bucket_fusion_rocket_shared_batch!(cases, value::Float64)
    for case in cases
        bucket_fusion_rocket_shared!(case, value)
    end
    return nothing
end

@inline function bucket_fusion_rocket_wave_batch!(cases, value::Float64)
    for case in cases
        bucket_fusion_rocket_wave!(case, value)
    end
    return nothing
end

function add_bucket_fusion_benchmarks(suite)
    haskey(suite, "Execution") ||
        (suite["Execution"] = BenchmarkGroup(["Execution"]))
    execution = suite["Execution"]
    execution["Bucket fusion"] =
        BenchmarkGroup(["Execution", "Bucket fusion"])
    group = execution["Bucket fusion"]

    for lanes in (4, 16, 64)
        name = "lanes=$lanes"
        group[name] = BenchmarkGroup(["Execution", "Bucket fusion", name])
        lane_group = group[name]

        lane_group["build"] = BenchmarkGroup()
        lane_group["build"]["bucket"] =
            @benchmarkable build_bucket_fusion_case($lanes) evals = 1
        lane_group["build"]["rocket"] =
            @benchmarkable build_bucket_fusion_rocket_case($lanes) evals = 1

        lane_group["first"] = BenchmarkGroup()
        lane_group["first"]["bucket"] =
            @benchmarkable bucket_fusion_wave!(case, 1.0) setup =
                (case = build_bucket_fusion_case($lanes)) evals = 1
        lane_group["first"]["rocket"] =
            @benchmarkable bucket_fusion_rocket_wave!(case, 1.0) setup =
                (case = build_bucket_fusion_rocket_case($lanes)) evals = 1

        bucket_cases = build_bucket_fusion_batch(
            lanes, BUCKET_FUSION_STEADY_REPEATS
        )
        rocket_cases = build_bucket_fusion_rocket_batch(
            lanes, BUCKET_FUSION_STEADY_REPEATS
        )
        bucket_fusion_wave_batch!(bucket_cases, 0.0)
        bucket_fusion_rocket_wave_batch!(rocket_cases, 0.0)
        lane_group["steady"] = BenchmarkGroup()
        lane_group["steady"]["bucket"] =
            @benchmarkable bucket_fusion_shared_batch!(cases, 1.0) setup = (
                bucket_fusion_prime_batch!($bucket_cases, 1.0);
                cases = $bucket_cases
            ) evals = 1
        lane_group["steady"]["rocket"] =
            @benchmarkable bucket_fusion_rocket_shared_batch!(cases, 1.0) setup = (
                bucket_fusion_rocket_prime_batch!($rocket_cases, 1.0);
                cases = $rocket_cases
            ) evals = 1

        lane_group["full steady wave"] = BenchmarkGroup()
        lane_group["full steady wave"]["bucket"] =
            @benchmarkable bucket_fusion_wave_batch!($bucket_cases, 1.0)
        lane_group["full steady wave"]["rocket"] =
            @benchmarkable bucket_fusion_rocket_wave_batch!($rocket_cases, 1.0)
    end
    return suite
end

"""
    run_bucket_fusion_gate(; samples = 200, seconds = 1.0)

Run the bounded phase-1 microkernel gate.  The comparison performs identical
two-input PushNew readiness waves: one private update per lane followed by one
shared source fanout.  Build and first-wave timings include their complete
phases.  The steady gate times the shared notification and all resulting lane
computations after both implementations are primed in benchmark setup; a full
wave speedup is also reported as a diagnostic.  Retention requires no shared
steady case regression, at least 3x geometric-mean speedup, and zero steady
allocations.
"""
function run_bucket_fusion_gate(; samples::Int = 200, seconds::Float64 = 1.0)
    rows = NamedTuple[]

    # Compile measurement and event paths before collecting any samples.
    for lanes in (4, 16, 64)
        bucket_case = build_bucket_fusion_case(lanes)
        rocket_case = build_bucket_fusion_rocket_case(lanes)
        bucket_fusion_wave!(bucket_case, 0.0)
        bucket_fusion_rocket_wave!(rocket_case, 0.0)
    end

    for lanes in (4, 16, 64)
        bucket_build = @benchmark build_bucket_fusion_case($lanes) samples =
            samples seconds = seconds evals = 1
        rocket_build = @benchmark build_bucket_fusion_rocket_case($lanes) samples =
            samples seconds = seconds evals = 1

        bucket_first = @benchmark bucket_fusion_wave!(case, 1.0) setup =
            (case = build_bucket_fusion_case($lanes)) samples = samples seconds =
            seconds evals = 1
        rocket_first = @benchmark bucket_fusion_rocket_wave!(case, 1.0) setup =
            (case = build_bucket_fusion_rocket_case($lanes)) samples = samples seconds =
            seconds evals = 1

        repeats = BUCKET_FUSION_STEADY_REPEATS
        bucket_cases = build_bucket_fusion_batch(lanes, repeats)
        rocket_cases = build_bucket_fusion_rocket_batch(lanes, repeats)
        bucket_fusion_wave_batch!(bucket_cases, 0.0)
        bucket_fusion_rocket_wave_batch!(rocket_cases, 0.0)
        bucket_steady =
            @benchmark bucket_fusion_shared_batch!(cases, 1.0) setup = (
                bucket_fusion_prime_batch!($bucket_cases, 1.0);
                cases = $bucket_cases
            ) samples = samples seconds = seconds evals = 1
        rocket_steady =
            @benchmark bucket_fusion_rocket_shared_batch!(cases, 1.0) setup = (
                bucket_fusion_rocket_prime_batch!($rocket_cases, 1.0);
                cases = $rocket_cases
            ) samples = samples seconds = seconds evals = 1

        bucket_full =
            @benchmark bucket_fusion_wave_batch!($bucket_cases, 1.0) samples =
                samples seconds = seconds
        rocket_full = @benchmark bucket_fusion_rocket_wave_batch!(
            $rocket_cases, 1.0
        ) samples = samples seconds = seconds evals = 1

        bucket_time = median(bucket_steady).time / repeats
        rocket_time = median(rocket_steady).time / repeats
        push!(
            rows,
            (
                lanes = lanes,
                build_bucket_ns = median(bucket_build).time,
                build_rocket_ns = median(rocket_build).time,
                first_bucket_ns = median(bucket_first).time,
                first_rocket_ns = median(rocket_first).time,
                steady_bucket_ns = bucket_time,
                steady_rocket_ns = rocket_time,
                steady_bucket_min_ns =
                    minimum(bucket_steady).time / repeats,
                steady_rocket_min_ns =
                    minimum(rocket_steady).time / repeats,
                speedup = rocket_time / bucket_time,
                full_bucket_ns = median(bucket_full).time / repeats,
                full_rocket_ns = median(rocket_full).time / repeats,
                full_speedup =
                    median(rocket_full).time / median(bucket_full).time,
                bucket_allocs = median(bucket_steady).allocs / repeats,
                bucket_memory = median(bucket_steady).memory / repeats,
                rocket_allocs = median(rocket_steady).allocs / repeats,
                rocket_memory = median(rocket_steady).memory / repeats,
                bucket_samples = length(bucket_steady),
                rocket_samples = length(rocket_steady),
            ),
        )
    end

    println(
        "lanes  build bucket/rocket (ns)  first bucket/rocket (ns)  ",
        "steady shared bucket/rocket (ns)  speedup  ",
        "full-wave speedup  allocations bucket/rocket",
    )
    for row in rows
        println(
            lpad(row.lanes, 5),
            "  ",
            lpad(round(Int, row.build_bucket_ns), 10),
            "/",
            lpad(round(Int, row.build_rocket_ns), 10),
            "  ",
            lpad(round(Int, row.first_bucket_ns), 10),
            "/",
            lpad(round(Int, row.first_rocket_ns), 10),
            "  ",
            lpad(round(Int, row.steady_bucket_ns), 10),
            "/",
            lpad(round(Int, row.steady_rocket_ns), 10),
            "  ",
            round(row.speedup; digits = 2),
            "x  full=",
            round(row.full_speedup; digits = 2),
            "x  ",
            row.bucket_allocs,
            "/",
            row.rocket_allocs,
            "  samples=",
            row.bucket_samples,
            "/",
            row.rocket_samples,
        )
    end

    speedups = getproperty.(rows, :speedup)
    geometric_mean = exp(sum(log, speedups) / length(speedups))
    minimum_speedup = minimum(speedups)
    zero_allocations =
        all(row -> row.bucket_allocs == 0 && row.bucket_memory == 0, rows)
    println(
        "steady geometric-mean speedup=",
        round(geometric_mean; digits = 3),
        "x minimum=",
        round(minimum_speedup; digits = 3),
        "x zero_allocations=",
        zero_allocations,
    )

    geometric_mean >= 3.0 ||
        error("bucket-fusion gate failed: geometric mean $geometric_mean < 3x")
    minimum_speedup >= 1.0 ||
        error("bucket-fusion gate failed: case regression $minimum_speedup < 1x")
    zero_allocations ||
        error("bucket-fusion gate failed: steady bucket path allocated")
    return rows
end
