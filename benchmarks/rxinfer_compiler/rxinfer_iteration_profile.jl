using Profile

# Load the archived notebook-representative workload and its correctness checks.
# Keep its regular sample count small because this script adds independent
# sampling and allocation profiles after the workload has been compiled.
ENV["PERF_SAMPLES"] = get(ENV, "PERF_SAMPLES", "1")
include(joinpath(@__DIR__, "rxinfer_hotpaths.jl"))

const PROFILE_ITERATIONS = parse(
    Int, get(ENV, "RXINFER_PROFILE_ITERATIONS", "3")
)
const PROFILE_DELAY = parse(
    Float64, get(ENV, "RXINFER_PROFILE_DELAY", "0.0001")
)
const PROFILE_ALLOC_SAMPLE_RATE = parse(
    Float64, get(ENV, "RXINFER_PROFILE_ALLOC_SAMPLE_RATE", "0.00001")
)
const PROFILE_CPU =
    lowercase(get(ENV, "RXINFER_PROFILE_CPU", "true")) == "true"
const PROFILE_ALLOCATIONS =
    lowercase(get(ENV, "RXINFER_PROFILE_ALLOCATIONS", "true")) == "true"

PROFILE_ITERATIONS > 0 ||
    throw(ArgumentError("RXINFER_PROFILE_ITERATIONS must be positive"))

if PROFILE_CPU
    for _ in 1:PROFILE_ITERATIONS
        initialization = rematerialize_for_iteration!()
        Profile.clear()
        Profile.init(; delay = PROFILE_DELAY)
        @profile run_iteration_wave!(initialization)
    end

    println("\nRXINFER_ITERATION_CPU_PROFILE")
    Profile.print(
        stdout;
        format = :flat,
        sortedby = :count,
        mincount = 3,
        maxdepth = 24,
        noisefloor = 2.0,
    )
end

if PROFILE_ALLOCATIONS
    initialization = rematerialize_for_iteration!()
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate = PROFILE_ALLOC_SAMPLE_RATE run_iteration_wave!(
        initialization
    )

    println("\nRXINFER_ITERATION_ALLOCATION_PROFILE")
    Profile.Allocs.print(
        stdout,
        Profile.Allocs.fetch();
        format = :flat,
        sortedby = :bytes,
        mincount = 1,
    )
end
