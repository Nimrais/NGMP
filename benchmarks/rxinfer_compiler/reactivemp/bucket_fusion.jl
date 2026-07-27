# Archived unintegrated compiler primitive. It passed its focused microbenchmark
# and correctness gates but never produced an RxInfer end-to-end improvement.
"""
    BucketFusion

An internal, allocation-free execution microkernel for a homogeneous plate of
message computations.  It is deliberately independent from ReactiveMP's
observable activation API: a future graph planner can either lower a complete
component to these buckets or keep that component on Rocket.

The microkernel separates:

  * a typed value plane, which owns and roots payloads; and
  * a CSR notification plane, which stores only integer fanout runs.

Consecutive edges from one source to consecutive lanes are represented by one
run.  Runs retain the input edge order supplied by the planner.  A lane output
and all of its descendants are delivered before the next lane in a run, which
matches synchronous depth-first observable delivery.
"""
module BucketFusion

export FusionEdge
export HomogeneousLaneBucket
export FusionExecutor
export NoopFusionHandler
export fuse_lane!
export reset_lane!
export on_lane_next!
export on_fusion_error!
export on_fusion_reset!
export notify_next!
export notify_error!
export reset!
export is_halted
export last_error
export source_value
export fanout_edge_count
export fanout_run_count

"""
    FusionEdge(source, lane, input)

Connect `source` to one `input` of a lane in a homogeneous bucket.  Edge order
is observable: fanout delivery follows the stable per-source order in which
edges were supplied to the executor.
"""
struct FusionEdge
    source::Int32
    lane::Int32
    input::UInt8

    function FusionEdge(source::Integer, lane::Integer, input::Integer)
        source > 0 || throw(ArgumentError("source indices must be positive"))
        lane > 0 || throw(ArgumentError("lane indices must be positive"))
        input > 0 || throw(ArgumentError("input indices must be positive"))
        source <= typemax(Int32) ||
            throw(ArgumentError("source index does not fit in Int32"))
        lane <= typemax(Int32) ||
            throw(ArgumentError("lane index does not fit in Int32"))
        input <= typemax(UInt8) ||
            throw(ArgumentError("input index does not fit in UInt8"))
        return new(Int32(source), Int32(lane), UInt8(input))
    end
end

"""
The payload plane is intentionally separate from notification topology.  An
uninitialised slot is never read; `initialised` is also useful to a future
component executor when it exposes recent source values.
"""
mutable struct TypedValuePlane{T}
    values::Vector{T}
    initialised::BitVector
end

function TypedValuePlane(::Type{T}, nsources::Int) where {T}
    isconcretetype(T) ||
        throw(ArgumentError("bucket value type must be concrete, got $T"))
    nsources > 0 || throw(ArgumentError("an executor needs at least one source"))
    return TypedValuePlane(Vector{T}(undef, nsources), falses(nsources))
end

"""
CSR rows address sources.  Each entry is a maximal stable run with equal input
index and consecutive lane indices.  This is the common plate fanout shape and
lets one source notification avoid one heap object and one dynamic actor call
per lane.
"""
struct CSRNotificationPlane
    rowptr::Vector{Int32}
    first_lane::Vector{Int32}
    run_length::Vector{Int32}
    input::Vector{UInt8}
    edge_count::Int
end

"""
    HomogeneousLaneBucket(::Type{T}, ::Val{N}, kernels, states;
                          output_sources)

Create a plate with `N` inputs per lane.  `kernels[lane]` and `states[lane]`
remain concretely typed.  Mutable kernel or state objects must be distinct for
every lane; aliasing them would turn an apparently independent plate into
order-dependent inference and is rejected at construction time.

`output_sources[lane] == 0` denotes a terminal output.  A positive source feeds
the concrete result back into the executor without type erasure.
"""
mutable struct HomogeneousLaneBucket{T,N,K,S}
    kernels::Vector{K}
    states::Vector{S}
    inputs::NTuple{N,Vector{T}}
    valid::Vector{UInt64}
    updated::Vector{UInt64}
    output_sources::Vector{Int32}

    function HomogeneousLaneBucket{T,N,K,S}(
        kernels::Vector{K},
        states::Vector{S},
        inputs::NTuple{N,Vector{T}},
        valid::Vector{UInt64},
        updated::Vector{UInt64},
        output_sources::Vector{Int32},
    ) where {T,N,K,S}
        return new{T,N,K,S}(
            kernels, states, inputs, valid, updated, output_sources
        )
    end
end

@inline function _all_inputs_mask(::Val{N}) where {N}
    return N == 64 ? typemax(UInt64) : (UInt64(1) << N) - UInt64(1)
end

function _reject_mutable_aliases(values::Vector{T}, label::AbstractString) where {T}
    ismutabletype(T) || return nothing
    seen = IdDict{Any,Nothing}()
    for (index, value) in pairs(values)
        haskey(seen, value) &&
            throw(
                ArgumentError(
                    "mutable $label object is shared by multiple lanes (first repeated at lane $index)",
                ),
            )
        seen[value] = nothing
    end
    return nothing
end

function HomogeneousLaneBucket(
    ::Type{T},
    ::Val{N},
    kernels::Vector{K},
    states::Vector{S};
    output_sources::AbstractVector{<:Integer} = zeros(Int, length(kernels)),
) where {T,N,K,S}
    1 <= N <= 64 ||
        throw(ArgumentError("a homogeneous bucket supports between 1 and 64 inputs"))
    isconcretetype(T) ||
        throw(ArgumentError("bucket value type must be concrete, got $T"))
    isconcretetype(K) ||
        throw(ArgumentError("lane kernel type must be concrete, got $K"))
    isconcretetype(S) ||
        throw(ArgumentError("lane state type must be concrete, got $S"))
    nlanes = length(kernels)
    nlanes > 0 || throw(ArgumentError("a homogeneous bucket needs at least one lane"))
    length(states) == nlanes ||
        throw(ArgumentError("kernels and states must have the same number of lanes"))
    length(output_sources) == nlanes ||
        throw(ArgumentError("output_sources must have one entry per lane"))

    _reject_mutable_aliases(kernels, "kernel")
    _reject_mutable_aliases(states, "state")

    outputs = Vector{Int32}(undef, nlanes)
    for (lane, source) in pairs(output_sources)
        source >= 0 || throw(ArgumentError("output source indices cannot be negative"))
        source <= typemax(Int32) ||
            throw(ArgumentError("output source index at lane $lane does not fit in Int32"))
        @inbounds outputs[lane] = Int32(source)
    end

    inputs = ntuple((_) -> Vector{T}(undef, nlanes), Val(N))
    return HomogeneousLaneBucket{T,N,K,S}(
        kernels,
        states,
        inputs,
        zeros(UInt64, nlanes),
        zeros(UInt64, nlanes),
        outputs,
    )
end

"""
    fuse_lane!(kernel, state, inputs::NTuple, lane)

Concrete lane computation hook.  Implementations should return the bucket's
value type.  The executor calls this only after every input has supplied a new
value (PushNew semantics).
"""
function fuse_lane! end

"""
    reset_lane!(kernel, state)

Optional hook for resetting mutable per-lane kernel state.
"""
@inline reset_lane!(kernel, state) = nothing

struct NoopFusionHandler end

"""
Called synchronously for a computed lane value, before delivering that value to
the lane's optional output source.
"""
@inline on_lane_next!(handler, executor, lane::Int, value) = nothing

"""
Called once when explicit notification or a thrown lane/callback error halts
the executor.
"""
@inline on_fusion_error!(handler, executor, error) = nothing

"""
Called after source readiness and per-lane state have been reset.
"""
@inline on_fusion_reset!(handler, executor) = nothing

mutable struct FusionExecutor{T,N,K,S,H}
    values::TypedValuePlane{T}
    notifications::CSRNotificationPlane
    bucket::HomogeneousLaneBucket{T,N,K,S}
    handler::H
    active::Bool
    halted::Bool
    error::Any
end

function _build_notification_plane(
    nsources::Int,
    nlanes::Int,
    ::Val{N},
    edges::AbstractVector{FusionEdge},
) where {N}
    counts = zeros(Int32, nsources)
    seen_inputs = Set{Tuple{Int32,UInt8}}()
    for edge in edges
        edge.source <= nsources ||
            throw(ArgumentError("edge source $(edge.source) exceeds $nsources sources"))
        edge.lane <= nlanes ||
            throw(ArgumentError("edge lane $(edge.lane) exceeds $nlanes lanes"))
        edge.input <= N ||
            throw(ArgumentError("edge input $(edge.input) exceeds bucket arity $N"))
        key = (edge.lane, edge.input)
        key in seen_inputs &&
            throw(ArgumentError("lane $(edge.lane) input $(edge.input) is connected twice"))
        push!(seen_inputs, key)
        @inbounds counts[edge.source] += Int32(1)
    end

    edge_rowptr = Vector{Int32}(undef, nsources + 1)
    edge_rowptr[1] = Int32(1)
    for source = 1:nsources
        @inbounds edge_rowptr[source + 1] = edge_rowptr[source] + counts[source]
    end

    ordered_lanes = Vector{Int32}(undef, length(edges))
    ordered_inputs = Vector{UInt8}(undef, length(edges))
    cursor = copy(@view(edge_rowptr[1:nsources]))
    # Counting-sort by source is stable, preserving each source's edge order.
    for edge in edges
        source = Int(edge.source)
        @inbounds destination = cursor[source]
        @inbounds ordered_lanes[destination] = edge.lane
        @inbounds ordered_inputs[destination] = edge.input
        @inbounds cursor[source] += Int32(1)
    end

    run_rowptr = Vector{Int32}(undef, nsources + 1)
    first_lane = Int32[]
    run_length = Int32[]
    run_input = UInt8[]
    sizehint!(first_lane, length(edges))
    sizehint!(run_length, length(edges))
    sizehint!(run_input, length(edges))

    for source = 1:nsources
        run_rowptr[source] = Int32(length(first_lane) + 1)
        edge_index = Int(edge_rowptr[source])
        edge_stop = Int(edge_rowptr[source + 1]) - 1
        while edge_index <= edge_stop
            lane = @inbounds ordered_lanes[edge_index]
            input = @inbounds ordered_inputs[edge_index]
            length = 1
            while edge_index + length <= edge_stop
                next_lane = @inbounds ordered_lanes[edge_index + length]
                next_input = @inbounds ordered_inputs[edge_index + length]
                (next_input == input && next_lane == lane + length) || break
                length += 1
            end
            push!(first_lane, lane)
            push!(run_length, Int32(length))
            push!(run_input, input)
            edge_index += length
        end
    end
    run_rowptr[end] = Int32(length(first_lane) + 1)

    return CSRNotificationPlane(
        run_rowptr,
        first_lane,
        run_length,
        run_input,
        length(edges),
    )
end

"""
    FusionExecutor(bucket, nsources, edges; handler = NoopFusionHandler())

Build stable CSR fanout runs for one homogeneous bucket.
"""
function FusionExecutor(
    bucket::HomogeneousLaneBucket{T,N,K,S},
    nsources::Integer,
    edges::AbstractVector{FusionEdge};
    handler::H = NoopFusionHandler(),
) where {T,N,K,S,H}
    nsources <= typemax(Int32) ||
        throw(ArgumentError("source count does not fit in Int32"))
    source_count = Int(nsources)
    values = TypedValuePlane(T, source_count)
    for (lane, source) in pairs(bucket.output_sources)
        source <= source_count ||
            throw(
                ArgumentError(
                    "output source $source at lane $lane exceeds $source_count sources",
                ),
            )
    end
    notifications =
        _build_notification_plane(source_count, length(bucket.kernels), Val(N), edges)
    return FusionExecutor(
        values,
        notifications,
        bucket,
        handler,
        false,
        false,
        nothing,
    )
end

@generated function _snapshot(inputs::I, lane::Int) where {I<:Tuple}
    values = [
        :(@inbounds inputs[$index][lane]) for index = 1:fieldcount(I)
    ]
    return Expr(:tuple, values...)
end

@inline function _offer_input!(
    executor::FusionExecutor{T,N},
    lane::Int,
    input::Int,
    value::T,
) where {T,N}
    bucket = executor.bucket
    @inbounds bucket.inputs[input][lane] = value
    bit = UInt64(1) << (input - 1)
    @inbounds bucket.valid[lane] |= bit
    @inbounds bucket.updated[lane] |= bit
    mask = _all_inputs_mask(Val(N))
    @inbounds ready =
        bucket.valid[lane] == mask && bucket.updated[lane] == mask
    ready || return nothing

    # Reset before callbacks: a synchronous reentrant notification observes
    # exactly the post-emission PushNew readiness state.
    @inbounds bucket.updated[lane] = UInt64(0)
    snapshot = _snapshot(bucket.inputs, lane)
    @inbounds value_out =
        fuse_lane!(
            bucket.kernels[lane], bucket.states[lane], snapshot, lane
        )::T
    on_lane_next!(executor.handler, executor, lane, value_out)
    executor.halted && return nothing

    @inbounds output_source = bucket.output_sources[lane]
    output_source == 0 ||
        _drain_next!(executor, Int(output_source), value_out)
    return nothing
end

@inline function _drain_next!(
    executor::FusionExecutor{T},
    source::Int,
    value::T,
) where {T}
    values = executor.values
    @inbounds values.values[source] = value
    @inbounds values.initialised[source] ||
        (@inbounds values.initialised[source] = true)

    plane = executor.notifications
    @inbounds run_index = Int(plane.rowptr[source])
    @inbounds run_stop = Int(plane.rowptr[source + 1]) - 1
    while run_index <= run_stop
        @inbounds lane = Int(plane.first_lane[run_index])
        @inbounds lane_stop = lane + Int(plane.run_length[run_index]) - 1
        @inbounds input = Int(plane.input[run_index])
        while lane <= lane_stop
            _offer_input!(executor, lane, input, value)
            # A lane callback/output recursively drains before this continuation.
            executor.halted && return nothing
            lane += 1
        end
        run_index += 1
    end
    return nothing
end

function _halt_with_error!(executor::FusionExecutor, error)
    executor.halted && return false
    executor.halted = true
    executor.error = error
    on_fusion_error!(executor.handler, executor, error)
    return false
end

"""
    notify_next!(executor, source, value) -> Bool

Deliver a typed source value.  Returns `false` if the executor is halted.
Synchronous reentrant calls join the active depth-first delivery; only the
outermost call installs the exception boundary.
"""
@inline function notify_next!(
    executor::FusionExecutor{T},
    source::Integer,
    value::T,
) where {T}
    executor.halted && return false
    1 <= source <= length(executor.values.values) ||
        throw(BoundsError(executor.values.values, source))

    if executor.active
        _drain_next!(executor, Int(source), value)
        return !executor.halted
    end

    executor.active = true
    try
        _drain_next!(executor, Int(source), value)
    catch error
        executor.active = false
        return _halt_with_error!(executor, error)
    end
    executor.active = false
    return !executor.halted
end

"""
    notify_error!(executor, error) -> Bool

Halt delivery immediately.  Later lanes in the current fanout run are skipped.
"""
notify_error!(executor::FusionExecutor, error) =
    _halt_with_error!(executor, error)

"""
    reset!(executor)

Clear source validity and PushNew readiness while retaining allocated, typed
storage.  Payload slots remain GC-safe roots but are never read until marked
valid again.
"""
function reset!(executor::FusionExecutor)
    executor.active &&
        throw(ArgumentError("cannot reset a bucket-fusion executor during delivery"))
    fill!(executor.values.initialised, false)
    fill!(executor.bucket.valid, UInt64(0))
    fill!(executor.bucket.updated, UInt64(0))
    for lane in eachindex(executor.bucket.kernels)
        @inbounds reset_lane!(
            executor.bucket.kernels[lane], executor.bucket.states[lane]
        )
    end
    executor.halted = false
    executor.error = nothing
    on_fusion_reset!(executor.handler, executor)
    return executor
end

is_halted(executor::FusionExecutor) = executor.halted
last_error(executor::FusionExecutor) = executor.error
fanout_edge_count(executor::FusionExecutor) = executor.notifications.edge_count
fanout_run_count(executor::FusionExecutor) =
    length(executor.notifications.first_lane)

function source_value(executor::FusionExecutor, source::Integer)
    1 <= source <= length(executor.values.values) ||
        throw(BoundsError(executor.values.values, source))
    @inbounds executor.values.initialised[source] ||
        throw(ArgumentError("source $source has not emitted since reset"))
    @inbounds return executor.values.values[source]
end

end
