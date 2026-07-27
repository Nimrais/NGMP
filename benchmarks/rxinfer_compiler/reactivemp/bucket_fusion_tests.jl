# Archived focused tests for the unintegrated bucket-fusion prototype.
include(joinpath(@__DIR__, "bucket_fusion.jl"))

@testitem "Bucket fusion preserves Rocket fanout semantics" begin
    using Rocket

    import Main.BucketFusion:
        FusionEdge,
        HomogeneousLaneBucket,
        FusionExecutor,
        fanout_edge_count,
        fanout_run_count,
        fuse_lane!,
        notify_next!,
        on_lane_next!

    struct SumKernel end
    fuse_lane!(::SumKernel, ::Nothing, inputs, lane) = inputs[1] + inputs[2]

    mutable struct RecordingHandler
        events::Vector{Tuple{Int,Float64}}
    end

    on_lane_next!(
        handler::RecordingHandler,
        executor,
        lane::Int,
        value::Float64,
    ) = push!(handler.events, (lane, value))

    function make_fused_plate(lanes::Int)
        handler = RecordingHandler(Tuple{Int,Float64}[])
        bucket = HomogeneousLaneBucket(
            Float64,
            Val(2),
            fill(SumKernel(), lanes),
            fill(nothing, lanes),
        )
        edges = FusionEdge[]
        append!(edges, (FusionEdge(1, lane, 1) for lane = 1:lanes))
        append!(
            edges,
            (FusionEdge(lane + 1, lane, 2) for lane = 1:lanes),
        )
        return FusionExecutor(bucket, lanes + 1, edges; handler), handler
    end

    function make_rocket_plate(lanes::Int)
        shared = Subject(Float64)
        private_sources = [Subject(Float64) for _ = 1:lanes]
        events = Tuple{Int,Float64}[]
        subscriptions = map(1:lanes) do lane
            subscribe!(
                combineLatest(
                    (shared, private_sources[lane]), PushNew()
                ),
                lambda(
                    on_next = (inputs) ->
                        push!(events, (lane, inputs[1] + inputs[2])),
                ),
            )
        end
        return shared, private_sources, events, subscriptions
    end

    for lanes in (4, 16, 64)
        fused, fused_handler = make_fused_plate(lanes)
        shared, private_sources, rocket_events, subscriptions =
            make_rocket_plate(lanes)

        for lane = 1:lanes
            value = Float64(lane)
            @test notify_next!(fused, lane + 1, value)
            next!(private_sources[lane], value)
        end
        @test notify_next!(fused, 1, 10.0)
        next!(shared, 10.0)
        @test fused_handler.events == rocket_events
        @test fused_handler.events ==
              [(lane, 10.0 + lane) for lane = 1:lanes]

        # A second PushNew wave verifies readiness reset as well as fanout order.
        for lane = 1:lanes
            value = Float64(100 + lane)
            @test notify_next!(fused, lane + 1, value)
            next!(private_sources[lane], value)
        end
        @test notify_next!(fused, 1, 20.0)
        next!(shared, 20.0)
        @test fused_handler.events == rocket_events

        @test fanout_edge_count(fused) == 2lanes
        # One shared run plus one singleton private-source run per lane.
        @test fanout_run_count(fused) == lanes + 1
        foreach(unsubscribe!, subscriptions)
    end
end

@testitem "Bucket fusion drains descendants depth-first and handles reentrancy" begin
    using Rocket

    import Main.BucketFusion:
        FusionEdge,
        HomogeneousLaneBucket,
        FusionExecutor,
        fuse_lane!,
        notify_next!,
        on_lane_next!

    struct SumKernel end
    fuse_lane!(::SumKernel, ::Nothing, inputs, lane) = inputs[1] + inputs[2]

    mutable struct LaneLog
        lanes::Vector{Int}
        values::Vector{Float64}
    end

    function on_lane_next!(
        log::LaneLog,
        executor,
        lane::Int,
        value::Float64,
    )
        push!(log.lanes, lane)
        push!(log.values, value)
        return nothing
    end

    # Lane 1 feeds lane 3.  Lane 3 must finish before source 1 continues to
    # lane 2, despite source 1 being represented by a single CSR run.
    log = LaneLog(Int[], Float64[])
    bucket = HomogeneousLaneBucket(
        Float64,
        Val(2),
        fill(SumKernel(), 3),
        fill(nothing, 3);
        output_sources = [4, 0, 0],
    )
    edges = [
        FusionEdge(1, 1, 1),
        FusionEdge(1, 2, 1),
        FusionEdge(2, 1, 2),
        FusionEdge(3, 2, 2),
        FusionEdge(4, 3, 1),
        FusionEdge(5, 3, 2),
    ]
    executor = FusionExecutor(bucket, 5, edges; handler = log)
    notify_next!(executor, 2, 1.0)
    notify_next!(executor, 3, 2.0)
    notify_next!(executor, 5, 100.0)
    notify_next!(executor, 1, 10.0)
    @test log.lanes == [1, 3, 2]
    @test log.values == [11.0, 111.0, 12.0]

    mutable struct ReentrantHandler
        events::Vector{Tuple{Int,Float64}}
        source::Int
        reentered::Bool
    end

    function on_lane_next!(
        handler::ReentrantHandler,
        executor,
        lane::Int,
        value::Float64,
    )
        push!(handler.events, (lane, value))
        if lane == 1 && !handler.reentered
            handler.reentered = true
            notify_next!(executor, handler.source, 50.0)
        end
        return nothing
    end

    # The same-source reentrant delivery is the demanding reference/order case:
    # lane 2 consumes the nested value before the outer source resumes.
    fused_handler = ReentrantHandler(Tuple{Int,Float64}[], 1, false)
    fused_bucket = HomogeneousLaneBucket(
        Float64,
        Val(2),
        fill(SumKernel(), 2),
        fill(nothing, 2),
    )
    fused_edges = [
        FusionEdge(1, 1, 1),
        FusionEdge(1, 2, 1),
        FusionEdge(2, 1, 2),
        FusionEdge(3, 2, 2),
    ]
    fused = FusionExecutor(
        fused_bucket, 3, fused_edges; handler = fused_handler
    )
    notify_next!(fused, 2, 1.0)
    notify_next!(fused, 3, 2.0)
    notify_next!(fused, 1, 10.0)

    rocket_shared = Subject(Float64)
    rocket_private = [Subject(Float64), Subject(Float64)]
    rocket_events = Tuple{Int,Float64}[]
    rocket_reentered = Ref(false)
    subscriptions = map(1:2) do lane
        subscribe!(
            combineLatest(
                (rocket_shared, rocket_private[lane]), PushNew()
            ),
            lambda(
                on_next = function (inputs)
                    push!(
                        rocket_events, (lane, inputs[1] + inputs[2])
                    )
                    if lane == 1 && !rocket_reentered[]
                        rocket_reentered[] = true
                        next!(rocket_shared, 50.0)
                    end
                end,
            ),
        )
    end
    next!(rocket_private[1], 1.0)
    next!(rocket_private[2], 2.0)
    next!(rocket_shared, 10.0)

    @test fused_handler.events == rocket_events
    @test fused_handler.events == [(1, 11.0), (2, 52.0)]
    foreach(unsubscribe!, subscriptions)
end

@testitem "Bucket fusion error and reset semantics are atomic" begin
    import Main.BucketFusion:
        FusionEdge,
        HomogeneousLaneBucket,
        FusionExecutor,
        fuse_lane!,
        is_halted,
        last_error,
        notify_next!,
        on_fusion_error!,
        on_fusion_reset!,
        on_lane_next!,
        reset!,
        reset_lane!,
        source_value

    struct FailingKernel
        fail::Bool
    end

    function fuse_lane!(
        kernel::FailingKernel,
        state::Nothing,
        inputs,
        lane,
    )
        kernel.fail && error("lane $lane failed")
        return inputs[1] + inputs[2]
    end

    mutable struct LifecycleLog
        lanes::Vector{Int}
        errors::Vector{String}
        resets::Int
    end

    on_lane_next!(
        log::LifecycleLog,
        executor,
        lane::Int,
        value::Float64,
    ) = push!(log.lanes, lane)
    on_fusion_error!(log::LifecycleLog, executor, error) =
        push!(log.errors, sprint(showerror, error))
    on_fusion_reset!(log::LifecycleLog, executor) = (log.resets += 1)

    handler = LifecycleLog(Int[], String[], 0)
    bucket = HomogeneousLaneBucket(
        Float64,
        Val(2),
        [FailingKernel(false), FailingKernel(true), FailingKernel(false)],
        fill(nothing, 3),
    )
    edges = FusionEdge[]
    append!(edges, (FusionEdge(1, lane, 1) for lane = 1:3))
    append!(edges, (FusionEdge(lane + 1, lane, 2) for lane = 1:3))
    executor = FusionExecutor(bucket, 4, edges; handler)

    for lane = 1:3
        notify_next!(executor, lane + 1, Float64(lane))
    end
    @test !notify_next!(executor, 1, 10.0)
    @test is_halted(executor)
    @test handler.lanes == [1]
    @test handler.errors == ["lane 2 failed"]
    @test last_error(executor) isa ErrorException
    @test !notify_next!(executor, 4, 99.0)
    @test handler.lanes == [1] # lane 3 and later events are stopped.

    @test source_value(executor, 1) == 10.0
    reset!(executor)
    @test !is_halted(executor)
    @test last_error(executor) === nothing
    @test handler.resets == 1
    @test_throws ArgumentError source_value(executor, 1)

    mutable struct CounterState
        count::Int
    end
    struct StatefulKernel end
    function fuse_lane!(::StatefulKernel, state::CounterState, inputs, lane)
        state.count += 1
        return inputs[1]
    end
    reset_lane!(::StatefulKernel, state::CounterState) = (state.count = 0)

    shared = CounterState(0)
    @test_throws ArgumentError HomogeneousLaneBucket(
        Float64,
        Val(1),
        fill(StatefulKernel(), 2),
        fill(shared, 2),
    )
    independent = [CounterState(0), CounterState(0)]
    independent_bucket = HomogeneousLaneBucket(
        Float64,
        Val(1),
        fill(StatefulKernel(), 2),
        independent,
    )
    independent_executor = FusionExecutor(
        independent_bucket,
        1,
        [FusionEdge(1, 1, 1), FusionEdge(1, 2, 1)],
    )
    notify_next!(independent_executor, 1, 1.0)
    @test getfield.(independent, :count) == [1, 1]
    reset!(independent_executor)
    @test getfield.(independent, :count) == [0, 0]
end

@testitem "Bucket fusion warm path is allocation-free and reference-safe" begin
    import Main.BucketFusion:
        FusionEdge,
        HomogeneousLaneBucket,
        FusionExecutor,
        NoopFusionHandler,
        fuse_lane!,
        notify_next!,
        on_lane_next!

    struct SumKernel end
    fuse_lane!(::SumKernel, ::Nothing, inputs, lane) = inputs[1] + inputs[2]

    lanes = 64
    bucket = HomogeneousLaneBucket(
        Float64,
        Val(2),
        fill(SumKernel(), lanes),
        fill(nothing, lanes),
    )
    edges = FusionEdge[]
    append!(edges, (FusionEdge(1, lane, 1) for lane = 1:lanes))
    append!(edges, (FusionEdge(lane + 1, lane, 2) for lane = 1:lanes))
    executor = FusionExecutor(
        bucket, lanes + 1, edges; handler = NoopFusionHandler()
    )

    function wave!(executor, lanes, value)
        for lane = 1:lanes
            notify_next!(executor, lane + 1, value + lane)
        end
        notify_next!(executor, 1, value)
        return nothing
    end

    wave!(executor, lanes, 1.0)
    @test @allocated(wave!(executor, lanes, 2.0)) == 0

    mutable struct Box
        value::Int
    end
    struct FirstKernel end
    fuse_lane!(::FirstKernel, ::Nothing, inputs, lane) = inputs[1]

    mutable struct BoxHandler
        seen::Vector{Tuple{Int,Box}}
        replacement::Box
        reentered::Bool
    end

    function on_lane_next!(
        handler::BoxHandler,
        executor,
        lane::Int,
        value::Box,
    )
        push!(handler.seen, (lane, value))
        if lane == 1 && !handler.reentered
            handler.reentered = true
            notify_next!(executor, 1, handler.replacement)
        end
        return nothing
    end

    original = Box(1)
    replacement = Box(2)
    box_handler = BoxHandler(Tuple{Int,Box}[], replacement, false)
    box_bucket = HomogeneousLaneBucket(
        Box,
        Val(2),
        fill(FirstKernel(), 2),
        fill(nothing, 2),
    )
    box_executor = FusionExecutor(
        box_bucket,
        3,
        [
            FusionEdge(1, 1, 1),
            FusionEdge(1, 2, 1),
            FusionEdge(2, 1, 2),
            FusionEdge(3, 2, 2),
        ];
        handler = box_handler,
    )
    notify_next!(box_executor, 2, Box(10))
    notify_next!(box_executor, 3, Box(20))
    notify_next!(box_executor, 1, original)

    @test length(box_handler.seen) == 2
    @test box_handler.seen[1][1] == 1
    @test box_handler.seen[1][2] === original
    @test box_handler.seen[2][1] == 2
    @test box_handler.seen[2][2] === replacement
end
