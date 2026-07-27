# Archived model-specific performance oracle. This is not a general compiler
# and is intentionally excluded from the SurrogateModelling production module.
using BayesBase: GenericProd, mean_var, weightedmean_precision
using Rocket: getrecent

@inline function static_natural_product(left, right)
    ξleft, Λleft = weightedmean_precision(left)
    ξright, Λright = weightedmean_precision(right)
    return NormalWeightedMeanPrecision(
        Float64(ξleft + ξright),
        Float64(Λleft + Λright),
    )
end

function static_typed_vector(f, length::Int)
    length > 0 || throw(ArgumentError("a static tile vector cannot be empty"))
    first_value = f(1)
    values = Vector{typeof(first_value)}(undef, length)
    @inbounds values[1] = first_value
    @inbounds for index in 2:length
        values[index] = f(index)
    end
    return values
end

struct StaticTileProgram{V,TC,I,O,P}
    v::Vector{V}
    τc::TC
    intercept::I
    obs_noise::O
    projection::P
end

"""
    StaticPredictionProgram(priors, projection, nneurons, lanes)

Immutable rule and fixed-prior metadata for the fused prediction plate

`ResidualSine → SoftDot → ManyPlus → (+ intercept) → NormalMeanPrecision`.
The graph topology is compiled once; each run extracts fresh messages and
NGMP edge state with [`compile_static_prediction`](@ref).
"""
const StaticPredictionProgram = StaticTileProgram

function StaticTileProgram(
    priors,
    projection,
    nneurons::Int,
    lanes::Int,
)
    lanes > 0 || throw(ArgumentError("a static program cannot have zero lanes"))
    v = static_typed_vector(neuron -> priors[:v][neuron], nneurons)
    return StaticTileProgram(
        v,
        priors[:τ_c],
        priors[:intercept],
        priors[:obs_noise],
        projection,
    )
end

mutable struct StaticTileState{
    ZF,
    RF,
    RB,
    CF,
    CB,
    HB,
    MP,
    OB,
    IB,
    IC,
    MN,
    MB,
    YF,
    YP,
    FS,
    BS,
}
    za_forward::Vector{ZF}
    residual_forward::Vector{RF}
    residual_backward::Vector{RB}
    c_forward::Vector{CF}
    c_backward::Vector{CB}
    h_backward::Vector{HB}
    qza::Vector{NormalWeightedMeanPrecision{Float64}}
    qh::Vector{NormalWeightedMeanPrecision{Float64}}
    manyplus_forward::MP
    out_backward::OB
    intercept_backward::IB
    intercept_cavity::IC
    mean_forward::MN
    mean_backward::MB
    y_forward::YF
    y_prior::YP
    qy::NormalWeightedMeanPrecision{Float64}
    forward_states::Vector{FS}
    backward_states::Vector{BS}
end

@inline static_variable(vardict, name) = vardict[name].variable
@inline static_variable(vardict, name, indices...) =
    vardict[name][indices...].variable

@inline function static_factor_message(variable, index::Int)
    return ReactiveMP.getdata(
        ReactiveMP.as_message(
            getrecent(
                ReactiveMP.get_stream_of_inbound_messages(variable, index),
            ),
        ),
    )
end

@inline function static_outbound_message(variable, index::Int)
    return ReactiveMP.getdata(
        ReactiveMP.as_message(
            getrecent(
                ReactiveMP.get_stream_of_outbound_messages(variable, index),
            ),
        ),
    )
end

@inline function static_current_marginal(variable)
    return ReactiveMP.getdata(
        getrecent(ReactiveMP.get_stream_of_marginals(variable)),
    )
end

@inline function static_natural_normal(distribution)
    ξ, Λ = weightedmean_precision(distribution)
    return NormalWeightedMeanPrecision(Float64(ξ), Float64(Λ))
end

function extract_static_tile(
    vardict,
    ngmp_states,
    lane::Int,
    nneurons::Int,
)
    za = static_typed_vector(
        neuron -> static_variable(vardict, :za, neuron, lane),
        nneurons,
    )
    h = static_typed_vector(
        neuron -> static_variable(vardict, :h, neuron, lane),
        nneurons,
    )
    c = static_typed_vector(
        neuron -> static_variable(vardict, :c, neuron, lane),
        nneurons,
    )
    out = static_variable(vardict, :out, lane)
    intercept = static_variable(vardict, :intercept)
    mean_output = static_variable(vardict, :mean_output, lane)
    y = static_variable(vardict, :y, lane)

    state_offset = 2 * nneurons * (lane - 1)
    forward_states = static_typed_vector(
        neuron -> deepcopy(ngmp_states[state_offset + 2neuron - 1]),
        nneurons,
    )
    backward_states = static_typed_vector(
        neuron -> deepcopy(ngmp_states[state_offset + 2neuron]),
        nneurons,
    )

    return StaticTileState(
        static_typed_vector(
            neuron -> static_factor_message(za[neuron], 1),
            nneurons,
        ),
        static_typed_vector(
            neuron -> static_factor_message(h[neuron], 1),
            nneurons,
        ),
        static_typed_vector(
            neuron -> static_factor_message(za[neuron], 2),
            nneurons,
        ),
        static_typed_vector(
            neuron -> static_factor_message(c[neuron], 1),
            nneurons,
        ),
        static_typed_vector(
            neuron -> static_factor_message(c[neuron], 2),
            nneurons,
        ),
        static_typed_vector(
            neuron -> static_factor_message(h[neuron], 2),
            nneurons,
        ),
        static_typed_vector(
            neuron -> static_natural_normal(
                static_current_marginal(za[neuron]),
            ),
            nneurons,
        ),
        static_typed_vector(
            neuron -> static_natural_normal(
                static_current_marginal(h[neuron]),
            ),
            nneurons,
        ),
        static_factor_message(out, 1),
        static_factor_message(out, 2),
        static_factor_message(intercept, lane + 1),
        static_outbound_message(intercept, lane + 1),
        static_factor_message(mean_output, 1),
        static_factor_message(mean_output, 2),
        static_factor_message(y, 1),
        static_factor_message(y, 2),
        static_natural_normal(static_current_marginal(y)),
        forward_states,
        backward_states,
    )
end

@inline function forward_static_tile!(
    state::StaticTileState,
    program::StaticTileProgram,
)
    nneurons = length(program.v)

    @inbounds for neuron in 1:nneurons
        state.qza[neuron] = static_natural_product(
            state.za_forward[neuron],
            state.residual_backward[neuron],
        )
        state.residual_forward[neuron] =
            @call_rule ResidualSine(
                :out,
                NaturalGradientMessage(program.projection),
            ) (
                m_in = state.za_forward[neuron],
                q_out = state.qh[neuron],
                meta = state.forward_states[neuron],
            )
        state.qh[neuron] = static_natural_product(
            state.residual_forward[neuron],
            state.h_backward[neuron],
        )
        state.c_forward[neuron] =
            @call_rule SoftDot(:y, Marginalisation) (
                q_θ = program.v[neuron],
                m_x = state.residual_forward[neuron],
                q_γ = program.τc,
            )
    end

    total_mean = 0.0
    total_variance = 0.0
    @inbounds for neuron in 1:nneurons
        message_mean, message_variance =
            mean_var(state.c_forward[neuron])
        total_mean += message_mean
        total_variance += message_variance
    end
    state.manyplus_forward =
        NormalMeanVariance(total_mean, total_variance)
    state.intercept_backward =
        @call_rule typeof(+)(:in2, Marginalisation) (
            m_out = state.mean_backward,
            m_in1 = state.manyplus_forward,
        )
    return nothing
end

struct StaticFrontierScratch
    prefix_ξ::Vector{Float64}
    prefix_Λ::Vector{Float64}
    suffix_ξ::Vector{Float64}
    suffix_Λ::Vector{Float64}
end

function StaticFrontierScratch(lanes::Int)
    return StaticFrontierScratch(
        Vector{Float64}(undef, lanes + 1),
        Vector{Float64}(undef, lanes + 1),
        Vector{Float64}(undef, lanes),
        Vector{Float64}(undef, lanes),
    )
end

function prepare_static_intercept_frontier!(
    states::Vector{S},
    scratch::StaticFrontierScratch,
) where {S<:StaticTileState}
    lanes = length(states)
    length(scratch.suffix_ξ) == lanes ||
        throw(ArgumentError("static frontier scratch has the wrong lane count"))

    # The reactive schedule visits tiles in lane order. Save the old right
    # suffix before producing current-iteration lane messages.
    last_ξ, last_Λ =
        weightedmean_precision(states[lanes].intercept_backward)
    scratch.suffix_ξ[lanes] = Float64(last_ξ)
    scratch.suffix_Λ[lanes] = Float64(last_Λ)
    @inbounds for lane in (lanes - 1):-1:1
        lane_ξ, lane_Λ =
            weightedmean_precision(states[lane].intercept_backward)
        scratch.suffix_ξ[lane] =
            Float64(lane_ξ + scratch.suffix_ξ[lane + 1])
        scratch.suffix_Λ[lane] =
            Float64(lane_Λ + scratch.suffix_Λ[lane + 1])
    end
    return nothing
end

function update_static_intercept_frontier!(
    states::Vector{S},
    program::StaticTileProgram,
    scratch::StaticFrontierScratch,
) where {S<:StaticTileState}
    lanes = length(states)
    prior_ξ, prior_Λ = weightedmean_precision(program.intercept)
    scratch.prefix_ξ[1] = Float64(prior_ξ)
    scratch.prefix_Λ[1] = Float64(prior_Λ)

    # EqualityChain forms the left side as msg[k] * prefix[k - 1].
    @inbounds for lane in 1:lanes
        lane_ξ, lane_Λ =
            weightedmean_precision(states[lane].intercept_backward)
        scratch.prefix_ξ[lane + 1] =
            Float64(lane_ξ + scratch.prefix_ξ[lane])
        scratch.prefix_Λ[lane + 1] =
            Float64(lane_Λ + scratch.prefix_Λ[lane])
    end

    @inbounds for lane in 1:lanes
        state = states[lane]
        cavity_ξ = scratch.prefix_ξ[lane]
        cavity_Λ = scratch.prefix_Λ[lane]
        if lane < lanes
            if lane == 1
                # The requested prediction actor keeps the first lane live
                # while later lane updates arrive. Its final cavity therefore
                # observes the current right suffix. Other lanes retain the
                # trace's Gauss-Seidel old-right/current-left frontier.
                right_ξ, right_Λ = weightedmean_precision(
                    states[lanes].intercept_backward,
                )
                for right_lane in (lanes - 1):-1:2
                    lane_ξ, lane_Λ = weightedmean_precision(
                        states[right_lane].intercept_backward,
                    )
                    right_ξ = lane_ξ + right_ξ
                    right_Λ = lane_Λ + right_Λ
                end
                cavity_ξ += right_ξ
                cavity_Λ += right_Λ
            else
                cavity_ξ += scratch.suffix_ξ[lane + 1]
                cavity_Λ += scratch.suffix_Λ[lane + 1]
            end
        end
        state.intercept_cavity = NormalWeightedMeanPrecision(
            Float64(cavity_ξ),
            Float64(cavity_Λ),
        )
        state.out_backward =
            @call_rule typeof(+)(:in1, Marginalisation) (
                m_out = state.mean_backward,
                m_in2 = state.intercept_cavity,
            )
    end
    return nothing
end

@inline function finish_static_tile!(
    state::StaticTileState,
    program::StaticTileProgram,
)
    state.mean_forward =
        @call_rule typeof(+)(:out, Marginalisation) (
            m_in1 = state.manyplus_forward,
            m_in2 = state.intercept_cavity,
        )
    state.y_forward =
        @call_rule NormalMeanPrecision(:out, Marginalisation) (
            m_μ = state.mean_forward,
            q_τ = program.obs_noise,
        )
    state.qy =
        prod(GenericProd(), state.y_forward, state.y_prior)

    total_mean, total_variance = mean_var(state.manyplus_forward)
    output_mean, output_variance = mean_var(state.out_backward)
    nneurons = length(program.v)
    @inbounds for neuron in 1:nneurons
        input_mean, input_variance = mean_var(state.c_forward[neuron])
        state.c_backward[neuron] = NormalMeanVariance(
            output_mean - (total_mean - input_mean),
            output_variance + (total_variance - input_variance),
        )
        state.h_backward[neuron] =
            @call_rule SoftDot(:x, Marginalisation) (
                m_y = state.c_backward[neuron],
                q_θ = program.v[neuron],
                q_γ = program.τc,
            )
        state.qh[neuron] = static_natural_product(
            state.residual_forward[neuron],
            state.h_backward[neuron],
        )
        state.residual_backward[neuron] =
            @call_rule ResidualSine(
                :in,
                NaturalGradientMessage(program.projection),
            ) (
                m_out = state.h_backward[neuron],
                q_in = state.qza[neuron],
                meta = state.backward_states[neuron],
            )
        state.qza[neuron] = static_natural_product(
            state.za_forward[neuron],
            state.residual_backward[neuron],
        )
    end
    return state.qy
end

function static_chunk_range(worker::Int, workers::Int, lanes::Int)
    first_lane = fld((worker - 1) * lanes, workers) + 1
    last_lane = fld(worker * lanes, workers)
    return first_lane:last_lane
end

function static_local_phase!(
    phase,
    states,
    program;
    threaded::Bool,
)
    lanes = length(states)
    if threaded && Threads.nthreads() > 1 && lanes > 1
        workers = min(Threads.nthreads(), lanes)
        Threads.@threads :static for worker in 1:workers
            for lane in static_chunk_range(worker, workers, lanes)
                @inbounds phase(states[lane], program)
            end
        end
    else
        @inbounds for lane in 1:lanes
            phase(states[lane], program)
        end
    end
    return nothing
end

function run_static_sweeps!(
    history::Matrix{NormalWeightedMeanPrecision{Float64}},
    states::Vector{S},
    program::StaticTileProgram,
    iterations::Int;
    threaded::Bool,
    bootstrap_iterations::Int = 1,
) where {S<:StaticTileState}
    lanes = length(states)
    bootstrap_iterations == 1 ||
        throw(ArgumentError("the static executor requires one traced bootstrap"))
    @inbounds for lane in 1:lanes
        history[1, lane] = states[lane].qy
    end
    frontier_scratch = StaticFrontierScratch(lanes)

    for iteration in 2:iterations
        prepare_static_intercept_frontier!(
            states,
            frontier_scratch,
        )
        static_local_phase!(
            forward_static_tile!,
            states,
            program;
            threaded,
        )
        update_static_intercept_frontier!(
            states,
            program,
            frontier_scratch,
        )
        static_local_phase!(
            finish_static_tile!,
            states,
            program;
            threaded,
        )
        @inbounds for lane in 1:lanes
            history[iteration, lane] = states[lane].qy
        end
    end
    return history
end

function extract_static_tiles(
    vardict,
    ngmp_states,
    lanes,
    nneurons,
    program,
)
    expected_states = 2 * lanes * nneurons
    length(ngmp_states) == expected_states ||
        throw(
            ArgumentError(
                "static tile guard failed: expected $expected_states NGMP states, got $(length(ngmp_states))",
            ),
        )
    states = static_typed_vector(
        lane -> extract_static_tile(
            vardict,
            ngmp_states,
            lane,
            nneurons,
        ),
        lanes,
    )
    for lane in 1:lanes
        @inbounds state = states[lane]
        for neuron in 1:nneurons
            @inbounds forward_state = state.forward_states[neuron]
            @inbounds backward_state = state.backward_states[neuron]
            @inbounds forward_message = state.residual_forward[neuron]
            @inbounds backward_message = state.residual_backward[neuron]
            forward_state.message == forward_message ||
                throw(
                    ArgumentError(
                        "static tile guard failed: forward NGMP state is misaligned at lane=$lane neuron=$neuron",
                    ),
                )
            backward_state.message == backward_message ||
                throw(
                    ArgumentError(
                        "static tile guard failed: backward NGMP state is misaligned at lane=$lane neuron=$neuron",
                    ),
                )
        end
    end
    return states
end

mutable struct StaticPredictionExecutor{P,S}
    program::P
    states::Vector{S}
    frontier::StaticFrontierScratch
    history::Matrix{NormalWeightedMeanPrecision{Float64}}
    iterations::Int
end

"""
    compile_static_prediction(program, vardict, ngmp_states;
                              lanes, nneurons, iterations)

Validate a materialized oracle iteration and extract fresh typed lane state.
The resulting executor contains no Rocket subjects, subscriptions, work
frames, or per-edge fanout for the remaining iterations.
"""
function compile_static_prediction(
    program::StaticPredictionProgram,
    vardict,
    ngmp_states;
    lanes::Int,
    nneurons::Int,
    iterations::Int,
)
    lanes > 0 || throw(ArgumentError("lanes must be positive"))
    iterations > 0 || throw(ArgumentError("iterations must be positive"))
    states = extract_static_tiles(
        vardict,
        ngmp_states,
        lanes,
        nneurons,
        program,
    )
    history = Matrix{NormalWeightedMeanPrecision{Float64}}(
        undef,
        iterations,
        lanes,
    )
    return StaticPredictionExecutor(
        program,
        states,
        StaticFrontierScratch(lanes),
        history,
        iterations,
    )
end

"""
    execute_static_prediction!(executor; threaded = false)

Execute the remaining prediction iterations as two disjoint local tile phases
around one deterministic O(N) intercept frontier. With `threaded = true`,
contiguous lane chunks write only their own state; the shared frontier is
replayed serially in trace order.

Returns an `iterations × lanes` matrix containing every requested `y`
posterior, including the oracle bootstrap iteration.
"""
function execute_static_prediction!(
    executor::StaticPredictionExecutor;
    threaded::Bool = false,
)
    history = executor.history
    states = executor.states
    program = executor.program
    lanes = length(states)

    @inbounds for lane in 1:lanes
        history[1, lane] = states[lane].qy
    end
    for iteration in 2:executor.iterations
        prepare_static_intercept_frontier!(
            states,
            executor.frontier,
        )
        static_local_phase!(
            forward_static_tile!,
            states,
            program;
            threaded,
        )
        update_static_intercept_frontier!(
            states,
            program,
            executor.frontier,
        )
        static_local_phase!(
            finish_static_tile!,
            states,
            program;
            threaded,
        )
        @inbounds for lane in 1:lanes
            history[iteration, lane] = states[lane].qy
        end
    end
    return history
end
