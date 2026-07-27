# Trace used to discover the shared intercept frontier in the archived oracle.
ENV["PERF_BATCH_SIZE"] = get(ENV, "PERF_BATCH_SIZE", "128")
ENV["PERF_SAMPLES"] = "1"

include(joinpath(@__DIR__, "rxinfer_hotpaths.jl"))

target_label = Ref{Any}(nothing)
current_iteration = Ref(0)
events = NamedTuple[]

function trace_addition_output(event)
    mapping = event.mapping
    ReactiveMP.message_mapping_fform(mapping) === (+) ||
        return nothing
    occursin(":out", string(mapping.vtag)) || return nothing
    push!(
        events,
        (
            iteration = current_iteration[],
            result = mean_var(event.result),
            messages = isnothing(event.messages) ? nothing :
                map(message -> mean_var(ReactiveMP.getdata(message)), event.messages),
        ),
    )
    return nothing
end

trace_callbacks = (
    before_iteration =
        event -> (current_iteration[] = event.iteration),
    after_message_rule_call = trace_addition_output,
)
trace_dependencies = make_perf_dependencies(ITERATION_CONFIG)
trace_model = prepare_inference(
    perf_prediction_model(
        n_neurons = ITERATION_CONFIG.n_neurons,
        priors = ITERATION_PRIORS,
        activation = ITERATION_ACTIVATION,
        activation_deps = trace_dependencies,
        y_prior_variance =
            ITERATION_CONFIG.prediction_prior_variance,
    );
    data = (features = ITERATION_FEATURES,),
    constraints =
        perf_prediction_constraints(ITERATION_PRIORS),
    initialization = iteration_initialization(),
    options = ITERATION_OPTIONS,
    callbacks = trace_callbacks,
)
vardict = GraphPPL.variables(RxInfer.getvardict(trace_model))
target_lane = parse(
    Int,
    get(ENV, "TRACE_LANE", string(length(ITERATION_FEATURES))),
)
target_label[] = vardict[:mean_output][target_lane].variable.label

infer!(
    trace_model;
    data = (features = ITERATION_FEATURES,),
    iterations = 2,
    free_energy = false,
    showprogress = false,
    returnvars = (y = KeepLast(),),
    callbacks = trace_callbacks,
    disable_inference_error_hint = true,
)

iteration_one = filter(event -> event.iteration == 1, events)
iteration_two = filter(event -> event.iteration == 2, events)
println(
    "STATIC_BOUNDARY_TRACE ",
    (
        lanes = length(ITERATION_FEATURES),
        target_lane,
        target_label = target_label[],
        counts = (
            initialization = count(event -> event.iteration == 0, events),
            iteration_one = length(iteration_one),
            iteration_two = length(iteration_two),
        ),
        iteration_one_target =
            length(iteration_one) >= target_lane ?
            iteration_one[target_lane] : nothing,
        iteration_two_target =
            length(iteration_two) >= target_lane ?
            iteration_two[target_lane] : nothing,
    ),
)
