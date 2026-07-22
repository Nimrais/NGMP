function _manyplus_negative_entropy(output, inputs)
    _, output_variance = BayesBase.mean_var(output)
    input_variances = map(input -> last(BayesBase.mean_var(input)), inputs)

    logdet_precision =
        mapreduce(variance -> -log(variance), +, input_variances) +
        log1p(sum(input_variances) / output_variance)

    dimension = length(input_variances)
    one_value = one(logdet_precision)
    two_value = one_value + one_value
    log_two_pi = log(two_value * oftype(logdet_precision, pi))

    return (
        logdet_precision - dimension * (one_value + log_two_pi)
    ) / two_value
end

function ReactiveMP.score(
    ::Type{T},
    ::ReactiveMP.FactorBoundFreeEnergy,
    ::ReactiveMP.Deterministic,
    node::ManyPlusFactorNode,
    meta,
    stream_postprocessors,
) where {T <: ReactiveMP.CountingReal}
    inbound_stream(interface) =
        ReactiveMP.get_stream_of_inbound_messages(interface) |>
        ReactiveMP.skip_initial()

    stream = Rocket.combineLatest(
        map(inbound_stream, ReactiveMP.getinterfaces(node)), Rocket.PushNew()
    )

    mapping = messages -> begin
        output = ReactiveMP.getdata(messages[1])
        inputs = map(ReactiveMP.getdata, Base.tail(messages))
        return convert(T, _manyplus_negative_entropy(output, inputs))
    end

    scores = stream |> Rocket.map(T, mapping)
    return ReactiveMP.postprocess_stream_of_scores(stream_postprocessors, scores)
end
