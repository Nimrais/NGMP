# Bethe free-energy contribution of the deterministic stack.
#
# `out = [in_1, ..., in_H]` is a bijection of R^H with unit Jacobian, so the
# node's local belief lives on H free dimensions and the contribution is exactly
# the negative entropy of that H-dimensional Gaussian:
#
#     Lambda = Lambda_out + Diagonal(1 / v_i)
#     -H[q] = (logdet Lambda - H (1 + log 2 pi)) / 2
#
# This is the same shape as `_manyplus_negative_entropy`, with the sum node's
# rank-one coupling `(1/v_out) 11'` replaced by the stack's full `Lambda_out`.
# Unlike `ManyPlus` -- whose messages assume independence while its free energy
# uses the exact dense joint -- the messages and the score here make the same
# (exact) assumption, so the objective and the updates agree.

function _mv_stack_negative_entropy(output, inputs)
    precision_matrix = Matrix(BayesBase.precision(output))
    for (index, input) in enumerate(inputs)
        precision_matrix[index, index] += inv(last(BayesBase.mean_var(input)))
    end

    logdet_precision = logdet(cholesky(Hermitian(precision_matrix)))
    dimension = length(inputs)
    one_value = one(logdet_precision)
    two_value = one_value + one_value
    log_two_pi = log(two_value * oftype(logdet_precision, pi))

    return (logdet_precision - dimension * (one_value + log_two_pi)) / two_value
end

function ReactiveMP.score(
    ::Type{T},
    ::ReactiveMP.FactorBoundFreeEnergy,
    ::ReactiveMP.Deterministic,
    node::MvStackFactorNode,
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
        return convert(T, _mv_stack_negative_entropy(output, inputs))
    end

    scores = stream |> Rocket.map(T, mapping)
    return ReactiveMP.postprocess_stream_of_scores(stream_postprocessors, scores)
end
