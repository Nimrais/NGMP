function _sum_mean_variance(messages)
    iterator = iterate(messages)
    isnothing(iterator) && throw(ArgumentError("Expected at least one Gaussian message."))

    message, state = iterator
    total_mean, total_variance = BayesBase.mean_var(message)

    while true
        iterator = iterate(messages, state)
        isnothing(iterator) && break

        message, state = iterator
        message_mean, message_variance = BayesBase.mean_var(message)
        total_mean = total_mean + message_mean
        total_variance = total_variance + message_variance
    end

    return total_mean, total_variance
end

@rule ManyPlus(:out, Marginalisation) (
    m_inputs::ManyOf{N, UnivariateNormalDistributionsFamily},
) where {N} = begin
    output_mean, output_variance = _sum_mean_variance(m_inputs)
    return NormalMeanVariance(output_mean, output_variance)
end

@rule ManyPlus((:inputs, k), Marginalisation) (
    m_out::UnivariateNormalDistributionsFamily,
    m_inputs::ManyOf{N, UnivariateNormalDistributionsFamily},
) where {N} = begin
    output_mean, output_variance = BayesBase.mean_var(m_out)
    other_mean, other_variance = _sum_mean_variance(m_inputs)
    return NormalMeanVariance(
        output_mean - other_mean, output_variance + other_variance
    )
end
