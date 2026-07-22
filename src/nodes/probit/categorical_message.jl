"""
    probit_binary_categorical_message(message::Categorical)

Convert the two-class categorical message emitted by a two-component mixture
into the Bernoulli convention used by `Probit`.  Category 2 is the Bernoulli
success state, so positive Probit scores select mixture component 2.
"""
function probit_binary_categorical_message(message::Categorical)
    probabilities = BayesBase.probvec(message)
    length(probabilities) == 2 || throw(DimensionMismatch(
        "Probit accepts a Categorical output message with exactly two " *
        "probabilities; received $(length(probabilities))",
    ))
    return Bernoulli(probabilities[2])
end

# NormalMixture always sends a Categorical message toward its switch, including
# for two components.  Delegate to ReactiveMP's Bernoulli Probit rules so the
# numerical moment-matching implementation remains defined in one place.
@rule Probit(:in, Marginalisation) (
    m_out::Categorical,
    meta::Union{ProbitMeta, Nothing},
) = @call_rule Probit(:in, Marginalisation) (
    m_out = probit_binary_categorical_message(m_out),
    meta = meta,
)

# Mean-field constraints may provide q(in) rather than a cavity message when
# computing the forward Bernoulli probability.  The stock Gaussian calculation
# depends only on the two moments, so delegate q(in) to that same rule.
@rule Probit(:out, Marginalisation) (
    q_in::UnivariateNormalDistributionsFamily,
    meta::Union{ProbitMeta, Nothing},
) = @call_rule Probit(:out, Marginalisation) (
    m_in = q_in,
    meta = meta,
)

@rule Probit(:in, Marginalisation) (
    m_out::Categorical,
    m_in::UnivariateNormalDistributionsFamily,
    meta::Union{ProbitMeta, Nothing},
) = @call_rule Probit(:in, Marginalisation) (
    m_out = probit_binary_categorical_message(m_out),
    m_in = m_in,
    meta = meta,
)

# Under q(switch), variational scheduling supplies the switch belief as q(out)
# and the score cavity as m(in).  It represents the same two probabilities as
# the factor-to-variable Categorical message handled above.
@rule Probit(:in, Marginalisation) (
    q_out::Categorical,
    m_in::UnivariateNormalDistributionsFamily,
    meta::Union{ProbitMeta, Nothing},
) = @call_rule Probit(:in, Marginalisation) (
    m_out = probit_binary_categorical_message(q_out),
    m_in = m_in,
    meta = meta,
)

# Depending on equality-chain product order, the same binary switch belief may
# be represented as Bernoulli after the first update.
@rule Probit(:in, Marginalisation) (
    q_out::Bernoulli,
    m_in::UnivariateNormalDistributionsFamily,
    meta::Union{ProbitMeta, Nothing},
) = @call_rule Probit(:in, Marginalisation) (
    m_out = q_out,
    m_in = m_in,
    meta = meta,
)
