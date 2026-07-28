# Exact belief propagation for the deterministic stack `out = [in_1, ..., in_H]`.
#
# The map is a bijection of R^H with unit Jacobian, so nothing here is an
# approximation. Contrast `src/ManyPlusNode/rules.jl`, whose backward rule adds
# the other inputs' variances and therefore assumes them independent of whatever
# the downstream factor learned.

"""
    _stack_forward(messages)

Forward message toward `out`: the joint of independent inbound scalars. The
covariance is diagonal because these are `H` separate messages -- exact, not an
independence assumption imposed on a joint.
"""
function _stack_forward(messages)
    moments = map(message -> BayesBase.mean_var(message), messages)
    means = collect(first.(moments))
    variances = collect(last.(moments))
    # A dense covariance, not a `Diagonal`: downstream consumers (the softdot
    # structured marginal, for one) accumulate off-diagonal mass in place, which
    # a `Diagonal` cannot represent and refuses to store.
    return MvNormalMeanCovariance(means, diagm(variances))
end

function _stack_forward_with_uninformative(messages)
    any(message -> message isa Uninformative, messages) &&
        return vague(
            MvNormalWeightedMeanPrecision,
            length(messages),
        )
    throw(ArgumentError(
        "MvStack inputs must be scalar Gaussian or Uninformative messages",
    ))
end

"""
    _stack_backward(m_out, others, target_index, dimension)

Message toward `in_k`, the `k`-th coordinate marginal of
`m_out * prod_{j != k} m_j(o_j)`.

In information form, with `Lambda_out, xi_out` from the output cavity and the
other inputs contributing a diagonal site with a zero in slot `k`:

    Lambda = Lambda_out + Diagonal(p),   p_k = 0
    xi     = xi_out     + xi_sites,      xi_sites[k] = 0

then marginalise to coordinate `k`, i.e. read `mean[k]` and `Lambda^-1[k, k]`.
Leaving slot `k` empty is what makes this a cavity: the target edge's own
contribution is excluded, so no information is double counted.
"""
function _stack_backward(m_out, others, target_index::Int, dimension::Int)
    precision_matrix = Matrix(BayesBase.precision(m_out))
    weighted_mean = collect(BayesBase.weightedmean(m_out))

    other_index = 0
    for slot in 1:dimension
        slot === target_index && continue
        other_index += 1
        mean_other, variance_other = BayesBase.mean_var(others[other_index])
        site_precision = inv(variance_other)
        precision_matrix[slot, slot] += site_precision
        weighted_mean[slot] += site_precision * mean_other
    end

    return _information_form_coordinate_marginal(
        precision_matrix,
        weighted_mean,
        target_index,
    )
end

function _information_form_coordinate_marginal(
    precision_matrix,
    weighted_mean,
    target_index,
)
    dimension = length(weighted_mean)
    other_indices = [
        index for index in 1:dimension if index != target_index
    ]
    isempty(other_indices) && return NormalWeightedMeanPrecision(
        weighted_mean[target_index],
        precision_matrix[target_index, target_index],
    )

    cross_precision =
        precision_matrix[other_indices, target_index]
    rest_precision = Symmetric(
        precision_matrix[other_indices, other_indices],
    )
    right_hand_side = hcat(
        cross_precision,
        weighted_mean[other_indices],
    )
    solved = try
        rest_precision \ right_hand_side
    catch exception
        exception isa SingularException || rethrow()
        scale = max(opnorm(rest_precision, Inf), one(eltype(precision_matrix)))
        regularized = Symmetric(
            Matrix(rest_precision) +
            sqrt(eps(eltype(precision_matrix))) * scale * I,
        )
        regularized \ right_hand_side
    end

    marginal_precision =
        precision_matrix[target_index, target_index] -
        dot(cross_precision, solved[:, 1])
    marginal_weighted_mean =
        weighted_mean[target_index] -
        dot(cross_precision, solved[:, 2])
    precision_floor = sqrt(eps(eltype(precision_matrix)))
    if !isfinite(marginal_precision) ||
       !isfinite(marginal_weighted_mean) ||
       marginal_precision <= precision_floor
        return vague(NormalWeightedMeanPrecision)
    end
    return NormalWeightedMeanPrecision(
        marginal_weighted_mean,
        marginal_precision,
    )
end

function _stack_backward_with_uninformative(
    m_out,
    others,
    target_index::Int,
    dimension::Int,
)
    precision_matrix = Matrix(BayesBase.precision(m_out))
    weighted_mean = collect(BayesBase.weightedmean(m_out))

    other_index = 0
    for slot in 1:dimension
        slot === target_index && continue
        other_index += 1
        message = others[other_index]
        message isa Uninformative && continue
        message isa UnivariateNormalDistributionsFamily || throw(
            ArgumentError(
                "MvStack inputs must be scalar Gaussian or " *
                "Uninformative messages",
            ),
        )
        mean_other, variance_other = BayesBase.mean_var(message)
        site_precision = inv(variance_other)
        precision_matrix[slot, slot] += site_precision
        weighted_mean[slot] += site_precision * mean_other
    end

    return _information_form_coordinate_marginal(
        precision_matrix,
        weighted_mean,
        target_index,
    )
end

@rule MvStack(:out, Marginalisation) (
    m_inputs::ManyOf{N, UnivariateNormalDistributionsFamily},
) where {N} = _stack_forward(m_inputs)

# An unobserved downstream graph can initially send no information to any
# coordinate. Keep that half-edge genuinely uninformative until all scalar
# Gaussian messages become available; do not manufacture a finite variance.
@rule MvStack(:out, Marginalisation) (
    m_inputs::ManyOf{N, Any},
) where {N} = _stack_forward_with_uninformative(m_inputs)

@rule MvStack((:inputs, k), Marginalisation) (
    m_out::MultivariateNormalDistributionsFamily,
    m_inputs::ManyOf{M, UnivariateNormalDistributionsFamily},
) where {M} = begin
    dimension = M + 1
    length(mean(m_out)) === dimension || throw(DimensionMismatch(
        "MvStack `out` has dimension $(length(mean(m_out))); expected $(dimension)",
    ))
    return _stack_backward(m_out, m_inputs, k, dimension)
end

@rule MvStack((:inputs, k), Marginalisation) (
    m_out::MultivariateNormalDistributionsFamily,
    m_inputs::ManyOf{M, Any},
) where {M} = begin
    dimension = M + 1
    length(mean(m_out)) === dimension || throw(DimensionMismatch(
        "MvStack `out` has dimension $(length(mean(m_out))); expected $(dimension)",
    ))
    return _stack_backward_with_uninformative(
        m_out,
        m_inputs,
        k,
        dimension,
    )
end
