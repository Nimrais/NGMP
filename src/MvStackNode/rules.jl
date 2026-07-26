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

    factorization = cholesky(Hermitian(precision_matrix))
    joint_mean = factorization \ weighted_mean
    # Only the target coordinate's variance is needed, so solve against a single
    # basis vector instead of forming the full inverse.
    basis = zeros(eltype(precision_matrix), dimension)
    basis[target_index] = one(eltype(precision_matrix))
    target_variance = (factorization \ basis)[target_index]

    return NormalMeanVariance(joint_mean[target_index], target_variance)
end

@rule MvStack(:out, Marginalisation) (
    m_inputs::ManyOf{N, UnivariateNormalDistributionsFamily},
) where {N} = _stack_forward(m_inputs)

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
