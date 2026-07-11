# Natural-gradient message toward the log-rate. The `out` edge is auto-factorized for
# data variables, so the observation arrives as a `q_out` marginal (PointMass), not `m_out`;
# `q_in` is the same-edge marginal injected by `NGMPDependencies`; `meta` is the per-edge
# damping state created at activation.
#
# NOTE: this rule hand-codes the EXACT closed-form Williams product (ρ = E_q[eᶻ]),
# so it ignores the `projection` field of the `NaturalGradientMessage` — every
# strategy would return the same site here.
@rule PoissonExp(:in, NaturalGradientMessage) (q_out::PointMass, q_in::UnivariateNormalDistributionsFamily, meta::NGMPEdgeState) = begin
    y    = mean(q_out)
    m, v = mean_var(q_in)
    rho  = exp(m + v / 2)
    return NaturalGradientMP.apply_damping!(meta, y + (m - 1) * rho, rho)
end
