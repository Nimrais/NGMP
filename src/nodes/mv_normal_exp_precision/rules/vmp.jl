# Mean-field VMP rules for MvNormalExpPrecision. All three interfaces see only
# the OTHER edges' marginals (required factorization q(out)q(μ)q(s)); each
# formula is the per-coordinate copy of the scalar Normal/Exp machinery with the
# lognormal moment ρⱼ = E_q(s)[e^{sⱼ}] = e^{mⱼ + Vⱼⱼ/2} — only the DIAGONAL of
# q(s)'s covariance enters, so a dense (MomentForm) q(s) composes correctly.
#
# `meta::Any` on every rule: the node carries `meta = DampingMeta(...)` for its
# NGMP s edge, and NGMPDependencies delivers that meta UNWRAPPED to the
# non-NGMP interfaces (the recurring softdot dispatch gotcha).

# toward y:  N(E[μ], diag(ρ)⁻¹)
@rule MvNormalExpPrecision(:out, Marginalisation) (q_μ::Any, q_s::MultivariateNormalDistributionsFamily, meta::Any) = begin
    ms, Vs = mean_cov(q_s)
    ρ = _mnep_rho(ms, diag(Vs))
    return MvNormalMeanPrecision(mean(q_μ), Matrix(Diagonal(ρ)))
end

# toward μ:  N(E[y], diag(ρ)⁻¹) — symmetric; under mean-field VMP the q(out)
# variance does not enter the μ message (it enters only the s site and the
# average energy)
@rule MvNormalExpPrecision(:μ, Marginalisation) (q_out::Any, q_s::MultivariateNormalDistributionsFamily, meta::Any) = begin
    ms, Vs = mean_cov(q_s)
    ρ = _mnep_rho(ms, diag(Vs))
    return MvNormalMeanPrecision(mean(q_out), Matrix(Diagonal(ρ)))
end

# toward s: the exact mean-field site Σⱼ (½sⱼ − bⱼe^{sⱼ}), bⱼ = ½E[(yⱼ−μⱼ)²],
# returned as a typed expression. This raw object cannot be multiplied into a
# Gaussian marginal directly — the practical inference path is the NGMP rule in
# rules/natural_gradient.jl, which projects it onto the Gaussian tangent space.
@rule MvNormalExpPrecision(:s, Marginalisation) (q_out::Any, q_μ::Any, meta::Any) = begin
    E = _mnep_expected_square_residuals(q_out, q_μ)
    E isa Real && return ExpGammaSiteMessage(one(E) / 2, E / 2)
    return MvExpGammaSiteMessage(fill(one(eltype(E)) / 2, length(E)), E ./ 2)
end
