# Information-form specializations of the structured softdot rules.
#
# Under a structured q(w, z) cluster the inbound message on the w edge (`m_x`)
# arrives as an `MvNormalWeightedMeanPrecision` — the equality-chain product is
# naturally accumulated in information form. The stock ReactiveMP rules then do
# redundant O(d³) work per softdot node per iteration:
#
#   - `@marginalrule SoftDot(:y_x)` calls `mean_cov(m_x)` (a d×d solve) and
#     immediately `cholinv`s the result BACK to the precision it started with;
#   - `@rule softdot(:y)` calls `mean_cov(m_x)` again on the same message.
#
# The methods below dispatch on the information form directly (they are strictly
# more specific than the stock `::Any`/`::NormalDistributionsFamily` signatures):
#
#   - the marginal rule reads (ξ, Λ) off `m_x` with ZERO matrix solves — with a
#     PointMass θ (features are data) the stock algebra reduces exactly to
#     W = [pᵧ+γ̄  −γ̄f′; −γ̄f  Λₓ+γ̄ff′], ξ = [ξᵧ; ξₓ];
#   - the y-rule replaces the full `mean_cov` inverse with a single Cholesky
#     factorization and two triangular solves (μ = Λₓ⁻¹ξₓ and f′Λₓ⁻¹f).
#
# This leaves exactly one unavoidable (d+1)-dim `mean_cov` per node per iteration,
# in the structured `softdot(:γ)` rule that consumes the joint marginal.
#
# NOTE: a regression here is silent — if these methods ever stop being the selected
# dispatch (upstream signature change, different message type on the w edge), Julia
# falls back to the stock rules with identical results and only the speed is lost.
# The test suite guards this by locating these methods in the ReactiveMP method
# table and asserting `Base.which(sig)` still selects them.

import LinearAlgebra: Hermitian, cholesky, dot
import ExponentialFamily: MvNormalWeightedMeanPrecision, NormalMeanVariance, weightedmean_precision
import BayesBase: PointMass, weightedmean

@marginalrule SoftDot(:y_x) (
    m_y::UnivariateNormalDistributionsFamily,
    m_x::MvNormalWeightedMeanPrecision,
    q_θ::PointMass,
    q_γ::Any,
) = begin
    f = mean(q_θ)
    mγ = mean(q_γ)
    ξy, py = weightedmean_precision(m_y)
    ξx = weightedmean(m_x)
    Λx = precision(m_x)

    W = [ (py + mγ)  (-mγ .* f)' ; (-mγ .* f)  (Λx .+ mγ .* (f * f')) ]
    ξ = [ ξy ; ξx ]

    return MvNormalWeightedMeanPrecision(ξ, W)
end

@rule softdot(:y, Marginalisation) (q_θ::PointMass, m_x::MvNormalWeightedMeanPrecision, q_γ::Any) = begin
    f = mean(q_θ)
    mγ = mean(q_γ)
    ξx = weightedmean(m_x)
    Λx = precision(m_x)

    C = cholesky(Hermitian(Λx))
    sol = C \ hcat(ξx, f)   # one factorization, two solves: Λₓ⁻¹ξₓ and Λₓ⁻¹f

    return NormalMeanVariance(dot(f, view(sol, :, 1)), dot(f, view(sol, :, 2)) + inv(mγ))
end
