# Unscented-transform tangent projections: the Williams product
# ∇_η E_q[ℓ] = Cov_q[T, ℓ] approximated with 3 sigma points (UnscentedTransforms
# submodule) — the middle ground between the 2-derivative second-order delta
# projection (biased for wide q) and dense quadrature (exact, ~10²–10³ evaluations).

import .UnscentedTransforms: UnscentedTransform, ut_parameters, gaussian_sigma_points, moment_matched_sigma_points
import ReactiveMP

# Bridge from ReactiveMP's exported `Unscented`/`UnscentedTransform` (runtime α, β, κ
# fields, used for Delta nodes) to our type-domain strategy, so notebooks can write
# `TangentProjection(type = Unscented(alpha = 1.0, beta = 0.0, kappa = 2.0))` without
# a name clash. Parameters are mapped VERBATIM — beware that ReactiveMP's `Unscented()`
# defaults (α = 1e-3, κ = 0) squeeze the sigma points to ±0.001σ, i.e. the delta
# regime; the bare type `TangentProjection(type = Unscented)` uses our GH(3)-equivalent
# defaults (α = 1, β = 0, κ = 2) instead.
TangentProjection(ut::ReactiveMP.Unscented) =
    TangentProjection{UnscentedTransform{Float64(ut.α), Float64(ut.β), Float64(ut.κ)}}()
TangentProjection(::Type{<:ReactiveMP.Unscented}) = TangentProjection{UnscentedTransform}()

"""
    project(TangentProjection(type = UnscentedTransform()), q::UnivariateGaussianDistributionsFamily, f::Logpdf)

Sigma-point tangent projection onto the Gaussian edge: classical scaled-UT points
at `q = N(m, v)`, covariance sums against `T = (x, x²)` with the UT covariance
weights and discrete-mean centering (for the default `α = 1, β = 0, κ = 2` the
statistic means are exact — `E[x] = m`, `E[x²] = m² + v` — and the rule coincides
with 3-point Gauss–Hermite). Returns the same unchecked
`ExponentialFamilyDistribution{NormalMeanVariance}` site `(ξ, -Λ/2)` as the other
strategies.
"""
function project(::TangentProjection{U}, q::_GaussianProjectionPoint, f::Logpdf) where {U <: UnscentedTransform}
    α, β, κ = ut_parameters(U)
    normal_q = convert(Distributions.Normal, q)
    m = Distributions.mean(normal_q)
    v = Distributions.var(normal_q)
    x, wm, wc = gaussian_sigma_points(α, β, κ, m, v)
    ℓ = map(xk -> _eval_logmessage(f, xk), x)
    Ex = sum(wm .* x)
    Ex2 = sum(wm .* x .^ 2)
    Eℓ = sum(wm .* ℓ)
    c1 = sum(wc .* (x .- Ex) .* (ℓ .- Eℓ))          # Cov_q[x, ℓ]
    c2 = sum(wc .* (x .^ 2 .- Ex2) .* (ℓ .- Eℓ))    # Cov_q[x², ℓ]
    ξ, Λ = _increments_from_williams_normal(c1, c2, m, v)
    η = promote(ξ, -Λ / 2)
    return ExponentialFamilyDistribution(NormalMeanVariance, collect(η), nothing, nothing)
end

"""
    project(TangentProjection(type = UnscentedTransform()), q::GammaDistributionsFamily, f::Logpdf)

Sigma-point tangent projection onto the Gamma edge via the **generalized** UT in
**log space** (mirroring the trapezoid `Quadrature` method): with `s = log τ`,
the log-Gamma central moments are exact polygamma expressions
(`E[s] = ψ(a) − log b`, `Var[s] = ψ₁(a)`, `μ₃ = ψ₂(a)`, `μ₄ = ψ₃(a) + 3ψ₁(a)²`),
the 3 asymmetric moment-matched points live on the whole real line (mapped back
through `exp`, so they are always inside the Gamma support — no clamping), and
the `log τ` sufficient statistic is exactly linear in the sigma variable.
Covariance sums against `T = (log τ, τ)` use the discrete means (self-normalizing
convention). The scaled-UT parameters `α, β, κ` are not used on this edge.
"""
function project(::TangentProjection{U}, q::_GammaProjectionPoint, f::Logpdf) where {U <: UnscentedTransform}
    q_ef = q isa ExponentialFamilyDistribution ? q : convert(ExponentialFamilyDistribution, q)
    a, b = _shape_rate(q_ef)
    ms = SpecialFunctions.digamma(a) - log(b)
    vs = trigamma(a)
    μ3s = SpecialFunctions.polygamma(2, a)
    μ4s = SpecialFunctions.polygamma(3, a) + 3 * vs^2
    s, w = moment_matched_sigma_points(ms, vs, μ3s, μ4s)
    τ = exp.(s)
    ℓ = map(τk -> _eval_logmessage(f, τk), τ)
    Es = sum(w .* s)
    Eτ = sum(w .* τ)
    Eℓ = sum(w .* ℓ)
    c1 = sum(w .* (s .- Es) .* (ℓ .- Eℓ))    # Cov_q[log τ, ℓ]
    c2 = sum(w .* (τ .- Eτ) .* (ℓ .- Eℓ))    # Cov_q[τ, ℓ]
    Δa, Δb = _increments_from_williams(c1, c2, a, b)
    η = promote(Δa, -Δb)
    return ExponentialFamilyDistribution(Distributions.Gamma, collect(η), nothing, nothing)
end

