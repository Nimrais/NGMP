"""
Sigma-point (unscented-transform) rules for approximating expectations and
covariances against exponential-family beliefs, ported from
ExpectationApproximations.jl (`methods/deterministic/quadrature/classical/
{unscented,gen_unscented}.jl`) — self-contained, no dependency on that package.

Two univariate rules, both with 3 points:

- the classical *scaled* UT for Gaussian beliefs (Julier/Uhlmann weights);
- the *generalized* UT — asymmetric sigma points moment-matched to the belief's
  3rd/4th CENTRAL moments — for skewed beliefs such as Gamma.

Used by the `TangentProjection{<:UnscentedTransform}` methods in
`src/tangent_projections/unscented.jl` to approximate the Williams product
`∇_η E_q[ℓ] = Cov_q[T, ℓ]` with 3 evaluations of ℓ, as a middle ground between
the 2-derivative second-order delta projection and dense quadrature.
"""
module UnscentedTransforms

export UnscentedTransform

"""
    UnscentedTransform(; alpha = 1.0, beta = 0.0, kappa = 2.0)

Tangent-projection strategy computing the Williams product with 3 sigma points.
The parameters live in the type domain (like `Quadrature{N}`), because
`TangentProjection` carries only the strategy TYPE: `UnscentedTransform(...)`
returns `UnscentedTransform{α, β, κ}()`, and the bare type
`TangentProjection(type = UnscentedTransform)` means the defaults.

Scaled-UT convention: `λ = α²(d + κ) − d` with `d = 1`; points at `m` and
`m ± √((1 + λ)v)`; mean weights `Wm₀ = λ/(d+λ)`, `Wmᵢ = 1/(2(d+λ))`; covariance
weights add `(1 − α² + β)` to the center. The defaults `α = 1, β = 0, κ = 2`
place the points at `m ± √3·σ` with weights `(2/3, 1/6, 1/6)` — identical to
3-point Gauss–Hermite, exact for polynomial ℓ up to degree 3 against a Gaussian
belief. (ExpectationApproximations.jl defaults to `α = 1e-3, κ = 0`, which
squeezes the points to `±0.001σ` — a finite-difference delta method; those
defaults are deliberately NOT copied, since the whole point of this strategy is
to escape the delta regime.)

On Gamma edges the parameters are ignored and the *generalized* UT is used
instead: asymmetric points moment-matched to the Gamma's skewness/kurtosis
([`moment_matched_sigma_points`](@ref)).
"""
struct UnscentedTransform{A, B, K} end

UnscentedTransform(; alpha::Real = 1.0, beta::Real = 0.0, kappa::Real = 2.0) =
    UnscentedTransform{Float64(alpha), Float64(beta), Float64(kappa)}()

"""
    ut_parameters(::Type{<:UnscentedTransform}) -> (α, β, κ)

Recover the scaled-UT parameters from the strategy type; the bare (unparameterized)
`UnscentedTransform` means the defaults `(1.0, 0.0, 2.0)`.
"""
ut_parameters(::Type{UnscentedTransform{A, B, K}}) where {A, B, K} = (A, B, K)
ut_parameters(::Type{UnscentedTransform}) = (1.0, 0.0, 2.0)

"""
    gaussian_sigma_points(α, β, κ, m, v) -> (points, wm, wc)

Classical scaled-UT sigma points and (mean, covariance) weights for a univariate
Gaussian belief `N(m, v)`; 3-tuples each.
"""
function gaussian_sigma_points(α::Real, β::Real, κ::Real, m::Real, v::Real)
    d = 1
    λ = α^2 * (d + κ) - d
    l = sqrt((d + λ) * v)
    w = 1 / (2 * (d + λ))
    points = (m, m + l, m - l)
    wm = (λ / (d + λ), w, w)
    wc = (λ / (d + λ) + (1 - α^2 + β), w, w)
    return points, wm, wc
end

"""
    moment_matched_sigma_points(m, v, μ3, μ4; lower_bound = -Inf, resampling = 0.9) -> (points, weights)

Generalized (asymmetric) 3-point UT matched to the first four moments of the
belief: mean `m`, variance `v`, and CENTRAL moments `μ3`, `μ4`. Points
`(m, m − u·L, m + v̂·L)` with `L = √v`,

    u = ½(−μ3/L³ + √(4μ4 − 3μ3²/v)/v),    v̂ = u + μ3/L³,

weights `w₃ = 1/(v̂(u + v̂))`, `w₂ = (v̂/u)·w₃`, `w₁ = 1 − w₂ − w₃` (they sum to 1
and reproduce m, v, μ3, μ4 exactly). Degenerates to the symmetric `m ± √3·σ`,
`(2/3, 1/6, 1/6)` rule for Gaussian moments (μ3 = 0, μ4 = 3v²). The kurtosis is
floored at `1.001·v·(μ3/L)²...` — the ExpectationApproximations.jl guard keeping
the discriminant positive.

For beliefs on a bounded support (Gamma: `lower_bound = 0`), a left sigma point
that would violate the bound is resampled as in ExpectationApproximations.jl's
`GenConstrainedUnscented`: `u ← resampling·(m − lower_bound)/L` and the weights
are recomputed from the same formulas — the mean and variance stay matched
exactly, the 3rd/4th moments only approximately (the price of the constraint;
this bites for small Gamma shapes, e.g. a < 2 where `u·L ≥ m`).
"""
function moment_matched_sigma_points(m::Real, v::Real, μ3::Real, μ4::Real; lower_bound::Real = -Inf, resampling::Real = 0.9)
    L = sqrt(v)
    invL3 = inv(L^3)
    minK = L^4 * (μ3 * invL3)^2
    μ4 = μ4 <= minK ? 1.001 * minK : μ4
    u = (1 / 2) * (-μ3 * invL3 + (1 / v) * sqrt(4 * μ4 - 3 * (μ3^2) / v))
    v̂ = u + μ3 * invL3
    if m - u * L <= lower_bound
        u = resampling * (m - lower_bound) / L
    end
    w3 = inv(v̂ * (u + v̂))
    w2 = (v̂ / u) * w3
    w1 = 1 - w2 - w3
    points = (m, m - u * L, m + v̂ * L)
    weights = (w1, w2, w3)
    return points, weights
end

end # module
