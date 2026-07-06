export PoissonExpression

"""
The energy object. PoissonExpression(y) represents the exact log-message

    ℓ(z) = log Poisson(y | eᶻ) = y·z − eᶻ − log y!

as a typed callable. It is log-concave but NOT Gaussian — that is exactly
the non-conjugate object the paper projects onto the Gaussian tangent
space. The −log y! constant is kept so it integrates against a Gaussian
cavity to the correct evidence.

Note: PoissonExpression(y) ↦ the equivalent `LogGamma`` message: `LogGamma(one(ℓ.y), ℓ.y; check_args = false)`.
"""
struct PoissonExpression{T<:Real, F<:Real}
    y::T
    loggamma_y::F
end

# loggamma(y + 1) = log y!  — the Poisson normaliser. `y + 1` keeps it finite at
# y = 0 (log 0! = 0), unlike loggamma(0) = ∞.
PoissonExpression(y::Real) = PoissonExpression(y, loggamma(y + 1))

function BayesBase.insupport(::PoissonExpression, ::Float64)
    return true
end 

# Evaluate ℓ(z) directly: `ℓ = PoissonExpression(y); ℓ(z)`.
(ℓ::PoissonExpression)(z) = ℓ.y * z - exp(z) - ℓ.loggamma_y

BayesBase.logpdf(ℓ::PoissonExpression, z) = ℓ(z)

_as_loggamma(ℓ::PoissonExpression) = LogGamma(one(ℓ.y), ℓ.y; check_args = false)

function ClosedFormExpectations.mean(
    e::ClosedFormExpectations.ClosedFormExpectation,
    f::ClosedFormExpectations.Logpdf{<:PoissonExpression},
    q::GaussianDistributionsFamily,
)
    return ClosedFormExpectations.mean(
        e, ClosedFormExpectations.Logpdf(_as_loggamma(f.dist)), q
    )
end

function ClosedFormExpectations.mean(
    e::ClosedFormExpectations.ClosedWilliamsProduct,
    f::ClosedFormExpectations.Logpdf{<:PoissonExpression},
    q::Normal,
)
    return ClosedFormExpectations.mean(
        e, ClosedFormExpectations.Logpdf(_as_loggamma(f.dist)), q
    )
end
