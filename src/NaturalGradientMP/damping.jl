"""
    DampingMeta(; alpha = 0.5, beta = 0.2, max_step = Inf)

Parameters of the damped heavy-ball update applied to natural-gradient messages,
in natural-parameter coordinates:

    v ← β·v + α·(η⋆ − η)
    η ← η + v

where `η⋆` is the undamped natural-gradient target and `η` the previously sent
message. Equivalently, with `β = 0`, the sent message is the normalized product
of powered messages `μ^(t) ∝ (μ^(t−1))^(1−α) · (μ⋆)^α`. Attach via the node
`where` clause or `@meta`. A finite `max_step` bounds the Euclidean norm of each
natural-parameter update after damping; the default `Inf` preserves the original
unbounded recursion.

`DampingMeta` is parameter-only by design: GraphPPL's `@meta` evaluates its
right-hand side once and shares the resulting instance across all matched nodes,
so per-node mutable state cannot live here. The mutable state lives in
[`NGMPEdgeState`](@ref), created per node per edge at activation time.

!!! warning
    With `beta > 0` the momentum step is not a convex combination of natural
    parameters and can leave the family's natural domain (e.g. a Gamma message
    with non-positive rate), which poisons downstream products. Damping-only
    (`beta = 0`) convex combinations of proper targets always stay proper —
    prefer it on Gamma-like edges.
"""
struct DampingMeta{T <: Real}
    α::T
    β::T
    max_step::T
end

function DampingMeta(; alpha::Real = 0.5, beta::Real = 0.2, max_step::Real = Inf)
    max_step > 0 || throw(ArgumentError("max_step must be positive"))
    return DampingMeta(promote(alpha, beta, max_step)...)
end

DampingMeta(alpha::Real, beta::Real) = DampingMeta(; alpha = alpha, beta = beta)

"""
    NGMPEdgeState(usermeta)

Per-node, per-edge mutable state of a natural-gradient message: the previously
sent message (as a distribution and as its natural-parameter vector `η`), the
heavy-ball momentum buffer in natural-parameter space, and a diagnostic counter
`nfired` of rule invocations. Family-agnostic: works for any exponential family
with a [`natural_parameters`](@ref)/[`from_natural`](@ref) bridge (univariate
Normal and Gamma are provided).

One instance is created for every NGMP-constrained interface inside
`ReactiveMP.activate!(::NGMPDependencies, ...)` and passed to the rule as `meta`;
`usermeta` holds whatever meta the user attached to the node (e.g. a
[`DampingMeta`](@ref), or `nothing`).
"""
mutable struct NGMPEdgeState{M}
    const usermeta::M
    message::Any               # last sent Distribution (nothing before the first firing)
    η::Vector{Float64}         # natural parameters of the last sent message
    momentum::Vector{Float64}  # heavy-ball momentum in natural-parameter space
    nfired::Int
end

NGMPEdgeState(usermeta) = NGMPEdgeState(usermeta, nothing, Float64[], Float64[], 0)

damping_parameters(state::NGMPEdgeState{<:DampingMeta}) = (state.usermeta.α, state.usermeta.β)
damping_parameters(state::NGMPEdgeState) = (0.5, 0.2)
damping_max_step(state::NGMPEdgeState{<:DampingMeta}) = state.usermeta.max_step
damping_max_step(state::NGMPEdgeState) = Inf

"""
    natural_parameters(target) -> (family_tag, η::Vector{Float64})

Extract the natural-parameter vector of a message target together with its
exponential-family tag, WITHOUT domain validation — natural-gradient sites are
legitimate improper messages (e.g. a Gamma site with negative rate increment),
and the checked `convert(ExponentialFamilyDistribution, ·)` path would throw on
them. Conventions follow ExponentialFamily.jl: Normal `η = (ξ, −Λ/2)`, Gamma
`η = (shape − 1, −rate)`; the implicit flat message `η = 0` is therefore
`(ξ = 0, Λ = 0)` and `Gamma(1, 0)` respectively.
"""
natural_parameters(target::UnivariateNormalDistributionsFamily) =
    (NormalMeanVariance, [weightedmean(target), -precision(target) / 2])
natural_parameters(target::GammaDistributionsFamily) =
    (Gamma, [shape(target) - 1.0, -rate(target)])
natural_parameters(target::ExponentialFamilyDistribution{T}) where {T} =
    (T, collect(Float64, getnaturalparameters(target)))
# Generic fallback — checked, proper distributions only.
natural_parameters(target) =
    (exponential_family_typetag(target), collect(Float64, getnaturalparameters(convert(ExponentialFamilyDistribution, target))))

"""
    from_natural(family_tag, η) -> Distribution

Reconstruct the message distribution from a (possibly improper) natural-parameter
vector, without domain validation. `NormalWeightedMeanPrecision` and
`GammaShapeRate` are plain structs with no argument checks, so improper messages
round-trip safely; the generic fallback uses the unchecked 4-argument
`ExponentialFamilyDistribution` constructor and is proper-only at conversion.
"""
from_natural(::Type{NormalMeanVariance}, η) = NormalWeightedMeanPrecision(η[1], -2 * η[2])
from_natural(::Type{Gamma}, η) = GammaShapeRate(η[1] + 1.0, -η[2])
from_natural(::Type{T}, η) where {T} =
    convert(Distribution, ExponentialFamilyDistribution(T, η, nothing, nothing))

"""
    apply_damping!(state::NGMPEdgeState, target) -> Distribution

Apply the heavy-ball update to the message state in natural-parameter space,
given the undamped natural-gradient `target` (a distribution or an
`ExponentialFamilyDistribution` site), and return the damped message to send.
Before the first firing the previous message is the implicit flat `η = 0`, so
the first sent message is `α · η_target`.

    apply_damping!(state, ξ, Λ)

Convenience wrapper for univariate Gaussian targets in weighted-mean/precision
coordinates; equivalent to `apply_damping!(state, NormalWeightedMeanPrecision(ξ, Λ))`.
"""
function apply_damping!(state::NGMPEdgeState, target)
    α, β = damping_parameters(state)
    T, ηt = natural_parameters(target)
    if state.nfired == 0
        state.η = zero(ηt)
        state.momentum = zero(ηt)
    end
    @. state.momentum = β * state.momentum + α * (ηt - state.η)
    max_step = damping_max_step(state)
    step_norm = sqrt(sum(abs2, state.momentum))
    if step_norm > max_step
        state.momentum .*= max_step / step_norm
    end
    @. state.η += state.momentum
    state.nfired += 1
    state.message = from_natural(T, state.η)
    return state.message
end

apply_damping!(state::NGMPEdgeState, ξ::Real, Λ::Real) =
    apply_damping!(state, NormalWeightedMeanPrecision(ξ, Λ))
