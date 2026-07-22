"""
    DampingMeta(; alpha = 0.5, beta = 0.2, max_step = Inf,
                  method = :damped, eps = 1e-8, metric_damping = 1e-6)

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

`method` selects the reusable natural-space optimizer:

- `:damped`: the original damped heavy-ball recursion;
- `:projected_nesterov`: projects the current direction onto the previous
  direction and applies the resulting Nesterov correction;
- `:vector_transport`: transports the previous update between the old and
  current diagonal Fisher metrics before adding momentum.

The latter two methods use the same per-edge state and `apply_damping!` entry
point as ordinary damping, so existing NGMP rules need no optimizer-specific
branches. `eps` stabilizes the projected-direction denominator and
`metric_damping` bounds the diagonal metric away from zero.
"""
struct DampingMeta{T <: Real}
    α::T
    β::T
    max_step::T
    method::Symbol
    eps::T
    metric_damping::T
end

function DampingMeta(;
    alpha::Real = 0.5,
    beta::Real = 0.2,
    max_step::Real = Inf,
    method::Symbol = :damped,
    eps::Real = 1e-8,
    metric_damping::Real = 1e-6,
)
    max_step > 0 || throw(ArgumentError("max_step must be positive"))
    eps > 0 || throw(ArgumentError("eps must be positive"))
    metric_damping > 0 || throw(ArgumentError("metric_damping must be positive"))
    method in (:damped, :projected_nesterov, :vector_transport) ||
        throw(ArgumentError(
            "method must be :damped, :projected_nesterov, or :vector_transport",
        ))
    α, β, step, stabilizer, metric_floor =
        promote(alpha, beta, max_step, eps, metric_damping)
    return DampingMeta(
        α, β, step, method, stabilizer, metric_floor,
    )
end

DampingMeta(alpha::Real, beta::Real) = DampingMeta(; alpha = alpha, beta = beta)

"""
    NGMPEdgeState(usermeta; damping = nothing)

Per-node, per-edge mutable state of a natural-gradient message: the previously
sent message (as a distribution and as its natural-parameter vector `η`), the
heavy-ball momentum buffer in natural-parameter space, and a diagnostic counter
`nfired` of rule invocations. Family-agnostic: works for any exponential family
with a [`natural_parameters`](@ref)/[`from_natural`](@ref) bridge (univariate
Normal and Gamma are provided).

One instance is created for every NGMP-constrained interface inside
`ReactiveMP.activate!(::NGMPDependencies, ...)` and passed to the rule as `meta`;
`usermeta` holds whatever meta the user attached to the node (e.g. a
[`DampingMeta`](@ref), `LinearReshapeMeta`, or `nothing`). The optional
`damping` override is kept separately so a dependency policy can choose damping
parameters without hiding metadata needed by the underlying factor rule.
"""
mutable struct NGMPEdgeState{M, D}
    const usermeta::M
    const damping::D
    message::Any               # last sent Distribution (nothing before the first firing)
    η::Vector{Float64}         # natural parameters of the last sent message
    momentum::Vector{Float64}  # heavy-ball momentum in natural-parameter space
    previous_direction::Vector{Float64}
    metric::Vector{Float64}    # diagonal Fisher metric at the last sent message
    nfired::Int
end

NGMPEdgeState(usermeta; damping = nothing) =
    NGMPEdgeState(usermeta, damping, nothing, Float64[], Float64[], Float64[], Float64[], 0)

# Preserve the original full-state positional constructor for callers that
# checkpoint or construct edge state explicitly. The optimizer buffers are
# rebuilt lazily: `apply_damping!` resets them on the first firing, and a
# restored state with `nfired > 0` gets direction/metric buffers shaped like
# its momentum.
NGMPEdgeState(usermeta, message, η, momentum, nfired) =
    NGMPEdgeState(usermeta, nothing, message, η, momentum, zero(momentum), zero(momentum), nfired)

function damping_configuration(state::NGMPEdgeState)
    if !isnothing(state.damping)
        return state.damping
    elseif state.usermeta isa DampingMeta
        return state.usermeta
    else
        return nothing
    end
end

function damping_parameters(state::NGMPEdgeState)
    damping = damping_configuration(state)
    return isnothing(damping) ? (0.5, 0.2) : (damping.α, damping.β)
end

function damping_max_step(state::NGMPEdgeState)
    damping = damping_configuration(state)
    return isnothing(damping) ? Inf : damping.max_step
end

function optimizer_method(state::NGMPEdgeState)
    damping = damping_configuration(state)
    return isnothing(damping) ? :damped : damping.method
end

function optimizer_eps(state::NGMPEdgeState)
    damping = damping_configuration(state)
    return isnothing(damping) ? 1e-8 : damping.eps
end

function optimizer_metric_damping(state::NGMPEdgeState)
    damping = damping_configuration(state)
    return isnothing(damping) ? 1e-6 : damping.metric_damping
end

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
function natural_parameters(target::MultivariateNormalDistributionsFamily)
    ξ, Λ = weightedmean_precision(target)
    return (MvNormalMeanCovariance, vcat(collect(Float64, ξ), vec(collect(Float64, -Λ ./ 2))))
end
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
function from_natural(::Type{MvNormalMeanCovariance}, η)
    d = div(isqrt(1 + 4 * length(η)) - 1, 2)
    return MvNormalWeightedMeanPrecision(η[1:d], -2 .* reshape(η[(d + 1):end], d, d))
end
from_natural(::Type{T}, η) where {T} =
    convert(Distribution, ExponentialFamilyDistribution(T, η, nothing, nothing))

function diagonal_fisher_metric(::Type{NormalMeanVariance}, η, damping)
    precision_value = -2η[2]
    if !(isfinite(precision_value) && precision_value > damping)
        return max.(abs.(η), damping)
    end
    variance = inv(precision_value)
    mean_value = η[1] * variance
    return [
        max(variance, damping),
        max(2variance^2 + 4mean_value^2 * variance, damping),
    ]
end

function diagonal_fisher_metric(::Type{Gamma}, η, damping)
    shape_value = η[1] + 1
    rate_value = -η[2]
    if !(isfinite(shape_value) && shape_value > 0 &&
         isfinite(rate_value) && rate_value > damping)
        return max.(abs.(η), damping)
    end
    return [
        max(trigamma(shape_value), damping),
        max(shape_value / rate_value^2, damping),
    ]
end

diagonal_fisher_metric(::Type, η, damping) = max.(abs.(η), damping)

function optimizer_step!(state::NGMPEdgeState, family, direction)
    α, β = damping_parameters(state)
    method = optimizer_method(state)
    if method === :damped
        @. state.momentum = β * state.momentum + α * direction
    elseif method === :projected_nesterov
        denominator = sum(abs2, state.previous_direction)
        coefficient = state.nfired > 0 && denominator > optimizer_eps(state) ?
            sum(state.previous_direction .* direction) / denominator : 0.0
        correction = β * coefficient
        @. state.momentum = α * (direction + correction * direction)
    elseif method === :vector_transport
        current_metric = diagonal_fisher_metric(
            family, state.η, optimizer_metric_damping(state))
        @. state.momentum =
            β * state.momentum * sqrt(state.metric / current_metric) +
            α * direction
        # The new update lives in the tangent space at the current message.
        # Keep that base metric so it can be transported on the next firing.
        state.metric = current_metric
    end
    state.previous_direction .= direction
    return state.momentum
end

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
    T, ηt = natural_parameters(target)
    if state.nfired == 0
        state.η = zero(ηt)
        state.momentum = zero(ηt)
        state.previous_direction = zero(ηt)
        state.metric = diagonal_fisher_metric(
            T, state.η, optimizer_metric_damping(state))
    end
    direction = ηt - state.η
    optimizer_step!(state, T, direction)
    max_step = damping_max_step(state)
    step_norm = sqrt(sum(abs2, state.momentum))
    if step_norm > max_step
        state.momentum .*= max_step / step_norm
    end
    @. state.η += state.momentum
    if optimizer_method(state) !== :vector_transport
        state.metric = diagonal_fisher_metric(
            T, state.η, optimizer_metric_damping(state))
    end
    state.nfired += 1
    state.message = from_natural(T, state.η)
    return state.message
end

apply_damping!(state::NGMPEdgeState, ξ::Real, Λ::Real) =
    apply_damping!(state, NormalWeightedMeanPrecision(ξ, Λ))
