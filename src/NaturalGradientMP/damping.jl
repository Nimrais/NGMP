"""
    DampingMeta(; alpha = 0.5, beta = 0.2)

Parameters of the damped heavy-ball update applied to natural-gradient messages,
in natural-parameter coordinates:

    v ← β·v + α·(η⋆ − η)
    η ← η + v

where `η⋆` is the undamped natural-gradient target and `η` the previously sent
message. Attach via the node `where` clause or `@meta`.

`DampingMeta` is parameter-only by design: GraphPPL's `@meta` evaluates its
right-hand side once and shares the resulting instance across all matched nodes,
so per-node mutable state cannot live here. The mutable state lives in
[`NGMPEdgeState`](@ref), created per node per edge at activation time.
"""
struct DampingMeta{T <: Real}
    α::T
    β::T
end

DampingMeta(; alpha::Real = 0.5, beta::Real = 0.2) = DampingMeta(promote(alpha, beta)...)

"""
    NGMPEdgeState(usermeta)

Per-node, per-edge mutable state of a natural-gradient message: the natural
parameters `(ξ, Λ)` of the previously sent message, the heavy-ball momentum
buffers `(vξ, vΛ)`, and a diagnostic counter `nfired` of rule invocations.
One instance is created for every NGMP-constrained interface inside
`ReactiveMP.activate!(::NGMPDependencies, ...)` and passed to the rule as `meta`;
`usermeta` holds whatever meta the user attached to the node (e.g. a
[`DampingMeta`](@ref), or `nothing`).
"""
mutable struct NGMPEdgeState{M}
    const usermeta::M
    ξ::Float64
    Λ::Float64
    vξ::Float64
    vΛ::Float64
    nfired::Int
end

NGMPEdgeState(usermeta) = NGMPEdgeState(usermeta, 0.0, 0.0, 0.0, 0.0, 0)

damping_parameters(state::NGMPEdgeState{<:DampingMeta}) = (state.usermeta.α, state.usermeta.β)
damping_parameters(state::NGMPEdgeState) = (0.5, 0.2)

"""
    apply_damping!(state::NGMPEdgeState, ξtarget, Λtarget)

Apply the heavy-ball update to the message state given the undamped
natural-gradient target `(ξtarget, Λtarget)` and return the damped natural
parameters `(ξ, Λ)` to be sent.
"""
function apply_damping!(state::NGMPEdgeState, ξtarget::Real, Λtarget::Real)
    α, β = damping_parameters(state)
    state.vξ = β * state.vξ + α * (ξtarget - state.ξ)
    state.ξ += state.vξ
    state.vΛ = β * state.vΛ + α * (Λtarget - state.Λ)
    state.Λ += state.vΛ
    state.nfired += 1
    return (state.ξ, state.Λ)
end
