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
  current diagonal Fisher metrics before adding heavy-ball momentum;
- `:vector_transport_nesterov`: transports the previous velocity between the
  old and current diagonal Fisher metrics, then applies a Nesterov momentum
  step:

      v ← β·Transport(v) + α·(η⋆ − η)
      Δη ← β·v + α·(η⋆ − η)

  This is the natural-parameter analogue of Nesterov SGD. The transported
  velocity is retained as optimizer state, while `Δη` is the step applied to
  the message.

- `:dual_transport` / `:dual_transport_nesterov`: like the two transport
  methods above, but the momentum is transported EXACTLY via Amari duality —
  `G(η_new)⁻¹ G(η_old) v` computed as two closed-form Jacobian-vector products
  through expectation coordinates (see [`transport_natural_vector`](@ref)) —
  instead of the diagonal-metric √-ratio approximation.

The transport methods use the same per-edge state and `apply_damping!` entry
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
    method in (
        :damped,
        :projected_nesterov,
        :vector_transport,
        :vector_transport_nesterov,
        :dual_transport,
        :dual_transport_nesterov,
    ) ||
        throw(ArgumentError(
            "method must be :damped, :projected_nesterov, " *
            ":vector_transport, :vector_transport_nesterov, " *
            ":dual_transport, or :dual_transport_nesterov",
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
    transport_point::Vector{Float64}  # η at which the momentum currently lives (:dual_transport*)
    nfired::Int
end

NGMPEdgeState(usermeta; damping = nothing) =
    NGMPEdgeState(usermeta, damping, nothing, Float64[], Float64[], Float64[], Float64[], Float64[], 0)

# Preserve the original full-state positional constructor for callers that
# checkpoint or construct edge state explicitly. The optimizer buffers are
# rebuilt lazily: `apply_damping!` resets them on the first firing, and a
# restored state with `nfired > 0` gets direction/metric buffers shaped like
# its momentum.
NGMPEdgeState(usermeta, message, η, momentum, nfired) =
    NGMPEdgeState(usermeta, nothing, message, η, momentum, zero(momentum), zero(momentum), zero(momentum), nfired)

"""
    PrecisionTempering(beta, inner)

β-NLL-style faithful-heteroscedastic node meta (Seitzer et al. 2022; Stirn et
al. 2023), message-passing form: rules that send the likelihood precision into
the MEAN pathway temper it to `ρ^(1-beta)` (`beta = 0` → full heteroscedastic
weighting, `beta = 1` → homoscedastic unit weighting, `beta = 0.5` recommended),
while the log-precision site itself stays untempered — the variance channel
trains on the residuals of the tempered-mean posterior, which is the
stop-gradient scheme. `inner` carries the ordinary optimizer meta (e.g. a
[`DampingMeta`](@ref)) for the node's NGMP edges.
"""
struct PrecisionTempering{T <: Real, M}
    beta::T
    inner::M
end

function PrecisionTempering(beta::Real, inner)
    0 <= beta <= 1 || throw(ArgumentError("tempering beta must lie in [0, 1]"))
    b = float(beta)
    return PrecisionTempering{typeof(b), typeof(inner)}(b, inner)
end

function damping_configuration(state::NGMPEdgeState)
    if !isnothing(state.damping)
        return state.damping
    elseif state.usermeta isa DampingMeta
        return state.usermeta
    elseif state.usermeta isa PrecisionTempering && state.usermeta.inner isa DampingMeta
        return state.usermeta.inner
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

# Diagonal of the multivariate Gaussian Fisher in the flat natural layout
# η = (ξ, vec(−Λ/2)) with sufficient statistics T = (x, vec(x xᵀ)):
# Var[xᵢ] = Vᵢᵢ and, by Wick's theorem,
# Var[xᵢxⱼ] = VᵢᵢVⱼⱼ + Vᵢⱼ² + mᵢ²Vⱼⱼ + mⱼ²Vᵢᵢ + 2mᵢmⱼVᵢⱼ
# (reduces to the univariate [v, 2v² + 4m²v] at d = 1). Falls back to the
# generic magnitude metric for improper (non-PD) message states.
function diagonal_fisher_metric(::Type{MvNormalMeanCovariance}, η, damping)
    d = div(isqrt(1 + 4 * length(η)) - 1, 2)
    all(isfinite, η) || return max.(abs.(η), damping)
    Λ = Matrix(Symmetric(-2 .* reshape(η[(d + 1):end], d, d)))
    F = cholesky(Symmetric(Λ); check = false)
    issuccess(F) || return max.(abs.(η), damping)
    V = F \ Matrix{Float64}(I, d, d)
    m = V * η[1:d]
    metric = Vector{Float64}(undef, length(η))
    metric[1:d] = diag(V)
    for j in 1:d, i in 1:d
        metric[d + i + (j - 1) * d] =
            V[i, i] * V[j, j] + V[i, j]^2 +
            m[i]^2 * V[j, j] + m[j]^2 * V[i, i] + 2 * m[i] * m[j] * V[i, j]
    end
    return max.(metric, damping)
end

diagonal_fisher_metric(::Type, η, damping) = max.(abs.(η), damping)

"""
    transport_natural_vector(family, η_from, η_to, v) -> Vector

EXACT m-connection (mixture) parallel transport of a natural-coordinate
tangent vector between two message states, via Amari duality: push `v` to
expectation coordinates with the closed-form Jacobian of `η ↦ μ` at the old
point (`u = G(η_from)·v`), where m-transport is the identity, then pull back
through the Jacobian of `μ ↦ η` at the new point (`G(η_to)⁻¹·u`). Both maps
are closed-form for the flat families here, so no Fisher matrix is ever
materialized and no diagonal truncation is made — this is
`G(η_to)⁻¹ G(η_from) v` computed as two Jacobian-vector products.

Improper endpoints (non-PD precision, non-positive Gamma parameters) fall back
to the identity (e-connection transport, which is exact in η coordinates).
"""
transport_natural_vector(::Type, η_from, η_to, v) = v

function transport_natural_vector(::Type{MvNormalMeanCovariance}, η_from, η_to, v)
    d = div(isqrt(1 + 4 * length(η_from)) - 1, 2)
    unpack(η) = begin
        all(isfinite, η) || return nothing
        Λ = Matrix(Symmetric(-2 .* reshape(η[(d + 1):end], d, d)))
        F = cholesky(Symmetric(Λ); check = false)
        issuccess(F) || return nothing
        V = F \ Matrix{Float64}(I, d, d)
        (V * η[1:d], V, Λ)
    end
    from = unpack(η_from)
    to = unpack(η_to)
    (isnothing(from) || isnothing(to)) && return v
    m₁, V₁, _ = from
    m₂, _, Λ₂ = to
    δξ = v[1:d]
    δΛ = Matrix(Symmetric(-2 .* reshape(v[(d + 1):end], d, d)))
    # η → μ at the old point: μ₁ = m, μ₂ = V + mmᵀ
    δm = V₁ * (δξ - δΛ * m₁)
    δμ₂ = -V₁ * δΛ * V₁ + δm * m₁' + m₁ * δm'
    # μ → η at the new point
    δV = δμ₂ - δm * m₂' - m₂ * δm'
    δΛ′ = -Λ₂ * δV * Λ₂
    δξ′ = δΛ′ * m₂ + Λ₂ * δm
    return vcat(δξ′, vec(-δΛ′ ./ 2))
end

function transport_natural_vector(::Type{NormalMeanVariance}, η_from, η_to, v)
    mv = transport_natural_vector(MvNormalMeanCovariance, η_from, η_to, v)
    return mv
end

function transport_natural_vector(::Type{Gamma}, η_from, η_to, v)
    a₁, b₁ = η_from[1] + 1, -η_from[2]
    a₂, b₂ = η_to[1] + 1, -η_to[2]
    (a₁ > 0 && b₁ > 0 && a₂ > 0 && b₂ > 0) || return v
    δa, δb = v[1], -v[2]
    # η → μ at the old point: μ = (ψ(a) − log b, a/b)
    δμ₁ = trigamma(a₁) * δa - δb / b₁
    δμ₂ = δa / b₁ - a₁ * δb / b₁^2
    # μ → η at the new point: solve the 2×2 Jacobian system
    determinant = (1 - a₂ * trigamma(a₂)) / b₂^2
    δa′ = (-a₂ * δμ₁ / b₂^2 + δμ₂ / b₂) / determinant
    δb′ = (trigamma(a₂) * δμ₂ - δμ₁ / b₂) / determinant
    return [δa′, -δb′]
end

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
    elseif method === :dual_transport
        transported = transport_natural_vector(
            family, state.transport_point, state.η, state.momentum)
        @. state.momentum = β * transported + α * direction
        state.transport_point = copy(state.η)
    elseif method === :dual_transport_nesterov
        transported = transport_natural_vector(
            family, state.transport_point, state.η, state.momentum)
        @. state.momentum = β * transported + α * direction
        nesterov_step = β .* state.momentum .+ α .* direction
        state.transport_point = copy(state.η)
        state.previous_direction .= direction
        return nesterov_step
    elseif method === :vector_transport_nesterov
        current_metric = diagonal_fisher_metric(
            family, state.η, optimizer_metric_damping(state))
        transported = state.momentum .* sqrt.(state.metric ./ current_metric)
        @. state.momentum = β * transported + α * direction
        nesterov_step = β .* state.momentum .+ α .* direction
        # The new update lives in the tangent space at the current message.
        # Keep that base metric so it can be transported on the next firing.
        state.metric = current_metric
        state.previous_direction .= direction
        return nesterov_step
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
        state.transport_point = zero(ηt)
        state.metric = diagonal_fisher_metric(
            T, state.η, optimizer_metric_damping(state))
    end
    direction = ηt - state.η
    step = optimizer_step!(state, T, direction)
    max_step = damping_max_step(state)
    step_norm = sqrt(sum(abs2, step))
    if step_norm > max_step
        step .*= max_step / step_norm
    end
    @. state.η += step
    if optimizer_method(state) ∉ (:vector_transport, :vector_transport_nesterov)
        state.metric = diagonal_fisher_metric(
            T, state.η, optimizer_metric_damping(state))
    end
    state.nfired += 1
    state.message = from_natural(T, state.η)
    return state.message
end

apply_damping!(state::NGMPEdgeState, ξ::Real, Λ::Real) =
    apply_damping!(state, NormalWeightedMeanPrecision(ξ, Λ))
