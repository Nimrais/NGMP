"""
    IVON(; lr = 0.1, ess, hess_init = 1.0, beta1 = 0.9, beta2 = 0.99999,
           weight_decay = 1e-4, mc_samples = 1, hess_approx = :price,
           clip_radius = Inf, rescale_lr = true, debias = true)

The **I**mproved **V**ariational **O**nline **N**ewton optimizer of Shen et al.
(ICML 2024). `IVON` is an `Optimisers.AbstractRule`, so it is used exactly like
`Optimisers.AdamW` in a Lux training loop — but instead of a point estimate it
learns a diagonal Gaussian posterior ``q(θ) = N(θ ∣ m, σ²)`` over the weights,
where the variance ``σ² = 1 / (λ (h + δ))`` is derived from an online estimate
``h`` of the Hessian diagonal.

Training minimizes the variational objective by evaluating gradients at
*sampled* weights (see [`ivon_train_step!`](@ref)); prediction averages over
posterior samples drawn with `rand(rng, ivon, opt_state, ps)` — an ensemble of
networks for the price of one training run.

# Keyword arguments

- `lr`: learning rate ``α`` (adjust during training with `Optimisers.adjust!`).
- `ess`: effective sample size ``λ`` (no default — typically the number of
  training examples, times a reweighting factor).
- `hess_init`: initialization ``h₀`` of the Hessian estimate.
- `beta1`, `beta2`: momentum coefficients for the gradient momentum ``g`` and
  the Hessian estimate ``h``.
- `weight_decay`: prior precision / weight decay ``δ``.
- `mc_samples`: number of weight samples per step used to estimate gradients.
- `hess_approx`: `:price` (default, the reparameterization-trick estimator
  ``ĥ = λ (h + δ) ⋅ noise ⋅ ĝ``) or `:gradsq` (squared-gradient estimator).
- `clip_radius`: elementwise clipping radius for the parameter update.
- `rescale_lr`: multiply `lr` by `(hess_init + weight_decay)` as in the
  official implementation.
- `debias`: apply Adam-style bias correction ``1 - β₁ᵗ`` to the momentum.

# Update rule

With averaged gradient ``ĝ`` and Hessian estimate ``ĥ`` at sampled weights
(see the paper, Algorithm 1, and `ivon/ivon/_ivon.py` in the reference repo):

```math
\\begin{aligned}
g &← β₁ g + (1-β₁) ĝ \\\\
h &← β₂ h + (1-β₂) ĥ + \\tfrac{1}{2}(1-β₂)² (h - ĥ)² / (h + δ) \\\\
m &← m - α \\, \\mathrm{clip}\\!\\left(\\frac{g/(1-β₁ᵗ) + δ m}{h + δ}\\right)
\\end{aligned}
```

# Examples

```julia
using Lux, Optimisers, IVONRepro

model = Chain(Dense(1 => 100, tanh), Dense(100 => 100, tanh), Dense(100 => 2))
ps, st = Lux.setup(rng, model)

opt   = IVON(lr = 0.1, ess = 15_000.0, hess_init = 0.1, beta2 = 0.9999)
state = Optimisers.setup(opt, ps)              # exactly like AdamW
```

# References

Shen, Daheim, Cong, Nickl, Marconi, Bazan, Yokota, Gurevych, Cremers, Khan,
Möllenhoff. *Variational Learning is Effective for Large Deep Networks.*
ICML 2024. <https://arxiv.org/abs/2402.17641>
"""
struct IVON <: Optimisers.AbstractRule
    eta::Float64
    ess::Float64
    hess_init::Float64
    beta1::Float64
    beta2::Float64
    weight_decay::Float64
    mc_samples::Int
    hess_approx::Symbol
    clip_radius::Float64
    rescale_lr::Bool
    debias::Bool
end

function IVON(;
    lr::Real = 0.1,
    ess::Real,
    hess_init::Real = 1.0,
    beta1::Real = 0.9,
    beta2::Real = 0.99999,
    weight_decay::Real = 1e-4,
    mc_samples::Integer = 1,
    hess_approx::Symbol = :price,
    clip_radius::Real = Inf,
    rescale_lr::Bool = true,
    debias::Bool = true,
)
    lr > 0 || throw(ArgumentError("invalid learning rate: $lr"))
    ess > 0 || throw(ArgumentError("invalid effective sample size: $ess"))
    hess_init > 0 || throw(ArgumentError("invalid Hessian initialization: $hess_init"))
    mc_samples >= 1 || throw(ArgumentError("invalid number of MC samples: $mc_samples"))
    0 <= beta1 <= 1 || throw(ArgumentError("invalid beta1: $beta1"))
    0 <= beta2 <= 1 || throw(ArgumentError("invalid beta2: $beta2"))
    hess_approx in (:price, :gradsq) ||
        throw(ArgumentError("invalid hess_approx: $hess_approx"))
    return IVON(lr, ess, hess_init, beta1, beta2, weight_decay, mc_samples,
        hess_approx, clip_radius, rescale_lr, debias)
end

"""
    Optimisers.init(o::IVON, x::AbstractArray)

Per-leaf optimizer state of [`IVON`](@ref):

- `m` — gradient momentum,
- `h` — Hessian diagonal estimate, initialized to `hess_init`,
- `avg_grad` — Welford running average of the gradient over MC samples,
- `avg_est` — running average of the Hessian estimator
  (`noise .* grad` for `:price`, `grad.^2` for `:gradsq`),
- `noise` — the scaled weight-noise of the current MC sample,
- `count` — number of accumulated MC samples,
- `t` — step counter (for the `debias` correction).
"""
function Optimisers.init(o::IVON, x::AbstractArray)
    return (;
        m = zero(x),
        h = fill!(similar(x), o.hess_init),
        avg_grad = zero(x),
        avg_est = zero(x),
        noise = zero(x),
        count = 0,
        t = 0,
    )
end

"""
    Optimisers.apply!(o::IVON, state, x, dx)

The IVON parameter update (torch reference: `IVON._update`,
`ivon/ivon/_ivon.py:226-309`). Consumes the MC-averaged gradient `avg_grad`
and Hessian estimator `avg_est` accumulated by [`ivon_train_step!`](@ref):

1. momentum: ``m ← β₁ m + (1-β₁)\\,\\overline{g}`` (`_new_momentum`, :288);
2. Hessian, with the second-order correction term
   (`_new_hess`, :292): the estimator is ``f = λ (h+δ)\\,\\overline{noise⋅g}``
   for `:price` or ``f = λ\\,\\overline{g²}`` for `:gradsq`, then
   ``h ← β₂ h + (1-β₂) f + \\tfrac{1}{2}(1-β₂)²(h-f)²/(h+δ)``;
3. parameter step (`_new_param_averages`, :302), returned as the update that
   `Optimisers.update!` subtracts:
   ``α_{\\mathrm{eff}}\\,\\mathrm{clip}\\big((m/(1-β₁ᵗ) + δ x)/(h+δ)\\big)``
   with ``α_{\\mathrm{eff}} = α (h₀+δ)`` when `rescale_lr`;
4. reset of the MC accumulators (`_reset_samples`, :111).
"""
function Optimisers.apply!(o::IVON, state, x::AbstractArray, dx)
    T = eltype(x)
    b1, b2 = T(o.beta1), T(o.beta2)
    wd, ess = T(o.weight_decay), T(o.ess)
    m, h = state.m, state.h
    t = state.t + 1

    @. m = b1 * m + (1 - b1) * state.avg_grad

    f = o.hess_approx === :price ? (@. state.avg_est * (h + wd) * ess) :
        (@. state.avg_est * ess)
    @. h = b2 * h + (1 - b2) * f + (T(0.5) * (1 - b2)^2) * (h - f)^2 / (h + wd)

    lr = T(o.eta) * (o.rescale_lr ? T(o.hess_init) + wd : one(T))
    debias = o.debias ? one(T) - b1^t : one(T)
    cr = T(o.clip_radius)
    Δ = @. lr * clamp((m / debias + wd * x) / (h + wd), -cr, cr)

    state.avg_grad .= 0
    state.avg_est .= 0
    return merge(state, (; count = 0, t = t)), Δ
end

"""
    _sigma(o::IVON, h::AbstractArray)

Posterior standard deviation ``σ = 1 / \\sqrt{λ (h + δ)}`` for a Hessian leaf
`h` (torch reference: `IVON._sample_params`, `ivon/ivon/_ivon.py:200-205`).
"""
function _sigma(o::IVON, h::AbstractArray)
    T = eltype(h)
    return 1 ./ sqrt.(T(o.ess) .* (h .+ T(o.weight_decay)))
end

"""
    _sample_for_train!(noise_source, o::IVON, tree, ps)

Draw one MC weight sample for training: for every parameter leaf, draw raw
standard-normal noise, scale it by the posterior standard deviation
[`_sigma`](@ref), *store the scaled noise in the leaf state* (needed by the
`:price` Hessian estimator) and return the perturbed parameters.

`noise_source` is either an `AbstractRNG` or a function
`(path_index, leaf) -> raw_noise::Array` used by the exact-replication tests to
inject noise recorded from the PyTorch implementation.
"""
function _sample_for_train!(rng::AbstractRNG, o::IVON, tree, ps)
    return _sample_for_train!((i, p) -> randn(rng, eltype(p), size(p)), o, tree, ps)
end

function _sample_for_train!(noise_fn, o::IVON, tree, ps)
    i = 0
    return fmap(ps, tree; exclude = Optimisers.isnumeric) do p, leaf
        i += 1
        leaf.state.noise .= noise_fn(i, p) .* _sigma(o, leaf.state.h)
        p .+ leaf.state.noise
    end
end

"""
    _accumulate!(o::IVON, tree, grads)

Fold the gradients of one MC sample into the Welford running averages stored in
the optimizer state (torch reference: `IVON._restore_param_average`,
`ivon/ivon/_ivon.py:134-167`): `avg_grad` accumulates the gradient, `avg_est`
accumulates `noise .* grad` (`:price`) or `grad.^2` (`:gradsq`).
"""
function _accumulate!(o::IVON, tree, grads)
    fmap(tree, grads; exclude = x -> x isa Optimisers.Leaf) do leaf, g
        st = leaf.state
        c = st.count + 1
        st.avg_grad .+= (g .- st.avg_grad) ./ c
        if o.hess_approx === :price
            st.avg_est .+= (st.noise .* g .- st.avg_est) ./ c
        else
            st.avg_est .+= (g .^ 2 .- st.avg_est) ./ c
        end
        leaf.state = merge(st, (; count = c))
        leaf
    end
    return tree
end

"""
    _averaged_gradients(tree)

Extract the tree of MC-averaged gradients `avg_grad` from the optimizer state
tree, shaped like the parameters (the `dx` passed to `Optimisers.update!`).
"""
function _averaged_gradients(tree)
    return fmap(leaf -> leaf.state.avg_grad, tree;
        exclude = x -> x isa Optimisers.Leaf)
end

"""
    ivon_train_step!(rng, backend, objective, data, ts::Training.TrainState)

Perform one IVON training step, mirroring `Lux.Training.single_train_step!`:

1. draw `mc_samples` weight samples from the current posterior
   (`_sample_for_train!`),
2. compute gradients of `objective` at each sampled weight via `backend`
   (e.g. `AutoEnzyme()`),
3. accumulate the Welford averages (`_accumulate!`),
4. apply the IVON update to the *mean* parameters via `Optimisers.update!`.

Returns `(loss, stats, ts)` with the updated `TrainState`. The learning-rate
schedule is applied externally with `Optimisers.adjust!`, exactly as for
`AdamW`.

# Examples

```julia
ts = Training.TrainState(model, ps, st, IVON(lr = 0.1, ess = 15_000.0))
for step in 1:n_steps
    Optimisers.adjust!(ts.optimizer_state, cosine_lr(step))
    loss, _, ts = ivon_train_step!(rng, AutoEnzyme(), objective, (X, y), ts)
end
```
"""
function ivon_train_step!(rng::AbstractRNG, backend, objective, data,
    ts::Training.TrainState)
    return _ivon_train_step!((i, p) -> randn(rng, eltype(p), size(p)),
        backend, objective, data, ts)
end

function _ivon_train_step!(noise_fn, backend, objective, data,
    ts::Training.TrainState)
    o = ts.optimizer
    o isa IVON || throw(ArgumentError("TrainState optimizer must be an IVON rule"))
    tree = ts.optimizer_state
    local loss = nothing
    local stats = nothing
    for _ in 1:o.mc_samples
        ps_noisy = _sample_for_train!(noise_fn, o, tree, ts.parameters)
        ts_noisy = @set ts.parameters = ps_noisy
        grads, loss, stats, _ = Training.compute_gradients(
            backend, objective, data, ts_noisy)
        _accumulate!(o, tree, grads)
    end
    tree, ps = Optimisers.update!(tree, ts.parameters, _averaged_gradients(tree))
    ts = @set ts.parameters = ps
    ts = @set ts.optimizer_state = tree
    ts = @set ts.step = ts.step + 1
    return loss, stats, ts
end

"""
    rand(rng::AbstractRNG, o::IVON, tree, ps)

Draw one weight sample from IVON's variational posterior
``q(θ) = N(θ ∣ ps, 1/(λ(h + δ)))`` — one member of the trained ensemble.
`tree` is the optimizer state (`Optimisers.setup`/`TrainState.optimizer_state`)
holding the Hessian estimate, `ps` the (mean) parameters. Returns a parameter
tree of the same structure; `ps` is not modified.

This is the prediction-time analogue of the PyTorch
`optimizer.sampled_params()` context manager: average predictions over many
draws to get IVON's posterior-averaged predictive distribution.

# Examples

```julia
probs = mean(1:64) do _
    θ = rand(rng, opt, ts.optimizer_state, ts.parameters)
    softmax(first(model(X, θ, ts.states)))
end
```
"""
function Base.rand(rng::AbstractRNG, o::IVON, tree, ps)
    return fmap(ps, tree; exclude = Optimisers.isnumeric) do p, leaf
        p .+ randn(rng, eltype(p), size(p)) .* _sigma(o, leaf.state.h)
    end
end

"""
    posterior_variance(o::IVON, tree)

The elementwise posterior variance ``σ² = 1/(λ (h + δ))`` as a tree shaped
like the parameters. Useful for diagnostics, model merging and sensitivity
analysis (see the paper, Sections 5-6).
"""
function posterior_variance(o::IVON, tree)
    return fmap(leaf -> _sigma(o, leaf.state.h) .^ 2, tree;
        exclude = x -> x isa Optimisers.Leaf)
end
