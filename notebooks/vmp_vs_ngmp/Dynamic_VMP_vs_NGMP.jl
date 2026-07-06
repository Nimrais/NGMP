### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 8bf2d1e2-8403-47f7-b21a-cc793969197c
md"""
# VMP vs NGMP for the dynamic β-ensemble (ETTh1, h96)

The `dynamic` ensemble model (per forecaster ``i``, observation ``j``):

```math
\begin{aligned}
w_i &\sim \mathcal{MVN}(0, (0.01)^{-1} I), \qquad
\tau_i \sim \Gamma(1, 10^{-3}), \qquad
\beta_i \sim \Gamma(1, 10^{3}),\\
z_{ij} &\sim \mathrm{softdot}(f_j, w_i, \tau_i), \qquad
\gamma_{ij} \sim \Gamma(1, \beta_i), \qquad
z_{ij} = \log \gamma_{ij},\\
y_j &\sim \mathcal N(\mathrm{pred}_{ij},\, \gamma_{ij}^{-1}).
\end{aligned}
```

Both arms use the same factorization ``q(w, z, \gamma, \tau, \beta) = q(w)\,q(z,\gamma)\,q(\tau)\,q(\beta)``
and differ only in how the non-conjugate `Log` link is handled:

1. **VMP** (the ProbabilisticEnsembling baseline): the `Log` node sends the exact
   `LogGamma` / `LogNormal` messages, and the marginals `q(z)`, `q(γ)` are
   form-constrained with `ProjectedTo(..., ClosedFormStrategy)`.
2. **NGMP**: the `Log` node's messages toward *both* edges are damped
   natural-gradient projections at the receiving marginals
   (`NGMPDependencies(out = ..., in = ...)`), so every message stays conjugate
   (Gaussian toward ``z``, Gamma toward ``γ``) and **no form constraints are
   needed** — the same `DampingMeta` machinery as the Poisson notebook, now
   family-generic (the damping state stores the previous message as a
   distribution and combines messages in natural-parameter space).
"""

# ╔═╡ 44ae68ef-3429-4942-aaa5-a4decaab2cd5
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ 053f43e5-dc03-46d4-a50b-8bed3748e130
begin
    using RxInfer
    using ExponentialFamily
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    using SurrogateModelling
    using ProbabilisticEnsembling
    using YAML
    using JLD2
end

# ╔═╡ c10f10ea-13b9-439c-8762-72cf3179be5b
begin
    using Plots
    using Statistics
    using Distributions
    using Printf
end

# ╔═╡ 48eaeab6-f543-42ad-be90-253faaf07760
begin
    ROOT = normpath(joinpath(@__DIR__, "..", ".."))
    spec = ProbabilisticEnsembling._parse_spec(
        YAML.load_file(joinpath(ROOT, "sessions", "dynamic", "vae", "dynamic_ETTh1_96.yaml")),
    )
    # Expert + VAE evaluation is expensive — cache the prepared arrays. PE resolves
    # `data/` and `models/` relative to the working directory, hence the cd(ROOT).
    cache_path = joinpath(@__DIR__, "dynamic_etth1_h96_cache.jld2")
    if !isfile(cache_path)
        prep = cd(() -> ProbabilisticEnsembling.before_rxinfer(spec), ROOT)
        jldsave(
            cache_path;
            y_val = prep[1], y_test = prep[2],
            predictions_val = prep[3], predictions_test = prep[4],
            features_val = prep[5], features_test = prep[6],
        )
    end
    dcache = load(cache_path)
    y_val = dcache["y_val"]
    y_test = dcache["y_test"]
    predictions_val = dcache["predictions_val"]
    predictions_test = dcache["predictions_test"]
    features_val = dcache["features_val"]
    features_test = dcache["features_test"]
    priors = spec.priors
    n_forecasters = size(predictions_val, 1);
end

# ╔═╡ b1d9d358-4807-49f6-96a3-82e9676108bf
begin
    n_obs = min(500, length(y_val))     # training subsample of the validation split
    iterations = 20                     # α = 0.2 damping needs ~20 sweeps to settle
    damping = DampingMeta(alpha = 0.2, beta = 0.0)  # beta = 0: Gamma momentum can leave the natural domain
    κ = 1.0                             # weight of the E[β] noise floor at prediction

    y_train = y_val[1:n_obs]
    f_train = features_val[1:n_obs]
    p_train = predictions_val[:, 1:n_obs];
end

# ╔═╡ 19fe300a-1294-455b-aa21-32058c80b2dc
md"""
## VMP baseline (ProjectedTo form constraints)

An inline copy of ProbabilisticEnsembling's `univariate_dynamic_ensemble`: the
`Log` node emits exact non-conjugate messages, and the `z`/`γ` marginals are
projected onto `NormalMeanVariance`/`Gamma` with the closed-form strategy.
"""

# ╔═╡ 1080a4cd-28f0-4585-9831-c4385bfef530
@model function dynamic_vmp(n_forecasters, n_obs, y, features, predictions, priors)
    local w, z, γ, τ, β
    for i in 1:n_forecasters
        w[i] ~ priors[:w][i]
        τ[i] ~ priors[:τ][i]
        β[i] ~ priors[:β][i]
    end
    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i]) where { meta = LowRankMeta() }
            γ[i, j] ~ GammaShapeRate(1.0, β[i])
            z[i, j] ~ Log(γ[i, j])
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
        end
    end
end

# ╔═╡ ab3a0d0d-9944-4810-b4b2-df882cd2ec45
begin
    @constraints function dynamic_vmp_constraints()
        q(w, z, γ, τ, β) = q(w)q(z, γ)q(τ)q(β)
        # keep q(w) in moment form: the softdot rules read mean/cov of q(w), so
        # converting the information-form marginal ONCE per update (instead of a
        # 65×65 solve in every rule call) is a ~7x wall-clock win
        q(w)::MomentForm()
        q(z)::ProjectedTo(
            NormalMeanVariance,
            parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
        )
        q(γ)::ProjectedTo(
            Gamma,
            parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
        )
    end

    @initialization function dynamic_init(priors)
        q(w) = deepcopy(priors[:w])
        q(z) = NormalMeanVariance(0.0, 1.0)
        q(γ) = GammaShapeScale(1.0, 1.0)
        q(τ) = priors[:τ]
        q(β) = priors[:β]
    end
end

# ╔═╡ 670dcdc6-75ce-490a-910e-7c52c688bf42
vmp_result = infer(
    model = dynamic_vmp(n_forecasters = n_forecasters, n_obs = n_obs, priors = priors),
    data = (y = y_train, features = f_train, predictions = p_train),
    constraints = dynamic_vmp_constraints(),
    initialization = dynamic_init(priors),
    iterations = iterations,
    free_energy = false,
    options = (limit_stack_depth = 500,),
    showprogress = true,
)

# ╔═╡ a0c3ccfe-dba0-4704-8434-4c47aaa6eb12
md"""
## Native NGMP

The only change to the model is the `where` clause on the `Log` link:
`NGMPDependencies(out = nothing, in = nothing)` makes each of its two outbound
messages additionally subscribe to the receiving edge's own marginal and
dispatches the `NaturalGradientMessage` rules
(`src/nodes/log/rules/natural_gradient.jl`):

- toward ``z``: the Gamma message on ``γ`` pulls back to a `LogGamma` log-message,
  projected at ``q(z)`` → a damped Gaussian message;
- toward ``γ``: the Gaussian message on ``z`` pushes forward to a `LogNormal`
  log-message, projected at ``q(γ)`` via the Gamma inverse-Fisher map → a damped
  Gamma message.

The constraints keep **only the factorization** — no `ProjectedTo` lines, since
every message is already in-family.
"""

# ╔═╡ 33c46a0a-3035-4aef-be05-2176dd3eb561
@model function dynamic_ngmp(n_forecasters, n_obs, y, features, predictions, priors, deps, damping)
    local w, z, γ, τ, β
    for i in 1:n_forecasters
        w[i] ~ priors[:w][i]
        τ[i] ~ priors[:τ][i]
        β[i] ~ priors[:β][i]
    end
    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i]) where { meta = LowRankMeta() }
            γ[i, j] ~ GammaShapeRate(1.0, β[i])
            z[i, j] ~ Log(γ[i, j]) where { dependencies = deps, meta = damping }
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
        end
    end
end

# ╔═╡ 64779645-a909-4725-a73b-76f909e90adf
@constraints function dynamic_ngmp_constraints()
    q(w, z, γ, τ, β) = q(w)q(z, γ)q(τ)q(β)
    q(w)::MomentForm()   # same moment-form marginal as the VMP arm (~7x faster)
end

# ╔═╡ d3ef2a41-05fe-4220-a9b2-aabdeab76b16
begin
    ngmp_deps = NGMPDependencies(out = nothing, in = nothing)
    ngmp_result = infer(
        model = dynamic_ngmp(
            n_forecasters = n_forecasters,
            n_obs = n_obs,
            priors = priors,
            deps = ngmp_deps,
            damping = damping,
        ),
        data = (y = y_train, features = f_train, predictions = p_train),
        constraints = dynamic_ngmp_constraints(),
        initialization = dynamic_init(priors),
        iterations = iterations,
        free_energy = false,
        options = (limit_stack_depth = 500,),
        showprogress = true,
    )
    # one damping state per Log edge: 2 (out + in) per forecaster per observation
    @assert length(ngmp_deps.states) == 2 * n_forecasters * n_obs
    ngmp_result
end

# ╔═╡ 163c4f31-3764-49c1-ba3d-51a813f564c2
begin
    τ_vmp = last(vmp_result.posteriors[:τ])
    β_vmp = last(vmp_result.posteriors[:β])
    w_vmp = last(vmp_result.posteriors[:w])
    τ_ngmp = last(ngmp_result.posteriors[:τ])
    β_ngmp = last(ngmp_result.posteriors[:β])
    w_ngmp = last(ngmp_result.posteriors[:w])
    Eβ_vmp = mean.(β_vmp)
    Eβ_ngmp = mean.(β_ngmp)
    Eτ_vmp = mean.(τ_vmp)
    Eτ_ngmp = mean.(τ_ngmp);
end

# ╔═╡ 83497909-fbe2-4041-97c2-8469ea660210
begin
    idx = 1:n_forecasters
    pβ = scatter(
        idx .- 0.1, Eβ_vmp;
        yscale = :log10, color = :darkorange, markersize = 6, label = "VMP",
        xlabel = "forecaster", ylabel = "E[β]  (noise-variance floor)",
        title = "Learned β per forecaster", xticks = idx,
    )
    scatter!(pβ, idx .+ 0.1, Eβ_ngmp; color = :dodgerblue, markersize = 6, label = "NGMP")
    pτ = scatter(
        idx .- 0.1, Eτ_vmp;
        yscale = :log10, color = :darkorange, markersize = 6, label = "VMP",
        xlabel = "forecaster", ylabel = "E[τ]  (softdot precision)",
        title = "Learned τ per forecaster", xticks = idx,
    )
    scatter!(pτ, idx .+ 0.1, Eτ_ngmp; color = :dodgerblue, markersize = 6, label = "NGMP")
    plot(pβ, pτ; layout = (1, 2), size = (900, 350))
end

# ╔═╡ d0f0204e-5f3b-414f-80a0-4317afb48913
md"""
## Test-set predictive comparison

Both arms are scored with the same predictive rule (ported from
`dynamic_ngmp_surrogate.jl`): a one-sweep conjugate `softdot` regression with the
trained `q(w)`, `q(τ)` gives the test-time `q(z)`; each forecaster's predictive
variance is the propagated ``z``-uncertainty plus the learned noise floor,

```math
V_{ij} = \exp(-m_{ij} + v_{ij}/2) + \kappa\, \mathbb E[\beta_i],
\qquad P_{ij} = 1/V_{ij},
```

and the ensemble is precision-weighted:
``\mu_j = \sum_i P_{ij}\,\mathrm{pred}_{ij} / \sum_i P_{ij}``,
``\sigma_j = (\sum_i P_{ij})^{-1/2}``.
"""

# ╔═╡ 20a3a5ee-8d27-4be0-b39f-bb4c53a29fcb
@model function inner_z(n_forecasters, n_obs, features, w_priors, τ_priors, obsz, Rz)
    local w, z, τ
    for i in 1:n_forecasters
        w[i] ~ w_priors[i]
        τ[i] ~ τ_priors[i]
    end
    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i])
            obsz[i, j] ~ NormalMeanVariance(z[i, j], Rz[i, j])
        end
    end
end

# ╔═╡ 2f784cf1-1b43-494c-8d69-6de4871110ca
begin
    inner_constraints = @constraints begin
        q(w, z, τ) = q(w, z)q(τ)
    end

    function infer_qz(features, w_priors, τ_priors, obsz, Rz; iters = 1)
        nf, no = size(obsz)
        init = @initialization begin
            q(w) = w_priors
            q(τ) = τ_priors
            q(z) = NormalMeanVariance(0.0, 1.0)
        end
        res = infer(
            model = inner_z(n_forecasters = nf, n_obs = no, w_priors = w_priors, τ_priors = τ_priors),
            data = (features = features, obsz = obsz, Rz = Rz),
            constraints = inner_constraints,
            initialization = init,
            iterations = iters,
            free_energy = false,
            options = (limit_stack_depth = 500,),
        )
        qz = last(res.posteriors[:z])
        return map(mean, qz), map(var, qz)
    end

    # predictive rule from dynamic_ngmp_surrogate.jl: V = E[1/γ] + κ·E[β]
    function dyn_predict(features, predictions, w_priors, τ_priors, Eβ; κ = 1.0)
        nf, no = length(w_priors), length(features)
        mz, vz = infer_qz(features, w_priors, τ_priors, zeros(nf, no), fill(1e12, nf, no))
        V = exp.(.-mz .+ vz ./ 2) .+ κ .* reshape(Float64.(Eβ), nf, 1)
        P = clamp.(1 ./ V, 1e-6, 1e6)
        μ = Vector{Float64}(undef, no)
        σ = Vector{Float64}(undef, no)
        for j in 1:no
            τc = sum(@view P[:, j])
            μ[j] = sum(P[i, j] * predictions[i, j] for i in 1:nf) / τc
            σ[j] = sqrt(1 / τc)
        end
        return μ, σ
    end

    function predictive_metrics(μ, σ, y; quantiles = (0.1, 0.9))
        n = length(y)
        ll_terms = [logpdf(Normal(μ[j], σ[j]), y[j]) for j in 1:n]
        zq95 = 1.959963984540054
        cov95 = mean((y .>= μ .- zq95 .* σ) .& (y .<= μ .+ zq95 .* σ))
        pin = Float64[]
        for q in quantiles
            zq = quantile(Normal(), q)
            qhat = μ .+ zq .* σ
            push!(pin, mean(max.(q .* (y .- qhat), (q - 1) .* (y .- qhat))))
        end
        return (; mae = mean(abs.(μ .- y)), rmse = sqrt(mean((μ .- y) .^ 2)),
                ll = mean(ll_terms), ll_std = std(ll_terms), cov95, pinball = mean(pin))
    end
end

# ╔═╡ 2d9a5277-deb4-471d-a286-86325b90549e
begin
    μ_vmp, σ_vmp = dyn_predict(features_test, predictions_test, w_vmp, τ_vmp, Eβ_vmp; κ = κ)
    m_vmp = predictive_metrics(μ_vmp, σ_vmp, y_test)
    μ_ngmp, σ_ngmp = dyn_predict(features_test, predictions_test, w_ngmp, τ_ngmp, Eβ_ngmp; κ = κ)
    m_ngmp = predictive_metrics(μ_ngmp, σ_ngmp, y_test)
    @info "test-set predictive" VMP = m_vmp NGMP = m_ngmp
end

# ╔═╡ 54be4609-e549-44e5-a873-880e87633ff6
Markdown.parse(
    """
    | method | MAE | RMSE | mean-LL | LL std | cov95 | pinball |
    |---|---|---|---|---|---|---|
    | VMP (ProjectedTo) | $(@sprintf("%.4f", m_vmp.mae)) | $(@sprintf("%.4f", m_vmp.rmse)) | $(@sprintf("%.4f", m_vmp.ll)) | $(@sprintf("%.3f", m_vmp.ll_std)) | $(@sprintf("%.4f", m_vmp.cov95)) | $(@sprintf("%.4f", m_vmp.pinball)) |
    | NGMP (native) | $(@sprintf("%.4f", m_ngmp.mae)) | $(@sprintf("%.4f", m_ngmp.rmse)) | $(@sprintf("%.4f", m_ngmp.ll)) | $(@sprintf("%.3f", m_ngmp.ll_std)) | $(@sprintf("%.4f", m_ngmp.cov95)) | $(@sprintf("%.4f", m_ngmp.pinball)) |

    Test set: $(length(y_test)) observations, trained on $(n_obs) validation
    observations, $(iterations) iterations, damping α = $(damping.α), β = $(damping.β), κ = $(κ).
    """,
)

# ╔═╡ be739f8e-3195-4c53-8578-faa9bb89525e
begin
    window = 1:min(200, length(y_test))
    plot(
        window, μ_vmp[window];
        ribbon = 1.96 .* σ_vmp[window], fillalpha = 0.18, lw = 2, color = :darkorange,
        label = "VMP (ProjectedTo)", xlabel = "test observation", ylabel = "scaled OT",
        title = "Precision-weighted ensemble predictive, 95% bands",
    )
    plot!(
        window, μ_ngmp[window];
        ribbon = 1.96 .* σ_ngmp[window], fillalpha = 0.18, lw = 2, color = :dodgerblue,
        label = "NGMP (native)",
    )
    scatter!(
        window, y_test[window];
        markersize = 2, markerstrokewidth = 0, alpha = 0.5, color = :black, label = "y",
    )
end

# ╔═╡ 53aaa6cb-febb-4a0f-962a-3e0bb69e6ed0
md"""
Both arms solve the same constrained-Bethe problem — the factorization
`q(w)q(z,γ)q(τ)q(β)` is identical; they differ in *where* the projection lives.
VMP projects the **marginals** after multiplying exact non-conjugate messages
(`ProjectedTo` with a Manopt descent per marginal per iteration), while NGMP
projects the **messages** at the receiving marginals in closed form (one Williams
product + inverse-Fisher map per message), so the whole graph stays conjugate and
no per-marginal optimization is required. The damping state is per-edge and
family-generic: the same `DampingMeta` heavy-ball update acts on the Gaussian
site toward ``z`` and the Gamma site toward ``γ`` through their natural
parameters — equivalently, the sent message is the product of powered messages
``\mu^{(t)} \propto (\mu^{(t-1)})^{1-\alpha} (\mu^\star)^{\alpha}`` when ``β = 0``.
"""

# ╔═╡ Cell order:
# ╟─8bf2d1e2-8403-47f7-b21a-cc793969197c
# ╠═44ae68ef-3429-4942-aaa5-a4decaab2cd5
# ╠═053f43e5-dc03-46d4-a50b-8bed3748e130
# ╠═c10f10ea-13b9-439c-8762-72cf3179be5b
# ╠═48eaeab6-f543-42ad-be90-253faaf07760
# ╠═b1d9d358-4807-49f6-96a3-82e9676108bf
# ╟─19fe300a-1294-455b-aa21-32058c80b2dc
# ╠═1080a4cd-28f0-4585-9831-c4385bfef530
# ╠═ab3a0d0d-9944-4810-b4b2-df882cd2ec45
# ╠═670dcdc6-75ce-490a-910e-7c52c688bf42
# ╟─a0c3ccfe-dba0-4704-8434-4c47aaa6eb12
# ╠═33c46a0a-3035-4aef-be05-2176dd3eb561
# ╠═64779645-a909-4725-a73b-76f909e90adf
# ╠═d3ef2a41-05fe-4220-a9b2-aabdeab76b16
# ╠═163c4f31-3764-49c1-ba3d-51a813f564c2
# ╠═83497909-fbe2-4041-97c2-8469ea660210
# ╟─d0f0204e-5f3b-414f-80a0-4317afb48913
# ╠═20a3a5ee-8d27-4be0-b39f-bb4c53a29fcb
# ╠═2f784cf1-1b43-494c-8d69-6de4871110ca
# ╠═2d9a5277-deb4-471d-a286-86325b90549e
# ╠═54be4609-e549-44e5-a873-880e87633ff6
# ╠═be739f8e-3195-4c53-8578-faa9bb89525e
# ╟─53aaa6cb-febb-4a0f-962a-3e0bb69e6ed0
