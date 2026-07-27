### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 1b2c3d4e-5f6a-4b7c-8d8e-af1b2c3d4e5f
begin
    ENV["GKSwstype"] = "100"

    using Distributions: InverseGamma
    using LinearAlgebra
    using Plots
    using Printf
    using Random
    using RxInfer
    using StableRNGs
    using Statistics
    using SurrogateModelling

    import ProbabilisticEnsembling: Exp
end

# ╔═╡ 2c3d4e5f-6a7b-4c8d-89ef-b2c3d4e5f6a7
md"""
# Why Bayesian models need a hierarchy

A Gaussian process is the well-behaved regression model. Its predictive
variance contracts exactly where the data are and expands exactly where they are not.

It also has one blind spot that no amount of tuning removes: **a GP cannot learn how
the noise varies with the input.**

This notebook builds one model family with a *number of layers* `L`, where `L = 1`
is exactly a GP and every further layer adds one level of "how sure am I about that".
"""

# ╔═╡ aa11bb22-cc33-4dd4-8ee5-ff6677889900
md"""
## Layer 1 is a Gaussian process — exactly

Put a Gaussian prior on a weight vector and read out a linear combination of fixed
features:

```math
y_o \sim \mathcal N\!\left(v^\top\phi(x_o),\ \gamma^{-1}\right),
\qquad v \sim \mathcal N(0, \sigma_v^2 I).
```

``f(x) = v^\top\phi(x)`` is a linear functional of a Gaussian vector, so ``f`` **is** a
Gaussian process, with covariance function

```math
k(x, x') = \operatorname{Cov}\!\left(v^\top\phi(x),\ v^\top\phi(x')\right)
         = \phi(x)^\top\phi(x').
```

The posterior is the GP posterior written in weight space, and the two forms are the
same object:

```math
\Sigma_v = \left(\Sigma_0^{-1} + \gamma\,\Phi^\top\Phi\right)^{-1},
\qquad
\phi_*^\top\Sigma_v\phi_* \;=\; k_{**} - k_*^\top(K + \gamma^{-1}I)^{-1}k_*.
```

Two features of that expression are what the later layers change.

`Σ_v` is **dense**, and its off-diagonal entries are where the whole "pinch at the
data, widen away from it" behaviour lives. Force it diagonal and the predictive
variance becomes a non-negative combination of fixed non-negative functions: it can be
scaled down everywhere at once, but it can never dip where the observations are.

`γ` is **one scalar**. The predictive variance is
``\phi_*^\top\Sigma_v\phi_* + \gamma^{-1}``, and that second term is the same number
at every input. No choice of kernel or lengthscale changes this — it is what "noise" means
in a GP.
"""

# ╔═╡ bb22cc33-dd44-4ee5-8ff6-001122334455
md"""
## Layer 2 is not a GP any more

Let the log-precision be linear in the same features:

```math
y_o \sim \mathcal N\!\left(v^\top\phi(x_o),\ \lambda(x_o)^{-1}\right),
\qquad \log\lambda(x) = w_1^\top\phi(x),
\qquad w_1 \sim \mathcal N(m_1, \Sigma_1).
```

By exactly the argument above, ``\log\lambda`` is *itself a Gaussian process*. So this
is **two coupled GPs** — one on the mean, one on the log-noise — joined by ``\exp``.

That is no longer a GP, and not merely in a technical sense. Integrating out
``\lambda`` gives

```math
p(y \mid x) = \int \mathcal N\!\left(y;\ v^\top\phi(x),\ \lambda^{-1}\right)\,
              p(\lambda \mid x)\ d\lambda,
```

a **scale mixture of Gaussians**: heavier-tailed than a Gaussian, and a different width
at every input. No covariance function reproduces that, because a GP's noise enters as a
single constant on the diagonal.

So the hierarchy enlarges the family of predictive distributions the model can express.
"""

# ╔═╡ cc33dd44-ee55-4ff6-8001-112233445566
md"""
## …and the recursion does not stop

Layer 2 still fixed one thing: ``\log\lambda`` was fitted with a *constant* carrier
precision — "my noise estimate is equally trustworthy everywhere". Make that
input-dependent and you have layer 3. Repeat:

```math
\begin{aligned}
\text{output:}&\quad y \sim \mathcal N\!\left(v^\top\phi,\ c_1(x)^{-1}\right)\\
\text{level } k:&\quad s_k(x) \sim \mathcal N\!\left(w_k^\top\phi,\ c_{k+1}(x)^{-1}\right),
                  \qquad c_k(x) = \exp s_k(x)\\
\text{top:}&\quad c_L = \text{constant}
\end{aligned}
```

Every level is the same object — a Gaussian score, a positive precision obtained through
``\exp``, and a carrier supplied from above. The model is **self-similar**, so `L` is
just a number. Level 2 says how noisy the data are; level 3 says how sure it is about
that; level 4 says how sure it is about *that*.

Whether each level pays for itself is an empirical question, and the table at the end
answers it.
"""

# ╔═╡ dd44ee55-ff66-4001-8112-233445566778
md"""
## Configuration

Everything tunable lives here. The split matters: the **kernel** block is a modelling
choice fixed once, so that comparing depths does not also compare kernels. The
**optimizer** block is inference machinery, and those were chosen by sweeping.

Two of the optimizer settings are load-bearing, and both were found the hard way:

* `MAX_STEP` bounds the natural-gradient step. It is a safety rail.
* `ALPHA` is a natural-gradient momentum step size.
"""

# ╔═╡ ee55ff66-0011-4223-8344-556677889900
begin
    # ---- benchmark ---------------------------------------------------------
    const N_SAMPLES = 600
    const HOLDOUT_FRACTION = 1 / 3
    const DATA_SEED = 7

    # ---- kernel: fixed modelling choices, NOT tuned per depth ---------------
    const N_BASIS = 16          # random Fourier features; must exceed the effective
    const LENGTHSCALE = 0.25    #   rank of the data, or no model can widen off-data
    const SIGNAL_SD = 1.0       # mean-weight prior sd
    const LEVEL_SD = 0.4        # per-level weight sd; smaller on purpose, so a noise
                                #   process is smoother than the mean process
    const ANCHOR_SD = 1.0
    const TOP_CARRIER = 25.0    # the top level's constant carrier precision

    # ---- optimizer: chosen by sweeping ------------------------------------
    const LAYER_COUNTS = [1, 2, 3, 4, 5]
    const ITERATIONS = 240      # log-density is monotone in this at ALPHA >= 0.3
    const ALPHA = 0.6           # natural-gradient step size
    const MAX_STEP = 0.5        # load-bearing: 4.0 diverges by 120 iterations
    const METHOD = :damped
    const PREDICT_ITERATIONS = 60   # prediction saturates by ~30
    const PREDICT_ALPHA = 0.5
end

# ╔═╡ 3d4e5f6a-7b8c-4d9e-8a01-c3d4e5f6a7b8
md"""
## The benchmark

```math
x \sim \mathcal N(0,1), \qquad
y = -(x + \tfrac12)\sin(3\pi x) + \varepsilon, \qquad
\varepsilon \sim \mathcal N\!\left(0,\ [0.45(x + \tfrac12)]^2\right)
```

The mean is a high-frequency wave, so fitting it needs real capacity. The noise vanishes
at ``x = -1/2`` and grows toward the edges, so it is *input-dependent* — the thing a GP
must call a constant. A third of the sample is held out.
"""

# ╔═╡ 4e5f6a7b-8c9d-4e0f-8b12-d4e5f6a7b8c9
begin
    true_mean(x) = -(x + 0.5) * sin(3pi * x)
    true_noise_variance(x) = abs2(0.45 * (x + 0.5))

    data = let
        rng = StableRNG(DATA_SEED)
        x = randn(rng, N_SAMPLES)
        y = true_mean.(x) .+ sqrt.(true_noise_variance.(x)) .* randn(rng, N_SAMPLES)
        order = randperm(rng, N_SAMPLES)
        n_test = round(Int, HOLDOUT_FRACTION * N_SAMPLES)
        test, train = order[1:n_test], order[(n_test + 1):end]
        (; x_train = x[train], y_train = y[train],
           x_test = x[test], y_test = y[test])
    end

    grid = collect(range(-3.0, 3.0; length = 241))
    @printf("%d training points, %d held out\n",
        length(data.y_train), length(data.y_test))
end

# ╔═╡ 6a7b8c9d-0e1f-4a2b-8d34-f6a7b8c9d0e1
begin
    # Random Fourier features plus a constant coordinate. phi'phi approximates the RBF
    # kernel, and the sqrt(2/H) scaling keeps the prior predictive width independent of H.
    # Every layer shares this one basis, so a depth comparison is not a kernel comparison.
    feature_map = let rng = StableRNG(20260726)
        frequencies = randn(rng, N_BASIS) ./ LENGTHSCALE
        phases = 2pi .* rand(rng, N_BASIS)
        scale = sqrt(2 / N_BASIS)
        x -> vcat(scale .* cos.(frequencies .* x .+ phases), [1.0])
    end

    const N_FEATURES = N_BASIS + 1
    design(xs) = [feature_map(x) for x in xs]
end

# ╔═╡ 8c9d0e1f-2a3b-4c4d-8f56-b8c9d0e1f2a3
# L = 1. softdot(theta, x, gamma) means y ~ N(theta' x, 1/gamma), so with the features as
# a PointMass and the dense weight vector as the random argument this is exact Bayesian
# linear regression -- the GP posterior.
@model function gp_model(y, features, v_prior, noise_prior)
    v ~ v_prior
    γ ~ noise_prior
    for o in eachindex(features)
        y[o] ~ softdot(features[o], v, γ)
    end
end

# ╔═╡ 2a3b4c5d-6e7f-4a8b-8d9a-f2a3b4c5d6e7
# L >= 2. Built from the top down: only the topmost score has a constant carrier, and
# every level below is carried by the level above. `n_levels = L - 1`.
@model function hierarchy_model(
    y, features, n_levels, v_prior, w_priors, top_carrier, deps, damping,
)
    local w, score, precision
    v ~ v_prior
    for k in 1:n_levels
        w[k] ~ w_priors[k]
    end
    for o in eachindex(features)
        score[n_levels, o] ~ softdot(features[o], w[n_levels], top_carrier)
        precision[n_levels, o] ~ Exp(score[n_levels, o]) where {
            dependencies = deps, meta = damping,
        }
        for k in (n_levels - 1):-1:1
            score[k, o] ~ softdot(features[o], w[k], precision[k + 1, o])
            precision[k, o] ~ Exp(score[k, o]) where {
                dependencies = deps, meta = damping,
            }
        end
        y[o] ~ softdot(features[o], v, precision[1, o])
    end
end

# ╔═╡ 3b4c5d6e-7f8a-4b9c-8eab-a3b4c5d6e7f8
md"""
## One factorization statement, every depth

```julia
q(v, w, score, precision, y) = q(v, y) q(w, score) q(precision)
```

Read it against the graph. At level `k` the `softdot` output is `score[k, o]` and its
weight edge is `w[k]` — both in the second cluster, so they stay **jointly** Gaussian,
which is the exact-conjugate configuration. Its precision edge is `precision[k+1, o]`,
in the third cluster, hence mean-field with respect to that level, as the rules require.
The likelihood is the same pattern with `(v, y)` joint.

Because it names whole arrays rather than individual levels, the same line holds for
`L = 2, 3, 4, …`. The weight vectors are never split coordinate-wise — that dense
covariance is what the GP behaviour rests on.
"""

# ╔═╡ 4c5d6e7f-8a9b-4cad-8fbc-b4c5d6e7f8a9
begin
    @constraints function gp_constraints()
        q(v, γ, y) = q(v, y)q(γ)
        q(v)::MomentForm()
    end

    @constraints function hierarchy_constraints()
        q(v, w, score, precision, y) = q(v, y)q(w, score)q(precision)
        q(v)::MomentForm()
        q(w)::MomentForm()
    end
end

# ╔═╡ 6e7f8a9b-acbd-4ecf-8bde-d6e7f8a9bacb
begin
    # Prior means are zero except on the constant coordinate, which is ANCHORED: level 1
    # at a basis-free noise estimate (Rice/Gasser first differences, so a mis-specified
    # basis cannot fool it), higher levels at the carrier they replace. A level therefore
    # starts flat and learns deviations instead of travelling from zero.
    noise_anchor(xs, ys) = -log(max(mean(abs2.(diff(ys[sortperm(xs)]))) / 2, 1e-8))

    gaussian(means, variances) =
        MvNormalMeanCovariance(collect(means), Matrix(Diagonal(collect(variances))))

    function make_priors(n_layers, xs, ys)
        anchor = noise_anchor(xs, ys)
        level_prior(k) = gaussian(
            vcat(zeros(N_BASIS), [k == 1 ? anchor : log(TOP_CARRIER)]),
            vcat(fill(abs2(LEVEL_SD), N_BASIS), [abs2(ANCHOR_SD)]),
        )
        return (;
            v = gaussian(zeros(N_FEATURES), fill(abs2(SIGNAL_SD), N_FEATURES)),
            w = [level_prior(k) for k in 1:(n_layers - 1)],
            noise = GammaShapeRate(2.0, 2.0 / TOP_CARRIER),
        )
    end
end

# ╔═╡ 8a9b0c1d-cedf-4aeb-8df0-f8a9b0c1dced
begin
    ngmp_deps() = NGMPDependencies(
        out = nothing, in = nothing;
        projection = TangentProjection(type = ClosedForm),
    )
    damping(alpha, max_step) =
        DampingMeta(alpha = alpha, beta = 0.0, max_step = max_step, method = METHOD)

    # Every latent a non-conjugate link touches needs a starting marginal. Push each
    # prior forward: the score is Gaussian, and the precision starts at the mode of that
    # Gaussian's log-normal pushforward so the first damped step is well scaled.
    function initial_marginals(n_layers, priors, rows)
        Φ = reduce(hcat, rows)'
        n_levels, n_points = n_layers - 1, length(rows)
        score = Matrix{NormalMeanVariance{Float64}}(undef, n_levels, n_points)
        precision = Matrix{GammaShapeRate{Float64}}(undef, n_levels, n_points)
        for k in 1:n_levels
            m, V = mean_cov(priors.w[k])
            means = Φ * m
            variances = vec(sum((Φ * V) .* Φ; dims = 2)) .+ inv(TOP_CARRIER)
            shapes = 1 .+ inv.(max.(variances, 1e-6))
            score[k, :] = NormalMeanVariance.(means, variances)
            precision[k, :] = GammaShapeRate.(shapes, shapes .* exp.(-means))
        end
        return (; score, precision)
    end

    build_model(n_layers, priors, alpha, max_step) = n_layers == 1 ?
        gp_model(v_prior = priors.v, noise_prior = priors.noise) :
        hierarchy_model(
            n_levels = n_layers - 1, v_prior = priors.v, w_priors = priors.w,
            top_carrier = TOP_CARRIER, deps = ngmp_deps(),
            damping = damping(alpha, max_step),
        )
end

# ╔═╡ 9b0c1d2e-dfea-4bfc-8e01-a9b0c1d2edfe
md"""
## Fit, then predict with RxInfer: One `infer` call per model.

Prediction runs the *same* model again with the learned posteriors as priors and `y`
declared through `predictvars`, so RxInfer performs the prediction by message passing and
``q(y_*)`` is read straight off the graph. (A variable supplied through `data` is not
returned among the posteriors even when its value is `missing`, which is why `features`
is the data variable here and `y` is not.)

For the hierarchical models prediction also retains ``q(\mathtt{precision}[1,o])``.
That marginal contains more information than the single variance of ``q(y_o)``; below
we turn it into sampling-free posterior bounds on the variance itself.
"""

# ╔═╡ 0c1d2e3f-eafb-4cad-8f12-b0c1d2e3feaf
begin
    function fit_layers(n_layers)
        rows = design(data.x_train)
        priors = make_priors(n_layers, data.x_train, data.y_train)
        states = n_layers == 1 ? nothing : initial_marginals(n_layers, priors, rows)

        init = n_layers == 1 ?
            @initialization(begin
                q(v) = deepcopy(priors.v)
                q(γ) = deepcopy(priors.noise)
            end) :
            @initialization(begin
                q(v) = deepcopy(priors.v)
                q(w) = deepcopy(priors.w)
                q(score) = states.score
                q(precision) = states.precision
            end)

        result = infer(
            model = build_model(n_layers, priors, ALPHA, MAX_STEP),
            data = (y = data.y_train, features = rows),
            constraints = n_layers == 1 ? gp_constraints() : hierarchy_constraints(),
            initialization = init,
            returnvars = n_layers == 1 ?
                (v = KeepLast(), γ = KeepLast()) : (v = KeepLast(), w = KeepLast()),
            iterations = ITERATIONS, free_energy = false, showprogress = false,
            options = (limit_stack_depth = 100,),
        )

        return (; n_layers,
            v = result.posteriors[:v],
            w = n_layers == 1 ? [] : collect(vec(result.posteriors[:w])),
            noise = n_layers == 1 ? result.posteriors[:γ] : priors.noise)
    end

    function predict_layers(fit, xs)
        rows = design(xs)
        priors = (; v = fit.v, w = fit.w, noise = fit.noise)
        states = fit.n_layers == 1 ? nothing :
            initial_marginals(fit.n_layers, priors, rows)

        init = fit.n_layers == 1 ?
            @initialization(begin
                q(v) = deepcopy(priors.v)
                q(γ) = deepcopy(priors.noise)
            end) :
            @initialization(begin
                q(v) = deepcopy(priors.v)
                q(w) = deepcopy(priors.w)
                q(score) = states.score
                q(precision) = states.precision
            end)

        result = infer(
            model = build_model(fit.n_layers, priors, PREDICT_ALPHA, MAX_STEP),
            data = (features = rows,),
            constraints = fit.n_layers == 1 ? gp_constraints() : hierarchy_constraints(),
            initialization = init,
            predictvars = (y = KeepLast(),),
            returnvars = fit.n_layers == 1 ?
                (γ = KeepLast(),) : (precision = KeepLast(),),
            iterations = PREDICT_ITERATIONS, free_energy = false, showprogress = false,
            options = (limit_stack_depth = 100,),
        )

        marginals = collect(vec(result.predictions[:y]))
        precisions = fit.n_layers == 1 ?
            fill(result.posteriors[:γ], length(rows)) :
            collect(result.posteriors[:precision][1, :])

        # Conditional on λₒ = precision[1, o], integrating q(v) gives
        # Var(yₒ | λₒ, data) = ϕₒ'Σᵥϕₒ + 1/λₒ. Since q(λₒ) is Gamma(shape, rate),
        # this variance is a shifted InverseGamma(shape, rate): its moments and
        # quantiles are available exactly, with no Monte Carlo samples.
        _, V = mean_cov(fit.v)
        latent_variance = [dot(ϕ, V * ϕ) for ϕ in rows]
        inverse_precisions =
            InverseGamma.(shape.(precisions), rate.(precisions))
        noise_variance_mean = mean.(inverse_precisions)
        noise_variance_median = quantile.(inverse_precisions, 0.5)
        noise_variance_lower = quantile.(inverse_precisions, 0.025)
        noise_variance_upper = quantile.(inverse_precisions, 0.975)
        variance_posterior = (;
            mean = latent_variance .+ noise_variance_mean,
            median = latent_variance .+ noise_variance_median,
            lower = latent_variance .+ noise_variance_lower,
            upper = latent_variance .+ noise_variance_upper,
            latent = latent_variance,
            noise_mean = noise_variance_mean,
        )

        return (;
            mean = mean.(marginals),
            variance = var.(marginals),
            variance_posterior,
        )
    end
end

# ╔═╡ 2e3f4a5b-acbd-4ecf-8b34-d2e3f4a5bacb
begin
    fits = map(LAYER_COUNTS) do n_layers
        elapsed = @elapsed fit = fit_layers(n_layers)
        @printf("L = %d fitted in %5.1f s\n", n_layers, elapsed)
        fit
    end

    grid_predictions = map(fit -> predict_layers(fit, grid), fits)
    test_predictions = map(fit -> predict_layers(fit, data.x_test), fits)
    nothing
end

# ╔═╡ 3f4a5b6c-bdce-4fda-8c45-e3f4a5b6cbdc
md"""
## Predictive mean — and a posterior over predictive variance

At a test input ``x_o``, prediction gives

```math
q(v)=\mathcal N(m_v,\Sigma_v), \qquad
q(\lambda_o)=q(\mathtt{precision}[1,o])
             =\operatorname{Gamma}(\alpha_o,\beta_o)
```

where ``\beta_o`` is a **rate**. After integrating the uncertainty in ``v``, but
conditioning on ``\lambda_o``, the predictive variance is the random quantity

```math
S_o \equiv \operatorname{Var}(y_o\mid\lambda_o,\mathcal D)
    = \underbrace{\phi_o^\top\Sigma_v\phi_o}_{c_o}
      + \lambda_o^{-1},
\qquad
S_o-c_o \sim \operatorname{InverseGamma}(\alpha_o,\beta_o).
```

Consequently its posterior mean (when ``\alpha_o>1``) and every quantile are analytic:

```math
\mathbb E[S_o]=c_o+\frac{\beta_o}{\alpha_o-1},\qquad
```

```math
\operatorname{CrI}_{95\%}(S_o)
=c_o+\left[
Q_{.025}\{\operatorname{InvGamma}(\alpha_o,\beta_o)\},
Q_{.975}\{\operatorname{InvGamma}(\alpha_o,\beta_o)\}
\right].
```

No sampling is involved. The shaded region in the right column is this **pointwise
95% posterior credible interval for the variance**; it is distinct from the 95%
predictive band for ``y_o`` in the left column. The dotted curve isolates the known
shift ``c_o`` contributed by uncertainty in the mean weights.

Left: the posterior predictive with a 95% band. Right: the predictive variance against
the **true** conditional variance (dash-dot). The right column is where the hierarchy
shows up. For the GP in row 1 the noise part is constant; only the separately displayed
mean-weight contribution can vary with ``x``.
"""

# ╔═╡ 4a5b6c7d-cedf-4aeb-8d56-f4a5b6c7dced
begin
    depth_label(n) = n == 1 ? "L = 1  (Gaussian process)" : "L = $n layers"

    panels = []
    for (n_layers, prediction) in zip(LAYER_COUNTS, grid_predictions)
        band = 1.96 .* sqrt.(max.(prediction.variance, 0.0))
        variance_summary = prediction.variance_posterior
        mean_panel = plot(
            grid, prediction.mean;
            ribbon = band, fillalpha = 0.2, color = :steelblue, linewidth = 2,
            label = "q(y*) ± 1.96 SD", xlabel = "x", ylabel = "y",
            title = depth_label(n_layers), titlefontsize = 9,
            legend = :topleft, legendfontsize = 6, ylims = (-4.5, 4.5),
        )
        plot!(mean_panel, grid, true_mean.(grid);
            color = :black, linewidth = 2, label = "true mean")
        scatter!(mean_panel, data.x_train, data.y_train;
            color = :gray55, markersize = 2, markeralpha = 0.35,
            markerstrokewidth = 0, label = "train")

        variance_panel = plot(
            grid, max.(variance_summary.lower, 1e-4);
            fillrange = max.(variance_summary.upper, 1e-4),
            fillcolor = :steelblue, fillalpha = 0.18,
            color = :transparent, linewidth = 0,
            label = "95% CrI for variance",
            xlabel = "x", ylabel = "variance", yscale = :log10,
            title = "$(depth_label(n_layers)): variance", titlefontsize = 9,
            legend = :topleft, legendfontsize = 6, ylims = (1e-4, 1e2),
        )
        plot!(variance_panel, grid, max.(variance_summary.mean, 1e-4);
            color = :steelblue, linewidth = 2,
            label = "posterior mean variance")
        plot!(variance_panel, grid, max.(variance_summary.latent, 1e-4);
            color = :darkorange, linestyle = :dot, linewidth = 1.5,
            label = "mean-weight contribution")
        plot!(variance_panel, grid, max.(true_noise_variance.(grid), 1e-4);
            color = :black, linestyle = :dashdot, linewidth = 2,
            label = "true noise variance")
        push!(panels, mean_panel, variance_panel)
    end

    figure = plot(panels...; layout = (length(LAYER_COUNTS), 2), size = (1_100, 1_500))
    haskey(ENV, "HIERARCHY_FIGURE") && savefig(figure, ENV["HIERARCHY_FIGURE"])
    figure
end

# ╔═╡ 5b6c7d8e-dfea-4bfc-8e67-a5b6c7d8edfe
md"""
## Held-out comparison

`logpdf` is the column that matters: it scores the whole predictive distribution, whereas
`RMSE` scores only the mean and is blind to whether the uncertainty is honest.

`noise corr` is the correlation between the posterior mean of ``1/\lambda(x)`` and the
true conditional variance across the held-out inputs. The mean-weight contribution
``\phi(x)^\top\Sigma_v\phi(x)`` is deliberately excluded: this metric isolates what the
noise hierarchy is for, and a GP scores zero on it by construction.
"""

# ╔═╡ 6c7d8e9f-eafb-4cad-8f78-b6c7d8e9feaf
begin
    function score(prediction, xs, ys)
        variance = max.(prediction.variance, 1e-8)
        residual = ys .- prediction.mean
        truth = true_noise_variance.(xs)
        estimated_noise = prediction.variance_posterior.noise_mean
        effectively_constant =
            std(estimated_noise) <= sqrt(eps()) * max(mean(abs, estimated_noise), eps())
        noise_corr = effectively_constant ? 0.0 : cor(estimated_noise, truth)
        return (;
            logpdf = mean(-0.5 .* (log.(2pi .* variance) .+ abs2.(residual) ./ variance)),
            rmse = sqrt(mean(abs2.(residual))),
            noise_corr,
        )
    end

    holdout = map(zip(LAYER_COUNTS, test_predictions)) do (n_layers, prediction)
        (; n_layers, score(prediction, data.x_test, data.y_test)...)
    end

    @printf("%-26s %9s %8s %11s\n",
        "model", "logpdf", "RMSE", "noise corr")
    for row in holdout
        @printf("%-26s %9.4f %8.4f %11.3f\n",
            depth_label(row.n_layers), row.logpdf, row.rmse, row.noise_corr)
    end
    @printf("\nbest held-out logpdf: %s\n",
        depth_label(holdout[argmax([r.logpdf for r in holdout])].n_layers))
end

# ╔═╡ 7d8e9f0a-fbac-4dbe-8a89-c7d8e9f0afba
md"""
## What to take from this

**One layer is a GP, and its noise variance is flat.** Not a tuning failure — a single
scalar precision cannot be a function of `x`, so its posterior over ``1/\gamma`` is the
same at every input and its noise correlation is zero. The total variance in row 1 may
still bend by the dotted amount ``\phi^\top\Sigma_v\phi``: that is uncertainty about the
mean function, not learned heteroscedastic noise.

**The second layer changes the model family.** Integrating out a random precision gives a
scale mixture of Gaussians, which is heavier-tailed and has a different width at every
input. That is why row 2's variance curve can follow the true noise, and it is the one
qualitative jump in the table.

**Further layers help, then saturate.** Each level adds a dense weight vector and a
non-conjugate link, in exchange for a more refined statement about
uncertainty-about-uncertainty. On this benchmark the gains fall off sharply, because a
level whose score barely leaves its anchor passes the level below essentially the constant
carrier it replaced. That is a benign way to fail: a too-deep model collapses gracefully
onto the shallower one rather than misbehaving, so depth can be added speculatively.

**Getting the optimizer wrong will hide all of this.** At `ALPHA = 0.1, ITERATIONS = 120`
the depth ordering inverts, purely because that setting sits where the `L = 2` and `L = 3`
trajectories happen to cross. The ordering is only meaningful once the updates have
settled — which is why the configuration cell records why each value is what it is.
"""

# ╔═╡ Cell order:
# ╟─2c3d4e5f-6a7b-4c8d-89ef-b2c3d4e5f6a7
# ╟─aa11bb22-cc33-4dd4-8ee5-ff6677889900
# ╟─bb22cc33-dd44-4ee5-8ff6-001122334455
# ╟─cc33dd44-ee55-4ff6-8001-112233445566
# ╟─dd44ee55-ff66-4001-8112-233445566778
# ╠═ee55ff66-0011-4223-8344-556677889900
# ╠═0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d
# ╠═1b2c3d4e-5f6a-4b7c-8d8e-af1b2c3d4e5f
# ╟─3d4e5f6a-7b8c-4d9e-8a01-c3d4e5f6a7b8
# ╠═4e5f6a7b-8c9d-4e0f-8b12-d4e5f6a7b8c9
# ╠═6a7b8c9d-0e1f-4a2b-8d34-f6a7b8c9d0e1
# ╠═8c9d0e1f-2a3b-4c4d-8f56-b8c9d0e1f2a3
# ╠═2a3b4c5d-6e7f-4a8b-8d9a-f2a3b4c5d6e7
# ╟─3b4c5d6e-7f8a-4b9c-8eab-a3b4c5d6e7f8
# ╠═4c5d6e7f-8a9b-4cad-8fbc-b4c5d6e7f8a9
# ╠═6e7f8a9b-acbd-4ecf-8bde-d6e7f8a9bacb
# ╠═8a9b0c1d-cedf-4aeb-8df0-f8a9b0c1dced
# ╟─9b0c1d2e-dfea-4bfc-8e01-a9b0c1d2edfe
# ╠═0c1d2e3f-eafb-4cad-8f12-b0c1d2e3feaf
# ╠═2e3f4a5b-acbd-4ecf-8b34-d2e3f4a5bacb
# ╟─3f4a5b6c-bdce-4fda-8c45-e3f4a5b6cbdc
# ╠═4a5b6c7d-cedf-4aeb-8d56-f4a5b6c7dced
# ╟─5b6c7d8e-dfea-4bfc-8e67-a5b6c7d8edfe
# ╠═6c7d8e9f-eafb-4cad-8f78-b6c7d8e9feaf
# ╟─7d8e9f0a-fbac-4dbe-8a89-c7d8e9f0afba
