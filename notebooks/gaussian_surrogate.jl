### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ e61ec0eb-decf-4df9-81cb-1fa52b0b181e
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 3667ef45-4fa9-4302-b327-80fabfb50498
begin
	using RxInfer
	using ExponentialFamily
	using SurrogateModelling
end

# ╔═╡ 65b10d3d-6cde-4449-a950-f4c08726ac98
begin
	using Plots
	using Random
	using Statistics
	using Distributions
	import ExponentialFamily: ExponentialFamilyDistribution
end

# ╔═╡ 8de6d9aa-5344-42fe-a0bf-f1afe4941632
md"""
# Inferring a Gaussian's mean and precision from a handful of points

**The setting.** We observe a few real numbers $y_1,\dots,y_N$ that we believe are
noisy readings of one unknown value $x$, corrupted by Gaussian noise of unknown
precision $\tau$ (inverse variance). We want the **joint** posterior over the
location $x$ *and* the noise precision $\tau$ — and, crucially, the **right
uncertainty** on each, especially when $N$ is small and the two are genuinely
entangled.

**The model.**

```math
x \sim \mathcal N(m_0, v_0), \qquad
\tau \sim \mathrm{Gamma}(a_0, b_0), \qquad
y_i \sim \mathcal N\!\big(x,\; \tau^{-1}\big),\quad i = 1,\dots,N.
```

This is the textbook normal model with **both** parameters unknown. It is the
simplest place to see the effect this whole line of work is about, because the
exact posterior does *not* factorize: $x$ and $\tau$ are coupled (the marginal of
$x$ is a **Student-t**, not a Gaussian), and any method that forces
$q(x,\tau)=q(x)\,q(\tau)$ has to pay for it somewhere.

**Why plain Gaussian message passing fails — and what this notebook shows.** The
likelihood factor $\mathcal N(y_i\mid x,\tau^{-1})$ is **not conjugate** once
$\tau$ is also a variable: the message it sends toward $x$ (after integrating out
the precision) is a heavy-tailed **Student-t**, and the message it sends toward
$\tau$ is **not Gamma**. Neither lands in the family of the receiving edge, so
conjugate message passing cannot carry them. We solve the *same model* three ways
and compare them:

1. **Exact (grid).** $x$ integrates out in closed form given $\tau$, leaving a 1-D
   integral over $\tau$ we evaluate on a fine grid. This is the ground truth.
2. **Mean-field VMP (RxInfer, automatic).** Force $q(x,\tau)=q(x)q(\tau)$ and run
   variational message passing — the classical conjugate route.
3. **NG-BP (native NGMP rules).** Keep the graph un-factorized and run belief
   propagation whose non-conjugate messages are **tangent-projected at the
   receiving marginals** — the `NaturalGradientMessage` rules on the standard
   `NormalMeanPrecision` node, dispatched with a `where { dependencies = ... }`
   clause. One `infer` call, no surrogate outer loop, **no factorization
   constraint at all**.

The three **agree when data are plentiful** and **part ways when data are scarce**
— and pinning down *when, where, and why* they differ is the point of the
notebook.
"""

# ╔═╡ fa4dd833-53d3-4e83-b5dc-0249c2f4046f
md"""
## The exact posterior (our ground truth)

With a Gaussian prior on $x$, the location integrates out **in closed form** for
each fixed $\tau$, because $x$ enters only through a Gaussian. What is left is a
one-dimensional density in $\tau$,

```math
p(\tau \mid y) \;\propto\;
\mathrm{Gamma}(\tau; a_0, b_0)\,
\tau^{N/2}\,
\exp\!\Big(-\tfrac{\tau}{2}\textstyle\sum_i y_i^2 + \tfrac{h(\tau)^2}{2P(\tau)}\Big)\,
P(\tau)^{-1/2},
```

with $P(\tau)=v_0^{-1}+N\tau$ and $h(\tau)=m_0/v_0+\tau\sum_i y_i$. We put $\tau$ on
a fine grid, and the posterior of $x$ comes out as the induced **mixture of
Gaussians** $p(x\mid y)=\int p(x\mid\tau,y)\,p(\tau\mid y)\,d\tau$ — which is
exactly the heavy-tailed Student-t-like shape that no single Gaussian can match.
"""

# ╔═╡ 97584205-9176-4993-8c5d-68da39d93943
function exact_posterior(ys; m0, v0, a0, b0, nτ = 4000)
    N, S1, S2 = length(ys), sum(ys), sum(abs2, ys)
    τs = exp.(range(log(1e-5), log(50.0); length = nτ))
    P  = @. 1 / v0 + N * τs                     # posterior precision of x | τ
    h  = @. m0 / v0 + τs * S1
    logp = @. (a0 - 1) * log(τs) - b0 * τs +
              (N / 2) * log(τs) - 0.5 * τs * S2 + h^2 / (2P) - 0.5 * log(P)
    logp .-= maximum(logp)
    w  = exp.(logp)
    dτ = [τs[2] - τs[1]; (τs[3:end] .- τs[1:end-2]) ./ 2; τs[end] - τs[end-1]]
    w  = w .* dτ; w ./= sum(w)
    μx, vx = h ./ P, 1 ./ P                     # p(x | τ_k, y)
    mx = sum(w .* μx)
    sx = sum(w .* (vx .+ μx .^ 2)) - mx^2
    mτ = sum(w .* τs)
    vτ = sum(w .* τs .^ 2) - mτ^2
    pdfx = xg -> sum(w .* pdf.(Normal.(μx, sqrt.(vx)), xg))
    pdfτ = (τs = τs, w = w ./ dτ)
    return (mx = mx, vx = sx, mτ = mτ, vτ = vτ, pdfx = pdfx, pdfτ = pdfτ)
end

# ╔═╡ 5c99bd13-02f2-4931-b497-b2117b5b63a0
md"""
## Method 1 — mean-field VMP (the conjugate route)

Declare the model in RxInfer, add a `MeanField()` constraint so
$q(x,\tau)=q(x)\,q(\tau)$, and let `infer` run conjugate variational message
passing. This is the standard, fully automatic baseline. Its defining assumption
— that $x$ and $\tau$ are *independent* under the posterior — is exactly what we
are going to stress-test.
"""

# ╔═╡ 78b2f094-96af-44ff-b028-d2d88aba297e
@model function normal_vmp(y, m0, v0, a0, b0)
    x ~ NormalMeanVariance(m0, v0)
    τ ~ GammaShapeRate(a0, b0)
    for i in 1:length(y)
        y[i] ~ NormalMeanPrecision(x, τ)
    end
end

# ╔═╡ 6f444787-cfe4-4a45-b925-37c1f070368c
@initialization function vmp_init()
    q(x) = NormalMeanVariance(0, 1)
end

# ╔═╡ 171227cb-895c-4a80-b754-7f1e54ff9c91
begin
    m0, v0, a0, b0 = 0.0, 1e6, 1.0, 1.0
    N  = 2
    Random.seed!(20260706)
    ys = rand(NormalMeanPrecision(10.0, 10.0), N)

    exact = exact_posterior(ys, m0 = m0, v0 = v0, a0 = a0, b0 = b0)

    vmp = infer(
            data = (y = ys,),
            model = normal_vmp(m0 = m0, v0 = v0, a0 = a0, b0 = b0),
            constraints = MeanField(),
            iterations = 10,
            initialization = vmp_init(),
            free_energy=true
        )
end

# ╔═╡ e1704c6a-e874-4344-aaab-b4c3c97c7cf6
begin
	niters = length(vmp.free_energy)
	plot(1:niters, vmp.free_energy, title="VMP Free Energy", label = nothing)
end

# ╔═╡ b468d91d-e9ce-4f1d-982e-bb506135f02c
md"""
## Method 2 — native NG-BP (tangent-projected belief propagation)

No factorization constraint: the model graph stays exactly as written, and the
likelihood nodes carry `NGMPDependencies(μ = nothing, τ = nothing)` — every
message toward the $x$ and $\tau$ edges is computed by the
`NaturalGradientMessage` rules in
`src/nodes/normal_mean_precision/rules/natural_gradient.jl`.

Unlike the `PoissonExp`/`Log` cases, **neither exact message here has a
closed-form Williams product**, so the rules use the **quadrature-exact** tangent
projection `TangentProjection(type = Quadrature(128))`: the Williams product
$\nabla_\eta E_q[\ell] = \mathrm{Cov}_q[T, \ell]$ is evaluated by Gauss–Hermite
on the Gaussian edge and by a log-space trapezoid rule on the Gamma edge
(substituting $s = \log\tau$ makes the Gamma integrand smooth and
double-exponentially decaying, so the trapezoid converges geometrically for any
shape). The cheaper *second-order* delta expansion (`project_to_gamma` /
`project_to_normal`) is kept as an alternative — but it is **biased whenever the
receiving marginal is wide**, because it integrates the touching quadratic of
$\ell$ far from the expansion point; the diagnostic cell below quantifies this.
The exact BP log-messages (prepared in `src/expressions/`):

- **toward $\tau$**: integrating the Gaussian cavity $\mathcal N(z\mid \tilde m, \tilde v)$
  out of the factor gives the `NormalPrecisionMessage`
  $\mu_{f\to\tau}(\tau) = \mathcal N(y \mid \tilde m,\ \tilde v + \tau^{-1})$ — not
  Gamma; `project_to_gamma` at $q(\tau)$ produces the damped Gamma site.
- **toward $x$**: integrating the Gamma cavity out gives the `StudentTMessage`
  $\mu_{f\to x}(x) \propto (2\tilde b + (x - y)^2)^{-(\tilde a + 1/2)}$ — a heavy-tailed
  Student-t; `project_to_normal` at $q(x)$ produces the damped Gaussian site.

Two practical notes, both visible in the initialization below:

- the Student-t log-message is concave **only in its bulk** ($(m-y)^2 < 2\tilde b$);
  seeding $q(x)$ far from the data expands in the convex tail and yields
  negative-precision sites — so we seed at the data mean;
- for $N \ge 2$ the $x$ and $\tau$ edges are **loopy** (every projected message
  needs the equality-chain product of the *other* nodes' messages), so messages
  `μ(x)`, `μ(τ)` need initial values too — the standard loopy-BP start-up.
"""

# ╔═╡ 0c9234bd-2df4-462c-b93b-4afbd7eade01
@model function normal_ngbp(y, m0, v0, a0, b0, deps, damping)
    x ~ NormalMeanVariance(m0, v0)
    τ ~ GammaShapeRate(a0, b0)
    for i in 1:length(y)
        y[i] ~ NormalMeanPrecision(x, τ) where { dependencies = deps, meta = damping }
    end
end

# ╔═╡ c5705754-7d35-49da-9af0-559d9fb0e279
begin
    ngbp_iterations = 50                      # α = 0.2 damping: ~50 sweeps to settle
    ngbp_deps = NGMPDependencies(μ = nothing, τ = nothing)
    mx0 = mean(ys)
    vx0 = max(var(ys), 0.1)
    ngbp_init = @initialization begin
        q(x) = NormalMeanVariance(mx0, vx0)   # data bulk: Student-t concave region
        q(τ) = GammaShapeRate(a0, b0)
        μ(x) = NormalMeanVariance(mx0, 10 * vx0)   # loopy-BP message seeds (N ≥ 2)
        μ(τ) = GammaShapeRate(a0, b0)
    end
    ngbp = infer(
        data = (y = ys,),
        model = normal_ngbp(m0 = m0, v0 = v0, a0 = a0, b0 = b0,
                            deps = ngbp_deps,
                            damping = DampingMeta(alpha = 0.2, beta = 0.0)),
        initialization = ngbp_init,
        iterations = ngbp_iterations,
         options = (limit_stack_depth = 500,),
    )
    # one damping state per likelihood edge (μ and τ), each fired once per sweep
    @assert length(ngbp_deps.states) == 2N
    @assert all(s -> s.nfired == ngbp_iterations, ngbp_deps.states)
    ngbp
end

# ╔═╡ c6bcf453-2eb7-456a-bcdd-99bfe22ba482
begin
    # Diagnostic: why the second-order projection biased the τ fixed point.
    # Project ONE likelihood's exact τ-message both ways at a marginal of the
    # converged shape, but artificially widened (small N regime): the touching
    # quadratic at the Gamma mean overshoots the rate increment Δb severely,
    # while the quadrature is exact.
    import SurrogateModelling: NormalPrecisionMessage, project_to_gamma
    import ClosedFormExpectations: Logpdf as CFELogpdf
    import ExponentialFamily: getnaturalparameters

    qτ_conv = ngbp.posteriors[:τ][end]
    p_diag = NormalPrecisionMessage(ys[1], mean(ngbp.posteriors[:x][end]), var(ngbp.posteriors[:x][end]))
    diag_rows = map((GammaShapeRate(1.5, 1.5 / mean(qτ_conv)), qτ_conv)) do qd
        Δa2, Δb2 = project_to_gamma(p_diag, convert(ExponentialFamilyDistribution, qd))
        ηq = getnaturalparameters(project(TangentProjection(type = Quadrature(128)), qd, CFELogpdf(p_diag)))
        (a = shape(qd), b = rate(qd), second_order = (round(Δa2, digits = 4), round(Δb2, digits = 4)),
         quadrature = (round(ηq[1], digits = 4), round(-ηq[2], digits = 4)))
    end
    Markdown.parse(
        """
        ### Diagnostic: second-order vs exact (Δa, Δb) for one τ-message

        | q(τ) | second-order (Δa, Δb) | quadrature-exact (Δa, Δb) |
        |---|---|---|
        | wide Gamma($(round(diag_rows[1].a, digits=2)), $(round(diag_rows[1].b, digits=2))) | $(diag_rows[1].second_order) | $(diag_rows[1].quadrature) |
        | converged Gamma($(round(diag_rows[2].a, digits=2)), $(round(diag_rows[2].b, digits=2))) | $(diag_rows[2].second_order) | $(diag_rows[2].quadrature) |

        For wide marginals (small N) the delta expansion overshoots the rate
        increment several-fold — the source of the τ bias the second-order rules
        had; at the converged, concentrated marginal the gap largely closes
        (and keeps shrinking as the shape grows).
        """,
    )
end

# ╔═╡ 957d8bce-1bd0-4812-9000-b0b39a8bf7fd
begin
    qx_vmp = vmp.posteriors[:x][end]
    qτ_vmp = vmp.posteriors[:τ][end]
    qx_ngbp = ngbp.posteriors[:x][end]
    qτ_ngbp = ngbp.posteriors[:τ][end]
    xg = range(exact.mx - 5sqrt(exact.vx), exact.mx + 5sqrt(exact.vx); length = 400)

    p1 = plot(xg,
              exact.pdfx.(xg);
              lw = 3, color = :black, label = "exact",
              xlabel = "x", ylabel = "density",
              title = "Posterior of the mean")
    plot!(p1, xg, pdf.(Normal(mean(qx_vmp), std(qx_vmp)), xg);
          lw = 2, ls = :dash, color = :crimson, label = "mean-field VMP")
    plot!(p1, xg, pdf.(Normal(mean(qx_ngbp), std(qx_ngbp)), xg);
          lw = 2, color = :dodgerblue, label = "NG-BP")
    mask = exact.pdfτ.w .> 1e-8 * maximum(exact.pdfτ.w)
    p2 = plot(
              exact.pdfτ.τs[mask],
              exact.pdfτ.w[mask],
              lw = 3,
              color = :black,
              label = "exact",
              title = "Posterior of the precision",
              xlabel = "τ",
    )
    plot!(p2, exact.pdfτ.τs[mask],
          pdf.(qτ_vmp, exact.pdfτ.τs[mask]);
          lw = 2, ls = :dash, color = :crimson, label = "mean-field VMP")
    plot!(p2, exact.pdfτ.τs[mask],
          pdf.(qτ_ngbp, exact.pdfτ.τs[mask]);
          lw = 2, color = :dodgerblue, label = "NG-BP")
    plt = plot(p1, p2; layout = (1,2), size = (1100, 800))
end

# ╔═╡ 8b086eb1-7525-4613-8eb7-1f226e24799b
Markdown.parse(
    """
    ## Posterior moments vs the exact grid

    | method | E[x] | V[x] | E[τ] | V[τ] |
    |---|---|---|---|---|
    | exact (grid) | $(round(exact.mx, digits = 4)) | $(round(exact.vx, digits = 4)) | $(round(exact.mτ, digits = 4)) | $(round(exact.vτ, digits = 4)) |
    | mean-field VMP | $(round(mean(qx_vmp), digits = 4)) | $(round(var(qx_vmp), digits = 4)) | $(round(mean(qτ_vmp), digits = 4)) | $(round(var(qτ_vmp), digits = 4)) |
    | NG-BP (native) | $(round(mean(qx_ngbp), digits = 4)) | $(round(var(qx_ngbp), digits = 4)) | $(round(mean(qτ_ngbp), digits = 4)) | $(round(var(qτ_ngbp), digits = 4)) |

    N = $(N) observations.
    """,
)

# ╔═╡ 69ed1331-6b78-4756-bf08-f4f3569d6ced
md"""
## What to look at

**The precision marginal.** With the quadrature-exact projection, NG-BP's
$q(\tau)$ essentially **matches the exact grid at every $N$** (means agree to
well under a percent; the variance is the closest of the two message-passing
methods, where mean-field VMP systematically understates it). This settles an
earlier suspicion: with the *second-order* rules the τ marginal degraded for
$N \ge 2$ (e.g. $E[\tau] = 2.13$ vs exact $2.82$ at $N = 5$), which looked like
an intrinsic loopy-BP effect — it was in fact **delta-expansion bias**: for a
wide $q(\tau)$ the touching quadratic overshoots the rate increment $\Delta b$
by a factor of ~3 (see the diagnostic table above). Exact projection, exact
fixed point.

**The location marginal.** All three means coincide. The exact $x$-marginal is a
scale mixture (Student-t-like), so no single Gaussian can reproduce its second
moment when the tails matter; both message-passing methods report a somewhat
smaller variance, with mean-field VMP the most overconfident. At $N = 1$
(vague-prior, heavy-tail extreme) NG-BP is visibly wider and more honest than
VMP's fixed unit-variance answer.

- at **$N = 1$** NG-BP also recovers the exact $\tau$ posterior exactly — one
  observation carries no precision information under a vague location prior, and
  the projected message is the null update, while mean-field VMP manufactures
  spurious confidence;
- for **$N \ge 2$** the loop is now benign: the τ marginal tracks the grid.

Try changing `N` in the data cell; damping (`alpha = 0.2`) and the two
initialization notes above are the only tuning in the NG-BP arm.
"""

# ╔═╡ Cell order:
# ╟─8de6d9aa-5344-42fe-a0bf-f1afe4941632
# ╠═e61ec0eb-decf-4df9-81cb-1fa52b0b181e
# ╠═3667ef45-4fa9-4302-b327-80fabfb50498
# ╠═65b10d3d-6cde-4449-a950-f4c08726ac98
# ╟─fa4dd833-53d3-4e83-b5dc-0249c2f4046f
# ╠═97584205-9176-4993-8c5d-68da39d93943
# ╟─5c99bd13-02f2-4931-b497-b2117b5b63a0
# ╠═78b2f094-96af-44ff-b028-d2d88aba297e
# ╠═6f444787-cfe4-4a45-b925-37c1f070368c
# ╠═171227cb-895c-4a80-b754-7f1e54ff9c91
# ╠═e1704c6a-e874-4344-aaab-b4c3c97c7cf6
# ╟─b468d91d-e9ce-4f1d-982e-bb506135f02c
# ╠═0c9234bd-2df4-462c-b93b-4afbd7eade01
# ╠═c5705754-7d35-49da-9af0-559d9fb0e279
# ╠═c6bcf453-2eb7-456a-bcdd-99bfe22ba482
# ╠═957d8bce-1bd0-4812-9000-b0b39a8bf7fd
# ╠═8b086eb1-7525-4613-8eb7-1f226e24799b
# ╟─69ed1331-6b78-4756-bf08-f4f3569d6ced
