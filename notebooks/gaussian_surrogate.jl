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
3. **NG-BP (by hand, your tangent projections).** Keep the exact non-conjugate
   messages and **project** each onto its receiving edge with the
   `SurrogateModelling` tangent projections: the Student-t onto the Gaussian $x$
   edge (`project_to_normal`), the precision message onto the Gamma $\tau$ edge
   (`project_to_gamma`).

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
    N  = 100
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
	plot(1:niters, vmp.free_energy)
end

# ╔═╡ 957d8bce-1bd0-4812-9000-b0b39a8bf7fd
begin
    qx_vmp = vmp.posteriors[:x][end]
    qτ_vmp = vmp.posteriors[:τ][end]
    xg = range(exact.mx - 5sqrt(exact.vx), exact.mx + 5sqrt(exact.vx); length = 400)
        
    p1 = plot(xg, 
              exact.pdfx.(xg); 
              lw = 3, color = :black, label = "exact",
              xlabel = "x", ylabel = "density", 
              title = "Posterior of the mean")
    plot!(p1, xg, pdf.(Normal(mean(qx_vmp), std(qx_vmp)), xg);
          lw = 2, ls = :dash, color = :crimson, label = "mean-field VMP")
    mask = exact.pdfτ.w .> 1e-8 * maximum(exact.pdfτ.w)
    p2 = plot(
              exact.pdfτ.τs[mask],
              exact.pdfτ.w[mask],
              lw = 3,
              color = :black,
              title = "Posterior of the precision",
    )
    plot!(p2, exact.pdfτ.τs[mask],
          pdf.(qτ_vmp, exact.pdfτ.τs[mask]);
          lw = 2, ls = :dash, color = :crimson, label = "mean-field VMP")
    plt = plot(p1, p2; layout = (1,2), size = (1100, 800))
end

# ╔═╡ 039006f5-482d-4382-a1dd-58c16d093438


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
# ╠═957d8bce-1bd0-4812-9000-b0b39a8bf7fd
# ╠═039006f5-482d-4382-a1dd-58c16d093438
