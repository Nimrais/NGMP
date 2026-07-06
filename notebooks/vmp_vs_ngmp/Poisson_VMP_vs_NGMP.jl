### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 9a56c2e0-4b5e-11f1-2b3c-6b9a6e90c111
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 206512e7-b5d2-45f8-b07a-3db6173bb01c
begin
    using RxInfer
    using ExponentialFamily
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    using SurrogateModelling
end

# ╔═╡ 98f5dfe2-d64b-4798-91ee-d2c83d33e078
begin
    using Plots
    using Statistics
end

# ╔═╡ e2b5458b-77e8-4d9c-9ffc-55df3ccace05
md"""
# VMP vs NGMP for a Poisson state-space model

This is the two-method version of `poisson_surrogate.jl`.

We use the same latent log-rate model for monthly sunspot counts:

```math
z_1 \sim \mathcal N(m_0, v_0), \qquad
z_k \sim \mathcal N(z_{k-1}, \sigma^2), \qquad
y_k \sim \mathrm{Poisson}(e^{z_k}).
```

The only difference is how the non-conjugate `PoissonExp` leaf is handled:

1. **VMP** uses RxInfer's mean-field projection route. The Poisson message is an
   exact `PoissonExpression`, then projected to `NormalMeanVariance` with
   `ClosedFormStrategy`.
2. **NGMP** runs on the original graph with `NGMPDependencies`. Each Poisson leaf
   sends the damped natural-gradient Gaussian message directly to the current
   log-rate marginal.
"""

# ╔═╡ 04d7c902-cc68-4062-8f8f-735942974c0e
begin
    sunspot_dataset = Sunspots()
    years = sunspot_dataset.features[!, :year]
    counts = round.(Int, sunspot_dataset.targets.average);
end

# ╔═╡ c7d24383-f0f8-4455-afb8-5f138332641f
begin
    sigma = 0.1
    m0 = 0.0
    v0 = 10.0
    iterations = 10
    damping_alpha = 0.5
    damping_beta = 0.2
end

# ╔═╡ ce7dc844-8ce4-4fdb-a355-24493f808db6
@model function poisson_vmp_ssm(y, sigma, m0, v0)
    z[1] ~ Normal(mean = m0, variance = v0)
    y[1] ~ PoissonExp(z[1])
    for k in 2:length(y)
        z[k] ~ Normal(mean = z[k - 1], variance = sigma)
        y[k] ~ PoissonExp(z[k])
    end
end

# ╔═╡ 99c5cb79-0a98-44c2-a178-dba7fe79ed21
begin
    @constraints function poisson_vmp_constraints()
        q(z) = MeanField()
        q(z) :: ProjectedTo(
            NormalMeanVariance,
            parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
        )
    end

    @initialization function poisson_vmp_init()
        q(z) = NormalMeanVariance(0.0, 100.0)
    end
end

# ╔═╡ c0530816-a335-4308-ba2f-f7809201325f
vmp_result = infer(
    model = poisson_vmp_ssm(sigma = sigma, m0 = m0, v0 = v0),
    constraints = poisson_vmp_constraints(),
    data = (y = counts,),
    initialization = poisson_vmp_init(),
    iterations = iterations,
    free_energy = true,
    options = (limit_stack_depth = 500,),
)

# ╔═╡ b19e0a7e-20b5-457f-ac40-fd7e51f2c9c0
md"""
## Native NGMP

The `dependencies = NGMPDependencies(in = nothing)` clause changes only the
message dispatched toward the `:in` interface of `PoissonExp`. It adds the
current marginal `q_in` as a dependency and selects the `NaturalGradientMessage`
rule from `src/nodes/poisson/rules/natural_gradient.jl`.
"""

# ╔═╡ acb7c5cb-cd31-4a25-b924-571d1dbbc1f8
@model function poisson_ngmp_ssm(y, sigma, m0, v0, deps, damping)
    z[1] ~ Normal(mean = m0, variance = v0)
    y[1] ~ PoissonExp(z[1]) where { dependencies = deps, meta = damping }
    for k in 2:length(y)
        z[k] ~ Normal(mean = z[k - 1], variance = sigma)
        y[k] ~ PoissonExp(z[k]) where { dependencies = deps, meta = damping }
    end
end

# ╔═╡ 33d8da7f-4c2e-420a-bb33-975d6e2eeffc
@initialization function poisson_ngmp_init(y)
    q(z) = NormalMeanVariance.(log.(coalesce.(y, 1) .+ 1.0), 1.0)
end

# ╔═╡ 2771939f-8552-4afd-87f9-932d958c4776
begin
    ngmp_deps = NGMPDependencies(in = nothing)
    ngmp_result = infer(
        model = poisson_ngmp_ssm(
            sigma = sigma,
            m0 = m0,
            v0 = v0,
            deps = ngmp_deps,
            damping = DampingMeta(alpha = damping_alpha, beta = damping_beta),
        ),
        data = (y = counts,),
        initialization = poisson_ngmp_init(counts),
        iterations = iterations,
        free_energy = true,
        options = (limit_stack_depth = 500,),
    )
end

# ╔═╡ 45851040-0d3e-4098-bbbe-af1447c3e280
begin
    vmp_post = last(vmp_result.posteriors[:z])
    ngmp_post = last(ngmp_result.posteriors[:z])

    vmp_mean = mean.(vmp_post)
    vmp_var = var.(vmp_post)
    ngmp_mean = mean.(ngmp_post)
    ngmp_var = var.(ngmp_post)
end

# ╔═╡ 9e4266b3-622f-46aa-9b46-7112d9562487
begin
    plot(
        years,
        vmp_mean;
        ribbon = 1.96 .* sqrt.(vmp_var),
        fillalpha = 0.18,
        lw = 2,
        color = :darkorange,
        label = "VMP mean-field",
        xlabel = "year",
        ylabel = "log-rate",
        title = "Posterior log-rate with 95% bands",
    )
    plot!(
        years,
        ngmp_mean;
        ribbon = 1.96 .* sqrt.(ngmp_var),
        fillalpha = 0.18,
        lw = 2,
        color = :dodgerblue,
        label = "NGMP native",
    )
    scatter!(
        years,
        log.(counts .+ 1.0);
        markersize = 1,
        markerstrokewidth = 0,
        alpha = 0.25,
        color = :black,
        label = "log(y+1)",
    )
end

# ╔═╡ 23076117-b364-46de-81af-a821d41e29af
md"""
The posterior means track the same solar-cycle signal. The visible difference is
the uncertainty: VMP factorizes `q(z)` across time, while NGMP keeps the Gaussian
chain exact after replacing each Poisson leaf by its natural-gradient message.
"""

# ╔═╡ 096e345d-8934-4d15-8bb4-a5499ba7b546
begin
    figure = 
    p1 = plot(
        1:iterations,
        vmp_result.free_energy;
        lw = 2,
        marker = :circle,
        color = :darkorange,
        label = "VMP variational FE",
        xlabel = "iteration",
        ylabel = "free energy",
        title = "Free-energy traces",
    )
    plot!(
        1:iterations,
        ngmp_result.free_energy;
        lw = 2,
        marker = :circle,
        color = :dodgerblue,
        label = "NGMP FE",
    )
    p2 = plot(
        1:iterations,
        ngmp_result.free_energy;
        lw = 2,
        marker = :circle,
        color = :dodgerblue,
        label = "NGMP FE",
    )
    plot(p1, p2, layout=(1,2), legend=false)
end

# ╔═╡ Cell order:
# ╟─e2b5458b-77e8-4d9c-9ffc-55df3ccace05
# ╠═9a56c2e0-4b5e-11f1-2b3c-6b9a6e90c111
# ╠═206512e7-b5d2-45f8-b07a-3db6173bb01c
# ╠═98f5dfe2-d64b-4798-91ee-d2c83d33e078
# ╠═04d7c902-cc68-4062-8f8f-735942974c0e
# ╠═c7d24383-f0f8-4455-afb8-5f138332641f
# ╠═ce7dc844-8ce4-4fdb-a355-24493f808db6
# ╠═99c5cb79-0a98-44c2-a178-dba7fe79ed21
# ╠═c0530816-a335-4308-ba2f-f7809201325f
# ╟─b19e0a7e-20b5-457f-ac40-fd7e51f2c9c0
# ╠═acb7c5cb-cd31-4a25-b924-571d1dbbc1f8
# ╠═33d8da7f-4c2e-420a-bb33-975d6e2eeffc
# ╠═2771939f-8552-4afd-87f9-932d958c4776
# ╠═45851040-0d3e-4098-bbbe-af1447c3e280
# ╠═9e4266b3-622f-46aa-9b46-7112d9562487
# ╟─23076117-b364-46de-81af-a821d41e29af
# ╠═096e345d-8934-4d15-8bb4-a5499ba7b546
