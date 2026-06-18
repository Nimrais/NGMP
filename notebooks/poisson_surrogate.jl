### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 7cd03948-6b0f-11f1-315c-87a678136b72
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 65ea2958-e50a-4942-a884-f76622160004
begin
	using RxInfer
	using ExponentialFamilyProjection
	using ClosedFormExpectations
	using SurrogateModelling
end

# ╔═╡ aa8a3f00-8551-485a-86a2-cde2973be07d
using Plots

# ╔═╡ 587a9f91-1091-4e11-8daa-6b1b8f49d435
sunspot_dataset = Sunspots()

# ╔═╡ c3b8e6a2-44d9-4f17-8aa5-1d9e7c0f2b41
# Monthly mean sunspot numbers as non-negative integer counts — the y_k that
# the Poisson leaf explains.
counts = round.(Int, sunspot_dataset.targets.average)

# ╔═╡ 9d2e1f70-5c83-4a0e-b6f4-8e72a1c5d930
md"""
## A Poisson state-space model with `PoissonExp`

An ordinary RxInfer model: a Gaussian random-walk on the latent **log-rate**
`z_k`, observed through the `PoissonExp` leaf from the package,

```math
z_1 \sim \mathcal N(m_0, v_0), \qquad
z_k \sim \mathcal N(z_{k-1}, \sigma^2), \qquad
y_k \sim \mathrm{PoissonExp}(z_k) \equiv \mathrm{Poisson}(e^{z_k}).
```
"""

# ╔═╡ 6f0a9b34-2d71-4e58-9c12-7b3f5a8e6d04
@model function poisson_state_space(y, σ, m0, v0)
	z[1] ~ Normal(mean = m0, variance = v0)
	y[1] ~ PoissonExp(z[1])
	for k in 2:length(y)
		z[k] ~ Normal(mean = z[k-1], variance = σ)
		y[k] ~ PoissonExp(z[k])
	end
end

# ╔═╡ 2a7c8e15-3b69-4d02-8f51-9e4a6c1b7f38
# Condition the model on the observed counts. This declares the full factor
# graph (Gaussian chain + PoissonExp leaves); inference on the non-conjugate
# PoissonExp message is the surrogate-projection step that comes next.
poisson_model = poisson_state_space(σ = 0.1, m0 = 0.0, v0 = 10.0) | (y = counts,)

# ╔═╡ a4d91587-22c6-4ee7-ae8d-2d20c95d3223
begin
	@constraints function mean_field_form_constraints()
		q(z) = MeanField()
		q(z) :: ProjectedTo(
						NormalMeanVariance, 
						parameters = ProjectionParameters(
							strategy = ClosedFormStrategy(),
						)
				)
	end
	
	@initialization function mean_field_init()
		q(z) = Normal(mean = 0, var = 100)
	end
	
	result = infer(
		model = poisson_state_space(σ = 0.1, m0 = 0.0, v0 = 10.0),
		constraints = mean_field_form_constraints(),
		data = ( y = counts,),
		initialization=mean_field_init(),
		options=(limit_stack_depth=100,),
		iterations = 10,
	)
end 

# ╔═╡ f5c554bc-7def-4580-8199-d6d71bd3a580
plot(
	sunspot_dataset.features[!, :year], 
	 mean.(result.posteriors[:z][end])
)

# ╔═╡ Cell order:
# ╠═7cd03948-6b0f-11f1-315c-87a678136b72
# ╠═65ea2958-e50a-4942-a884-f76622160004
# ╠═587a9f91-1091-4e11-8daa-6b1b8f49d435
# ╠═c3b8e6a2-44d9-4f17-8aa5-1d9e7c0f2b41
# ╟─9d2e1f70-5c83-4a0e-b6f4-8e72a1c5d930
# ╠═6f0a9b34-2d71-4e58-9c12-7b3f5a8e6d04
# ╠═2a7c8e15-3b69-4d02-8f51-9e4a6c1b7f38
# ╠═a4d91587-22c6-4ee7-ae8d-2d20c95d3223
# ╠═aa8a3f00-8551-485a-86a2-cde2973be07d
# ╠═f5c554bc-7def-4580-8199-d6d71bd3a580
