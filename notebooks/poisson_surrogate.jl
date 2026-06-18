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
		free_energy=true
	)
end 

# ╔═╡ f5c554bc-7def-4580-8199-d6d71bd3a580
plot(
	 sunspot_dataset.features[!, :year], 
	 mean.(result.posteriors[:z][end])
)

# ╔═╡ ffdf6c8e-6cf8-497c-a4f5-d285b110be9e
plot(1:length(result.free_energy), result.free_energy)

# ╔═╡ cb0918a8-8f86-45c1-aa1f-ddb26848a97e
md"""
Here we need to explain how surrogate inference works
"""

# ╔═╡ 2d3634c4-1b1b-42ab-bba9-0e0731f38af5
@model function lg_chain(y, R, σ, m0, v0)
    x[1] ~ Normal(mean = m0, variance = v0)
    y[1] ~ Normal(mean = x[1], variance = R[1])
    for k in 2:length(y)
        x[k] ~ Normal(mean = x[k-1], variance = σ)
        y[k] ~ Normal(mean = x[k],   variance = R[k])
    end
end

# ╔═╡ 0b00d09a-0422-4ac4-af57-1c4d7e278d8f
function poisson_surrogate(y::Real, m::Real, v::Real)
    ∂m, ∂σ = mean(
        ClosedWilliamsProduct(), 
        Logpdf(PoissonExpression(y)), 
        Normal(m, sqrt(v))
    )
    Λ = -∂σ / sqrt(v)
    ξ = ∂m + m * Λ
    return ξ, Λ
end

# ╔═╡ 11bded3d-294b-47c7-a29e-3a89b0c8bb76
function ngmp_poisson_smoother(y; σ = 0.1, m0 = 0.0, v0 = 10.0,
                               iters = 20, α = 0.5, β = 0.2)
    N = length(y)
    m = log.(y .+ 1.0); v = fill(1.0, N)          # initial edge marginals
    ξ = zeros(N); Λ = zeros(N)                     # surrogate message state η
    vξ = zeros(N); vΛ = zeros(N)                   # momentum buffers
    surrogate_free_energy = Float64[]              # Bethe FE of each frozen surrogate
    update_norm = Float64[]                        # fixed-point residual ‖Φ(λ)-λ‖∞
    local inference                                 # last inner BP result, kept below
    for _ in 1:iters
        s = poisson_surrogate.(y, m, v)            # project every leaf at q_k
        ηξ = first.(s); ηΛ = last.(s)
        @. vξ = β * vξ + α * (ηξ - ξ); @. ξ += vξ
        @. vΛ = β * vΛ + α * (ηΛ - Λ); @. Λ += vΛ
        inference = infer(
                     model = lg_chain(σ = σ, m0 = m0, v0 = v0),
                     data = (y = ξ ./ Λ, R = 1.0 ./ Λ),
                     options = (limit_stack_depth = 100,),
                     free_energy = true,
        )
        post = inference.posteriors[:x]
        mnew = mean.(post); vnew = var.(post)
        push!(surrogate_free_energy, inference.free_energy[end])
        push!(update_norm, max(maximum(abs.(mnew .- m)), maximum(abs.(vnew .- v))))
        m, v = mnew, vnew
    end
    return (inference = inference, means = m, variances = v,
            surrogate_free_energy = surrogate_free_energy, update_norm = update_norm)
end

# ╔═╡ af2b5341-66fe-4ff5-ad4b-bb7c4ab9b309
ngmp_result = ngmp_poisson_smoother(counts)

# ╔═╡ b7150c92-698e-4410-9fff-62b5652071ff
begin
	plot(
		 sunspot_dataset.features[!, :year], 
		 ngmp_result[:means]
	)
	plot!(
		 sunspot_dataset.features[!, :year], 
		 mean.(result.posteriors[:z][end])
	)
end

# ╔═╡ e2a7b3c1-5d64-4f08-9b12-7c3e8a1f6d29
# Surrogate Bethe free energy. NOTE: each outer iteration freezes a *different*
# Gaussian surrogate, so this is a diagnostic, not a single objective being
# minimized — it need not decrease monotonically (see §Damping and Momentum).
# However it converges anyway :ballon:
plot(1:length(ngmp_result.surrogate_free_energy), ngmp_result.surrogate_free_energy,
     xlabel = "outer iteration", ylabel = "surrogate Bethe free energy",
     legend = false)

# ╔═╡ Cell order:
# ╠═7cd03948-6b0f-11f1-315c-87a678136b72
# ╠═65ea2958-e50a-4942-a884-f76622160004
# ╠═587a9f91-1091-4e11-8daa-6b1b8f49d435
# ╠═c3b8e6a2-44d9-4f17-8aa5-1d9e7c0f2b41
# ╠═9d2e1f70-5c83-4a0e-b6f4-8e72a1c5d930
# ╠═6f0a9b34-2d71-4e58-9c12-7b3f5a8e6d04
# ╠═a4d91587-22c6-4ee7-ae8d-2d20c95d3223
# ╠═aa8a3f00-8551-485a-86a2-cde2973be07d
# ╠═f5c554bc-7def-4580-8199-d6d71bd3a580
# ╠═ffdf6c8e-6cf8-497c-a4f5-d285b110be9e
# ╠═cb0918a8-8f86-45c1-aa1f-ddb26848a97e
# ╠═2d3634c4-1b1b-42ab-bba9-0e0731f38af5
# ╠═0b00d09a-0422-4ac4-af57-1c4d7e278d8f
# ╠═11bded3d-294b-47c7-a29e-3a89b0c8bb76
# ╠═af2b5341-66fe-4ff5-ad4b-bb7c4ab9b309
# ╠═b7150c92-698e-4410-9fff-62b5652071ff
# ╠═e2a7b3c1-5d64-4f08-9b12-7c3e8a1f6d29
