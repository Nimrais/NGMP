using SurrogateModelling
using RxInfer
using StableRNGs
using Plots

using LinearAlgebra, StatsPlots, LaTeXStrings, DataFrames, CSV


function generate_data(a, b, v, nr_samples; rng=StableRNG(1234))
    x = float.(collect(1:nr_samples))
    y = a .* x .+ b .+ randn(rng, nr_samples) .* sqrt(v)
    return x, y
end;

@model function linear_regression(x, y)
    a ~ Normal(mean = 0.0, variance = 1.0)
    b ~ Normal(mean = 0.0, variance = 100.0)    
    for i in 1:length(y)
        scaled_a[i] := a * x[i]
        intercept[i] := b * 1.0
        sum_res[i] ~ ManyPlus(inputs = [scaled_a[i], intercept[i]])
        y[i] ~ Normal(mean = sum_res[i], variance = 1.0)
    end
end

x_data, y_data = generate_data(0.5, 25.0, 1.0, 250)

# begin
#     scatter(x_data, y_data, title = "Dataset (City road)", legend=false)
#     xlabel!("Speed")
#     ylabel!("Fuel consumption")
# end

results = infer(
    model          = linear_regression(), 
    data           = (y = y_data, x = x_data), 
    initialization = @initialization(μ(b) = NormalMeanVariance(0.0, 100.0)), 
    returnvars     = (a = KeepLast(), b = KeepLast()),
    iterations     = 20,
    free_energy    = true
)

@show convert(NormalMeanVariance, results.posteriors[:a])
@show convert(NormalMeanVariance, results.posteriors[:b])

function generate_data_2coefs(a, b, c, v, nr_samples; rng=StableRNG(1234))
    x1 = float.(rand(rng, nr_samples))
    x2 = float.(rand(rng, nr_samples))
    y = a .* x1 .+ b .* x2 .+ c .+ randn(rng, nr_samples) .* sqrt(v)
    return x1, x2, y
end;

x1_data, x2_data, y_data = generate_data_2coefs(0.5, -1.0, 3.0, 1.0, 100)

@model function linear_regression_2coefs(x1, x2, y)
    a ~ Normal(mean = 0.0, variance = 1e12)
    b ~ Normal(mean = 0.0, variance = 1e12)
    c ~ Normal(mean = 0.0, variance = 1e12)
    for i in 1:length(y)
        scaled_a[i] := a * x1[i]
        scaled_b[i] := b * x2[i]
        intercept[i] := c * 1.0
        sum_res[i] ~ ManyPlus(inputs = [scaled_a[i], scaled_b[i], intercept[i]])
        y[i] ~ Normal(mean = sum_res[i], variance = 1.0)
    end
end

results = infer(
    model          = linear_regression_2coefs(), 
    data           = (y = y_data, x1 = x1_data, x2 = x2_data),
    initialization = (@initialization begin
      μ(b) = NormalMeanVariance(0.0, 1e12)
      μ(c) = NormalMeanVariance(0.0, 1e12)
    end),
    returnvars     = (a = KeepLast(), b = KeepLast(), c = KeepLast()),
    iterations     = 20,
    free_energy    = true,
    options = (limit_stack_depth = 100,)
)

@show convert(NormalMeanVariance, results.posteriors[:a])
@show convert(NormalMeanVariance, results.posteriors[:b])
@show convert(NormalMeanVariance, results.posteriors[:c])

begin
    prb = plot(range(-40, 40, length = 1000), (x) -> pdf(NormalMeanVariance(0.0, 100.0), x), title=L"Prior for $c$ parameter", fillalpha=0.3, fillrange = 0, label=L"p(c)", c=1, legend = :topleft)
    prb = vline!(prb, [ 3 ], label=L"True $c$", c = 3)
    psb = plot(range(2, 4, length = 1000), (x) -> pdf(results.posteriors[:c], x), title=L"Posterior for $c$ parameter", fillalpha=0.3, fillrange = 0, label=L"p(c\mid y)", c=2, legend = :topleft)
    psb = vline!(psb, [ 3 ], label=L"True $c$", c = 3)
    plot(prb, psb, size = (1000, 200), xlabel=L"$c$", ylabel=L"$p(c)$", ylims=[0, Inf])
end