using RxInfer
using StableRNGs

@model function normal_node_with_noise_learning(y, priors)
    x ~ priors[:μ]
    τ ~ priors[:τ]
    for i in 1:length(y)
        y[i] ~ NormalMeanPrecision(x, τ)
    end
end

@initialization function marginal_init()
    q(x) = NormalMeanVariance(0, 1)
end

ys = rand(StableRNG(42), NormalMeanVariance(10, 10), 1000)

priors = (
    μ = NormalMeanVariance(0, 1),
    τ = GammaShapeRate(1, 1)
)

result = infer(
    data = (y = ys,),
    model = normal_node_with_noise_learning(priors = priors),
    constraints = MeanField(),
    iterations = 10,
    initialization = marginal_init()
)

@show convert(NormalMeanVariance, result.posteriors[:x][end])
@show mean(result.posteriors[:τ][end])