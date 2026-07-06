module NaturalGradientMP

using ReactiveMP, Rocket, TupleTools
using Distributions: Distribution, Gamma, shape, rate
using BayesBase: weightedmean, precision
using ExponentialFamily:
    ExponentialFamilyDistribution,
    getnaturalparameters,
    exponential_family_typetag,
    NormalMeanVariance,
    NormalWeightedMeanPrecision,
    GammaShapeRate,
    UnivariateNormalDistributionsFamily,
    GammaDistributionsFamily

export NaturalGradientMessage, NGMPDependencies, DampingMeta, NGMPEdgeState

include("constraint.jl")
include("damping.jl")
include("dependencies.jl")

end
