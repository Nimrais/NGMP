module NaturalGradientMP

using ReactiveMP, Rocket, TupleTools
using Distributions: Distribution, Gamma, shape, rate
using BayesBase: weightedmean, precision
using SpecialFunctions: trigamma
using ExponentialFamily:
    ExponentialFamilyDistribution,
    getnaturalparameters,
    exponential_family_typetag,
    NormalMeanVariance,
    NormalWeightedMeanPrecision,
    GammaShapeRate,
    UnivariateNormalDistributionsFamily,
    GammaDistributionsFamily

export NaturalGradientMessage, NGMPDependencies, DampingMeta, NGMPEdgeState, ClosedFormDefault, getprojection

include("constraint.jl")
include("damping.jl")
include("dependencies.jl")

end
