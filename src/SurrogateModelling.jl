module SurrogateModelling

using DataDeps
import CSV
import DataFrames

# RxInfer custom factor nodes. We pull the modelling macros / distributions
# from RxInfer (which re-exports ReactiveMP, ExponentialFamily and BayesBase)
# so the nodes live with the rest of the package rather than in a notebook.
using RxInfer
using Distributions
using SpecialFunctions: loggamma
import BayesBase
import RxInfer: @node, @rule, @average_energy

# Closed-form projection support. Loading `ClosedFormExpectations` alongside
# `ExponentialFamilyProjection` activates the latter's `ClosedFormStrategy`
# extension; the `PoissonExpression` hooks for it live in `nodes/poisson_exp.jl`.
import ExponentialFamilyProjection
import ClosedFormExpectations
import ClosedFormExpectations: LogGamma
import ExponentialFamily: GaussianDistributionsFamily

include("datasets/sunspots.jl")
export Sunspots

include("nodes/poisson_exp.jl")
export PoissonExp, PoissonExpression

include("expressions/normal_precision_message.jl")
export NormalPrecisionMessage

function __init__()
    __init__sunspots()
end

end # module
