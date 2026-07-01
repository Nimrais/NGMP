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
include("datasets/etth1.jl")
export Sunspots, ETTh1

include("nodes/poisson_exp.jl")
export PoissonExp, PoissonExpression

include("expressions/normal_precision_message.jl")
export NormalPrecisionMessage

include("expressions/student_t_message.jl")
export StudentTMessage

include("tangent_projections/common.jl")

include("tangent_projections/gamma.jl")

include("tangent_projections/normal.jl")

# ---------------------------------------------------------------------------
# model_zoo — NGMP surrogate models that plug into ProbabilisticEnsembling's YAML
# `run_experiment` pipeline by extending its model-type hooks. (Template stage:
# inner model + projections are real; the outer-loop driver in pipeline.jl is a
# stub to be ported from the repo-root dynamic_exp_ngmp_surrogate.jl.)
# ---------------------------------------------------------------------------
import ProbabilisticEnsembling
include("model_zoo/dynamic_exp_ngmp/model_type.jl")


function __init__()
    __init__sunspots()
    __init__etth1()
end

end # module
