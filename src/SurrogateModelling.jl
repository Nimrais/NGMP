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
import RxInfer: @node, @rule, @average_energy, @marginalrule

# Closed-form projection support. Loading `ClosedFormExpectations` alongside
# `ExponentialFamilyProjection` activates the latter's `ClosedFormStrategy`
# extension; the `PoissonExpression` hooks for it live in `nodes/poisson/expression.jl`.
import ExponentialFamilyProjection
import ClosedFormExpectations
import ClosedFormExpectations: LogGamma
import ExponentialFamily: GaussianDistributionsFamily

# ProbabilisticEnsembling supplies the `Log` node (z = log γ) of the dynamic
# ensemble model; the natural-gradient rules for it live in nodes/log/.
import ProbabilisticEnsembling

include("NaturalGradientMP/NaturalGradientMP.jl")
using .NaturalGradientMP: NaturalGradientMessage, NGMPDependencies, DampingMeta, NGMPEdgeState, ClosedFormDefault, getprojection
export NaturalGradientMP, NaturalGradientMessage, NGMPDependencies, DampingMeta, NGMPEdgeState, getprojection

# NOTE: the strategy struct `UnscentedTransforms.UnscentedTransform` is deliberately
# NOT re-exported — ReactiveMP already exports `Unscented`/`UnscentedTransform` for
# its Delta-node approximations and re-exporting ours would make the name ambiguous.
# Users select the strategy through ReactiveMP's familiar type instead:
# `TangentProjection(type = Unscented)` (our GH(3) defaults α=1, β=0, κ=2) or
# `TangentProjection(type = Unscented(alpha = ..., beta = ..., kappa = ...))`
# (bridged verbatim in tangent_projections/unscented.jl).
include("UnscentedTransforms/UnscentedTransforms.jl")
using .UnscentedTransforms: UnscentedTransform
export UnscentedTransforms

include("datasets/sunspots.jl")
include("datasets/etth1.jl")
export Sunspots, ETTh1

include("nodes/poisson/poisson_exp.jl")
include("nodes/poisson/expression.jl")
include("nodes/poisson/rules/marginal.jl")
include("nodes/poisson/rules/natural_gradient.jl")

include("expressions/normal_precision_message.jl")

include("expressions/student_t_message.jl")

include("tangent_projections/common.jl")

include("tangent_projections/closed_form_tangent.jl")

include("tangent_projections/gamma.jl")

include("tangent_projections/normal.jl")

include("tangent_projections/unscented.jl")

include("nodes/log/rules/natural_gradient.jl")
include("nodes/normal_mean_precision/rules/natural_gradient.jl")
include("nodes/softdot/rules/structured_info_form.jl")

include("moment_form.jl")

function __init__()
    __init__sunspots()
    __init__etth1()
end

end # module
