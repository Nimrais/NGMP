module SurrogateModelling

using DataDeps
import CSV
import DataFrames
using Random
using StableRNGs
using Statistics

# RxInfer custom factor nodes. We pull the modelling macros / distributions
# from RxInfer (which re-exports ReactiveMP, ExponentialFamily and BayesBase)
# so the nodes live with the rest of the package rather than in a notebook.
using RxInfer
using Distributions
using SpecialFunctions: loggamma
import BayesBase
import RxInfer: @node, @rule, @call_rule, @average_energy, @marginalrule, @call_marginalrule

# Closed-form projection support. Loading `ClosedFormExpectations` alongside
# `ExponentialFamilyProjection` activates the latter'g `ClosedFormStrategy`
# extension; the `PoissonExpression` hooks for it live in `nodes/poisson/expression.jl`.
import ExponentialFamilyProjection
import ClosedFormExpectations
import ClosedFormExpectations: LogGamma
import ExponentialFamily: GaussianDistributionsFamily

# ProbabilisticEnsembling supplies the `Log` node (z = log γ) of the dynamic
# ensemble model; the natural-gradient rules for it live in nodes/log/.
import ProbabilisticEnsembling

include("NaturalGradientMP/NaturalGradientMP.jl")
using .NaturalGradientMP: NaturalGradientMessage, NGMPDependencies, DampingMeta, NGMPEdgeState, ClosedFormDefault, getprojection, PrecisionTempering
export NaturalGradientMP, NaturalGradientMessage, NGMPDependencies, DampingMeta, NGMPEdgeState, getprojection, PrecisionTempering

include("ManyPlusNode/ManyPlusNode.jl")
using .ManyPlusNode: ManyPlus
export ManyPlusNode, ManyPlus

# `MvStack` is the lossless counterpart of `ManyPlus`: it gathers scalar neurons
# into a vector edge instead of pre-summing them, so a dense output-weight
# posterior can act on the hidden layer. See src/MvStackNode/node.jl.
include("MvStackNode/MvStackNode.jl")
using .MvStackNode: MvStack
export MvStackNode, MvStack

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
include("datasets/uci_regression.jl")
include("datasets/uci_benchmark.jl")
export Sunspots, ETTh1
export UCIRegressionDataset, Yacht, Concrete, EnergyEfficiency, BostonHousing, PowerPlant, WineQualityRed
export UCISplitSpec,
    UCI_SPLIT_PROTOCOL_VERSION,
    UCI_DEFAULT_N_SPLITS,
    UCI_DEFAULT_SPLIT_SEED,
    UCI_DEFAULT_TEST_FRACTION,
    UCI_DEFAULT_VALIDATION_FRACTION,
    uci_regression_split,
    uci_regression_splits,
    fit_uci_standardizer,
    transform_uci_features,
    transform_uci_targets,
    prepare_uci_regression_partition

include("nodes/poisson/poisson_exp.jl")
include("nodes/poisson/expression.jl")
include("nodes/poisson/rules/marginal.jl")
include("nodes/poisson/rules/natural_gradient.jl")

include("expressions/normal_precision_message.jl")

include("expressions/mv_normal_precision_message.jl")

include("expressions/product_messages.jl")

include("expressions/student_t_message.jl")

include("expressions/gaussian_student_t_message.jl")

include("expressions/normal_log_precision_message.jl")

include("expressions/gaussian_lognormal_scale_message.jl")

include("features/gauss_hermite_fourier.jl")

include("distributions/mv_inverse_softplus_normal.jl")

include("tangent_projections/common.jl")

include("tangent_projections/closed_form_tangent.jl")

include("tangent_projections/gamma.jl")

include("tangent_projections/normal.jl")

include("tangent_projections/unscented.jl")

include("tangent_projections/mv_normal.jl")

include("tangent_projections/mv_inverse_softplus_normal.jl")

include("nodes/softplus/softplus.jl")
include("nodes/squareplus/squareplus.jl")
include("nodes/mv_softplus/node.jl")
include("nodes/mv_softplus/rules/natural_gradient.jl")
include("nodes/mv_softplus/rules/marginal.jl")
include("nodes/mv_residual_sine/node.jl")
include("nodes/mv_residual_sine/rules/natural_gradient.jl")
include("nodes/mv_residual_sine/rules/marginal.jl")
include("nodes/residual_sine/node.jl")
include("nodes/residual_sine/rules/natural_gradient.jl")
include("nodes/residual_sine/rules/marginal.jl")
include("nodes/ContinuousTransition/linear_low_rank_meta.jl")
include("nodes/ContinuousTransition/linear_reshape_meta.jl")
include("nodes/ContinuousTransition/pointmass_transition.jl")
include("nodes/exp/rules/natural_gradient.jl")
include("nodes/log/rules/natural_gradient.jl")
include("nodes/probit/categorical_message.jl")
include("nodes/normal_mean_precision/joint_belief.jl")
include("nodes/normal_mean_precision/rules/natural_gradient.jl")
include("nodes/softdot/rules/structured_info_form.jl")
include("nodes/softdot/rules/vmp_with_dumping.jl")
include("nodes/softdot/rules/relaxed_structure.jl")
include("nodes/softdot/rules/natural_gradient.jl")
include("nodes/mv_normal_exp_precision/node.jl")
include("nodes/mv_normal_exp_precision/rules/vmp.jl")
include("nodes/mv_normal_exp_precision/rules/natural_gradient.jl")
include("nodes/mv_normal_exp_precision/rules/univariate.jl")
include("nodes/mv_normal_exp_precision/rules/structured.jl")
include("nodes/mv_normal_exp_precision/rules/tempering.jl")
include("nodes/mv_normal_exp_precision/rules/cavity.jl")

include("moment_form.jl")

include("kl_divergences.jl")

function __init__()
    __init__sunspots()
    __init__etth1()
    __init__uci_regression()
end

end # module
