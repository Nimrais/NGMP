using Test

using ClosedFormExpectations
using Distributions
using ExponentialFamily
using RxInfer
using ReactiveMP
using SurrogateModelling
using BayesBase
using Random
using SpecialFunctions
using Statistics

import ClosedFormExpectations: ClosedWilliamsProduct, Logpdf
import ExponentialFamily: ExponentialFamilyDistribution, NormalMeanVariance, getnaturalparameters
import ReactiveMP: @call_rule, Marginal

struct ConvexLogMessage <: ClosedFormExpectations.Expression end

function ClosedFormExpectations.mean(
    ::ClosedWilliamsProduct,
    ::Logpdf{<:ConvexLogMessage},
    q::Normal,
)
    return (zero(Distributions.mean(q)), Distributions.std(q))
end

# @testset "UCI hierarchy benchmark" begin
#     include("uci_hierarchy_deep_kernel_tests.jl")
# end

@testset "closed-form tangent projection" begin
    m = 0.4
    v = 0.7
    α = 2.0
    β = 3.0
    q = NormalMeanVariance(m, v)
    f = Logpdf(LogGamma(α, β; check_args = false))

    site = project(TangentProjection(type = ClosedForm), q, f)

    Λ = exp(m + v / 2) / α
    ξ = β + (m - 1) * Λ
    η = getnaturalparameters(site)

    @test site isa ExponentialFamilyDistribution{NormalMeanVariance}
    @test η[1] ≈ ξ
    @test η[2] ≈ -Λ / 2

    qef = convert(ExponentialFamilyDistribution, q)
    site_from_ef = project(TangentProjection(ClosedForm()), qef, f)
    @test getnaturalparameters(site_from_ef) ≈ η

    improper_site = project(TangentProjection(), q, Logpdf(ConvexLogMessage()))
    improper_η = getnaturalparameters(improper_site)

    @test improper_site isa ExponentialFamilyDistribution{NormalMeanVariance}
    @test improper_η[1] ≈ -m
    @test improper_η[2] ≈ 0.5
end

@testset "ManyPlusNode" begin
    include("manyplus_tests.jl")
end

@testset "NaturalGradientMP integration" begin
    include("ngmp/reference.jl")
    include("ngmp/damping_tests.jl")
    include("ngmp/integration_tests.jl")
    include("ngmp/log_rule_tests.jl")
    include("ngmp/probit_categorical_rule_tests.jl")
    include("ngmp/softplus_rule_tests.jl")
    include("ngmp/squareplus_rule_tests.jl")
    include("ngmp/mv_softplus_rule_tests.jl")
    include("ngmp/mv_residual_sine_rule_tests.jl")
    include("ngmp/residual_sine_rule_tests.jl")
    include("ngmp/mv_inverse_softplus_normal_tests.jl")
    include("ngmp/linear_reshape_meta_tests.jl")
    include("ngmp/linear_low_rank_meta_tests.jl")
    include("ngmp/normal_mean_precision_rule_tests.jl")
    include("ngmp/normal_mean_precision_joint_belief_tests.jl")
    include("ngmp/softdot_info_form_tests.jl")
    include("ngmp/softdot_relaxed_structure_tests.jl")
    include("ngmp/dense_last_layer_gp_tests.jl")
    include("mv_stack_tests.jl")
    include("ngmp/unscented_tests.jl")
    #TODO for now I do not need MvNormalScalePrecision rules, test shows how it works
    # but so far implementation is so hard that I leave it for the future
    # include("ngmp/gated_weights_tests.jl")
    # include("ngmp/softdot_precision_gate_tests.jl")
end
