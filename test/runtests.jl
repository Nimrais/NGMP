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

@testset "NaturalGradientMP integration" begin
    include("ngmp/reference.jl")
    include("ngmp/rule_tests.jl")
    include("ngmp/integration_tests.jl")
end
