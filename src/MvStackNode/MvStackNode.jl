module MvStackNode

using ReactiveMP, Rocket
using BayesBase
using ExponentialFamily
using LinearAlgebra

export MvStack

include("node.jl")
include("rules.jl")
include("score.jl")

end
