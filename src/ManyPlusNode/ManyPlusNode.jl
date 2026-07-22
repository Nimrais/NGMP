module ManyPlusNode

using ReactiveMP, Rocket
using BayesBase
using ExponentialFamily

export ManyPlus

include("node.jl")
include("rules.jl")
include("score.jl")

end
