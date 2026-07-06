module NaturalGradientMP

using ReactiveMP, Rocket, TupleTools

export NaturalGradientMessage, NGMPDependencies, DampingMeta, NGMPEdgeState

include("constraint.jl")
include("damping.jl")
include("dependencies.jl")

end
