module IVONPGE

using ADTypes: AutoEnzyme
using Dates
using Distributions
using Enzyme
using IVONRepro
using JLD2
using LinearAlgebra
using Lux
using Optimisers
using Printf
using ProbabilisticEnsembling
using Random
using SHA
using StableRNGs
using Statistics
using TOML
using YAML

const PE = ProbabilisticEnsembling

include("config.jl")
include("gates.jl")
include("data.jl")
include("training.jl")
include("metrics.jl")
include("artifacts.jl")
include("benchmark.jl")
include("cli.jl")

export main, build_gate, parameter_count, train_ivon_gate, posterior_components,
    mixture_metrics, verify_frozen_hashes, posterior_variances_valid,
    expected_parameter_count, DEFAULT_SEED

end
