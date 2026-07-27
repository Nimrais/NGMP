module BBBUCI

using CSV
using DataFrames
using Dates
using JLD2
using LinearAlgebra
using LogExpFunctions: logsumexp
using Optimisers
import Plots
using Printf
using Random
using StableRNGs
using Statistics
using SurrogateModelling:
    Yacht, Concrete, EnergyEfficiency, BostonHousing, PowerPlant, WineQualityRed
using TOML
using Zygote

include("config.jl")
include("data.jl")
include("model.jl")
include("metrics.jl")
include("training.jl")
include("artifacts.jl")
include("benchmark.jl")

export BBBConfig,
    load_config,
    validate_config,
    parse_datasets,
    parse_likelihoods,
    deterministic_split,
    fit_standardizer,
    prepare_split,
    inverse_softplus,
    stable_softplus,
    initialize_model,
    sample_epsilon,
    forward_sample,
    gaussian_kl,
    elbo_loss,
    predictive_samples,
    predictive_metrics,
    train_with_validation,
    refit_model,
    run_configuration,
    run_benchmark,
    main

end
