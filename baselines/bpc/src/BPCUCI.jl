module BPCUCI

using CSV
import CUDA
using DataFrames
using DataDeps
using Dates
using Distributions: Chisq
using JLD2
using LinearAlgebra
using Printf
using Random
using StableRNGs
using Statistics
using TOML

const SHARED_UCI_SOURCE_DIRECTORY = normpath(joinpath(
    @__DIR__, "..", "..", "..", "src", "datasets",
))
include(joinpath(SHARED_UCI_SOURCE_DIRECTORY, "uci_regression.jl"))
include(joinpath(SHARED_UCI_SOURCE_DIRECTORY, "uci_benchmark.jl"))

function __init__()
    __init__uci_regression()
end

include("config.jl")
include("data.jl")
include("backend.jl")
include("model.jl")
include("metrics.jl")
include("training.jl")
include("artifacts.jl")
include("benchmark.jl")

export BPCConfig,
    BPCLayer,
    BPCModel,
    load_config,
    validate_config,
    parse_datasets,
    load_dataset,
    outer_and_inner_splits,
    backend_available,
    initialize_model,
    posterior_update!,
    infer_latent_states,
    expected_energy,
    predictive_samples,
    predictive_metrics,
    train_with_validation,
    refit_model,
    load_posterior_checkpoint,
    predict_holdout,
    run_configuration,
    run_benchmark,
    main

end
