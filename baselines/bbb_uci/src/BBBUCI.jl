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
    Yacht,
    Concrete,
    EnergyEfficiency,
    BostonHousing,
    PowerPlant,
    WineQualityRed,
    UCISplitSpec,
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
using TOML
using Zygote

include("config.jl")
include("data.jl")
include("model.jl")
include("metrics.jl")
include("training.jl")
include("posteriors.jl")
include("artifacts.jl")
include("benchmark.jl")

export BBBConfig,
    load_config,
    validate_config,
    parse_datasets,
    parse_likelihoods,
    uci_regression_split,
    uci_regression_splits,
    prepare_uci_regression_partition,
    inverse_softplus,
    stable_softplus,
    initialize_model,
    sample_epsilon,
    forward_sample,
    sampled_complexity_cost,
    elbo_loss,
    predictive_samples,
    predictive_metrics,
    train_with_validation,
    refit_model,
    load_posterior_checkpoint,
    predict_posterior,
    predict_holdout,
    run_configuration,
    run_benchmark,
    main

end
