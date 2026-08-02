module DVIUCI

using CSV
using DataFrames
using Dates
using JLD2
using LinearAlgebra
using Optimisers
import Plots
using Printf
using Random
using SpecialFunctions: erf
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
    prepare_uci_regression_partition
using TOML
using Zygote

include("config.jl")
include("data.jl")
include("moments.jl")
include("model.jl")
include("metrics.jl")
include("training.jl")
include("artifacts.jl")
include("benchmark.jl")

export DVIConfig,
    NumericalTracker,
    DVINumericalError,
    safe_exp,
    tracker_record,
    load_config,
    validate_config,
    parse_datasets,
    parse_split_ids,
    uci_regression_split,
    uci_regression_splits,
    prepare_uci_regression_partition,
    standard_gaussian,
    gaussian_cdf,
    softrelu,
    dvi_relu_delta,
    relu_moments_full,
    relu_moments_diagonal,
    initialize_model,
    propagate_dvi,
    empirical_bayes_kl,
    empirical_bayes_prior_variances,
    expected_log_likelihood,
    dvi_loss,
    kl_weight,
    predictive_distribution,
    predictive_metrics,
    train_with_validation,
    refit_model,
    run_configuration,
    run_benchmark,
    main

end
