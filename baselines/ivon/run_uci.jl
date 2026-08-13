#!/usr/bin/env julia

module IVONUCI

using ADTypes: AutoEnzyme
using CSV
using DataFrames
using Dates
using Enzyme
using IVONRepro: IVON, ivon_train_step!, posterior_variance
using JLD2
using LogExpFunctions: logsumexp
using Lux
using Optimisers
using Printf
using Random
using SHA
using StableRNGs
using Statistics
using SurrogateModelling:
    Yacht,
    Concrete,
    EnergyEfficiency,
    BostonHousing,
    PowerPlant,
    WineQualityRed,
    UCI_SPLIT_PROTOCOL_VERSION,
    uci_regression_split,
    prepare_uci_regression_partition
using TOML

export UCIConfig,
    DATASET_REGISTRY,
    LEARNING_RATES,
    ESS_MULTIPLIERS,
    model_and_evaluation_seeds,
    build_model,
    train_with_validation,
    refit_model,
    posterior_components,
    predictive_metrics,
    select_pilot_configuration,
    replay_checkpoint,
    validate_final_rows,
    run_smoke,
    run_pilot,
    run_full,
    summarize,
    main

const BENCHMARK_ROOT = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(BENCHMARK_ROOT, "..", ".."))
const RESULTS_ROOT = joinpath(REPOSITORY_ROOT, "paper_materials", "ivon", "uci")
const SCHEMA_VERSION = "ivon-uci-v1"
const PILOT_SCHEMA_VERSION = "ivon-uci-pilot-v1"
const CHECKPOINT_SCHEMA_VERSION = "ivon-uci-checkpoint-v1"
const SPLIT_BASE_SEED = 20_260_726
const MODEL_BASE_SEED = 20_260_727
const LEARNING_RATES = (0.001, 0.01, 0.1)
const ESS_MULTIPLIERS = (1, 100)
const FIXED_IVON = (
    hess_init = 0.1,
    beta1 = 0.9,
    beta2 = 0.9999,
    weight_decay = 1.0e-4,
    mc_samples = 1,
    hess_approx = :price,
    rescale_lr = true,
    debias = true,
)
const DATASET_REGISTRY = (
    (key = "yacht", name = "Yacht", constructor = Yacht),
    (key = "concrete", name = "Concrete", constructor = Concrete),
    (key = "energy", name = "Energy", constructor = EnergyEfficiency),
    (key = "housing", name = "Boston Housing", constructor = BostonHousing),
    (key = "power", name = "Power Plant", constructor = PowerPlant),
    (key = "wine", name = "Wine", constructor = WineQualityRed),
)
const DATASET_KEYS = Tuple(spec.key for spec in DATASET_REGISTRY)
const LOG2PI_F32 = Float32(log(2pi))
const METRIC_NAMES = (
    :lpd_standardized,
    :nll_standardized,
    :lpd_original,
    :nll_original,
    :rmse_standardized,
    :rmse_original,
    :coverage_50,
    :coverage_80,
    :coverage_95,
    :mean_total_variance_standardized,
    :mean_epistemic_variance_standardized,
    :mean_aleatoric_variance_standardized,
    :mean_total_variance_original,
    :mean_epistemic_variance_original,
    :mean_aleatoric_variance_original,
)

Base.@kwdef struct UCIConfig
    datasets::Vector{String} = collect(DATASET_KEYS)
    n_splits::Int = 20
    hidden_units::Int = 50
    batch_size::Int = 100
    max_epochs::Int = 500
    min_epochs::Int = 25
    validation_every::Int = 5
    patience::Int = 20
    eval_samples::Int = 20
    test_fraction::Float64 = 0.1
    validation_fraction::Float64 = 0.1
    split_base_seed::Int = SPLIT_BASE_SEED
    model_base_seed::Int = MODEL_BASE_SEED
    output_root::String = RESULTS_ROOT
    show_progress::Bool = true
end

function validate_config(config::UCIConfig)
    all(key -> key in DATASET_KEYS, config.datasets) ||
        throw(ArgumentError("UCI configuration contains an unknown dataset"))
    length(unique(config.datasets)) == length(config.datasets) ||
        throw(ArgumentError("UCI datasets must be unique"))
    config.n_splits >= 1 || throw(ArgumentError("n_splits must be positive"))
    config.hidden_units >= 1 || throw(ArgumentError("hidden_units must be positive"))
    config.batch_size >= 1 || throw(ArgumentError("batch_size must be positive"))
    config.max_epochs >= 1 || throw(ArgumentError("max_epochs must be positive"))
    1 <= config.min_epochs <= config.max_epochs ||
        throw(ArgumentError("min_epochs must lie in 1:max_epochs"))
    config.validation_every >= 1 ||
        throw(ArgumentError("validation_every must be positive"))
    config.patience >= 1 || throw(ArgumentError("patience must be positive"))
    config.eval_samples >= 1 || throw(ArgumentError("eval_samples must be positive"))
    0 < config.test_fraction < 1 || throw(ArgumentError("invalid test fraction"))
    0 < config.validation_fraction < 1 ||
        throw(ArgumentError("invalid validation fraction"))
    return config
end

function result_affecting_config(config::UCIConfig)
    return (
        schema_version = SCHEMA_VERSION,
        datasets = Tuple(config.datasets),
        n_splits = config.n_splits,
        architecture = (hidden_layers = 2, hidden_units = config.hidden_units,
            activation = "relu", output_units = 1),
        likelihood = "homoscedastic_gaussian_standardized_variance_1",
        batch_size = config.batch_size,
        train_posterior_samples = 1,
        eval_samples = config.eval_samples,
        max_epochs = config.max_epochs,
        min_epochs = config.min_epochs,
        validation_every = config.validation_every,
        patience = config.patience,
        test_fraction = config.test_fraction,
        validation_fraction = config.validation_fraction,
        split_protocol = UCI_SPLIT_PROTOCOL_VERSION,
        split_base_seed = config.split_base_seed,
        model_base_seed = config.model_base_seed,
        learning_rates = LEARNING_RATES,
        ess_multipliers = ESS_MULTIPLIERS,
        fixed_ivon = FIXED_IVON,
    )
end

config_fingerprint(config::UCIConfig) =
    bytes2hex(sha256(repr(result_affecting_config(config))))

selected_fingerprint(config::UCIConfig, selected) = bytes2hex(sha256(repr((
    protocol = config_fingerprint(config),
    learning_rate = Float64(selected.learning_rate),
    ess_multiplier = Int(selected.ess_multiplier),
))))

function dataset_spec(key::AbstractString)
    index = findfirst(spec -> spec.key == key, DATASET_REGISTRY)
    isnothing(index) && throw(ArgumentError("unknown UCI dataset '$key'"))
    return DATASET_REGISTRY[index], index
end

function load_dataset(key::AbstractString)
    spec, index = dataset_spec(key)
    raw = spec.constructor(as_df = false)
    features, targets = raw[:]
    return (
        key = spec.key,
        name = spec.name,
        index = index,
        features = Matrix{Float64}(features),
        targets = vec(Float64.(targets)),
    )
end

function split_for(dataset, split_id::Int, config::UCIConfig)
    return uci_regression_split(
        size(dataset.features, 1),
        split_id;
        base_seed = config.split_base_seed,
        test_fraction = config.test_fraction,
        validation_fraction = config.validation_fraction,
    )
end

function inner_partition(dataset, split_spec)
    # Deliberately does not slice split_spec.test_indices. Pilot fitting and
    # epoch selection therefore cannot inspect outer holdout values.
    return prepare_uci_regression_partition(
        dataset.features,
        dataset.targets,
        split_spec.inner_train_indices,
        split_spec.validation_indices,
    )
end

function outer_partition(dataset, split_spec)
    return prepare_uci_regression_partition(
        dataset.features,
        dataset.targets,
        split_spec.train_indices,
        split_spec.test_indices,
    )
end

function split_spec_record(spec)
    return (
        protocol_version = String(spec.protocol_version),
        split_id = Int(spec.split_id),
        n_observations = Int(spec.n_observations),
        outer_seed = Int(spec.outer_seed),
        inner_seed = Int(spec.inner_seed),
        test_fraction = Float64(spec.test_fraction),
        validation_fraction = Float64(spec.validation_fraction),
        train_indices = Int.(spec.train_indices),
        test_indices = Int.(spec.test_indices),
        inner_train_indices = Int.(spec.inner_train_indices),
        validation_indices = Int.(spec.validation_indices),
    )
end

function split_records_equal(left, right)
    return all(name -> getproperty(left, name) == getproperty(right, name),
        propertynames(left)) && propertynames(left) == propertynames(right)
end

function model_and_evaluation_seeds(
    dataset_index::Int,
    split_id::Int;
    model_base_seed::Int = MODEL_BASE_SEED,
)
    dataset_index in eachindex(DATASET_REGISTRY) ||
        throw(ArgumentError("dataset_index is out of range"))
    split_id >= 1 || throw(ArgumentError("split_id must be positive"))
    model = model_base_seed + 10_000 * dataset_index + 100 * split_id + 1
    return (model = model, batches = model + 10, validation = model + 20,
        test = model + 30)
end

build_model(input_dimension::Int; hidden_units::Int = 50) = Lux.Chain(
    Lux.Dense(input_dimension => hidden_units, Lux.relu),
    Lux.Dense(hidden_units => hidden_units, Lux.relu),
    Lux.Dense(hidden_units => 1),
)

function ivon_optimizer(learning_rate::Real, n_training::Int, ess_multiplier::Int)
    n_training >= 1 || throw(ArgumentError("n_training must be positive"))
    ess_multiplier in ESS_MULTIPLIERS ||
        throw(ArgumentError("unsupported ESS multiplier $ess_multiplier"))
    return IVON(
        lr = Float64(learning_rate),
        ess = Float64(n_training * ess_multiplier),
        hess_init = FIXED_IVON.hess_init,
        beta1 = FIXED_IVON.beta1,
        beta2 = FIXED_IVON.beta2,
        weight_decay = FIXED_IVON.weight_decay,
        mc_samples = FIXED_IVON.mc_samples,
        hess_approx = FIXED_IVON.hess_approx,
        rescale_lr = FIXED_IVON.rescale_lr,
        debias = FIXED_IVON.debias,
    )
end

function gaussian_objective(model, ps, st, data)
    features, targets = data
    predictions, st_new = model(features, ps, st)
    loss = 0.5f0 * sum(abs2, predictions .- targets) / length(targets) +
        0.5f0 * LOG2PI_F32
    return loss, st_new, (;)
end

function initialize_training(
    input_dimension::Int,
    n_training::Int,
    learning_rate::Real,
    ess_multiplier::Int,
    model_seed::Int,
    hidden_units::Int,
)
    model = build_model(input_dimension; hidden_units)
    parameters, states = Lux.setup(StableRNG(model_seed), model)
    optimizer = ivon_optimizer(learning_rate, n_training, ess_multiplier)
    train_state = Lux.Training.TrainState(model, parameters, states, optimizer)
    return model, optimizer, train_state
end

function train_epoch(
    train_state,
    features::AbstractMatrix,
    targets::AbstractVector,
    batch_size::Int,
    rng::AbstractRNG,
)
    n_training = size(features, 1)
    length(targets) == n_training || throw(DimensionMismatch("training data disagree"))
    order = randperm(rng, n_training)
    weighted_loss = 0.0
    for first_index in 1:batch_size:n_training
        last_index = min(first_index + batch_size - 1, n_training)
        indices = @view order[first_index:last_index]
        x_batch = permutedims(features[indices, :])
        y_batch = reshape(targets[indices], 1, :)
        loss, _, train_state = ivon_train_step!(
            rng, AutoEnzyme(), gaussian_objective, (x_batch, y_batch), train_state,
        )
        isfinite(loss) || error("non-finite IVON minibatch loss")
        weighted_loss += length(indices) * Float64(loss)
    end
    return train_state, weighted_loss / n_training
end

posterior_variances_valid(x::AbstractArray) = all(value -> isfinite(value) && value > 0, x)
posterior_variances_valid(x::NamedTuple) = all(posterior_variances_valid, values(x))
posterior_variances_valid(x::Tuple) = all(posterior_variances_valid, x)
posterior_variances_valid(::Any) = true

function snapshot(train_state)
    return (
        parameters = deepcopy(train_state.parameters),
        states = deepcopy(train_state.states),
        optimizer_state = deepcopy(train_state.optimizer_state),
        step = Int(train_state.step),
    )
end

function posterior_components(
    model,
    optimizer,
    saved_state,
    standardized_features::AbstractMatrix;
    n_samples::Int,
    seed::Int,
)
    n_samples >= 1 || throw(ArgumentError("n_samples must be positive"))
    variances = posterior_variance(optimizer, saved_state.optimizer_state)
    posterior_variances_valid(variances) || error("invalid IVON posterior variance")
    n_observations = size(standardized_features, 1)
    means = Matrix{Float64}(undef, n_samples, n_observations)
    component_variances = ones(Float64, n_samples, n_observations)
    inputs = permutedims(Float32.(standardized_features))
    test_states = Lux.testmode(saved_state.states)
    rng = StableRNG(seed)
    for sample_id in 1:n_samples
        parameters = rand(
            rng, optimizer, saved_state.optimizer_state, saved_state.parameters,
        )
        prediction, _ = model(inputs, parameters, test_states)
        means[sample_id, :] .= Float64.(vec(prediction))
    end
    all(isfinite, means) || error("non-finite IVON posterior prediction")
    return (means = means, variances = component_variances)
end

function row_logmeanexp(values::AbstractMatrix)
    result = Vector{Float64}(undef, size(values, 2))
    for column in axes(values, 2)
        result[column] = logsumexp(@view values[:, column]) - log(size(values, 1))
    end
    return result
end

function predictive_metrics(
    components,
    targets_standardized::AbstractVector,
    targets_original::AbstractVector,
    standardizer,
)
    n_samples, n_observations = size(components.means)
    size(components.variances) == (n_samples, n_observations) ||
        throw(DimensionMismatch("component matrices disagree"))
    length(targets_standardized) == n_observations ||
        throw(DimensionMismatch("standardized targets disagree"))
    length(targets_original) == n_observations ||
        throw(DimensionMismatch("original targets disagree"))
    all(value -> isfinite(value) && value > 0, components.variances) ||
        throw(ArgumentError("component variances must be finite and positive"))

    log_components = Matrix{Float64}(undef, n_samples, n_observations)
    for sample_id in 1:n_samples
        variance = @view components.variances[sample_id, :]
        residual_squared = (targets_standardized .-
            @view(components.means[sample_id, :])) .^ 2
        log_components[sample_id, :] .= -0.5 .* (
            log(2pi) .+ log.(variance) .+ residual_squared ./ variance
        )
    end
    pointwise_lpd = row_logmeanexp(log_components)
    predictive_mean = vec(mean(components.means; dims = 1))
    epistemic = vec(mean(
        (components.means .- transpose(predictive_mean)) .^ 2; dims = 1,
    ))
    aleatoric = vec(mean(components.variances; dims = 1))
    total = epistemic + aleatoric
    all(value -> isfinite(value) && value > 0, total) ||
        error("non-positive predictive variance")

    y_center = Float64(standardizer.y_center)
    y_scale = Float64(standardizer.y_scale)
    y_scale > 0 || throw(ArgumentError("target scale must be positive"))
    predictive_mean_original = y_center .+ y_scale .* predictive_mean
    epistemic_original = y_scale^2 .* epistemic
    aleatoric_original = y_scale^2 .* aleatoric
    total_original = epistemic_original + aleatoric_original
    standard_deviation_original = sqrt.(total_original)
    coverage(z) = mean(abs.(targets_original .- predictive_mean_original) .<=
        z .* standard_deviation_original)
    lpd_standardized = mean(pointwise_lpd)

    metrics = (
        lpd_standardized = lpd_standardized,
        nll_standardized = -lpd_standardized,
        lpd_original = lpd_standardized - log(y_scale),
        nll_original = -lpd_standardized + log(y_scale),
        rmse_standardized = sqrt(mean((targets_standardized .- predictive_mean) .^ 2)),
        rmse_original = sqrt(mean((targets_original .- predictive_mean_original) .^ 2)),
        coverage_50 = coverage(0.6744897501960817),
        coverage_80 = coverage(1.2815515655446004),
        coverage_95 = coverage(1.959963984540054),
        mean_total_variance_standardized = mean(total),
        mean_epistemic_variance_standardized = mean(epistemic),
        mean_aleatoric_variance_standardized = mean(aleatoric),
        mean_total_variance_original = mean(total_original),
        mean_epistemic_variance_original = mean(epistemic_original),
        mean_aleatoric_variance_original = mean(aleatoric_original),
        predictive_mean_original = predictive_mean_original,
        total_variance_original = total_original,
        pointwise_lpd_standardized = pointwise_lpd,
    )
    all(isfinite, (metrics.lpd_standardized, metrics.rmse_original,
        metrics.mean_total_variance_original)) || error("non-finite predictive metric")
    return metrics
end

function scalar_metrics(metrics)
    return (; (name => getproperty(metrics, name) for name in METRIC_NAMES)...)
end

function component_digest(components)
    means_digest = bytes2hex(sha256(reinterpret(UInt8, vec(components.means))))
    variances_digest = bytes2hex(sha256(reinterpret(UInt8, vec(components.variances))))
    return bytes2hex(sha256(means_digest * variances_digest))
end

function evaluate_state(
    model,
    optimizer,
    saved_state,
    partition,
    config::UCIConfig;
    seed::Int,
)
    components = posterior_components(
        model,
        optimizer,
        saved_state,
        partition.x_test_standardized;
        n_samples = config.eval_samples,
        seed,
    )
    metrics = predictive_metrics(
        components,
        partition.y_test_standardized,
        partition.y_test,
        partition.standardizer,
    )
    return metrics, components
end

"""
    train_with_validation(partition, config; learning_rate, ess_multiplier, seeds)

Train on an inner-training partition and choose an epoch from validation LPD.
The validation posterior draws are fixed across checks. Patience is counted in
validation checks, exactly as in the BBB UCI runner.
"""
function train_with_validation(
    partition,
    config::UCIConfig;
    learning_rate::Real,
    ess_multiplier::Int,
    seeds,
)
    validate_config(config)
    n_training = size(partition.x_train_standardized, 1)
    input_dimension = size(partition.x_train_standardized, 2)
    model, optimizer, train_state = initialize_training(
        input_dimension,
        n_training,
        learning_rate,
        ess_multiplier,
        Int(seeds.model),
        config.hidden_units,
    )
    rng = StableRNG(Int(seeds.batches))
    history = NamedTuple[]
    best_epoch = 0
    best_lpd = -Inf
    best_metrics = nothing
    best_components = nothing
    best_state = nothing
    non_improving_checks = 0
    stopped_early = false

    for epoch in 1:config.max_epochs
        train_state, training_loss = train_epoch(
            train_state,
            partition.x_train_standardized,
            partition.y_train_standardized,
            config.batch_size,
            rng,
        )
        should_validate = epoch % config.validation_every == 0 ||
            epoch == config.max_epochs
        should_validate || continue

        current_state = snapshot(train_state)
        metrics, components = evaluate_state(
            model, optimizer, current_state, partition, config;
            seed = Int(seeds.validation),
        )
        push!(history, (
            epoch = epoch,
            training_nll = training_loss,
            validation_lpd_standardized = metrics.lpd_standardized,
            validation_lpd_original = metrics.lpd_original,
            validation_rmse_standardized = metrics.rmse_standardized,
            validation_rmse_original = metrics.rmse_original,
        ))
        if config.show_progress
            @printf(
                "      epoch %4d  train NLL %.6f  val LPD(std) %.6f  val RMSE(std) %.6f\n",
                epoch, training_loss, metrics.lpd_standardized,
                metrics.rmse_standardized,
            )
        end

        epoch < config.min_epochs && continue
        if isfinite(metrics.lpd_standardized) &&
           metrics.lpd_standardized > best_lpd
            best_epoch = epoch
            best_lpd = metrics.lpd_standardized
            best_metrics = metrics
            best_components = components
            best_state = current_state
            non_improving_checks = 0
        else
            non_improving_checks += 1
        end
        if non_improving_checks >= config.patience
            stopped_early = true
            break
        end
    end

    best_epoch > 0 || error("validation did not produce a finite IVON model")
    variances = posterior_variance(optimizer, best_state.optimizer_state)
    posterior_variances_valid(variances) || error("invalid selected posterior variance")
    return (
        model = model,
        optimizer = optimizer,
        state = best_state,
        best_epoch = best_epoch,
        best_lpd = best_lpd,
        best_metrics = best_metrics,
        best_components = best_components,
        history = DataFrame(history),
        stopped_early = stopped_early,
        ess = n_training * ess_multiplier,
    )
end

"""Reinitialize at the model seed and refit on the complete outer training set."""
function refit_model(
    partition,
    config::UCIConfig;
    learning_rate::Real,
    ess_multiplier::Int,
    seeds,
    epochs::Int,
)
    epochs >= 1 || throw(ArgumentError("epochs must be positive"))
    n_training = size(partition.x_train_standardized, 1)
    input_dimension = size(partition.x_train_standardized, 2)
    model, optimizer, train_state = initialize_training(
        input_dimension,
        n_training,
        learning_rate,
        ess_multiplier,
        Int(seeds.model),
        config.hidden_units,
    )
    rng = StableRNG(Int(seeds.batches))
    losses = Vector{Float64}(undef, epochs)
    for epoch in 1:epochs
        train_state, losses[epoch] = train_epoch(
            train_state,
            partition.x_train_standardized,
            partition.y_train_standardized,
            config.batch_size,
            rng,
        )
    end
    saved_state = snapshot(train_state)
    variances = posterior_variance(optimizer, saved_state.optimizer_state)
    posterior_variances_valid(variances) || error("invalid refitted posterior variance")
    return (
        model = model,
        optimizer = optimizer,
        state = saved_state,
        losses = losses,
        ess = n_training * ess_multiplier,
    )
end

function atomic_csv_write(path::AbstractString, table)
    mkpath(dirname(path))
    temporary = path * ".tmp-" * string(getpid())
    try
        CSV.write(temporary, table)
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return path
end

function atomic_jld2_write(path::AbstractString, record::NamedTuple)
    mkpath(dirname(path))
    temporary = path * ".tmp-" * string(getpid())
    try
        JLD2.jldsave(temporary; record...)
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return path
end

function atomic_toml_write(path::AbstractString, data::AbstractDict)
    mkpath(dirname(path))
    temporary = path * ".tmp-" * string(getpid())
    try
        open(temporary, "w") do io
            TOML.print(io, data; sorted = true)
        end
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return path
end

function ensure_directories(config::UCIConfig)
    for relative in (
        "", "pilot", joinpath("pilot", "histories"),
        joinpath("pilot", "checkpoints"), "histories", "checkpoints",
    )
        mkpath(joinpath(config.output_root, relative))
    end
    return config.output_root
end

function protocol_dictionary(config::UCIConfig; selected = nothing)
    data = Dict{String,Any}(
        "benchmark" => "IVON on the BBB homoscedastic UCI protocol",
        "schema_version" => SCHEMA_VERSION,
        "config_fingerprint" => config_fingerprint(config),
        "output_root" => relpath(config.output_root, REPOSITORY_ROOT),
        "datasets" => config.datasets,
        "n_splits" => config.n_splits,
        "split_protocol" => UCI_SPLIT_PROTOCOL_VERSION,
        "split_base_seed" => config.split_base_seed,
        "model_base_seed" => config.model_base_seed,
        "hidden_layers" => 2,
        "hidden_units" => config.hidden_units,
        "activation" => "relu",
        "likelihood" => "homoscedastic Gaussian",
        "standardized_observation_variance" => 1.0,
        "batch_size" => config.batch_size,
        "train_posterior_samples" => 1,
        "eval_posterior_samples" => config.eval_samples,
        "max_epochs" => config.max_epochs,
        "min_epochs" => config.min_epochs,
        "validation_every" => config.validation_every,
        "patience" => config.patience,
        "learning_rates" => collect(LEARNING_RATES),
        "ess_multipliers" => collect(ESS_MULTIPLIERS),
        "julia_version" => string(VERSION),
    )
    data["ivon"] = Dict(
        "hess_init" => FIXED_IVON.hess_init,
        "beta1" => FIXED_IVON.beta1,
        "beta2" => FIXED_IVON.beta2,
        "weight_decay" => FIXED_IVON.weight_decay,
        "mc_samples" => FIXED_IVON.mc_samples,
        "hess_approx" => string(FIXED_IVON.hess_approx),
        "rescale_lr" => FIXED_IVON.rescale_lr,
        "debias" => FIXED_IVON.debias,
    )
    if selected !== nothing
        data["selected_ivon"] = Dict(
            "learning_rate" => Float64(selected.learning_rate),
            "ess_multiplier" => Int(selected.ess_multiplier),
            "selection_fingerprint" => selected_fingerprint(config, selected),
        )
    end
    return data
end

function write_protocol_config(config::UCIConfig; selected = nothing)
    path = joinpath(config.output_root, "config.toml")
    if isfile(path)
        existing = TOML.parsefile(path)
        get(existing, "config_fingerprint", "") == config_fingerprint(config) ||
            throw(ArgumentError("existing UCI result configuration is incompatible: $path"))
    end
    return atomic_toml_write(path, protocol_dictionary(config; selected))
end

function write_split_manifest(config::UCIConfig; force::Bool = false)
    path = joinpath(config.output_root, "split_manifest.jld2")
    if isfile(path) && !force
        saved = JLD2.load(path)
        get(saved, "schema_version", "") == "ivon-uci-split-manifest-v1" ||
            error("invalid split manifest: $path")
        get(saved, "config_fingerprint", "") == config_fingerprint(config) ||
            error("split manifest configuration mismatch: $path")
        return path
    end
    datasets = Dict{String,Any}()
    for key in config.datasets
        dataset = load_dataset(key)
        datasets[key] = [
            split_spec_record(split_for(dataset, split_id, config))
            for split_id in 1:config.n_splits
        ]
    end
    return atomic_jld2_write(path, (
        schema_version = "ivon-uci-split-manifest-v1",
        config_fingerprint = config_fingerprint(config),
        split_protocol = UCI_SPLIT_PROTOCOL_VERSION,
        base_seed = config.split_base_seed,
        n_splits = config.n_splits,
        datasets = datasets,
    ))
end

function initialize_results(config::UCIConfig; selected = nothing)
    validate_config(config)
    ensure_directories(config)
    write_protocol_config(config; selected)
    write_split_manifest(config)
    return config.output_root
end

function dataframe_or_empty(path::AbstractString)
    return isfile(path) && filesize(path) > 0 ? CSV.read(path, DataFrame) : DataFrame()
end

function replace_matching_row(table::DataFrame, row::NamedTuple, keys)
    if !isempty(table) && all(key -> key in propertynames(table), keys)
        keep = map(eachrow(table)) do existing
            !all(key -> isequal(getproperty(existing, key), getproperty(row, key)), keys)
        end
        table = table[keep, :]
    end
    incoming = DataFrame([row])
    return isempty(table) ? incoming : vcat(table, incoming; cols = :union)
end

pilot_stem(dataset::AbstractString, learning_rate::Real, ess_multiplier::Int) =
    "$(dataset)_lr$(replace(@sprintf("%.3g", learning_rate), "." => "p"))_ess$(ess_multiplier)"

pilot_checkpoint_path(config::UCIConfig, dataset, learning_rate, ess_multiplier) =
    joinpath(config.output_root, "pilot", "checkpoints",
        pilot_stem(dataset, learning_rate, ess_multiplier) * ".jld2")

pilot_history_path(config::UCIConfig, dataset, learning_rate, ess_multiplier) =
    joinpath(config.output_root, "pilot", "histories",
        pilot_stem(dataset, learning_rate, ess_multiplier) * ".csv")

final_stem(dataset::AbstractString, split_id::Int) =
    @sprintf("%s_split%02d_homoscedastic", dataset, split_id)

final_checkpoint_path(config::UCIConfig, dataset, split_id) =
    joinpath(config.output_root, "checkpoints", final_stem(dataset, split_id) * ".jld2")

final_history_path(config::UCIConfig, dataset, split_id) =
    joinpath(config.output_root, "histories", final_stem(dataset, split_id) * "_selection.csv")

refit_history_path(config::UCIConfig, dataset, split_id) =
    joinpath(config.output_root, "histories", final_stem(dataset, split_id) * "_refit.csv")

function checkpoint_basic_valid(path::AbstractString; phase::AbstractString)
    isfile(path) || return false
    return try
        saved = JLD2.load(path)
        get(saved, "complete", false) === true &&
            get(saved, "schema_version", "") == CHECKPOINT_SCHEMA_VERSION &&
            get(saved, "phase", "") == phase &&
            get(saved, "prediction_digest", "") ==
                component_digest(saved["posterior_components"])
    catch error_value
        @warn "Ignoring invalid IVON UCI checkpoint" path exception =
            (error_value, catch_backtrace())
        false
    end
end

function pilot_checkpoint_valid(
    path::AbstractString,
    config::UCIConfig,
    dataset::AbstractString,
    learning_rate::Real,
    ess_multiplier::Int,
)
    checkpoint_basic_valid(path; phase = "pilot") || return false
    saved = JLD2.load(path)
    _, dataset_index = dataset_spec(dataset)
    expected_seeds = model_and_evaluation_seeds(
        dataset_index, 1; model_base_seed = config.model_base_seed,
    )
    return saved["config_fingerprint"] == config_fingerprint(config) &&
        saved["dataset"] == dataset &&
        saved["split_id"] == 1 &&
        saved["learning_rate"] == Float64(learning_rate) &&
        saved["ess_multiplier"] == ess_multiplier &&
        saved["seeds"] == expected_seeds
end

function pilot_failure_row(dataset, learning_rate, ess_multiplier, seeds, error_value)
    return (
        dataset = dataset.key,
        dataset_name = dataset.name,
        split = 1,
        learning_rate = Float64(learning_rate),
        ess_multiplier = Int(ess_multiplier),
        status = "failure",
        error = sprint(showerror, error_value, catch_backtrace()),
        model_seed = seeds.model,
        batch_seed = seeds.batches,
        validation_seed = seeds.validation,
        best_epoch = 0,
        validation_lpd_standardized = NaN,
        validation_lpd_original = NaN,
        validation_rmse_standardized = NaN,
        validation_rmse_original = NaN,
        stopped_early = false,
        elapsed_seconds = NaN,
        checkpoint = "",
    )
end

function run_pilot_fit(
    dataset,
    split_spec,
    config::UCIConfig,
    learning_rate::Real,
    ess_multiplier::Int,
)
    seeds = model_and_evaluation_seeds(
        dataset.index, 1; model_base_seed = config.model_base_seed,
    )
    # Only inner_train_indices and validation_indices are materialized here.
    partition = inner_partition(dataset, split_spec)
    started = time()
    selection = train_with_validation(
        partition,
        config;
        learning_rate,
        ess_multiplier,
        seeds,
    )
    elapsed_seconds = time() - started
    history_path = pilot_history_path(
        config, dataset.key, learning_rate, ess_multiplier,
    )
    atomic_csv_write(history_path, selection.history)
    checkpoint_path = pilot_checkpoint_path(
        config, dataset.key, learning_rate, ess_multiplier,
    )
    metrics = selection.best_metrics
    row = (
        dataset = dataset.key,
        dataset_name = dataset.name,
        split = 1,
        learning_rate = Float64(learning_rate),
        ess_multiplier = Int(ess_multiplier),
        status = "success",
        error = "",
        model_seed = seeds.model,
        batch_seed = seeds.batches,
        validation_seed = seeds.validation,
        best_epoch = selection.best_epoch,
        validation_lpd_standardized = metrics.lpd_standardized,
        validation_lpd_original = metrics.lpd_original,
        validation_rmse_standardized = metrics.rmse_standardized,
        validation_rmse_original = metrics.rmse_original,
        stopped_early = selection.stopped_early,
        elapsed_seconds = elapsed_seconds,
        checkpoint = relpath(checkpoint_path, config.output_root),
    )
    atomic_jld2_write(checkpoint_path, (
        complete = true,
        schema_version = CHECKPOINT_SCHEMA_VERSION,
        phase = "pilot",
        config_fingerprint = config_fingerprint(config),
        dataset = dataset.key,
        dataset_name = dataset.name,
        dataset_index = dataset.index,
        split_id = 1,
        split_spec = split_spec_record(split_spec),
        learning_rate = Float64(learning_rate),
        ess_multiplier = Int(ess_multiplier),
        seeds = seeds,
        n_train = length(partition.train_indices),
        n_validation = length(partition.test_indices),
        standardizer = partition.standardizer,
        validation_features_standardized = partition.x_test_standardized,
        validation_targets_standardized = partition.y_test_standardized,
        validation_targets_original = partition.y_test,
        model_parameters_mean = selection.state.parameters,
        model_states = selection.state.states,
        optimizer_state = selection.state.optimizer_state,
        training_step = selection.state.step,
        ess = selection.ess,
        best_epoch = selection.best_epoch,
        history = selection.history,
        metrics = scalar_metrics(metrics),
        posterior_components = selection.best_components,
        prediction_digest = component_digest(selection.best_components),
        pilot_row = row,
    ))
    pilot_checkpoint_valid(
        checkpoint_path, config, dataset.key, learning_rate, ess_multiplier,
    ) || error("pilot checkpoint round-trip validation failed: $checkpoint_path")
    return row
end

function expected_pilot_keys(config::UCIConfig)
    return Set(
        (dataset, Float64(learning_rate), Int(ess_multiplier))
        for dataset in config.datasets
        for learning_rate in LEARNING_RATES
        for ess_multiplier in ESS_MULTIPLIERS
    )
end

function pilot_rows_digest(rows::DataFrame, config::UCIConfig)
    successful = filter(row -> string(row.status) == "success", rows)
    pieces = String[]
    for row in eachrow(successful)
        push!(pieces, join((
            string(row.dataset),
            repr(Float64(row.learning_rate)),
            string(Int(row.ess_multiplier)),
            repr(Float64(row.validation_lpd_standardized)),
            repr(Float64(row.validation_rmse_standardized)),
            string(Int(row.best_epoch)),
        ), "|"))
    end
    sort!(pieces)
    length(pieces) == length(expected_pilot_keys(config)) ||
        error("pilot rows are incomplete")
    return bytes2hex(sha256(join(pieces, '\n')))
end

"""
    select_pilot_configuration(rows; dataset_keys=DATASET_KEYS)

Rank complete configurations by mean standardized validation LPD (descending),
then mean standardized RMSE, learning rate, and ESS multiplier (ascending).
"""
function select_pilot_configuration(
    rows::DataFrame;
    dataset_keys = DATASET_KEYS,
)
    required_columns = (
        :dataset, :learning_rate, :ess_multiplier, :status,
        :validation_lpd_standardized, :validation_rmse_standardized,
    )
    all(name -> name in propertynames(rows), required_columns) ||
        throw(ArgumentError("pilot table is missing required columns"))
    requested = Set(String.(dataset_keys))
    successful = filter(rows) do row
        string(row.status) == "success" && string(row.dataset) in requested &&
            isfinite(Float64(row.validation_lpd_standardized)) &&
            isfinite(Float64(row.validation_rmse_standardized))
    end
    rankings = NamedTuple[]
    for learning_rate in LEARNING_RATES, ess_multiplier in ESS_MULTIPLIERS
        matching = filter(successful) do row
            Float64(row.learning_rate) == learning_rate &&
                Int(row.ess_multiplier) == ess_multiplier
        end
        Set(string.(matching.dataset)) == requested || continue
        nrow(matching) == length(requested) || continue
        push!(rankings, (
            learning_rate = Float64(learning_rate),
            ess_multiplier = Int(ess_multiplier),
            mean_validation_lpd_standardized =
                mean(Float64.(matching.validation_lpd_standardized)),
            mean_validation_rmse_standardized =
                mean(Float64.(matching.validation_rmse_standardized)),
            n_datasets = nrow(matching),
        ))
    end
    length(rankings) == length(LEARNING_RATES) * length(ESS_MULTIPLIERS) ||
        error("pilot does not contain one successful fit per dataset and configuration")
    sort!(rankings; by = row -> (
        -row.mean_validation_lpd_standardized,
        row.mean_validation_rmse_standardized,
        row.learning_rate,
        row.ess_multiplier,
    ))
    return (selected = first(rankings), rankings = DataFrame(rankings))
end

function save_pilot_selection(rows::DataFrame, config::UCIConfig)
    result = select_pilot_configuration(rows; dataset_keys = config.datasets)
    path = joinpath(config.output_root, "pilot", "selection.jld2")
    digest = pilot_rows_digest(rows, config)
    atomic_jld2_write(path, (
        complete = true,
        schema_version = PILOT_SCHEMA_VERSION,
        config_fingerprint = config_fingerprint(config),
        pilot_rows_digest = digest,
        selected = result.selected,
        rankings = result.rankings,
    ))
    atomic_csv_write(joinpath(config.output_root, "pilot", "ranking.csv"),
        result.rankings)
    write_protocol_config(config; selected = result.selected)
    return result.selected
end

function load_pilot_selection(config::UCIConfig)
    path = joinpath(config.output_root, "pilot", "selection.jld2")
    isfile(path) || error("full requires a saved pilot selection; run `pilot` first")
    saved = JLD2.load(path)
    get(saved, "complete", false) === true || error("pilot selection is incomplete")
    get(saved, "schema_version", "") == PILOT_SCHEMA_VERSION ||
        error("unsupported pilot selection schema")
    saved["config_fingerprint"] == config_fingerprint(config) ||
        error("pilot selection configuration does not match the requested full run")
    rows_path = joinpath(config.output_root, "pilot", "pilot_rows.csv")
    rows = dataframe_or_empty(rows_path)
    digest = pilot_rows_digest(rows, config)
    saved["pilot_rows_digest"] == digest || error("pilot selection audit digest mismatch")
    recomputed = select_pilot_configuration(rows; dataset_keys = config.datasets).selected
    selected = saved["selected"]
    Float64(selected.learning_rate) == Float64(recomputed.learning_rate) &&
        Int(selected.ess_multiplier) == Int(recomputed.ess_multiplier) ||
        error("saved pilot selection is not the current deterministic winner")
    return selected
end

function run_pilot(; force::Bool = false, config::UCIConfig = UCIConfig())
    validate_config(config)
    Set(config.datasets) == Set(DATASET_KEYS) ||
        error("the benchmark pilot must use all six UCI datasets")
    initialize_results(config)
    rows_path = joinpath(config.output_root, "pilot", "pilot_rows.csv")
    rows = dataframe_or_empty(rows_path)
    total = length(config.datasets) * length(LEARNING_RATES) * length(ESS_MULTIPLIERS)
    cell = 0
    for key in config.datasets
        dataset = load_dataset(key)
        split_spec = split_for(dataset, 1, config)
        for learning_rate in LEARNING_RATES, ess_multiplier in ESS_MULTIPLIERS
            cell += 1
            path = pilot_checkpoint_path(config, key, learning_rate, ess_multiplier)
            history_path = pilot_history_path(config, key, learning_rate, ess_multiplier)
            if force
                isfile(path) && rm(path; force = true)
                isfile(history_path) && rm(history_path; force = true)
            end
            if !force && pilot_checkpoint_valid(
                path, config, key, learning_rate, ess_multiplier,
            )
                row = JLD2.load(path)["pilot_row"]
                rows = replace_matching_row(rows, row,
                    (:dataset, :learning_rate, :ess_multiplier))
                atomic_csv_write(rows_path, rows)
                println("[$cell/$total] skip pilot $(dataset.name) lr=$learning_rate ess×$ess_multiplier")
                continue
            end
            println("[$cell/$total] pilot $(dataset.name) lr=$learning_rate ess×$ess_multiplier")
            seeds = model_and_evaluation_seeds(
                dataset.index, 1; model_base_seed = config.model_base_seed,
            )
            row = try
                run_pilot_fit(dataset, split_spec, config, learning_rate, ess_multiplier)
            catch error_value
                @error "IVON UCI pilot fit failed" dataset = key learning_rate ess_multiplier exception =
                    (error_value, catch_backtrace())
                pilot_failure_row(dataset, learning_rate, ess_multiplier, seeds, error_value)
            end
            rows = replace_matching_row(rows, row,
                (:dataset, :learning_rate, :ess_multiplier))
            atomic_csv_write(rows_path, rows)
        end
    end
    selected = save_pilot_selection(rows, config)
    @printf("Selected global IVON configuration: lr=%.3g, ESS multiplier=%d\n",
        selected.learning_rate, selected.ess_multiplier)
    return selected
end

function final_checkpoint_valid(
    path::AbstractString,
    config::UCIConfig,
    dataset::AbstractString,
    split_id::Int,
    selected,
)
    checkpoint_basic_valid(path; phase = "full") || return false
    return try
        saved = JLD2.load(path)
        _, dataset_index = dataset_spec(dataset)
        expected_seeds = model_and_evaluation_seeds(
            dataset_index, split_id; model_base_seed = config.model_base_seed,
        )
        saved["config_fingerprint"] == config_fingerprint(config) &&
            saved["selection_fingerprint"] == selected_fingerprint(config, selected) &&
            saved["dataset"] == dataset && saved["split_id"] == split_id &&
            saved["learning_rate"] == Float64(selected.learning_rate) &&
            saved["ess_multiplier"] == Int(selected.ess_multiplier) &&
            saved["seeds"] == expected_seeds &&
            begin
                expected = uci_regression_split(
                    Int(saved["n_observations"]), split_id;
                    base_seed = config.split_base_seed,
                    test_fraction = config.test_fraction,
                    validation_fraction = config.validation_fraction,
                )
                split_records_equal(saved["split_spec"], split_spec_record(expected))
            end
    catch error_value
        @warn "Ignoring invalid final IVON UCI checkpoint" path exception =
            (error_value, catch_backtrace())
        false
    end
end

function final_failure_row(dataset, split_spec, selected, seeds, error_value)
    metrics = (; (name => NaN for name in METRIC_NAMES)...)
    return (
        method = "IVON",
        dataset = dataset.key,
        dataset_name = dataset.name,
        split = split_spec.split_id,
        split_protocol = UCI_SPLIT_PROTOCOL_VERSION,
        likelihood = "homoscedastic",
        status = "failure",
        error = sprint(showerror, error_value, catch_backtrace()),
        n_observations = size(dataset.features, 1),
        n_features = size(dataset.features, 2),
        n_train = length(split_spec.train_indices),
        n_test = length(split_spec.test_indices),
        outer_split_seed = split_spec.outer_seed,
        inner_split_seed = split_spec.inner_seed,
        model_seed = seeds.model,
        batch_seed = seeds.batches,
        validation_seed = seeds.validation,
        test_seed = seeds.test,
        learning_rate = Float64(selected.learning_rate),
        ess_multiplier = Int(selected.ess_multiplier),
        best_epoch = 0,
        stopped_early = false,
        selection_seconds = NaN,
        refit_seconds = NaN,
        evaluation_seconds = NaN,
        total_seconds = NaN,
        y_center = NaN,
        y_scale = NaN,
        checkpoint = "",
        metrics...,
    )
end

function run_final_configuration(
    dataset,
    split_id::Int,
    config::UCIConfig,
    selected,
)
    split_spec = split_for(dataset, split_id, config)
    seeds = model_and_evaluation_seeds(
        dataset.index, split_id; model_base_seed = config.model_base_seed,
    )

    # Epoch selection is completed before the outer partition is materialized.
    inner = inner_partition(dataset, split_spec)
    selection_started = time()
    selection = train_with_validation(
        inner,
        config;
        learning_rate = Float64(selected.learning_rate),
        ess_multiplier = Int(selected.ess_multiplier),
        seeds,
    )
    selection_seconds = time() - selection_started
    atomic_csv_write(final_history_path(config, dataset.key, split_id),
        selection.history)

    outer = outer_partition(dataset, split_spec)
    refit_started = time()
    refit = refit_model(
        outer,
        config;
        learning_rate = Float64(selected.learning_rate),
        ess_multiplier = Int(selected.ess_multiplier),
        seeds,
        epochs = selection.best_epoch,
    )
    refit_seconds = time() - refit_started
    atomic_csv_write(refit_history_path(config, dataset.key, split_id),
        DataFrame(epoch = collect(1:selection.best_epoch), training_nll = refit.losses))

    evaluation_started = time()
    components = posterior_components(
        refit.model,
        refit.optimizer,
        refit.state,
        outer.x_test_standardized;
        n_samples = config.eval_samples,
        seed = seeds.test,
    )
    metrics = predictive_metrics(
        components,
        outer.y_test_standardized,
        outer.y_test,
        outer.standardizer,
    )
    evaluation_seconds = time() - evaluation_started
    checkpoint_path = final_checkpoint_path(config, dataset.key, split_id)
    row = (
        method = "IVON",
        dataset = dataset.key,
        dataset_name = dataset.name,
        split = split_id,
        split_protocol = UCI_SPLIT_PROTOCOL_VERSION,
        likelihood = "homoscedastic",
        status = "success",
        error = "",
        n_observations = size(dataset.features, 1),
        n_features = size(dataset.features, 2),
        n_train = length(split_spec.train_indices),
        n_test = length(split_spec.test_indices),
        outer_split_seed = split_spec.outer_seed,
        inner_split_seed = split_spec.inner_seed,
        model_seed = seeds.model,
        batch_seed = seeds.batches,
        validation_seed = seeds.validation,
        test_seed = seeds.test,
        learning_rate = Float64(selected.learning_rate),
        ess_multiplier = Int(selected.ess_multiplier),
        best_epoch = selection.best_epoch,
        stopped_early = selection.stopped_early,
        selection_seconds = selection_seconds,
        refit_seconds = refit_seconds,
        evaluation_seconds = evaluation_seconds,
        total_seconds = selection_seconds + refit_seconds + evaluation_seconds,
        y_center = outer.standardizer.y_center,
        y_scale = outer.standardizer.y_scale,
        checkpoint = relpath(checkpoint_path, config.output_root),
        scalar_metrics(metrics)...,
    )
    atomic_jld2_write(checkpoint_path, (
        complete = true,
        schema_version = CHECKPOINT_SCHEMA_VERSION,
        phase = "full",
        config_fingerprint = config_fingerprint(config),
        selection_fingerprint = selected_fingerprint(config, selected),
        dataset = dataset.key,
        dataset_name = dataset.name,
        dataset_index = dataset.index,
        n_observations = size(dataset.features, 1),
        n_features = size(dataset.features, 2),
        hidden_units = config.hidden_units,
        split_id = split_id,
        split_spec = split_spec_record(split_spec),
        learning_rate = Float64(selected.learning_rate),
        ess_multiplier = Int(selected.ess_multiplier),
        fixed_ivon = FIXED_IVON,
        seeds = seeds,
        n_train = length(split_spec.train_indices),
        n_test = length(split_spec.test_indices),
        standardizer = outer.standardizer,
        test_features_standardized = outer.x_test_standardized,
        test_targets_standardized = outer.y_test_standardized,
        test_targets_original = outer.y_test,
        model_parameters_mean = refit.state.parameters,
        model_states = refit.state.states,
        optimizer_state = refit.state.optimizer_state,
        training_step = refit.state.step,
        ess = refit.ess,
        best_epoch = selection.best_epoch,
        stopped_early = selection.stopped_early,
        selection_history = selection.history,
        refit_losses = refit.losses,
        metrics = scalar_metrics(metrics),
        posterior_components = components,
        prediction_digest = component_digest(components),
        run_row = row,
    ))
    final_checkpoint_valid(
        checkpoint_path, config, dataset.key, split_id, selected,
    ) || error("final checkpoint round-trip validation failed: $checkpoint_path")
    return row
end

"""
    replay_checkpoint(path)

Recompute the 20-component posterior mixture and its metrics from a saved final
checkpoint. A digest mismatch is an error.
"""
function replay_checkpoint(path::AbstractString)
    checkpoint_basic_valid(path; phase = "full") ||
        error("invalid IVON UCI checkpoint: $path")
    saved = JLD2.load(path)
    n_training = Int(saved["n_train"])
    learning_rate = Float64(saved["learning_rate"])
    ess_multiplier = Int(saved["ess_multiplier"])
    model = build_model(Int(saved["n_features"]);
        hidden_units = Int(saved["hidden_units"]))
    optimizer = ivon_optimizer(learning_rate, n_training, ess_multiplier)
    state = (
        parameters = saved["model_parameters_mean"],
        states = saved["model_states"],
        optimizer_state = saved["optimizer_state"],
        step = Int(saved["training_step"]),
    )
    n_samples = size(saved["posterior_components"].means, 1)
    components = posterior_components(
        model,
        optimizer,
        state,
        saved["test_features_standardized"];
        n_samples,
        seed = Int(saved["seeds"].test),
    )
    digest = component_digest(components)
    digest == saved["prediction_digest"] || error("checkpoint prediction replay mismatch")
    metrics = predictive_metrics(
        components,
        saved["test_targets_standardized"],
        saved["test_targets_original"],
        saved["standardizer"],
    )
    for name in METRIC_NAMES
        isapprox(getproperty(metrics, name), getproperty(saved["metrics"], name);
            rtol = 1e-12, atol = 1e-12) ||
            error("checkpoint metric replay mismatch for $name")
    end
    return (components = components, metrics = metrics, digest = digest)
end

function successful_checkpoint_rows(config::UCIConfig, selected)
    rows = NamedTuple[]
    for key in config.datasets, split_id in 1:config.n_splits
        path = final_checkpoint_path(config, key, split_id)
        final_checkpoint_valid(path, config, key, split_id, selected) || continue
        saved_row = JLD2.load(path)["run_row"]
        replay = replay_checkpoint(path)
        push!(rows, merge(saved_row, scalar_metrics(replay.metrics)))
    end
    sort!(rows; by = row -> (findfirst(==(row.dataset), config.datasets), row.split))
    return rows
end

function summary_table(runs::DataFrame, config::UCIConfig)
    isempty(runs) && return DataFrame()
    successful = filter(row -> string(row.status) == "success", runs)
    rows = NamedTuple[]
    for key in config.datasets
        matching = filter(row -> string(row.dataset) == key, successful)
        isempty(matching) && continue
        pairs = Pair{Symbol,Any}[]
        for metric in METRIC_NAMES
            values = Float64.(matching[!, metric])
            push!(pairs, Symbol(metric, "_mean") => mean(values))
            standard_deviation = length(values) > 1 ? std(values; corrected = true) : 0.0
            push!(pairs, Symbol(metric, "_std") => standard_deviation)
            push!(pairs, Symbol(metric, "_se") => standard_deviation / sqrt(length(values)))
        end
        _, index = dataset_spec(key)
        push!(rows, (;
            dataset = key,
            dataset_name = DATASET_REGISTRY[index].name,
            n = nrow(matching),
            pairs...,
        ))
    end
    return DataFrame(rows)
end

format_ci(mean_value, standard_error) =
    @sprintf("%.4f ± %.4f", mean_value, 1.96 * standard_error)

function write_markdown_table(summary_table::DataFrame, config::UCIConfig)
    path = joinpath(config.output_root, "table.md")
    mkpath(dirname(path))
    temporary = path * ".tmp-" * string(getpid())
    try
        open(temporary, "w") do io
            println(io, "| Dataset | Runs | LPD original (95% CI) | RMSE (95% CI) | LPD standardized (95% CI) | Coverage 95% |")
            println(io, "|---|---:|---:|---:|---:|---:|")
            for row in eachrow(summary_table)
                println(io,
                    "| $(row.dataset_name) | $(row.n) | " *
                    "$(format_ci(row.lpd_original_mean, row.lpd_original_se)) | " *
                    "$(format_ci(row.rmse_original_mean, row.rmse_original_se)) | " *
                    "$(format_ci(row.lpd_standardized_mean, row.lpd_standardized_se)) | " *
                    "$(format_ci(row.coverage_95_mean, row.coverage_95_se)) |")
            end
        end
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return path
end

function summarize(; config::UCIConfig = UCIConfig(), require_selection::Bool = true)
    selected = require_selection ? load_pilot_selection(config) : nothing
    rows = require_selection ? successful_checkpoint_rows(config, selected) : NamedTuple[]
    runs = DataFrame(rows)
    atomic_csv_write(joinpath(config.output_root, "runs.csv"), runs)
    summary = summary_table(runs, config)
    atomic_csv_write(joinpath(config.output_root, "summary.csv"), summary)
    write_markdown_table(summary, config)
    return (runs = runs, summary = summary)
end

function validate_final_rows(
    runs::DataFrame,
    config::UCIConfig = UCIConfig();
    require_complete::Bool = true,
)
    required = (
        :dataset, :split, :status, :split_protocol, :outer_split_seed,
        :model_seed, :batch_seed, :validation_seed, :test_seed,
        :lpd_original, :rmse_original, :mean_total_variance_original,
    )
    all(name -> name in propertynames(runs), required) ||
        throw(ArgumentError("final run table is missing required columns"))
    successful = filter(row -> string(row.status) == "success", runs)
    for key in config.datasets
        matches = filter(row -> string(row.dataset) == key, successful)
        expected_count = require_complete ? config.n_splits : nrow(matches)
        nrow(matches) == expected_count ||
            error("dataset $key has $(nrow(matches)) successful rows, expected $expected_count")
        _, dataset_index = dataset_spec(key)
        for row in eachrow(matches)
            split_id = Int(row.split)
            string(row.split_protocol) == UCI_SPLIT_PROTOCOL_VERSION ||
                error("unexpected split protocol for $key split $split_id")
            Int(row.outer_split_seed) == config.split_base_seed + split_id - 1 ||
                error("unexpected outer split seed for $key split $split_id")
            seeds = model_and_evaluation_seeds(
                dataset_index, split_id; model_base_seed = config.model_base_seed,
            )
            all((
                Int(row.model_seed) == seeds.model,
                Int(row.batch_seed) == seeds.batches,
                Int(row.validation_seed) == seeds.validation,
                Int(row.test_seed) == seeds.test,
            )) || error("unexpected model/evaluation seeds for $key split $split_id")
            all(isfinite, (Float64(row.lpd_original), Float64(row.rmse_original),
                Float64(row.mean_total_variance_original))) ||
                error("non-finite final metric for $key split $split_id")
            Float64(row.mean_total_variance_original) > 0 ||
                error("non-positive final predictive variance for $key split $split_id")
        end
    end
    require_complete && nrow(successful) == length(config.datasets) * config.n_splits ||
        (!require_complete || error("final run table contains duplicate successful rows"))
    return true
end

function run_full(; force::Bool = false, config::UCIConfig = UCIConfig())
    validate_config(config)
    Set(config.datasets) == Set(DATASET_KEYS) && config.n_splits == 20 ||
        error("the complete benchmark is fixed to six datasets and 20 splits")
    selected = load_pilot_selection(config)
    initialize_results(config; selected)
    runs_path = joinpath(config.output_root, "runs.csv")
    runs = dataframe_or_empty(runs_path)
    total = length(config.datasets) * config.n_splits
    cell = 0
    for key in config.datasets
        dataset = load_dataset(key)
        for split_id in 1:config.n_splits
            cell += 1
            path = final_checkpoint_path(config, key, split_id)
            if force
                for matching_path in (
                    path,
                    final_history_path(config, key, split_id),
                    refit_history_path(config, key, split_id),
                )
                    isfile(matching_path) && rm(matching_path; force = true)
                end
            end
            if !force && final_checkpoint_valid(path, config, key, split_id, selected)
                row = JLD2.load(path)["run_row"]
                runs = replace_matching_row(runs, row, (:dataset, :split))
                atomic_csv_write(runs_path, runs)
                println("[$cell/$total] skip $(dataset.name) split $split_id")
                continue
            end
            println("[$cell/$total] $(dataset.name) split $split_id")
            split_spec = split_for(dataset, split_id, config)
            seeds = model_and_evaluation_seeds(
                dataset.index, split_id; model_base_seed = config.model_base_seed,
            )
            row = try
                run_final_configuration(dataset, split_id, config, selected)
            catch error_value
                @error "IVON UCI final fit failed" dataset = key split_id exception =
                    (error_value, catch_backtrace())
                final_failure_row(dataset, split_spec, selected, seeds, error_value)
            end
            runs = replace_matching_row(runs, row, (:dataset, :split))
            atomic_csv_write(runs_path, runs)
        end
    end
    artifacts = summarize(; config)
    validate_final_rows(artifacts.runs, config; require_complete = true)
    println("IVON UCI benchmark complete: 120 successful fits in $(config.output_root)")
    return artifacts
end

function run_smoke(; force::Bool = false)
    config = UCIConfig(
        datasets = ["yacht"],
        n_splits = 1,
        max_epochs = 2,
        min_epochs = 1,
        validation_every = 1,
        patience = 2,
        eval_samples = 3,
        output_root = joinpath(RESULTS_ROOT, "smoke"),
        show_progress = true,
    )
    selected = (learning_rate = 0.01, ess_multiplier = 1)
    initialize_results(config; selected)
    dataset = load_dataset("yacht")
    path = final_checkpoint_path(config, "yacht", 1)
    if force
        for matching_path in (
            path,
            final_history_path(config, "yacht", 1),
            refit_history_path(config, "yacht", 1),
        )
            isfile(matching_path) && rm(matching_path; force = true)
        end
    end
    if !force && final_checkpoint_valid(path, config, "yacht", 1, selected)
        println("Smoke checkpoint already complete: $path")
        return JLD2.load(path)["run_row"]
    end
    row = run_final_configuration(dataset, 1, config, selected)
    println("Smoke complete: $path")
    return row
end

function usage(io::IO = stdout)
    println(io, """
    IVON UCI benchmark (BBB-compatible homoscedastic protocol)

    Usage:
      julia --project=benchmarks/ivon benchmarks/ivon/run_uci.jl smoke [--force]
      julia --project=benchmarks/ivon benchmarks/ivon/run_uci.jl pilot [--force]
      julia --project=benchmarks/ivon benchmarks/ivon/run_uci.jl full [--force]
      julia --project=benchmarks/ivon benchmarks/ivon/run_uci.jl all [--force]
      julia --project=benchmarks/ivon benchmarks/ivon/run_uci.jl summarize
    """)
end

function main(args = ARGS)
    isempty(args) && (usage(stderr); return 2)
    command = first(args)
    flags = Set(args[2:end])
    unknown = setdiff(flags, Set(["--force"]))
    isempty(unknown) || error("unknown flags: $(join(unknown, ", "))")
    force = "--force" in flags
    if command == "smoke"
        run_smoke(; force)
    elseif command == "pilot"
        run_pilot(; force)
    elseif command == "full"
        run_full(; force)
    elseif command == "all"
        run_pilot(; force)
        run_full(; force)
    elseif command == "summarize"
        isempty(flags) || error("summarize does not accept --force")
        summarize()
    elseif command in ("help", "--help", "-h")
        usage()
    else
        usage(stderr)
        error("unknown command: $command")
    end
    return 0
end

end # module IVONUCI

if abspath(PROGRAM_FILE) == @__FILE__
    exit(IVONUCI.main(ARGS))
end
