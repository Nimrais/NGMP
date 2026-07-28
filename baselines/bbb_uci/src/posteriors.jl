const BBB_POSTERIOR_SCHEMA_VERSION = "bbb-uci-posterior-v1"

function split_spec_record(spec::UCISplitSpec)
    return (
        protocol_version = spec.protocol_version,
        split_id = spec.split_id,
        n_observations = spec.n_observations,
        outer_seed = spec.outer_seed,
        inner_seed = spec.inner_seed,
        test_fraction = spec.test_fraction,
        validation_fraction = spec.validation_fraction,
        train_indices = spec.train_indices,
        test_indices = spec.test_indices,
        inner_train_indices = spec.inner_train_indices,
        validation_indices = spec.validation_indices,
    )
end

function posterior_prediction_config(config::BBBConfig)
    return (
        hidden_units = config.hidden_units,
        homo_log_variance = config.homo_log_variance,
        minimum_log_variance = config.minimum_log_variance,
        maximum_log_variance = config.maximum_log_variance,
        eval_samples = config.eval_samples,
    )
end

function bbb_posterior_record(
    dataset,
    prepared,
    likelihood::String,
    posterior_params,
    selection,
    seeds,
    metrics,
    config::BBBConfig,
)
    return (
        schema_version = BBB_POSTERIOR_SCHEMA_VERSION,
        split_protocol_version = UCI_SPLIT_PROTOCOL_VERSION,
        method = "Bayes by Backprop (sampled free energy)",
        method_reference =
            "Blundell et al., PMLR 37:1613-1622 (2015)",
        dataset = dataset.key,
        dataset_name = dataset.display_name,
        n_observations = size(dataset.features, 1),
        n_features = size(dataset.features, 2),
        split_id = prepared.split_spec.split_id,
        split_spec = split_spec_record(prepared.split_spec),
        likelihood = likelihood,
        posterior_params = posterior_params,
        standardizer = prepared.outer.standardizer,
        prediction_config = posterior_prediction_config(config),
        training_config = config_dictionary(config),
        best_epoch = selection.best_epoch,
        model_seed = seeds.model,
        test_seed = seeds.test,
        test_metrics = scalar_metrics(metrics),
    )
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

function _require_checkpoint_keys(data::AbstractDict, required)
    missing_keys = filter(key -> !haskey(data, key), required)
    isempty(missing_keys) || throw(ArgumentError(
        "posterior checkpoint is missing keys: $(join(missing_keys, ", "))",
    ))
    return nothing
end

"""
    load_posterior_checkpoint(path)

Load and validate a versioned BBB posterior checkpoint.
"""
function load_posterior_checkpoint(path::AbstractString)
    isfile(path) || throw(ArgumentError("posterior checkpoint does not exist: $path"))
    data = JLD2.load(path)
    required = (
        "schema_version",
        "split_protocol_version",
        "dataset",
        "dataset_name",
        "n_observations",
        "n_features",
        "split_id",
        "split_spec",
        "likelihood",
        "posterior_params",
        "standardizer",
        "prediction_config",
        "training_config",
        "best_epoch",
        "model_seed",
        "test_seed",
        "test_metrics",
    )
    _require_checkpoint_keys(data, required)
    data["schema_version"] == BBB_POSTERIOR_SCHEMA_VERSION ||
        throw(ArgumentError(
            "unsupported BBB posterior schema $(data["schema_version"])",
        ))
    data["split_protocol_version"] == UCI_SPLIT_PROTOCOL_VERSION ||
        throw(ArgumentError(
            "unsupported UCI split protocol $(data["split_protocol_version"])",
        ))
    data["likelihood"] in ("homoscedastic", "heteroscedastic") ||
        throw(ArgumentError("checkpoint has an invalid likelihood"))

    return (
        path = abspath(path),
        schema_version = data["schema_version"],
        split_protocol_version = data["split_protocol_version"],
        dataset = String(data["dataset"]),
        dataset_name = String(data["dataset_name"]),
        n_observations = Int(data["n_observations"]),
        n_features = Int(data["n_features"]),
        split_id = Int(data["split_id"]),
        split_spec = data["split_spec"],
        likelihood = String(data["likelihood"]),
        posterior_params = data["posterior_params"],
        standardizer = data["standardizer"],
        prediction_config = data["prediction_config"],
        training_config = data["training_config"],
        best_epoch = Int(data["best_epoch"]),
        model_seed = Int(data["model_seed"]),
        test_seed = Int(data["test_seed"]),
        test_metrics = data["test_metrics"],
    )
end

_resolve_posterior(checkpoint::AbstractString) =
    load_posterior_checkpoint(checkpoint)
_resolve_posterior(checkpoint::NamedTuple) = checkpoint

function _prediction_bbb_config(checkpoint, n_samples::Int)
    prediction = checkpoint.prediction_config
    return BBBConfig(
        datasets = [checkpoint.dataset],
        n_splits = 1,
        likelihoods = [checkpoint.likelihood],
        hidden_units = Int(prediction.hidden_units),
        homo_log_variance = Float64(prediction.homo_log_variance),
        minimum_log_variance = Float64(prediction.minimum_log_variance),
        maximum_log_variance = Float64(prediction.maximum_log_variance),
        eval_samples = n_samples,
        show_progress = false,
        make_plot = false,
        save_checkpoints = false,
    )
end

"""
    predict_posterior(checkpoint_or_path, features; n_samples, seed)

Evaluate a saved BBB posterior on original-unit feature rows. The result
contains both the standardized finite-mixture components and their
original-target-unit transformation.
"""
function predict_posterior(
    checkpoint_or_path,
    features::AbstractMatrix;
    n_samples::Union{Nothing, Int} = nothing,
    seed::Union{Nothing, Int} = nothing,
)
    checkpoint = _resolve_posterior(checkpoint_or_path)
    size(features, 2) == checkpoint.n_features ||
        throw(DimensionMismatch("features and posterior checkpoint disagree"))
    sample_count = isnothing(n_samples) ?
        Int(checkpoint.prediction_config.eval_samples) : n_samples
    sample_count >= 1 || throw(ArgumentError("n_samples must be positive"))
    prediction_seed = isnothing(seed) ? checkpoint.test_seed : seed

    standardized_features = transform_uci_features(
        features, checkpoint.standardizer,
    )
    config = _prediction_bbb_config(checkpoint, sample_count)
    samples_standardized = predictive_samples(
        checkpoint.posterior_params,
        standardized_features,
        checkpoint.likelihood,
        config;
        seed = prediction_seed,
        n_samples = sample_count,
    )
    moments = predictive_moments(samples_standardized)
    y_center = checkpoint.standardizer.y_center
    y_scale = checkpoint.standardizer.y_scale

    return (
        dataset = checkpoint.dataset,
        split_id = checkpoint.split_id,
        likelihood = checkpoint.likelihood,
        seed = prediction_seed,
        n_samples = sample_count,
        samples_standardized = samples_standardized,
        component_means_original =
            y_center .+ y_scale .* samples_standardized.means,
        component_variances_original =
            y_scale^2 .* samples_standardized.variances,
        predictive_mean_original =
            y_center .+ y_scale .* moments.mean,
        epistemic_variance_original =
            y_scale^2 .* moments.epistemic_variance,
        aleatoric_variance_original =
            y_scale^2 .* moments.aleatoric_variance,
        total_variance_original =
            y_scale^2 .* moments.total_variance,
    )
end

"""
    predict_holdout(checkpoint_or_path; n_samples, seed)

Reload the checkpoint's UCI dataset, validate its recorded split, and replay
posterior prediction on exactly the saved outer holdout rows.
"""
function predict_holdout(
    checkpoint_or_path;
    n_samples::Union{Nothing, Int} = nothing,
    seed::Union{Nothing, Int} = nothing,
)
    checkpoint = _resolve_posterior(checkpoint_or_path)
    dataset = load_dataset(checkpoint.dataset)
    size(dataset.features) ==
        (checkpoint.n_observations, checkpoint.n_features) ||
        throw(DimensionMismatch("dataset and posterior checkpoint disagree"))
    split_spec = checkpoint.split_spec
    Int(split_spec.split_id) == checkpoint.split_id ||
        throw(ArgumentError("checkpoint split identifiers disagree"))
    Int(split_spec.n_observations) == checkpoint.n_observations ||
        throw(ArgumentError("checkpoint split row count disagrees"))
    test_indices = Int.(split_spec.test_indices)
    all(index -> 1 <= index <= checkpoint.n_observations, test_indices) ||
        throw(ArgumentError("checkpoint contains invalid holdout indices"))

    features = dataset.features[test_indices, :]
    targets_original = dataset.targets[test_indices]
    prediction = predict_posterior(
        checkpoint,
        features;
        n_samples = n_samples,
        seed = seed,
    )
    targets_standardized = transform_uci_targets(
        targets_original, checkpoint.standardizer,
    )
    metrics = predictive_metrics(
        prediction.samples_standardized,
        targets_standardized,
        targets_original,
        checkpoint.standardizer,
    )
    return merge(prediction, (
        test_indices = test_indices,
        targets_original = targets_original,
        metrics = metrics,
    ))
end
