function model_and_evaluation_seeds(
    config::BPCConfig,
    dataset_index::Int,
    split_id::Int,
)
    model_seed = config.model_seed + 10_000 * dataset_index + 100 * split_id + 1
    return (
        model = model_seed,
        batches = model_seed + 10,
        validation = model_seed + 20,
        test = model_seed + 30,
    )
end

function device_training_data(
    features::AbstractMatrix,
    targets::AbstractVector,
    config::BPCConfig,
)
    feature_columns = Matrix{Float32}(transpose(features))
    target_columns = reshape(Float32.(targets), 1, :)
    return (
        features = to_device(feature_columns, config),
        targets = to_device(target_columns, config),
    )
end

function parameters_are_finite(model::BPCModel)
    posterior = host_model(model)
    return all(posterior.layers) do layer
        all(isfinite, layer.K) && all(isfinite, layer.H) &&
            all(isfinite, layer.G) && all(isfinite, layer.M) &&
            all(isfinite, layer.V) && all(isfinite, layer.Psi) &&
            isfinite(layer.nu)
    end
end

function natural_learning_rate(model::BPCModel, config::BPCConfig)
    step = model.update_step + 1
    return step^(-config.natural_learning_exponent)
end

function train_epoch!(
    model::BPCModel,
    features::AbstractMatrix,
    targets::AbstractMatrix,
    config::BPCConfig,
    rng::AbstractRNG,
)
    observations = size(features, 2)
    order = randperm(rng, observations)
    batches = 0
    for first_index in 1:config.batch_size:observations
        last_index = min(first_index + config.batch_size - 1, observations)
        indices = order[first_index:last_index]
        feature_batch = features[:, indices]
        target_batch = targets[:, indices]
        states = infer_latent_states(model, feature_batch, target_batch, config)
        posterior_update!(
            model,
            states,
            natural_learning_rate(model, config),
            config.minibatch_stat_scale,
        )
        batches += 1
    end
    synchronize_backend(config)
    return batches
end

function evaluate_standardized_model(
    model::BPCModel,
    features::AbstractMatrix,
    targets_standardized::AbstractVector,
    targets_original::AbstractVector,
    standardizer,
    config::BPCConfig;
    seed::Int,
)
    samples = predictive_samples(
        model, features, config; seed = seed, n_samples = config.eval_samples,
    )
    metrics = predictive_metrics(
        samples, targets_standardized, targets_original, standardizer,
    )
    return metrics, samples
end

"""
Fit BPC on the inner split and select an epoch by original-unit validation
LPD. Selection uses fixed posterior-sampling seeds at every checkpoint.
"""
function train_with_validation(
    inner_split,
    input_dimension::Int,
    config::BPCConfig;
    seed::Int,
    validation_seed::Int = seed + 1,
)
    model = initialize_model(input_dimension, config; seed = seed)
    training = device_training_data(
        inner_split.x_train_standardized,
        inner_split.y_train_standardized,
        config,
    )
    rng = StableRNG(seed + 10)
    history = NamedTuple[]
    best_lpd = -Inf
    best_epoch = 0
    non_improving_checks = 0
    stopped_early = false

    for epoch in 1:config.max_epochs
        batches = train_epoch!(
            model, training.features, training.targets, config, rng,
        )
        should_validate =
            epoch % config.validation_every == 0 || epoch == config.max_epochs
        should_validate || continue
        parameters_are_finite(model) ||
            throw(ErrorException("non-finite BPC posterior parameters"))
        validation_metrics, _ = evaluate_standardized_model(
            model,
            inner_split.x_test_standardized,
            inner_split.y_test_standardized,
            inner_split.y_test,
            inner_split.standardizer,
            config;
            seed = validation_seed,
        )
        push!(history, (
            epoch = epoch,
            posterior_updates = model.update_step,
            batches = batches,
            validation_lpd_original = validation_metrics.lpd_original,
            validation_lpd_standardized = validation_metrics.lpd_standardized,
            validation_rmse_original = validation_metrics.rmse_original,
        ))
        if config.show_progress
            @printf(
                "    epoch %4d  updates %5d  val LPD %.5f  val RMSE %.5f\n",
                epoch,
                model.update_step,
                validation_metrics.lpd_original,
                validation_metrics.rmse_original,
            )
        end

        epoch < config.min_epochs && continue
        if isfinite(validation_metrics.lpd_original) &&
           validation_metrics.lpd_original > best_lpd
            best_lpd = validation_metrics.lpd_original
            best_epoch = epoch
            non_improving_checks = 0
        else
            non_improving_checks += 1
        end
        if non_improving_checks >= config.patience
            stopped_early = true
            break
        end
    end
    best_epoch > 0 || throw(ErrorException(
        "validation did not produce a finite BPC model",
    ))
    return (
        best_epoch = best_epoch,
        best_lpd = best_lpd,
        history = DataFrame(history),
        stopped_early = stopped_early,
    )
end

"""Reinitialize and fit all outer-training rows for the selected epoch count."""
function refit_model(
    outer_split,
    input_dimension::Int,
    config::BPCConfig;
    seed::Int,
    epochs::Int,
)
    epochs >= 1 || throw(ArgumentError("epochs must be positive"))
    model = initialize_model(input_dimension, config; seed = seed)
    training = device_training_data(
        outer_split.x_train_standardized,
        outer_split.y_train_standardized,
        config,
    )
    rng = StableRNG(seed + 10)
    for _ in 1:epochs
        train_epoch!(model, training.features, training.targets, config, rng)
    end
    parameters_are_finite(model) ||
        throw(ErrorException("non-finite BPC posterior after refit"))
    return model
end
