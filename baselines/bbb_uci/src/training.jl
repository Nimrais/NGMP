function model_and_evaluation_seeds(
    config::BBBConfig,
    dataset_index::Int,
    split_id::Int,
    likelihood::String,
)
    mode_offset = likelihood == "homoscedastic" ? 1 : 2
    model_seed =
        config.model_seed + 10_000 * dataset_index + 100 * split_id + mode_offset
    return (
        model = model_seed,
        batches = model_seed + 10,
        validation = model_seed + 20,
        test = model_seed + 30,
    )
end

function gradient_step(
    params,
    optimizer_state,
    features::AbstractMatrix,
    targets::AbstractVector,
    likelihood::String,
    config::BBBConfig,
    n_training::Int,
    rng::AbstractRNG,
)
    epsilon_samples = [
        sample_epsilon(params, rng) for _ in 1:config.train_samples
    ]
    loss, gradients = Zygote.withgradient(params) do candidate
        elbo_loss(
            candidate,
            epsilon_samples,
            features,
            targets,
            likelihood,
            config,
            n_training,
        )
    end
    gradient = clip_gradient(gradients[1], config.gradient_clip)
    optimizer_state, params = Optimisers.update(
        optimizer_state, params, gradient,
    )
    isfinite(loss) || throw(ErrorException("non-finite ELBO loss"))
    parameters_are_finite(params) ||
        throw(ErrorException("non-finite model parameters"))
    return params, optimizer_state, Float64(loss)
end

function train_epoch(
    params,
    optimizer_state,
    features::AbstractMatrix,
    targets::AbstractVector,
    likelihood::String,
    config::BBBConfig,
    rng::AbstractRNG,
)
    n_training = size(features, 1)
    order = randperm(rng, n_training)
    weighted_loss = 0.0
    for first_index in 1:config.batch_size:n_training
        last_index = min(first_index + config.batch_size - 1, n_training)
        batch_indices = @view order[first_index:last_index]
        x_batch = @view features[batch_indices, :]
        y_batch = @view targets[batch_indices]
        params, optimizer_state, loss = gradient_step(
            params,
            optimizer_state,
            x_batch,
            y_batch,
            likelihood,
            config,
            n_training,
            rng,
        )
        weighted_loss += length(batch_indices) * loss
    end
    return params, optimizer_state, weighted_loss / n_training
end

function evaluate_standardized_model(
    params,
    features::AbstractMatrix,
    targets_standardized::AbstractVector,
    targets_original::AbstractVector,
    standardizer,
    likelihood::String,
    config::BBBConfig;
    seed::Int,
)
    samples = predictive_samples(
        params,
        features,
        likelihood,
        config;
        seed = seed,
        n_samples = config.eval_samples,
    )
    return predictive_metrics(
        samples, targets_standardized, targets_original, standardizer,
    ), samples
end

"""
    train_with_validation(inner_split, input_dimension, likelihood, config; seed)

Fit on the inner training partition and select the epoch with highest
original-unit validation LPD. The same fixed Monte Carlo draws are reused at
each validation checkpoint. Patience is counted in validation checks.
"""
function train_with_validation(
    inner_split,
    input_dimension::Int,
    likelihood::String,
    config::BBBConfig;
    seed::Int,
    validation_seed::Int = seed + 1,
)
    params = initialize_model(input_dimension, likelihood, config; seed = seed)
    optimizer_state = Optimisers.setup(
        Optimisers.Adam(config.learning_rate), params,
    )
    rng = StableRNG(seed + 10)
    history = NamedTuple[]
    best_lpd = -Inf
    best_epoch = 0
    best_params = nothing
    non_improving_checks = 0
    stopped_early = false

    for epoch in 1:config.max_epochs
        params, optimizer_state, training_loss = train_epoch(
            params,
            optimizer_state,
            inner_split.x_train_standardized,
            inner_split.y_train_standardized,
            likelihood,
            config,
            rng,
        )

        should_validate =
            epoch % config.validation_every == 0 || epoch == config.max_epochs
        should_validate || continue
        validation_metrics, _ = evaluate_standardized_model(
            params,
            inner_split.x_test_standardized,
            inner_split.y_test_standardized,
            inner_split.y_test,
            inner_split.standardizer,
            likelihood,
            config;
            seed = validation_seed,
        )
        push!(history, (
            epoch = epoch,
            training_elbo_loss = training_loss,
            validation_lpd_original = validation_metrics.lpd_original,
            validation_lpd_standardized = validation_metrics.lpd_standardized,
            validation_rmse_original = validation_metrics.rmse_original,
        ))

        if config.show_progress
            @printf(
                "    epoch %4d  loss %.5f  val LPD %.5f  val RMSE %.5f\n",
                epoch,
                training_loss,
                validation_metrics.lpd_original,
                validation_metrics.rmse_original,
            )
        end

        epoch < config.min_epochs && continue
        if isfinite(validation_metrics.lpd_original) &&
           validation_metrics.lpd_original > best_lpd
            best_lpd = validation_metrics.lpd_original
            best_epoch = epoch
            best_params = deepcopy(params)
            non_improving_checks = 0
        else
            non_improving_checks += 1
        end
        if non_improving_checks >= config.patience
            stopped_early = true
            break
        end
    end

    best_epoch > 0 ||
        throw(ErrorException("validation did not produce a finite model"))
    return (
        best_epoch = best_epoch,
        best_lpd = best_lpd,
        best_params = best_params,
        history = DataFrame(history),
        stopped_early = stopped_early,
    )
end

"""
    refit_model(outer_split, input_dimension, likelihood, config; seed, epochs)

Reinitialize from the same model seed and train on all outer-training rows for
the epoch count selected by inner validation.
"""
function refit_model(
    outer_split,
    input_dimension::Int,
    likelihood::String,
    config::BBBConfig;
    seed::Int,
    epochs::Int,
)
    epochs >= 1 || throw(ArgumentError("epochs must be positive"))
    params = initialize_model(input_dimension, likelihood, config; seed = seed)
    optimizer_state = Optimisers.setup(
        Optimisers.Adam(config.learning_rate), params,
    )
    rng = StableRNG(seed + 10)
    losses = Vector{Float64}(undef, epochs)
    for epoch in 1:epochs
        params, optimizer_state, losses[epoch] = train_epoch(
            params,
            optimizer_state,
            outer_split.x_train_standardized,
            outer_split.y_train_standardized,
            likelihood,
            config,
            rng,
        )
    end
    return (params = params, losses = losses)
end
