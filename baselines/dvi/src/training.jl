function model_and_batch_seeds(
    config::DVIConfig,
    dataset_index::Int,
    split_id::Int,
    likelihood::String,
)
    likelihood_offset = likelihood == "homoscedastic" ? 1 : 2
    model_seed =
        config.model_seed +
        10_000 * dataset_index +
        100 * split_id +
        likelihood_offset
    return (model = model_seed, batches = model_seed + 10)
end

function gradient_step(
    params,
    optimizer_state,
    features::AbstractMatrix,
    targets::AbstractVector,
    likelihood::String,
    config::DVIConfig,
    n_training::Int,
    epoch::Int,
)
    loss, gradients = Zygote.withgradient(params) do candidate
        dvi_loss(
            candidate,
            features,
            targets,
            likelihood,
            config,
            n_training;
            epoch = epoch,
        )
    end
    parameters_are_finite(gradients[1]) ||
        throw(ErrorException("non-finite DVI gradient"))
    gradient = clip_gradient(gradients[1], config.gradient_clip)
    optimizer_state, params = Optimisers.update(
        optimizer_state, params, gradient,
    )
    isfinite(loss) || throw(ErrorException("non-finite DVI loss"))
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
    config::DVIConfig,
    rng::AbstractRNG,
    epoch::Int,
)
    n_training = size(features, 1)
    order = randperm(rng, n_training)
    weighted_loss = 0.0
    for first_index in 1:config.batch_size:n_training
        last_index = min(first_index + config.batch_size - 1, n_training)
        batch_indices = @view order[first_index:last_index]
        params, optimizer_state, loss = gradient_step(
            params,
            optimizer_state,
            @view(features[batch_indices, :]),
            @view(targets[batch_indices]),
            likelihood,
            config,
            n_training,
            epoch,
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
    config::DVIConfig,
)
    return predictive_metrics(
        params,
        features,
        targets_standardized,
        targets_original,
        standardizer,
        likelihood,
        config,
    )
end

earliest_selection_epoch(config::DVIConfig) = max(
    config.min_epochs,
    config.kl_warmup_epochs + config.kl_anneal_epochs,
)

function train_with_validation(
    inner_split,
    input_dimension::Int,
    likelihood::String,
    config::DVIConfig;
    seed::Int,
)
    params = initialize_model(
        input_dimension, likelihood, config; seed = seed,
    )
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
    selection_start_epoch = earliest_selection_epoch(config)

    for epoch in 1:config.max_epochs
        params, optimizer_state, training_loss = train_epoch(
            params,
            optimizer_state,
            inner_split.x_train_standardized,
            inner_split.y_train_standardized,
            likelihood,
            config,
            rng,
            epoch,
        )
        should_validate =
            epoch % config.validation_every == 0 ||
            epoch == config.max_epochs
        should_validate || continue

        validation_metrics = evaluate_standardized_model(
            params,
            inner_split.x_test_standardized,
            inner_split.y_test_standardized,
            inner_split.y_test,
            inner_split.standardizer,
            likelihood,
            config,
        )
        push!(history, (
            epoch = epoch,
            kl_weight = kl_weight(epoch, config),
            training_dvi_loss = training_loss,
            validation_lpd_original = validation_metrics.lpd_original,
            validation_lpd_standardized =
                validation_metrics.lpd_standardized,
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

        epoch < selection_start_epoch && continue
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

function refit_model(
    outer_split,
    input_dimension::Int,
    likelihood::String,
    config::DVIConfig;
    seed::Int,
    epochs::Int,
)
    epochs >= 1 || throw(ArgumentError("epochs must be positive"))
    params = initialize_model(
        input_dimension, likelihood, config; seed = seed,
    )
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
            epoch,
        )
    end
    return (params = params, losses = losses)
end
