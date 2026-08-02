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

struct DVINumericalError <: Exception
    kind::String
    diagnostics::NamedTuple
end

function Base.showerror(io::IO, error::DVINumericalError)
    diagnostics = error.diagnostics
    print(
        io,
        error.kind,
        " during ",
        diagnostics.phase,
        " at epoch ",
        diagnostics.epoch,
        ", batch ",
        diagnostics.batch,
        ", optimizer step ",
        diagnostics.optimizer_step,
        "; first non-finite tensor: ",
        diagnostics.first_nonfinite_tensor,
    )
end

function first_nonfinite_path(value, path::String)
    value === nothing && return nothing
    if value isa NamedTuple
        for name in keys(value)
            found = first_nonfinite_path(
                getfield(value, name), isempty(path) ? string(name) :
                    "$path.$name",
            )
            isnothing(found) || return found
        end
        return nothing
    elseif value isa AbstractArray
        index = findfirst(element -> !isfinite(element), value)
        return isnothing(index) ? nothing : "$path[$index]"
    elseif value isa Number
        return isfinite(value) ? nothing : path
    end
    return nothing
end

function finite_extrema(values)
    finite_values = filter(isfinite, vec(Float64.(values)))
    isempty(finite_values) && return (minimum = NaN, maximum = NaN)
    return (
        minimum = minimum(finite_values),
        maximum = maximum(finite_values),
    )
end

function numerical_diagnostics(
    params,
    features::AbstractMatrix,
    likelihood::String,
    config::DVIConfig;
    phase::String,
    epoch::Int,
    batch::Int,
    optimizer_step::Int,
    loss = NaN,
    gradient = nothing,
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    first_nonfinite = first_nonfinite_path(gradient, "gradient")
    isnothing(first_nonfinite) &&
        (first_nonfinite = first_nonfinite_path(params, "parameters"))
    output_error = ""
    output_mean_extrema = (minimum = NaN, maximum = NaN)
    output_variance_extrema = (minimum = NaN, maximum = NaN)
    log_variance_extrema = (minimum = NaN, maximum = NaN)
    log_variance_variance_extrema = (minimum = NaN, maximum = NaN)
    precision_exponent_extrema = (minimum = NaN, maximum = NaN)
    try
        output = propagate_dvi(params, features, config)
        output_mean_extrema = finite_extrema(output.mean)
        output_variance = config.propagation == "full" ?
            covariance_diagonal(output.covariance) : output.variance
        output_variance_extrema = finite_extrema(output_variance)
        output_nonfinite = first_nonfinite_path(output, "output")
        isnothing(first_nonfinite) && (first_nonfinite = output_nonfinite)
        if likelihood == "heteroscedastic"
            log_variance = vec(output.mean[:, 2])
            log_variance_variance = output_covariance_entry(
                output, 2, 2, config,
            )
            log_variance_extrema = finite_extrema(log_variance)
            log_variance_variance_extrema = finite_extrema(
                log_variance_variance,
            )
            precision_exponent_extrema = finite_extrema(
                -log_variance .+ 0.5f0 .* log_variance_variance,
            )
        end
    catch error
        output_error = sprint(showerror, error)
        isnothing(first_nonfinite) && (first_nonfinite = "forward-pass-error")
    end
    log_stds = vcat(
        vec(params.hidden.weight_log_std),
        vec(params.hidden.bias_log_std),
        vec(params.output.weight_log_std),
        vec(params.output.bias_log_std),
    )
    log_std_extrema = finite_extrema(log_stds)
    tracker_state = tracker === nothing ? (
        clamp_count = 0,
        clamp_rate = 0.0,
    ) : tracker_record(tracker)
    if isnothing(first_nonfinite) && !isfinite(loss)
        first_nonfinite = "loss"
    end
    return (
        phase = phase,
        epoch = epoch,
        batch = batch,
        optimizer_step = optimizer_step,
        loss = Float64(loss),
        first_nonfinite_tensor = something(first_nonfinite, "unknown"),
        gradient_maximum_absolute = gradient === nothing ? NaN :
            Float64(gradient_maximum_absolute(gradient)),
        safe_exp_clamp_count = tracker_state.clamp_count,
        safe_exp_clamp_rate = tracker_state.clamp_rate,
        posterior_log_std_minimum = log_std_extrema.minimum,
        posterior_log_std_maximum = log_std_extrema.maximum,
        output_mean_minimum = output_mean_extrema.minimum,
        output_mean_maximum = output_mean_extrema.maximum,
        output_variance_minimum = output_variance_extrema.minimum,
        output_variance_maximum = output_variance_extrema.maximum,
        log_variance_minimum = log_variance_extrema.minimum,
        log_variance_maximum = log_variance_extrema.maximum,
        log_variance_variance_minimum =
            log_variance_variance_extrema.minimum,
        log_variance_variance_maximum =
            log_variance_variance_extrema.maximum,
        precision_exponent_minimum = precision_exponent_extrema.minimum,
        precision_exponent_maximum = precision_exponent_extrema.maximum,
        output_error = output_error,
    )
end

function throw_numerical_error(
    kind::String,
    params,
    features,
    likelihood,
    config;
    kwargs...,
)
    throw(DVINumericalError(
        kind,
        numerical_diagnostics(
            params, features, likelihood, config; kwargs...,
        ),
    ))
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
    optimizer_step::Int;
    batch::Int,
    phase::String,
    tracker::NumericalTracker,
)
    next_optimizer_step = optimizer_step + 1
    loss, gradients = Zygote.withgradient(params) do candidate
        dvi_loss(
            candidate,
            features,
            targets,
            likelihood,
            config,
            n_training;
            epoch = epoch,
            optimizer_step = next_optimizer_step,
            tracker = tracker,
        )
    end
    isfinite(loss) || throw_numerical_error(
        "non-finite DVI loss",
        params,
        features,
        likelihood,
        config;
        phase = phase,
        epoch = epoch,
        batch = batch,
        optimizer_step = next_optimizer_step,
        loss = loss,
        gradient = gradients[1],
        tracker = tracker,
    )
    parameters_are_finite(gradients[1]) || throw_numerical_error(
        "non-finite DVI gradient",
        params,
        features,
        likelihood,
        config;
        phase = phase,
        epoch = epoch,
        batch = batch,
        optimizer_step = next_optimizer_step,
        loss = loss,
        gradient = gradients[1],
        tracker = tracker,
    )
    gradient = clip_gradient(gradients[1], config.gradient_clip)
    optimizer_state, params = Optimisers.update(
        optimizer_state, params, gradient,
    )
    parameters_are_finite(params) || throw_numerical_error(
        "non-finite model parameters",
        params,
        features,
        likelihood,
        config;
        phase = phase,
        epoch = epoch,
        batch = batch,
        optimizer_step = next_optimizer_step,
        loss = loss,
        gradient = gradient,
        tracker = tracker,
    )
    return params, optimizer_state, Float64(loss), next_optimizer_step
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
    optimizer_step::Int;
    phase::String,
    tracker::NumericalTracker,
)
    n_training = size(features, 1)
    order = randperm(rng, n_training)
    weighted_loss = 0.0
    for (batch, first_index) in enumerate(1:config.batch_size:n_training)
        last_index = min(first_index + config.batch_size - 1, n_training)
        batch_indices = @view order[first_index:last_index]
        params, optimizer_state, loss, optimizer_step = gradient_step(
            params,
            optimizer_state,
            @view(features[batch_indices, :]),
            @view(targets[batch_indices]),
            likelihood,
            config,
            n_training,
            epoch,
            optimizer_step;
            batch = batch,
            phase = phase,
            tracker = tracker,
        )
        weighted_loss += length(batch_indices) * loss
    end
    return params, optimizer_state, weighted_loss / n_training, optimizer_step
end

function evaluate_standardized_model(
    params,
    features::AbstractMatrix,
    targets_standardized::AbstractVector,
    targets_original::AbstractVector,
    standardizer,
    likelihood::String,
    config::DVIConfig,
    tracker::Union{Nothing, NumericalTracker} = nothing,
)
    return predictive_metrics(
        params,
        features,
        targets_standardized,
        targets_original,
        standardizer,
        likelihood,
        config,
        tracker,
    )
end

function earliest_selection_epoch(config::DVIConfig, n_training::Int)
    schedule_epoch = if config.kl_schedule_unit == "steps"
        batches_per_epoch = cld(n_training, config.batch_size)
        cld(config.kl_warmup_steps + config.kl_anneal_steps, batches_per_epoch)
    else
        config.kl_warmup_epochs + config.kl_anneal_epochs
    end
    return max(config.min_epochs, schedule_epoch)
end

function train_with_validation(
    inner_split,
    input_dimension::Int,
    likelihood::String,
    config::DVIConfig;
    seed::Int,
    history_path::Union{Nothing, AbstractString} = nothing,
)
    initial_params = initialize_model(
        input_dimension, likelihood, config; seed = seed,
    )
    backend = initialize_training_backend(
        initial_params,
        inner_split.x_train_standardized,
        inner_split.y_train_standardized,
        likelihood,
        config,
    )
    rng = StableRNG(seed + 10)
    history = NamedTuple[]
    best_lpd = -Inf
    best_epoch = 0
    best_params = nothing
    non_improving_checks = 0
    stopped_early = false
    optimizer_step = 0
    tracker = NumericalTracker()
    selection_start_epoch = earliest_selection_epoch(
        config, size(inner_split.x_train_standardized, 1),
    )

    for epoch in 1:config.max_epochs
        backend, training_loss, optimizer_step = training_backend_epoch(
            backend,
            rng,
            epoch,
            optimizer_step;
            phase = "selection",
            tracker = tracker,
        )
        should_validate =
            epoch % config.validation_every == 0 ||
            epoch == config.max_epochs
        should_validate || continue

        validation_params = training_backend_parameters(backend)
        validation_metrics = evaluate_standardized_model(
            validation_params,
            inner_split.x_test_standardized,
            inner_split.y_test_standardized,
            inner_split.y_test,
            inner_split.standardizer,
            likelihood,
            config,
            tracker,
        )
        tracker_state = tracker_record(tracker)
        progress = kl_progress(epoch, optimizer_step, config)
        push!(history, (
            epoch = epoch,
            optimizer_step = optimizer_step,
            kl_weight = kl_weight(progress, config),
            training_dvi_loss = training_loss,
            validation_lpd_original = validation_metrics.lpd_original,
            validation_lpd_standardized =
                validation_metrics.lpd_standardized,
            validation_rmse_original = validation_metrics.rmse_original,
            safe_exp_clamp_count = tracker_state.clamp_count,
            safe_exp_clamp_rate = tracker_state.clamp_rate,
        ))
        history_path === nothing || atomic_csv_write(
            history_path, DataFrame(history),
        )

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
            best_params = deepcopy(validation_params)
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
        optimizer_steps = optimizer_step,
        numerical = tracker_record(tracker),
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
    initial_params = initialize_model(
        input_dimension, likelihood, config; seed = seed,
    )
    backend = initialize_training_backend(
        initial_params,
        outer_split.x_train_standardized,
        outer_split.y_train_standardized,
        likelihood,
        config,
    )
    rng = StableRNG(seed + 10)
    losses = Vector{Float64}(undef, epochs)
    optimizer_step = 0
    tracker = NumericalTracker()
    for epoch in 1:epochs
        backend, losses[epoch], optimizer_step = training_backend_epoch(
            backend,
            rng,
            epoch,
            optimizer_step;
            phase = "refit",
            tracker = tracker,
        )
    end
    params = training_backend_parameters(backend)
    return (
        params = params,
        losses = losses,
        optimizer_steps = optimizer_step,
        numerical = tracker_record(tracker),
    )
end
