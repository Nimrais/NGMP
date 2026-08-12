function float32_gate_data(data)
    predictions = [
        Float32.(data.predictions[i, j]) for
        i in axes(data.predictions, 1), j in axes(data.predictions, 2)
    ]
    features = [Float32.(x) for x in data.features]
    targets = [Float32.(y) for y in data.targets]
    return (; predictions, features, targets)
end

function average_gate_loss(data, gate, ps, st)
    st_test = Lux.testmode(st)
    total = 0.0
    for j in eachindex(data.features)
        logits, _ = gate(data.features[j], ps, st_test)
        probabilities = stable_softmax(vec(logits))
        losses = [sum(abs2, data.predictions[i, j] .- data.targets[j]) for
                  i in axes(data.predictions, 1)]
        total += dot(probabilities, losses)
    end
    return total / length(data.features)
end

function ivon_optimizer(learning_rate, ess)
    return IVON(
        lr = learning_rate,
        ess = Float64(ess),
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

function train_ivon_gate(fit_data, monitor_data, architecture::Symbol;
    learning_rate::Real, ess_multiplier::Integer, seed::Integer = DEFAULT_SEED,
    n_epochs::Integer = 100, patience::Integer = 1, min_delta::Real = 1.0e-3)
    fit = float32_gate_data(fit_data)
    monitor = float32_gate_data(monitor_data)
    rng = StableRNG(seed)
    gate = build_gate(architecture)
    ps, st = Lux.setup(rng, gate)
    validate_gate(architecture, gate, ps, st)
    ess = length(fit.features) * Int(ess_multiplier)
    optimizer = ivon_optimizer(learning_rate, ess)
    train_state = Lux.Training.TrainState(gate, ps, st, optimizer)
    backend = AutoEnzyme()

    best_loss = Inf
    best_epoch = 0
    best = nothing
    patience_counter = 0
    history = NamedTuple[]
    for epoch = 1:Int(n_epochs)
        epoch_loss = 0.0
        for j in eachindex(fit.features)
            item = (fit.predictions[:, j], fit.features[j], fit.targets[j])
            loss, _, train_state = ivon_train_step!(
                rng, backend, PE.moe_objective, item, train_state)
            isfinite(loss) || error("Non-finite IVON training loss at epoch $epoch")
            epoch_loss += Float64(loss)
        end
        train_loss = average_gate_loss(fit, gate, train_state.parameters,
            train_state.states)
        monitor_loss = average_gate_loss(monitor, gate, train_state.parameters,
            train_state.states)
        all(isfinite, (train_loss, monitor_loss)) ||
            error("Non-finite IVON epoch metric")
        push!(history, (; epoch, online_loss = epoch_loss / length(fit.features),
            train_loss, monitor_loss))
        @info "IVON gate epoch" architecture epoch train_loss monitor_loss ess

        if monitor_loss < best_loss - Float64(min_delta)
            best_loss = monitor_loss
            best_epoch = epoch
            best = (
                parameters = deepcopy(train_state.parameters),
                states = deepcopy(train_state.states),
                optimizer_state = deepcopy(train_state.optimizer_state),
                step = train_state.step,
            )
            patience_counter = 0
        else
            patience_counter += 1
            patience_counter >= patience && break
        end
    end
    best === nothing && error("IVON produced no valid checkpoint")
    variances = posterior_variance(optimizer, best.optimizer_state)
    posterior_variances_valid(variances) || error("Invalid IVON posterior variance")
    return (
        gate = gate,
        optimizer = optimizer,
        parameters = best.parameters,
        states = best.states,
        optimizer_state = best.optimizer_state,
        step = best.step,
        best_epoch = best_epoch,
        best_monitor_loss = best_loss,
        history = history,
        ess = ess,
        seed = Int(seed),
    )
end

posterior_variances_valid(x::AbstractArray) = all(v -> isfinite(v) && v > 0, x)
posterior_variances_valid(x::NamedTuple) = all(posterior_variances_valid, values(x))
posterior_variances_valid(x::Tuple) = all(posterior_variances_valid, x)
posterior_variances_valid(::Any) = true
