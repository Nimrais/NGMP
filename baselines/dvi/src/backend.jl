struct ZygoteTrainingBackend{P, O, F, T, C}
    params::P
    optimizer_state::O
    features::F
    targets::T
    likelihood::String
    config::C
end

const REACTANT_BACKEND_MODULE = Ref{Union{Nothing, Module}}(nothing)

function load_reactant_backend()
    loaded = REACTANT_BACKEND_MODULE[]
    loaded === nothing || return loaded
    Base.include(@__MODULE__, joinpath(@__DIR__, "reactant_backend.jl"))
    loaded = getfield(@__MODULE__, :DVIReactantBackend)
    REACTANT_BACKEND_MODULE[] = loaded
    return loaded
end

function call_reactant_backend(name::Symbol, arguments...; kwargs...)
    backend = load_reactant_backend()
    function_reference = getfield(backend, name)
    return Base.invokelatest(function_reference, arguments...; kwargs...)
end

function initialize_training_backend(
    params,
    features::AbstractMatrix,
    targets::AbstractVector,
    likelihood::String,
    config::DVIConfig,
)
    if config.execution_backend == "zygote"
        optimizer_state = Optimisers.setup(
            Optimisers.Adam(config.learning_rate), params,
        )
        return ZygoteTrainingBackend(
            params, optimizer_state, features, targets, likelihood, config,
        )
    end
    return call_reactant_backend(
        :initialize_training_backend,
        params,
        features,
        targets,
        likelihood,
        config,
    )
end

function training_backend_epoch(
    backend::ZygoteTrainingBackend,
    rng::AbstractRNG,
    epoch::Int,
    optimizer_step::Int;
    phase::String,
    tracker::NumericalTracker,
)
    params, optimizer_state, loss, next_optimizer_step = train_epoch(
        backend.params,
        backend.optimizer_state,
        backend.features,
        backend.targets,
        backend.likelihood,
        backend.config,
        rng,
        epoch,
        optimizer_step;
        phase = phase,
        tracker = tracker,
    )
    return ZygoteTrainingBackend(
        params,
        optimizer_state,
        backend.features,
        backend.targets,
        backend.likelihood,
        backend.config,
    ), loss, next_optimizer_step
end

function training_backend_epoch(
    backend,
    rng::AbstractRNG,
    epoch::Int,
    optimizer_step::Int;
    phase::String,
    tracker::NumericalTracker,
)
    return call_reactant_backend(
        :training_backend_epoch,
        backend,
        rng,
        epoch,
        optimizer_step;
        phase = phase,
        tracker = tracker,
    )
end

training_backend_parameters(backend::ZygoteTrainingBackend) = backend.params
training_backend_parameters(backend) = call_reactant_backend(
    :training_backend_parameters, backend,
)
