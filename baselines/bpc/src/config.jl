const DATASET_REGISTRY = [
    ("yacht", "Yacht", Yacht),
    ("concrete", "Concrete", Concrete),
    ("energy", "Energy", EnergyEfficiency),
    ("housing", "Boston", BostonHousing),
    ("power", "Power", PowerPlant),
    ("wine", "Wine", WineQualityRed),
]

const DATASET_ALIASES = Dict(
    "yacht" => "yacht", "yach" => "yacht",
    "concrete" => "concrete", "conc" => "concrete",
    "energy" => "energy", "ener" => "energy",
    "energyefficiency" => "energy",
    "housing" => "housing", "boston" => "housing",
    "bost" => "housing", "bostonhousing" => "housing",
    "power" => "power", "powe" => "power", "powerplant" => "power",
    "wine" => "wine", "winered" => "wine", "winequalityred" => "wine",
)

env_int(key, default) = parse(Int, get(ENV, key, string(default)))
env_float(key, default) = parse(Float64, get(ENV, key, string(default)))

function env_bool(key, default)
    value = lowercase(strip(get(ENV, key, string(default))))
    value in ("1", "true", "yes", "on") && return true
    value in ("0", "false", "no", "off") && return false
    throw(ArgumentError("$key must be true/false, yes/no, on/off, or 1/0"))
end

function parse_datasets(value::AbstractString)
    normalized = lowercase(strip(value))
    normalized == "all" && return first.(DATASET_REGISTRY)
    result = String[]
    for token in split(normalized, ',')
        key = replace(strip(token), r"[^a-z0-9]" => "")
        isempty(key) && continue
        haskey(DATASET_ALIASES, key) ||
            throw(ArgumentError("unknown dataset '$token'"))
        push!(result, DATASET_ALIASES[key])
    end
    isempty(result) && throw(ArgumentError("at least one dataset is required"))
    return unique(result)
end

function default_output_directory()
    stamp = Dates.format(Dates.now(), dateformat"yyyymmddTHHMMSS")
    return normpath(joinpath(
        @__DIR__, "..", "..", "..", "results", "bpc", stamp,
    ))
end

"""Configuration for the paper-faithful homoscedastic UCI BPC baseline."""
Base.@kwdef struct BPCConfig
    datasets::Vector{String} = first.(DATASET_REGISTRY)
    n_splits::Int = UCI_DEFAULT_N_SPLITS
    hidden_units::Int = 50
    batch_size::Int = 100
    max_epochs::Int = 100
    min_epochs::Int = 3
    validation_every::Int = 1
    patience::Int = 10
    latent_steps::Int = 10
    latent_learning_rate::Float64 = 1e-2
    latent_adam_beta1::Float64 = 0.9
    latent_adam_beta2::Float64 = 0.999
    latent_adam_epsilon::Float64 = 1e-8
    natural_learning_exponent::Float64 = 0.25
    minibatch_stat_scale::Float64 = 1.0
    prior_v_scale::Float64 = 10.0
    prior_psi_scale::Float64 = 1000.0
    prior_nu_offset::Float64 = 2.0
    posterior_jitter::Float64 = 1e-5
    activate_input::Bool = true
    eval_samples::Int = 20
    test_fraction::Float64 = UCI_DEFAULT_TEST_FRACTION
    validation_fraction::Float64 = UCI_DEFAULT_VALIDATION_FRACTION
    split_seed::Int = UCI_DEFAULT_SPLIT_SEED
    model_seed::Int = 20260805
    backend::String = "cpu"
    output_dir::String = default_output_directory()
    resume::Bool = true
    save_checkpoints::Bool = true
    show_progress::Bool = true
end

function load_config()
    config = BPCConfig(
        datasets = parse_datasets(get(ENV, "BPC_DATASETS", "all")),
        n_splits = env_int("BPC_SPLITS", UCI_DEFAULT_N_SPLITS),
        hidden_units = env_int("BPC_HIDDEN_UNITS", 50),
        batch_size = env_int("BPC_BATCH_SIZE", 100),
        max_epochs = env_int("BPC_MAX_EPOCHS", 100),
        min_epochs = env_int("BPC_MIN_EPOCHS", 3),
        validation_every = env_int("BPC_VALIDATION_EVERY", 1),
        patience = env_int("BPC_PATIENCE", 10),
        latent_steps = env_int("BPC_LATENT_STEPS", 10),
        latent_learning_rate = env_float("BPC_LATENT_LEARNING_RATE", 1e-2),
        latent_adam_beta1 = env_float("BPC_LATENT_ADAM_BETA1", 0.9),
        latent_adam_beta2 = env_float("BPC_LATENT_ADAM_BETA2", 0.999),
        latent_adam_epsilon = env_float("BPC_LATENT_ADAM_EPSILON", 1e-8),
        natural_learning_exponent =
            env_float("BPC_NATURAL_LEARNING_EXPONENT", 0.25),
        minibatch_stat_scale = env_float("BPC_MINIBATCH_STAT_SCALE", 1.0),
        prior_v_scale = env_float("BPC_PRIOR_V_SCALE", 10.0),
        prior_psi_scale = env_float("BPC_PRIOR_PSI_SCALE", 1000.0),
        prior_nu_offset = env_float("BPC_PRIOR_NU_OFFSET", 2.0),
        posterior_jitter = env_float("BPC_POSTERIOR_JITTER", 1e-5),
        activate_input = env_bool("BPC_ACTIVATE_INPUT", true),
        eval_samples = env_int("BPC_EVAL_SAMPLES", 20),
        test_fraction = env_float("BPC_TEST_FRACTION", UCI_DEFAULT_TEST_FRACTION),
        validation_fraction = env_float(
            "BPC_VALIDATION_FRACTION", UCI_DEFAULT_VALIDATION_FRACTION,
        ),
        split_seed = env_int("BPC_SPLIT_SEED", UCI_DEFAULT_SPLIT_SEED),
        model_seed = env_int("BPC_MODEL_SEED", 20260805),
        backend = lowercase(strip(get(ENV, "BPC_BACKEND", "cpu"))),
        output_dir = abspath(get(ENV, "BPC_OUTPUT_DIR", default_output_directory())),
        resume = env_bool("BPC_RESUME", true),
        save_checkpoints = env_bool("BPC_SAVE_CHECKPOINTS", true),
        show_progress = env_bool("BPC_SHOW_PROGRESS", true),
    )
    return validate_config(config)
end

function validate_config(config::BPCConfig)
    config.n_splits >= 1 || throw(ArgumentError("BPC_SPLITS must be positive"))
    config.hidden_units >= 1 || throw(ArgumentError("BPC_HIDDEN_UNITS must be positive"))
    config.batch_size >= 1 || throw(ArgumentError("BPC_BATCH_SIZE must be positive"))
    config.max_epochs >= 1 || throw(ArgumentError("BPC_MAX_EPOCHS must be positive"))
    1 <= config.min_epochs <= config.max_epochs ||
        throw(ArgumentError("BPC_MIN_EPOCHS must lie in 1:BPC_MAX_EPOCHS"))
    config.validation_every >= 1 ||
        throw(ArgumentError("BPC_VALIDATION_EVERY must be positive"))
    config.patience >= 1 || throw(ArgumentError("BPC_PATIENCE must be positive"))
    config.latent_steps >= 0 || throw(ArgumentError("BPC_LATENT_STEPS cannot be negative"))
    config.latent_learning_rate > 0 ||
        throw(ArgumentError("BPC_LATENT_LEARNING_RATE must be positive"))
    0 <= config.natural_learning_exponent <= 1 ||
        throw(ArgumentError("BPC_NATURAL_LEARNING_EXPONENT must be in [0, 1]"))
    config.minibatch_stat_scale > 0 ||
        throw(ArgumentError("BPC_MINIBATCH_STAT_SCALE must be positive"))
    all(>(0), (config.prior_v_scale, config.prior_psi_scale,
        config.prior_nu_offset, config.posterior_jitter)) ||
        throw(ArgumentError("BPC prior scales, nu offset, and jitter must be positive"))
    config.eval_samples >= 2 || throw(ArgumentError("BPC_EVAL_SAMPLES must be at least 2"))
    0 < config.test_fraction < 1 ||
        throw(ArgumentError("BPC_TEST_FRACTION must be in (0, 1)"))
    0 < config.validation_fraction < 1 ||
        throw(ArgumentError("BPC_VALIDATION_FRACTION must be in (0, 1)"))
    config.backend in ("cpu", "cuda") ||
        throw(ArgumentError("BPC_BACKEND must be cpu or cuda"))
    config.backend == "cpu" || backend_available("cuda") ||
        throw(ArgumentError("BPC_BACKEND=cuda requested, but CUDA is not functional"))
    return config
end

config_dictionary(config::BPCConfig) = Dict(
    string(name) => getfield(config, name) for name in fieldnames(BPCConfig)
)
