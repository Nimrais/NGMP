const DATASET_REGISTRY = [
    ("yacht", "Yacht", Yacht),
    ("concrete", "Concrete", Concrete),
    ("energy", "Energy", EnergyEfficiency),
    ("housing", "Boston", BostonHousing),
    ("power", "Power", PowerPlant),
    ("wine", "Wine", WineQualityRed),
]

const DATASET_ALIASES = Dict(
    "yacht" => "yacht",
    "yach" => "yacht",
    "concrete" => "concrete",
    "conc" => "concrete",
    "energy" => "energy",
    "ener" => "energy",
    "energyefficiency" => "energy",
    "housing" => "housing",
    "boston" => "housing",
    "bost" => "housing",
    "bostonhousing" => "housing",
    "power" => "power",
    "powe" => "power",
    "powerplant" => "power",
    "wine" => "wine",
    "winered" => "wine",
    "winequalityred" => "wine",
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
    normalized_value = lowercase(strip(value))
    normalized_value == "all" && return first.(DATASET_REGISTRY)
    result = String[]
    for token in split(normalized_value, ',')
        normalized = replace(strip(token), r"[^a-z0-9]" => "")
        isempty(normalized) && continue
        haskey(DATASET_ALIASES, normalized) ||
            throw(ArgumentError("unknown dataset '$token'"))
        push!(result, DATASET_ALIASES[normalized])
    end
    isempty(result) && throw(ArgumentError("at least one dataset is required"))
    return unique(result)
end

function parse_likelihoods(value::AbstractString)
    result = String[]
    for token in split(lowercase(value), ',')
        likelihood = strip(token)
        isempty(likelihood) && continue
        likelihood in ("homoscedastic", "heteroscedastic") ||
            throw(ArgumentError(
                "DVI_LIKELIHOODS must contain homoscedastic and/or heteroscedastic",
            ))
        push!(result, likelihood)
    end
    isempty(result) && throw(ArgumentError("at least one likelihood is required"))
    return unique(result)
end

function default_output_directory()
    stamp = Dates.format(Dates.now(), dateformat"yyyymmddTHHMMSS")
    return normpath(joinpath(
        @__DIR__, "..", "..", "..", "results", "dvi", stamp,
    ))
end

Base.@kwdef struct DVIConfig
    datasets::Vector{String} = first.(DATASET_REGISTRY)
    n_splits::Int = 20
    likelihoods::Vector{String} = ["heteroscedastic"]
    propagation::String = "full"
    hidden_units::Int = 50
    batch_size::Int = 100
    learning_rate::Float64 = 1e-3
    eb_alpha::Float64 = 1.0
    eb_beta::Float64 = 10.0
    initialization_scale::Float64 = 5.0
    bias_variance_divisor::Float64 = 10.0
    homo_log_variance::Float64 = 0.0
    max_epochs::Int = 10_000
    min_epochs::Int = 25
    validation_every::Int = 5
    patience::Int = 500
    kl_warmup_epochs::Int = 7_000
    kl_anneal_epochs::Int = 500
    test_fraction::Float64 = 0.1
    validation_fraction::Float64 = 0.1
    split_seed::Int = 20260726
    model_seed::Int = 20260727
    gradient_clip::Float64 = 0.1
    output_dir::String = default_output_directory()
    resume::Bool = true
    save_checkpoints::Bool = true
    make_plot::Bool = true
    show_progress::Bool = true
end

function load_config()
    config = DVIConfig(
        datasets = parse_datasets(get(ENV, "DVI_DATASETS", "all")),
        n_splits = env_int("DVI_SPLITS", 20),
        likelihoods = parse_likelihoods(
            get(ENV, "DVI_LIKELIHOODS", "heteroscedastic"),
        ),
        propagation = lowercase(strip(get(ENV, "DVI_PROPAGATION", "full"))),
        hidden_units = env_int("DVI_HIDDEN_UNITS", 50),
        batch_size = env_int("DVI_BATCH_SIZE", 100),
        learning_rate = env_float("DVI_LEARNING_RATE", 1e-3),
        eb_alpha = env_float("DVI_EB_ALPHA", 1.0),
        eb_beta = env_float("DVI_EB_BETA", 10.0),
        initialization_scale = env_float("DVI_INITIALIZATION_SCALE", 5.0),
        bias_variance_divisor =
            env_float("DVI_BIAS_VARIANCE_DIVISOR", 10.0),
        homo_log_variance = env_float("DVI_HOMO_LOG_VARIANCE", 0.0),
        max_epochs = env_int("DVI_MAX_EPOCHS", 10_000),
        min_epochs = env_int("DVI_MIN_EPOCHS", 25),
        validation_every = env_int("DVI_VALIDATION_EVERY", 5),
        patience = env_int("DVI_PATIENCE", 500),
        kl_warmup_epochs = env_int("DVI_KL_WARMUP_EPOCHS", 7_000),
        kl_anneal_epochs = env_int("DVI_KL_ANNEAL_EPOCHS", 500),
        test_fraction = env_float("DVI_TEST_FRACTION", 0.1),
        validation_fraction = env_float("DVI_VALIDATION_FRACTION", 0.1),
        split_seed = env_int("DVI_SPLIT_SEED", 20260726),
        model_seed = env_int("DVI_MODEL_SEED", 20260727),
        gradient_clip = env_float("DVI_GRADIENT_CLIP", 0.1),
        output_dir = abspath(get(
            ENV, "DVI_OUTPUT_DIR", default_output_directory(),
        )),
        resume = env_bool("DVI_RESUME", true),
        save_checkpoints = env_bool("DVI_SAVE_CHECKPOINTS", true),
        make_plot = env_bool("DVI_MAKE_PLOT", true),
        show_progress = env_bool("DVI_SHOW_PROGRESS", true),
    )
    return validate_config(config)
end

function validate_config(config::DVIConfig)
    config.n_splits >= 1 || throw(ArgumentError("DVI_SPLITS must be positive"))
    config.propagation in ("full", "diagonal") ||
        throw(ArgumentError("DVI_PROPAGATION must be full or diagonal"))
    config.hidden_units >= 1 ||
        throw(ArgumentError("DVI_HIDDEN_UNITS must be positive"))
    config.batch_size >= 1 ||
        throw(ArgumentError("DVI_BATCH_SIZE must be positive"))
    config.learning_rate > 0 ||
        throw(ArgumentError("DVI_LEARNING_RATE must be positive"))
    config.eb_alpha > 0 ||
        throw(ArgumentError("DVI_EB_ALPHA must be positive"))
    config.eb_beta > 0 ||
        throw(ArgumentError("DVI_EB_BETA must be positive"))
    config.initialization_scale > 0 ||
        throw(ArgumentError("DVI_INITIALIZATION_SCALE must be positive"))
    config.bias_variance_divisor > 0 ||
        throw(ArgumentError("DVI_BIAS_VARIANCE_DIVISOR must be positive"))
    isfinite(config.homo_log_variance) ||
        throw(ArgumentError("DVI_HOMO_LOG_VARIANCE must be finite"))
    config.max_epochs >= 1 ||
        throw(ArgumentError("DVI_MAX_EPOCHS must be positive"))
    1 <= config.min_epochs <= config.max_epochs ||
        throw(ArgumentError("DVI_MIN_EPOCHS must lie in 1:DVI_MAX_EPOCHS"))
    config.validation_every >= 1 ||
        throw(ArgumentError("DVI_VALIDATION_EVERY must be positive"))
    config.patience >= 1 ||
        throw(ArgumentError("DVI_PATIENCE must be positive"))
    config.kl_warmup_epochs >= 0 ||
        throw(ArgumentError("DVI_KL_WARMUP_EPOCHS cannot be negative"))
    config.kl_anneal_epochs >= 0 ||
        throw(ArgumentError("DVI_KL_ANNEAL_EPOCHS cannot be negative"))
    config.kl_warmup_epochs + config.kl_anneal_epochs <= config.max_epochs ||
        throw(ArgumentError(
            "KL warmup plus annealing cannot exceed DVI_MAX_EPOCHS",
        ))
    0 < config.test_fraction < 1 ||
        throw(ArgumentError("DVI_TEST_FRACTION must be in (0, 1)"))
    0 < config.validation_fraction < 1 ||
        throw(ArgumentError("DVI_VALIDATION_FRACTION must be in (0, 1)"))
    (isinf(config.gradient_clip) || config.gradient_clip > 0) ||
        throw(ArgumentError("DVI_GRADIENT_CLIP must be positive or Inf"))
    return config
end

function config_dictionary(config::DVIConfig)
    return Dict(
        string(name) => getfield(config, name)
        for name in fieldnames(DVIConfig)
    )
end
