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
        mode = strip(token)
        isempty(mode) && continue
        mode in ("homoscedastic", "heteroscedastic") ||
            throw(ArgumentError(
                "BBB_LIKELIHOODS must contain homoscedastic and/or heteroscedastic",
            ))
        push!(result, mode)
    end
    isempty(result) && throw(ArgumentError("at least one likelihood is required"))
    return unique(result)
end

function default_output_directory()
    stamp = Dates.format(Dates.now(), dateformat"yyyymmddTHHMMSS")
    return normpath(joinpath(
        @__DIR__, "..", "..", "..", "results", "bbb_uci", stamp,
    ))
end

Base.@kwdef struct BBBConfig
    datasets::Vector{String} = first.(DATASET_REGISTRY)
    n_splits::Int = 5
    likelihoods::Vector{String} = ["heteroscedastic"]
    hidden_units::Int = 50
    batch_size::Int = 100
    learning_rate::Float64 = 1e-3
    prior_mean::Float64 = 0.0
    prior_std::Float64 = 1.0
    initial_posterior_std::Float64 = 0.05
    homo_log_variance::Float64 = 0.0
    minimum_log_variance::Float64 = -20.0
    maximum_log_variance::Float64 = 20.0
    max_epochs::Int = 500
    min_epochs::Int = 25
    validation_every::Int = 5
    patience::Int = 20
    train_samples::Int = 1
    eval_samples::Int = 20
    test_fraction::Float64 = 0.1
    validation_fraction::Float64 = 0.1
    split_seed::Int = 20260726
    model_seed::Int = 20260727
    gradient_clip::Float64 = Inf
    output_dir::String = default_output_directory()
    resume::Bool = true
    save_checkpoints::Bool = true
    make_plot::Bool = true
    show_progress::Bool = true
end

function load_config()
    config = BBBConfig(
        datasets = parse_datasets(get(ENV, "BBB_DATASETS", "all")),
        n_splits = env_int("BBB_SPLITS", 5),
        likelihoods = parse_likelihoods(
            get(ENV, "BBB_LIKELIHOODS", "heteroscedastic"),
        ),
        hidden_units = env_int("BBB_HIDDEN_UNITS", 50),
        batch_size = env_int("BBB_BATCH_SIZE", 100),
        learning_rate = env_float("BBB_LEARNING_RATE", 1e-3),
        prior_mean = env_float("BBB_PRIOR_MEAN", 0.0),
        prior_std = env_float("BBB_PRIOR_STD", 1.0),
        initial_posterior_std =
            env_float("BBB_INITIAL_POSTERIOR_STD", 0.05),
        homo_log_variance = env_float("BBB_HOMO_LOG_VARIANCE", 0.0),
        minimum_log_variance =
            env_float("BBB_MINIMUM_LOG_VARIANCE", -20.0),
        maximum_log_variance =
            env_float("BBB_MAXIMUM_LOG_VARIANCE", 20.0),
        max_epochs = env_int("BBB_MAX_EPOCHS", 500),
        min_epochs = env_int("BBB_MIN_EPOCHS", 25),
        validation_every = env_int("BBB_VALIDATION_EVERY", 5),
        patience = env_int("BBB_PATIENCE", 20),
        train_samples = env_int("BBB_TRAIN_SAMPLES", 1),
        eval_samples = env_int("BBB_EVAL_SAMPLES", 20),
        test_fraction = env_float("BBB_TEST_FRACTION", 0.1),
        validation_fraction = env_float("BBB_VALIDATION_FRACTION", 0.1),
        split_seed = env_int("BBB_SPLIT_SEED", 20260726),
        model_seed = env_int("BBB_MODEL_SEED", 20260727),
        gradient_clip = env_float("BBB_GRADIENT_CLIP", Inf),
        output_dir = abspath(get(
            ENV, "BBB_OUTPUT_DIR", default_output_directory(),
        )),
        resume = env_bool("BBB_RESUME", true),
        save_checkpoints = env_bool("BBB_SAVE_CHECKPOINTS", true),
        make_plot = env_bool("BBB_MAKE_PLOT", true),
        show_progress = env_bool("BBB_SHOW_PROGRESS", true),
    )
    return validate_config(config)
end

function validate_config(config::BBBConfig)
    config.n_splits >= 1 || throw(ArgumentError("BBB_SPLITS must be positive"))
    config.hidden_units >= 1 ||
        throw(ArgumentError("BBB_HIDDEN_UNITS must be positive"))
    config.batch_size >= 1 || throw(ArgumentError("BBB_BATCH_SIZE must be positive"))
    config.learning_rate > 0 ||
        throw(ArgumentError("BBB_LEARNING_RATE must be positive"))
    isfinite(config.prior_mean) ||
        throw(ArgumentError("BBB_PRIOR_MEAN must be finite"))
    config.prior_std > 0 ||
        throw(ArgumentError("BBB_PRIOR_STD must be positive"))
    config.initial_posterior_std > 0 ||
        throw(ArgumentError("BBB_INITIAL_POSTERIOR_STD must be positive"))
    isfinite(config.homo_log_variance) ||
        throw(ArgumentError("BBB_HOMO_LOG_VARIANCE must be finite"))
    all(isfinite, (
        config.minimum_log_variance,
        config.maximum_log_variance,
    )) || throw(ArgumentError("BBB log-variance bounds must be finite"))
    config.minimum_log_variance < config.maximum_log_variance ||
        throw(ArgumentError(
            "BBB_MINIMUM_LOG_VARIANCE must be below BBB_MAXIMUM_LOG_VARIANCE",
        ))
    config.max_epochs >= 1 ||
        throw(ArgumentError("BBB_MAX_EPOCHS must be positive"))
    1 <= config.min_epochs <= config.max_epochs ||
        throw(ArgumentError("BBB_MIN_EPOCHS must lie in 1:BBB_MAX_EPOCHS"))
    config.validation_every >= 1 ||
        throw(ArgumentError("BBB_VALIDATION_EVERY must be positive"))
    config.patience >= 1 || throw(ArgumentError("BBB_PATIENCE must be positive"))
    config.train_samples >= 1 ||
        throw(ArgumentError("BBB_TRAIN_SAMPLES must be positive"))
    config.eval_samples >= 1 ||
        throw(ArgumentError("BBB_EVAL_SAMPLES must be positive"))
    0 < config.test_fraction < 1 ||
        throw(ArgumentError("BBB_TEST_FRACTION must be in (0, 1)"))
    0 < config.validation_fraction < 1 ||
        throw(ArgumentError("BBB_VALIDATION_FRACTION must be in (0, 1)"))
    (isinf(config.gradient_clip) || config.gradient_clip > 0) ||
        throw(ArgumentError("BBB_GRADIENT_CLIP must be positive or Inf"))
    return config
end

function config_dictionary(config::BBBConfig)
    return Dict(
        string(name) => getfield(config, name)
        for name in fieldnames(BBBConfig)
    )
end
