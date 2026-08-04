#!/usr/bin/env julia

using DVIUCI
using Random
using StableRNGs
using Statistics

const PAPER_STEP_ESTIMATE = 24_808_675
const SAFETY_MARGIN = 1.25
const TARGET_DAYS = 3.0

env_int(name, default) = parse(Int, get(ENV, name, string(default)))

function requested_backends()
    return String.(filter(!isempty, strip.(split(
        lowercase(get(ENV, "DVI_PROFILE_BACKENDS", "zygote")), ",",
    ))))
end

function requested_dimensions()
    return parse.(Int, filter(!isempty, strip.(split(
        get(ENV, "DVI_PROFILE_INPUT_DIMS", "6,13"), ",",
    ))))
end

function profile_configuration(
    backend_name::String,
    input_dimension::Int;
    batch_size::Int,
    warmup::Int,
    samples::Int,
)
    device = backend_name == "reactant" ?
        lowercase(get(ENV, "DVI_DEVICE", "gpu")) : "cpu"
    config = DVIConfig(
        datasets = ["synthetic"],
        n_splits = 1,
        split_ids = [1],
        likelihoods = ["heteroscedastic"],
        propagation = "full",
        execution_backend = backend_name,
        execution_device = device,
        hidden_units = 50,
        batch_size = batch_size,
        max_epochs = warmup + samples + 1,
        min_epochs = 1,
        validation_every = warmup + samples + 1,
        patience = warmup + samples + 1,
        show_progress = false,
        save_checkpoints = false,
        make_plot = false,
    )
    validate_config(config)

    data_rng = StableRNG(20260802 + input_dimension)
    features = randn(data_rng, Float32, batch_size, input_dimension)
    targets = randn(data_rng, Float32, batch_size)
    params = initialize_model(
        input_dimension, "heteroscedastic", config; seed = 20260802,
    )
    state = DVIUCI.initialize_training_backend(
        params, features, targets, "heteroscedastic", config,
    )
    batch_rng = StableRNG(20260812)
    tracker = NumericalTracker()
    optimizer_step = 0

    compile_timing = @timed begin
        state, _, optimizer_step = DVIUCI.training_backend_epoch(
            state,
            batch_rng,
            1,
            optimizer_step;
            phase = "profile",
            tracker = tracker,
        )
    end
    for epoch in 2:(warmup + 1)
        state, _, optimizer_step = DVIUCI.training_backend_epoch(
            state,
            batch_rng,
            epoch,
            optimizer_step;
            phase = "profile",
            tracker = tracker,
        )
    end

    times = Float64[]
    allocations = Int[]
    for sample in 1:samples
        timing = @timed begin
            state, _, optimizer_step = DVIUCI.training_backend_epoch(
                state,
                batch_rng,
                warmup + sample + 1,
                optimizer_step;
                phase = "profile",
                tracker = tracker,
            )
        end
        push!(times, timing.time)
        push!(allocations, timing.bytes)
    end

    workers_default = backend_name == "zygote" ? 4 : 1
    workers = env_int("DVI_PROFILE_WORKERS", workers_default)
    seconds_per_step = median(times)
    projected_days = (
        compile_timing.time +
        SAFETY_MARGIN * PAPER_STEP_ESTIMATE * seconds_per_step / workers
    ) / 86_400
    return (
        backend = backend_name,
        device = device,
        input_dimension = input_dimension,
        batch_size = batch_size,
        compile_seconds = compile_timing.time,
        seconds_per_step = seconds_per_step,
        bytes_per_step = median(allocations),
        projected_days = projected_days,
        passes_gate = projected_days <= TARGET_DAYS,
    )
end

function main()
    batch_size = env_int("DVI_PROFILE_BATCH_SIZE", 100)
    warmup = env_int("DVI_PROFILE_WARMUP", 3)
    samples = env_int("DVI_PROFILE_SAMPLES", 20)
    warmup >= 1 || throw(ArgumentError("DVI_PROFILE_WARMUP must be positive"))
    samples >= 1 || throw(ArgumentError("DVI_PROFILE_SAMPLES must be positive"))

    println(
        "DVI steady-state profile; estimated paper steps=",
        PAPER_STEP_ESTIMATE,
        ", safety margin=",
        SAFETY_MARGIN,
        ", launch gate=",
        TARGET_DAYS,
        " days",
    )
    all_pass = true
    for backend in requested_backends(), dimension in requested_dimensions()
        result = profile_configuration(
            backend,
            dimension;
            batch_size = batch_size,
            warmup = warmup,
            samples = samples,
        )
        gate = result.passes_gate ? "PASS" : "FAIL"
        all_pass &= result.passes_gate
        println(
            "backend=$(result.backend) device=$(result.device) ",
            "shape=$(result.batch_size)x$(result.input_dimension) ",
            "compile_seconds=$(round(result.compile_seconds; digits=3)) ",
            "seconds_per_step=$(round(result.seconds_per_step; digits=6)) ",
            "bytes_per_step=$(round(Int, result.bytes_per_step)) ",
            "projected_days=$(round(result.projected_days; digits=2)) ",
            "gate=$gate",
        )
    end
    all_pass || println(stderr, "DVI paper launch gate failed.")
    return all_pass
end

if abspath(PROGRAM_FILE) == abspath(String(@__FILE__))
    main() || exit(1)
end
