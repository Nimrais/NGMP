#!/usr/bin/env julia
using DVIUCI, CSV, DataFrames, Dates, JLD2, StableRNGs, Statistics, TOML, SHA

module ReferenceCampaign
include("two_hidden_experiment.jl")
end
const Table = ReferenceCampaign.FinalTable
const ROOT = abspath(get(ENV, "DVI_BIG_OUTPUT_ROOT", Table.DVI_BIG_ROOT))
const SMALL_DVI = only(s.directory for s in Table.SOURCES if s.method == "DVI-Hetero")
const BIG_DDVI = only(s.directory for s in Table.SOURCES if s.method == "dDVI-2HL-Hetero")
const DATASETS = ["yacht", "concrete", "energy", "housing", "power", "wine"]

function config_at(directory; kwargs...)
    base = ReferenceCampaign.experiment_config(directory)
    fields = (; (name => getfield(base, name) for name in fieldnames(DVIConfig))...)
    settings = merge(fields, (propagation = "full",
        execution_backend = get(ENV, "DVI_BIG_BACKEND", "reactant"),
        execution_device = get(ENV, "DVI_BIG_DEVICE", "gpu"),
        implementation_version = DVIUCI.DVI_FULL_TWO_HIDDEN_VERSION,
        full_selection_patience_steps = 50000), (; kwargs...))
    return validate_config(DVIConfig(; settings...))
end

function save_toml(path, values)
    mkpath(dirname(path))
    temporary = path * ".tmp-$(getpid())"
    open(io -> TOML.print(io, values; sorted = true), temporary, "w")
    mv(temporary, path; force = true)
end

function source_fingerprint()
    paths = sort(vcat(filter(p -> endswith(p, ".jl"), readdir(joinpath(@__DIR__, "src"); join = true)),
        [@__FILE__, joinpath(@__DIR__, "two_hidden_experiment.jl"),
         joinpath(@__DIR__, "run_dvi_two_hidden_paper.sh"),
         joinpath(@__DIR__, "..", "..", "paper_materials", "make_final_baseline_table.jl"),
         joinpath(@__DIR__, "test", "full_two_hidden_reactant.jl"),
         joinpath(@__DIR__, "Project.toml")],
        filter(isfile, [joinpath(@__DIR__, "Manifest.toml"),
            joinpath(@__DIR__, "..", "..", "Project.toml"),
            joinpath(@__DIR__, "..", "..", "Manifest.toml")])) )
    return bytes2hex(sha256(join([bytes2hex(sha256(read(p))) for p in paths], "\n")))
end

function preflight()
    config = config_at(joinpath(ROOT, "preflight"))
    DVIUCI.ensure_output_directories(config)
    DVIUCI.save_config(config)
    DVIUCI.write_split_manifest(config)
    sources = Table.table_sources(include_dvi_big = true, dvi_big_directory = config.output_dir)
    Table.validate_split_manifests(sources)
    Table.load_results(Table.SOURCES)
    println("Verified the six datasets and all 20 paired split manifests.")
end

function check_complete(config)
    path = joinpath(config.output_dir, "runs.csv")
    isfile(path) || error("missing runs: $path")
    DVIUCI.validate_resume_config(joinpath(config.output_dir, "config.toml"), config)
    runs = CSV.read(path, DataFrame)
    DVIUCI.validate_complete_runs(runs, config.datasets, config.split_ids, config.likelihoods, "full")
    for row in eachrow(runs)
        DVIUCI.configuration_succeeded(runs, String(row.dataset), row.split,
            "heteroscedastic", "full", config) || error("missing or incompatible checkpoint")
    end
    return runs
end

function canonical_config()
    directory = joinpath(ROOT, "dvi")
    saved = TOML.parsefile(joinpath(directory, "config.toml"))
    # Verification uses host-side prediction and does not need the training GPU.
    return config_at(directory; execution_backend = saved["execution_backend"],
        execution_device = saved["execution_device"])
end

function paired_comparison(runs, directory, label)
    original = CSV.read(joinpath(directory, "runs.csv"), DataFrame)
    paired = innerjoin(select(runs, :dataset, :split, :nll_original => :new_nll,
        :rmse_original => :new_rmse), select(original, :dataset, :split,
        :nll_original => :old_nll, :rmse_original => :old_rmse);
        on = [:dataset, :split], validate = (true, true))
    nrow(paired) == 120 || error("incomplete paired comparison")
    paired.nll_change = paired.new_nll .- paired.old_nll
    paired.rmse_change = paired.new_rmse .- paired.old_rmse
    CSV.write(joinpath(ROOT, "paired_$(label)_differences.csv"), paired)
    summary = combine(groupby(paired, :dataset),
        :nll_change => mean => :nll_change_mean,
        :nll_change => (x -> 1.96std(x) / sqrt(length(x))) => :nll_change_ci95,
        :rmse_change => mean => :rmse_change_mean,
        :rmse_change => (x -> 1.96std(x) / sqrt(length(x))) => :rmse_change_ci95)
    CSV.write(joinpath(ROOT, "paired_$(label)_summary.csv"), summary)
end

function finish()
    config = canonical_config()
    runs = check_complete(config)
    ReferenceCampaign.replay_checkpoints(config, runs)
    sources = Table.table_sources(include_dvi_big = true, dvi_big_directory = config.output_dir)
    Table.validate_split_manifests(sources)
    results = Table.load_results(sources)
    open(joinpath(ROOT, "final_baseline_table.md"), "w") do io
        Table.render_markdown(io, sources, results)
    end
    paired_comparison(runs, SMALL_DVI, "small_dvi")
    paired_comparison(runs, BIG_DDVI, "big_ddvi")
    save_toml(joinpath(ROOT, "status.toml"), Dict("status" => "complete",
        "updated_utc" => string(now(UTC)), "verified_fits" => 120))
    println("Verified all 120 DVI-big checkpoints and saved the ten-method comparison.")
end

function profile(backend_name, device)
    directory = joinpath(ROOT, "pilots", "$(backend_name)_$(device)")
    mkpath(directory)
    rows = NamedTuple[]
    for dataset_key in DATASETS
        config = config_at(joinpath(directory, dataset_key);
            execution_backend = backend_name, execution_device = device)
        dataset = DVIUCI.load_dataset(dataset_key)
        prepared = DVIUCI.outer_and_inner_splits(dataset, 1, config)
        partition = prepared.inner
        x, y = partition.x_train_standardized, partition.y_train_standardized
        params = initialize_model(size(x, 2), "heteroscedastic", config; seed = 811)
        initialization = @elapsed state = DVIUCI.initialize_training_backend(params, x, y,
            "heteroscedastic", config)
        rng, tracker, step = StableRNG(812), NumericalTracker(), 15000
        compile_seconds = @elapsed begin
            state, loss, step = DVIUCI.training_backend_epoch(state, rng, 1, step;
                phase = "profile", tracker = tracker)
            isfinite(loss) || error("nonfinite pilot loss")
        end
        for epoch in 2:4
            state, _, step = DVIUCI.training_backend_epoch(state, rng, epoch, step;
                phase = "profile", tracker = tracker)
        end
        steps_per_epoch = cld(size(x, 1), config.batch_size)
        times = Float64[]
        for epoch in 5:(4 + max(8, cld(100, steps_per_epoch)))
            elapsed = @elapsed begin
                state, loss, step = DVIUCI.training_backend_epoch(state, rng, epoch, step;
                    phase = "profile", tracker = tracker)
                isfinite(loss) || error("nonfinite pilot loss")
            end
            push!(times, elapsed / steps_per_epoch)
        end
        val_compile = @elapsed DVIUCI.training_backend_validation_output(state,
            partition.x_test_standardized; tracker = tracker)
        val_times = [@elapsed DVIUCI.training_backend_validation_output(state,
            partition.x_test_standardized; tracker = tracker) for _ in 1:3]
        # Deliberately short artifact pilot: it is not a paper-quality fit.
        pilot = config_at(config.output_dir; execution_backend = backend_name,
            execution_device = device, datasets = [dataset_key], split_ids = [1],
            max_epochs = 2, min_epochs = 1, validation_every = 1, patience = 2,
            kl_warmup_steps = 0, kl_anneal_steps = 0,
            training_budget_protocol = "pilot-two-epochs-not-paper-results")
        DVIUCI.ensure_output_directories(pilot)
        DVIUCI.save_config(pilot)
        DVIUCI.write_split_manifest(pilot)
        artifact_seconds = @elapsed begin
            run_benchmark(pilot)
            runs = check_complete(pilot)
            ReferenceCampaign.replay_checkpoints(pilot, runs)
        end
        # A second invocation must skip the successful fit without modifying its checkpoint.
        checkpoint_path = DVIUCI.posterior_checkpoint_path(pilot, dataset_key, 1, "heteroscedastic")
        before = sha256(read(checkpoint_path))
        resume_seconds = @elapsed run_benchmark(pilot)
        before == sha256(read(checkpoint_path)) || error("resume overwrote checkpoint")
        push!(rows, (dataset = dataset_key, backend = backend_name, device = device,
            seconds_per_step = median(times), initialization_seconds = initialization,
            compile_seconds = compile_seconds, validation_compile_seconds = val_compile,
            validation_seconds = median(val_times), artifact_pilot_seconds = artifact_seconds,
            resume_seconds = resume_seconds, status = "verified"))
        CSV.write(joinpath(directory, "profile.csv"), DataFrame(rows))
        println(dataset_key, ": ", round(1000median(times); digits = 3), " ms/step; artifacts and resume verified")
        flush(stdout)
    end
end

# Each worker compiles first, then waits for a shared start signal. The controller
# measures simultaneous sustained throughput rather than dividing single-worker time.
function throughput_worker(backend_name, device, group, id)
    directory = joinpath(ROOT, "pilots", group)
    mkpath(directory)
    config = config_at(joinpath(directory, "worker_$id");
        execution_backend = backend_name, execution_device = device)
    dataset = DVIUCI.load_dataset("power")
    partition = DVIUCI.outer_and_inner_splits(dataset, 1, config).inner
    x, y = partition.x_train_standardized, partition.y_train_standardized
    params = initialize_model(size(x, 2), "heteroscedastic", config; seed = 911 + parse(Int, id))
    state = DVIUCI.initialize_training_backend(params, x, y, "heteroscedastic", config)
    rng, tracker, step = StableRNG(912), NumericalTracker(), 15000
    for epoch in 1:4
        state, _, step = DVIUCI.training_backend_epoch(state, rng, epoch, step;
            phase = "profile", tracker = tracker)
    end
    write(joinpath(directory, "ready_$id"), "ready")
    deadline = time() + 300
    while !isfile(joinpath(directory, "go"))
        time() < deadline || error("throughput barrier timed out")
        sleep(0.1)
    end
    start_at = parse(Float64, read(joinpath(directory, "go"), String))
    sleep(max(0.0, start_at - time()))
    started, first_step, epoch = time(), step, 5
    while time() - started < 15
        state, loss, step = DVIUCI.training_backend_epoch(state, rng, epoch, step;
            phase = "profile", tracker = tracker)
        isfinite(loss) || error("nonfinite throughput loss")
        epoch += 1
    end
    save_toml(joinpath(directory, "worker_$id.toml"), Dict(
        "steps" => step - first_step, "seconds" => time() - started,
        "backend" => backend_name, "device" => device))
end

function throughput(group)
    directory = joinpath(ROOT, "pilots", group)
    records = [TOML.parsefile(joinpath(directory, p)) for p in readdir(directory)
        if startswith(p, "worker_") && endswith(p, ".toml")]
    isempty(records) && error("missing throughput measurements: $group")
    return sum(r["steps"] for r in records) / maximum(r["seconds"] for r in records)
end

function report()
    gpu = CSV.read(joinpath(ROOT, "pilots", "reactant_gpu", "profile.csv"), DataFrame)
    cpu = CSV.read(joinpath(ROOT, "pilots", "zygote_cpu", "profile.csv"), DataFrame)
    sort(gpu.dataset) == sort(cpu.dataset) == sort(DATASETS) || error("incomplete profiles")
    all(gpu.status .== "verified") && all(cpu.status .== "verified") || error("failed pilots")
    gpu_single, gpu_pair, cpu_four = throughput.( ["gpu_one", "gpu_two", "cpu_four"] )
    workers = gpu_pair >= 1.1gpu_single ? 2 : 1
    gpu_rate = workers == 2 ? gpu_pair : gpu_single
    concurrency_gain = gpu_rate / gpu_single
    rows = NamedTuple[]
    for (basis, path) in [("small_dvi", SMALL_DVI), ("big_ddvi", BIG_DDVI)]
        historical = CSV.read(joinpath(path, "runs.csv"), DataFrame)
        for key in DATASETS
            records = filter(r -> r.dataset == key, historical)
            timing = only(eachrow(filter(r -> r.dataset == key, gpu)))
            cpu_timing = only(eachrow(filter(r -> r.dataset == key, cpu)))
            steps = sum(records.selection_optimizer_steps + records.refit_optimizer_steps)
            # Historical epochs approximate validation frequency; deep-model stopping is unknown.
            inner_count = size(DVIUCI.outer_and_inner_splits(DVIUCI.load_dataset(key), 1, config_at(ROOT)).inner.x_train_standardized, 1)
            validations = sum(ceil.(records.selection_optimizer_steps ./ cld(inner_count, 100) ./ 5))
            overhead = 2 * workers * (timing.compile_seconds + timing.initialization_seconds + timing.validation_compile_seconds)
            seconds = 1.25 * (steps * timing.seconds_per_step / concurrency_gain +
                validations * timing.validation_seconds / workers + overhead)
            cpu_gain = cpu_four * only(filter(r -> r.dataset == "power", cpu).seconds_per_step)
            cpu_seconds = 1.25 * (steps * cpu_timing.seconds_per_step / cpu_gain +
                validations * cpu_timing.validation_seconds / 4)
            push!(rows, (basis = basis, dataset = key, historical_steps = steps,
                gpu_workers = workers, gpu_hours = seconds / 3600, cpu_four_hours = cpu_seconds / 3600))
        end
    end
    table = DataFrame(rows)
    CSV.write(joinpath(ROOT, "pilots", "runtime_projection.csv"), table)
    gpu_hours = [sum(filter(r -> r.basis == basis, table).gpu_hours) for basis in ("small_dvi", "big_ddvi")]
    cpu_hours = [sum(filter(r -> r.basis == basis, table).cpu_four_hours) for basis in ("small_dvi", "big_ddvi")]
    prefer_gpu = sum(gpu_hours) < sum(cpu_hours)
    memory_samples = Int[]
    for path in readdir(joinpath(ROOT, "logs"); join = true)
        endswith(path, "gpu_memory.csv") || continue
        for line in eachline(path)
            matched = match(r",\s*(\d+)\s+MiB,", line)
            matched === nothing || push!(memory_samples, parse(Int, matched.captures[1]))
        end
    end
    recommendation = Dict("status" => "passed", "source_fingerprint" => source_fingerprint(),
        "gpu_hardware" => strip(read(joinpath(ROOT, "logs", "gpu.txt"), String)),
        "implementation_version" => DVIUCI.DVI_FULL_TWO_HIDDEN_VERSION,
        "backend" => prefer_gpu ? "reactant" : "zygote", "device" => prefer_gpu ? "gpu" : "cpu",
        "workers" => prefer_gpu ? workers : 4,
        "gpu_workers" => workers, "gpu_one_steps_per_second" => gpu_single,
        "gpu_two_steps_per_second" => gpu_pair, "cpu_four_steps_per_second" => cpu_four,
        "gpu_hours_small_dvi_steps" => gpu_hours[1], "gpu_hours_big_ddvi_steps" => gpu_hours[2],
        "cpu_hours_small_dvi_steps" => cpu_hours[1], "cpu_hours_big_ddvi_steps" => cpu_hours[2],
        "updated_utc" => string(now(UTC)))
    isempty(memory_samples) || (recommendation["peak_whole_gpu_memory_mib"] = maximum(memory_samples))
    save_toml(joinpath(ROOT, "pilots", "recommendation.toml"), recommendation)
    open(joinpath(ROOT, "pilots", "report.md"), "w") do io
        println(io, "# DVI big pilot report\n\nAll six short artifact pilots and checkpoint replays passed on both backends.")
        println(io, "\nRecommended: $(recommendation["backend"])/$(recommendation["device"]), $(recommendation["workers"]) workers.")
        isempty(memory_samples) || println(io, "Peak sampled whole-device GPU memory: $(maximum(memory_samples)) MiB (includes other applications).")
        println(io, "\nGPU projections: $(round.(gpu_hours; digits=2)) hours; four-worker CPU: $(round.(cpu_hours; digits=2)) hours.")
        println(io, "The two projections use small-DVI and dDVI-big step counts, respectively, with a 25% margin. They are scenarios, not confidence bounds. Early stopping can differ.")
        println(io, "Concurrency scaling is measured on Power and extrapolated to the other datasets. Two-epoch pilots validate the pipeline, not final predictive quality.")
        println(io, "\n| Basis | Dataset | Steps | GPU hours | CPU hours |\n|---|---|---:|---:|---:|")
        for r in eachrow(table)
            println(io, "| $(r.basis) | $(r.dataset) | $(r.historical_steps) | $(round(r.gpu_hours; digits=2)) | $(round(r.cpu_four_hours; digits=2)) |")
        end
    end
    println(read(joinpath(ROOT, "pilots", "report.md"), String))
end

function gate()
    path = joinpath(ROOT, "pilots", "recommendation.toml")
    isfile(path) || error("Run the pilot first; no successful recommendation is saved")
    record = TOML.parsefile(path)
    record["status"] == "passed" || error("pilot did not pass")
    record["source_fingerprint"] == source_fingerprint() || error("source changed since pilot; rerun pilot")
    println(record["backend"], " ", record["device"], " ", record["workers"])
end

function main(args)
    if args == ["preflight"]
        preflight()
    elseif args == ["finish"]
        finish()
    elseif args == ["check-complete"]
        check_complete(canonical_config())
    elseif args == ["report"]
        report()
    elseif args == ["gate"]
        gate()
    elseif length(args) == 3 && args[1] == "profile"
        profile(args[2], args[3])
    elseif length(args) == 5 && args[1] == "throughput"
        throughput_worker(args[2], args[3], args[4], args[5])
    elseif length(args) == 3 && args[1] == "run"
        config = config_at(joinpath(ROOT, "dvi_shards", args[2]);
            split_ids = parse_split_ids(args[3], 20))
        run_benchmark(config)
        check_complete(config)
    else
        error("usage: full_two_hidden_experiment.jl preflight|finish|check-complete|report|gate|profile BACKEND DEVICE|throughput BACKEND DEVICE GROUP ID|run SHARD SPLITS")
    end
end
abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
