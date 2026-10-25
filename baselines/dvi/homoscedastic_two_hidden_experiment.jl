#!/usr/bin/env julia
using DVIUCI, CSV, DataFrames, Dates, JLD2, SHA, Statistics, TOML

module HomoReference
include("two_hidden_experiment.jl")
end
const HomoTable = HomoReference.FinalTable
const HOMO_ROOT = abspath(get(ENV, "DVI_HOMO_BIG_OUTPUT_ROOT", HomoTable.HOMO_BIG_ROOT))
const HOMO_LIKELIHOOD = "homoscedastic"

function method_propagation(method)
    method in ("ddvi", "dvi") || error("method must be ddvi or dvi")
    return method == "ddvi" ? "diagonal" : "full"
end
method_directory(method) = (method_propagation(method); joinpath(HOMO_ROOT, method))

function homo_config(method, directory = method_directory(method); kwargs...)
    propagation = method_propagation(method)
    base = HomoReference.experiment_config(directory)
    fields = (; (key => getfield(base, key) for key in fieldnames(DVIConfig))...)
    settings = merge(fields, (likelihoods = [HOMO_LIKELIHOOD], propagation = propagation,
        execution_backend = get(ENV, "DVI_HOMO_BIG_BACKEND", "reactant"),
        execution_device = get(ENV, "DVI_HOMO_BIG_DEVICE", "gpu"),
        implementation_version = DVIUCI.dvi_implementation_version(propagation, 2),
        selection_max_optimizer_steps = 50000, refit_max_optimizer_steps = 50000,
        training_budget_protocol = "selection-refit-max-50000-v1"), (; kwargs...))
    return validate_config(DVIConfig(; settings...))
end

function write_record(path, record)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(io -> TOML.print(io, record; sorted = true), temporary, "w")
    mv(temporary, path; force = true)
end

function homo_source_fingerprint()
    paths = vcat(filter(p -> endswith(p, ".jl"), readdir(joinpath(@__DIR__, "src"); join = true)),
        [joinpath(@__DIR__, name) for name in ("homoscedastic_two_hidden_experiment.jl",
            "run_homoscedastic_two_hidden_common.sh", "run_ddvi_homoscedastic_two_hidden_paper.sh",
            "run_dvi_homoscedastic_two_hidden_paper.sh", "two_hidden_experiment.jl", "merge_shards.jl",
            "Project.toml", "Manifest.toml")],
        [joinpath(@__DIR__, "..", "..", name) for name in
            ("Project.toml", "Manifest.toml", "paper_materials/make_final_baseline_table.jl")])
    return bytes2hex(sha256(join([bytes2hex(sha256(read(p))) for p in sort(paths)], "\n")))
end

function completed_homo_runs(config)
    DVIUCI.validate_resume_config(joinpath(config.output_dir, "config.toml"), config)
    runs = CSV.read(joinpath(config.output_dir, "runs.csv"), DataFrame)
    DVIUCI.validate_complete_runs(runs, config.datasets, config.split_ids,
        config.likelihoods, config.propagation)
    for row in eachrow(runs)
        DVIUCI.configuration_succeeded(runs, String(row.dataset), Int(row.split),
            HOMO_LIKELIHOOD, config.propagation, config) || error("missing or incompatible checkpoint")
    end
    return runs
end

function saved_homo_config(method)
    saved = TOML.parsefile(joinpath(method_directory(method), "config.toml"))
    return homo_config(method; execution_backend = saved["execution_backend"],
        execution_device = saved["execution_device"])
end

function replay_homo(config, runs)
    verified = NamedTuple[]
    for key in config.datasets
        dataset = DVIUCI.load_dataset(key)
        for row in eachrow(filter(r -> r.dataset == key, runs))
            checkpoint = JLD2.load(DVIUCI.posterior_checkpoint_path(config, key, row.split, HOMO_LIKELIHOOD))
            prepared = DVIUCI.outer_and_inner_splits(dataset, row.split, config)
            isequal(checkpoint["split_spec"], DVIUCI.split_spec_record(prepared.split_spec)) ||
                error("checkpoint split mismatch")
            metrics = predictive_metrics(checkpoint["posterior_params"], prepared.outer.x_test_standardized,
                prepared.outer.y_test_standardized, prepared.outer.y_test,
                checkpoint["standardizer"], HOMO_LIKELIHOOD, config)
            for name in DVIUCI.DVI_SCALAR_METRIC_NAMES
                actual = getproperty(metrics, name)
                isfinite(actual) && isapprox(actual, row[name]; rtol = 1e-9, atol = 1e-10) ||
                    error("checkpoint replay differs: $key split $(row.split), $name")
            end
            push!(verified, (dataset = key, split = row.split, status = "verified"))
        end
    end
    DVIUCI.atomic_csv_write(joinpath(config.output_dir, "checkpoint_replay.csv"), DataFrame(verified))
end

function homo_preflight(method)
    # Temporary manifests let check validate every split without creating campaign results.
    mktempdir() do directory
        config = homo_config(method, directory)
        DVIUCI.write_split_manifest(config)
        sources = HomoTable.table_sources(include_dvi_big = true)
        push!(sources, HomoTable.ResultSource("check", HOMO_LIKELIHOOD,
            config.propagation, directory; hidden_layers = 2))
        HomoTable.validate_split_manifests(sources)
        HomoTable.load_results(sources[1:end-1])
        saved = joinpath(method_directory(method), "config.toml")
        isfile(saved) && DVIUCI.validate_resume_config(saved, config)
    end
    println("$method: verified homoscedastic 50 -> 50 -> 1, 50k budgets, six datasets and 20 paired splits.")
end

function homo_smoke(method)
    reference = nothing
    backend = get(ENV, "DVI_HOMO_BIG_BACKEND", "reactant")
    device = get(ENV, "DVI_HOMO_BIG_DEVICE", "gpu")
    layouts = unique([("zygote", "cpu"), (backend, device)])
    for (backend, device) in layouts
        directory = joinpath(method_directory(method), "smoke", homo_source_fingerprint(), "$(backend)_$(device)")
        config = homo_config(method, directory; execution_backend = backend, execution_device = device,
            datasets = ["yacht"], split_ids = [1], max_epochs = 2, min_epochs = 1,
            validation_every = 1, patience = 2, kl_warmup_steps = 0, kl_anneal_steps = 0)
        run_benchmark(config)
        runs = completed_homo_runs(config)
        replay_homo(config, runs)
        path = DVIUCI.posterior_checkpoint_path(config, "yacht", 1, HOMO_LIKELIHOOD)
        before = sha256(read(path))
        run_benchmark(config)
        before == sha256(read(path)) || error("resume changed the checkpoint")
        row = only(eachrow(runs))
        if reference !== nothing
            for name in (:nll_original, :rmse_original)
                isapprox(row[name], reference[name]; rtol = 1e-3, atol = 1e-4) || error("CPU/backend parity failed")
            end
        end
        reference = NamedTuple(row)
    end
    write_record(joinpath(method_directory(method), "smoke", "passed.toml"),
        Dict("source_fingerprint" => homo_source_fingerprint(), "backend" => backend,
            "device" => device, "julia_version" => string(VERSION), "status" => "passed"))
    println("$method smoke passed: fit, checkpoint replay, resume and CPU/backend agreement.")
end

function prepare_homo(method, workers)
    workers in (1, 2, 4) || error("workers must be 1, 2, or 4")
    config = homo_config(method)
    config.execution_device == "gpu" && workers > 2 && error("use one or two GPU workers")
    homo_preflight(method)
    record = Dict("source_fingerprint" => homo_source_fingerprint(), "workers" => workers,
        "backend" => config.execution_backend, "device" => config.execution_device,
        "julia_version" => string(VERSION))
    path = joinpath(method_directory(method), "campaign.toml")
    if isfile(path)
        TOML.parsefile(path) == record || error("campaign source or execution settings changed; restore them or use a new output root")
    elseif isdir(joinpath(HOMO_ROOT, method * "_shards"))
        error("shards exist without campaign provenance; restore campaign.toml before resuming")
    end
    smoke_path = joinpath(method_directory(method), "smoke", "passed.toml")
    smoke = isfile(smoke_path) ? TOML.parsefile(smoke_path) : Dict()
    all(get(smoke, key, nothing) == record[key] for key in
        ("source_fingerprint", "backend", "device", "julia_version")) && get(smoke, "status", "") == "passed" ||
        homo_smoke(method)
    write_record(path, record)
end

function finish_homo(method)
    config = saved_homo_config(method)
    runs = completed_homo_runs(config)
    replay_homo(config, runs)
    sources = HomoTable.table_sources(include_dvi_big = true,
        include_ddvi_big_homo = method == "ddvi", include_dvi_big_homo = method == "dvi",
        ddvi_big_homo_directory = method_directory("ddvi"), dvi_big_homo_directory = method_directory("dvi"))
    HomoTable.validate_split_manifests(sources)
    results = HomoTable.load_results(sources)
    open(io -> HomoTable.render_markdown(io, sources, results),
        joinpath(config.output_dir, "final_baseline_table.md"), "w")
    small_label = method == "ddvi" ? "dDVI-Homo" : "DVI-Homo"
    small_directory = only(s.directory for s in sources if s.method == small_label)
    original = CSV.read(joinpath(small_directory, "runs.csv"), DataFrame)
    paired = innerjoin(select(runs, :dataset, :split, :nll_original => :new_nll, :rmse_original => :new_rmse),
        select(original, :dataset, :split, :nll_original => :old_nll, :rmse_original => :old_rmse);
        on = [:dataset, :split], validate = (true, true))
    nrow(paired) == 120 || error("incomplete paired comparison")
    paired.nll_change = paired.new_nll .- paired.old_nll
    paired.rmse_change = paired.new_rmse .- paired.old_rmse
    summary = combine(groupby(paired, :dataset), :nll_change => mean => :nll_change_mean,
        :nll_change => (x -> 1.96std(x) / sqrt(length(x))) => :nll_change_ci95,
        :rmse_change => mean => :rmse_change_mean,
        :rmse_change => (x -> 1.96std(x) / sqrt(length(x))) => :rmse_change_ci95)
    DVIUCI.atomic_csv_write(joinpath(config.output_dir, "paired_small_differences.csv"), paired)
    DVIUCI.atomic_csv_write(joinpath(config.output_dir, "paired_small_summary.csv"), summary)
    write_record(joinpath(config.output_dir, "status.toml"), Dict("status" => "complete",
        "verified_fits" => 120, "updated_utc" => string(now(UTC))))
    println("$method complete: all 120 homoscedastic fits verified; saved the 11-method comparison.")
end

function homo_main(args)
    length(args) >= 2 || error("usage: homoscedastic_two_hidden_experiment.jl METHOD check|smoke|prepare WORKERS|run SHARD SPLITS|check-complete|finish")
    method, action = args[1:2]
    method_propagation(method)
    if length(args) == 2 && action == "check"
        homo_preflight(method)
    elseif length(args) == 2 && action == "smoke"
        homo_preflight(method)
        homo_smoke(method)
    elseif length(args) == 3 && action == "prepare"
        prepare_homo(method, parse(Int, args[3]))
    elseif length(args) == 4 && action == "run"
        args[3] in ("01", "02", "03", "04") || error("invalid shard ID")
        config = homo_config(method, joinpath(HOMO_ROOT, method * "_shards", args[3]);
            split_ids = parse_split_ids(args[4], 20))
        run_benchmark(config)
        completed_homo_runs(config)
    elseif length(args) == 2 && action == "check-complete"
        completed_homo_runs(saved_homo_config(method))
    elseif length(args) == 2 && action == "finish"
        finish_homo(method)
    else
        error("invalid campaign command")
    end
end
abspath(PROGRAM_FILE) == abspath(@__FILE__) && homo_main(ARGS)
