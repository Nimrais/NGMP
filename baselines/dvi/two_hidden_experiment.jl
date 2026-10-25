#!/usr/bin/env julia

using DVIUCI
using CSV
using DataFrames
using Dates
using JLD2
using LinearAlgebra
using StableRNGs
using Statistics
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const EXPERIMENT_ROOT = joinpath(REPO_ROOT, "paper_materials", "dvi_uci",
    "wu2019_repeated_holdout_v1_20splits_heteroscedastic_2hidden50_v1_20261024")
const ORIGINAL_DDVI = joinpath(REPO_ROOT, "paper_materials", "dvi_uci",
    "wu2019_repeated_holdout_v1_20splits_robust_v1_20260731", "ddvi")

# Explicit settings protect this campaign from ambient DVI_* overrides.
function experiment_config(output_dir; split_ids = collect(1:20))
    return validate_config(DVIConfig(
        datasets = ["yacht", "concrete", "energy", "housing", "power", "wine"],
        n_splits = 20, split_ids = split_ids,
        hidden_layers = 2, hidden_units = 50,
        likelihoods = ["heteroscedastic"], propagation = "diagonal",
        execution_backend = "zygote", execution_device = "cpu",
        batch_size = 100, learning_rate = 0.0003,
        eb_alpha = 1.0, eb_beta = 10.0, initialization_scale = 5.0,
        bias_variance_divisor = 10.0, log_variance_head_posterior_std = 0.05,
        log_variance_head_initial_bias = 0.0, homo_log_variance = 0.0,
        max_epochs = 10000, min_epochs = 25, validation_every = 5, patience = 500,
        selection_max_optimizer_steps = 0, refit_max_optimizer_steps = 0,
        training_budget_protocol = "unlimited-v1", kl_schedule_unit = "steps",
        kl_warmup_steps = 14000, kl_anneal_steps = 1000,
        test_fraction = 0.1, validation_fraction = 0.1,
        split_seed = 20260726, model_seed = 20260727, gradient_clip = 0.1,
        numerical_protocol = "bounded-exp-step-kl-v1", safe_exp_min = -20.0,
        safe_exp_max = 20.0, output_dir = abspath(output_dir), resume = true,
        save_checkpoints = true, make_plot = false, show_progress = false,
    ))
end

function preflight()
    directory = joinpath(EXPERIMENT_ROOT, "preflight")
    config = experiment_config(directory)
    DVIUCI.ensure_output_directories(config)
    DVIUCI.save_config(config)
    DVIUCI.write_split_manifest(config)
    reference = JLD2.load(joinpath(ORIGINAL_DDVI, "split_manifest.jld2"))
    observed = JLD2.load(joinpath(directory, "split_manifest.jld2"))
    isequal(reference, observed) || error("split manifest differs from original dDVI")
    existing_sources = filter(source -> source.method != "dDVI-2HL-Hetero", FinalTable.SOURCES)
    FinalTable.validate_split_manifests(existing_sources)
    FinalTable.load_results(existing_sources)
    for key in config.datasets
        dataset = DVIUCI.load_dataset(key)
        println(key, ": ", size(dataset.features), ", coefficients=",
            DVIUCI.model_coefficient_count(size(dataset.features, 2), "heteroscedastic", config))
    end
    old_runs = CSV.read(joinpath(ORIGINAL_DDVI, "runs.csv"), DataFrame)
    profile_rows = NamedTuple[]
    for d in (4, 6, 8, 11, 13)
        rng = StableRNG(321 + d)
        features = randn(rng, Float32, 100, d)
        targets = randn(rng, Float32, 100)
        params = initialize_model(d, "heteroscedastic", config; seed = 322)
        backend = DVIUCI.initialize_training_backend(params, features, targets, "heteroscedastic", config)
        tracker = NumericalTracker()
        times = Float64[]
        for step in 1:23
            elapsed = @elapsed begin
                backend, loss, _ = DVIUCI.training_backend_epoch(
                    backend, rng, step, 15000 + step - 1; phase = "profile", tracker = tracker)
                isfinite(loss) || error("non-finite profiling loss")
            end
            step > 3 && push!(times, elapsed)
        end
        push!(profile_rows, (input_dimension = d, seconds_per_step = median(times)))
    end
    profile = DataFrame(profile_rows)
    CSV.write(joinpath(directory, "profile.csv"), profile)
    times = Dict(row.input_dimension => row.seconds_per_step for row in eachrow(profile))
    shard_seconds = zeros(4)
    for row in eachrow(old_runs)
        shard_seconds[mod1(row.split, 4)] += times[row.n_features] *
            (row.selection_optimizer_steps + row.refit_optimizer_steps)
    end
    projected_hours = 1.25 * maximum(shard_seconds) / 3600
    open(joinpath(directory, "timing.toml"), "w") do io
        TOML.print(io, Dict("projected_hours" => projected_hours, "workers" => 4,
            "basis" => "measured two-layer steps times historical one-layer step counts",
            "caveat" => "early stopping, compilation and concurrent-worker contention can change runtime",
            "julia_version" => string(VERSION), "safety_margin" => 1.25))
    end
    println("Projected four-worker runtime: ", round(projected_hours; digits = 2),
        " hours (historical step counts; actual stopping may differ)")
end

function replay_checkpoints(config, runs)
    replay_rows = NamedTuple[]
    for dataset_key in config.datasets
        dataset = DVIUCI.load_dataset(dataset_key)
        for row in eachrow(filter(r -> r.dataset == dataset_key, runs))
            checkpoint = JLD2.load(DVIUCI.posterior_checkpoint_path(
                config, dataset_key, row.split, "heteroscedastic"))
            checkpoint["schema_version"] == DVIUCI.DVI_POSTERIOR_SCHEMA_VERSION ||
                error("wrong checkpoint schema")
            checkpoint["hidden_layers"] == 2 && checkpoint["hidden_units"] == 50 ||
                error("wrong checkpoint architecture")
            prepared = DVIUCI.outer_and_inner_splits(dataset, row.split, config)
            isequal(checkpoint["split_spec"], DVIUCI.split_spec_record(prepared.split_spec)) ||
                error("checkpoint split mismatch")
            metrics = predictive_metrics(checkpoint["posterior_params"],
                prepared.outer.x_test_standardized, prepared.outer.y_test_standardized,
                prepared.outer.y_test, checkpoint["standardizer"], "heteroscedastic", config)
            for name in DVIUCI.DVI_SCALAR_METRIC_NAMES
                actual, expected = getproperty(metrics, name), row[name]
                isfinite(actual) && isapprox(actual, expected; rtol = 1e-9, atol = 1e-10) ||
                    error("checkpoint replay differs: $dataset_key split $(row.split), $name")
            end
            metrics.mean_total_variance_original > 0 || error("nonpositive predictive variance")
            push!(replay_rows, (dataset = dataset_key, split = row.split, status = "verified"))
        end
    end
    CSV.write(joinpath(config.output_dir, "checkpoint_replay.csv"), DataFrame(replay_rows))
end

module FinalTable
include(joinpath(@__DIR__, "..", "..", "paper_materials", "make_final_baseline_table.jl"))
end

function finish()
    config = experiment_config(joinpath(EXPERIMENT_ROOT, "ddvi"))
    runs = CSV.read(joinpath(config.output_dir, "runs.csv"), DataFrame)
    DVIUCI.validate_complete_runs(runs, config.datasets, collect(1:20), ["heteroscedastic"], "diagonal")
    replay_checkpoints(config, runs)
    FinalTable.validate_split_manifests(FinalTable.SOURCES)
    results = FinalTable.load_results(FinalTable.SOURCES)
    open(joinpath(EXPERIMENT_ROOT, "final_baseline_table.md"), "w") do io
        FinalTable.render_markdown(io, FinalTable.SOURCES, results)
    end
    original = CSV.read(joinpath(ORIGINAL_DDVI, "runs.csv"), DataFrame)
    paired = innerjoin(select(runs, :dataset, :split, :nll_original => :new_nll,
        :rmse_original => :new_rmse), select(original, :dataset, :split,
        :nll_original => :old_nll, :rmse_original => :old_rmse); on = [:dataset, :split],
        validate = (true, true))
    nrow(paired) == 120 || error("paired comparison is incomplete")
    paired.nll_change = paired.new_nll .- paired.old_nll
    paired.rmse_change = paired.new_rmse .- paired.old_rmse
    CSV.write(joinpath(EXPERIMENT_ROOT, "paired_differences.csv"), paired)
    comparison = combine(groupby(paired, :dataset),
        :nll_change => mean => :nll_change_mean,
        :nll_change => (x -> 1.96std(x) / sqrt(length(x))) => :nll_change_ci95,
        :rmse_change => mean => :rmse_change_mean,
        :rmse_change => (x -> 1.96std(x) / sqrt(length(x))) => :rmse_change_ci95)
    CSV.write(joinpath(EXPERIMENT_ROOT, "paired_summary.csv"), comparison)
    open(joinpath(EXPERIMENT_ROOT, "comparison.md"), "w") do io
        println(io, "Differences are two-hidden-layer minus original one-hidden-layer dDVI; negative is better. Intervals are 1.96 × paired standard error across 20 splits.\n")
        println(io, "| Dataset | NLL change (95% CI) | RMSE change (95% CI) |\n|---|---:|---:|")
        for row in eachrow(comparison)
            println(io, "| ", row.dataset, " | ", DVIUCI.format_ci95(row.nll_change_mean, row.nll_change_ci95 / 1.96),
                " | ", DVIUCI.format_ci95(row.rmse_change_mean, row.rmse_change_ci95 / 1.96), " |")
        end
    end
    println("Verified all 120 checkpoints; saved full table and paired comparison in ", EXPERIMENT_ROOT)
end

function main(args)
    isempty(args) && error("usage: two_hidden_experiment.jl preflight | run SHARD_ID SPLIT_IDS | finish")
    if args == ["preflight"]
        preflight()
    elseif length(args) == 3 && args[1] == "run"
        args[2] in ("01", "02", "03", "04") || error("invalid shard ID")
        config = experiment_config(joinpath(EXPERIMENT_ROOT, "ddvi_shards", args[2]);
            split_ids = parse_split_ids(args[3], 20))
        run_benchmark(config)
        runs = CSV.read(joinpath(config.output_dir, "runs.csv"), DataFrame)
        DVIUCI.validate_complete_runs(runs, config.datasets, config.split_ids, config.likelihoods, config.propagation)
    elseif args == ["finish"]
        finish()
    else
        error("invalid arguments")
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
