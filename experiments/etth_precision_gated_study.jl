#!/usr/bin/env julia

# Validation-selected precision-gated ensemble benchmark for ETTh1/ETTh2.
#
# Selection is deliberately a separate data path: `load_validation_only` opens
# exactly the three validation datasets in each prepared cache.  Test targets
# are only reachable from `load_refit_data`, which is called by the `refit`
# command after `selected_configurations.csv` exists.
#
# Usage (run selection cells sequentially; the RFF candidates are CPU-heavy):
#   julia --project=. experiments/etth_precision_gated_study.jl select ETTh1 96
#   julia --project=. experiments/etth_precision_gated_study.jl choose
#   julia --project=. experiments/etth_precision_gated_study.jl refit ETTh1 96
#   julia --project=. experiments/etth_precision_gated_study.jl verify

using CSV
using DataFrames
using Distributions
using JLD2
using Printf
using Serialization
using SHA
using Statistics
using TOML

const STUDY_ROOT = normpath(joinpath(@__DIR__, ".."))
const STUDY_OUTPUT = get(
    ENV,
    "ETTH_STUDY_OUTPUT",
    joinpath(STUDY_ROOT, "paper_materials", "etth_precision_gated"),
)
const STUDY_DATASETS = ("ETTh1", "ETTh2")
const STUDY_HORIZONS = (96, 192, 336, 720)
const STUDY_TEST_COUNTS = Dict(96 => 3446, 192 => 3426, 336 => 3398, 720 => 3321)
const STUDY_Z95 = 1.959963984540054

# Freeze the predeclared consensus-x protocol before loading its implementation.
ENV["EDH_RESULTS_DIR"] = joinpath(STUDY_OUTPUT, "raw_consensus")
ENV["EDH_XMSG"] = "student"
ENV["EDH_TOP"] = "fixed"
ENV["EDH_CALIB"] = "0"
ENV["EDH_ITERATIONS"] = get(ENV, "ETTH_STUDY_ITERATIONS", "60")
ENV["EDH_RFF_SEED"] = "12345"

const CONSENSUS = Module(gensym(:ETThConsensusStudy), true, true)
Core.eval(CONSENSUS, :(include(path) = Base.include($CONSENSUS, path)))
Base.include(
    CONSENSUS,
    joinpath(@__DIR__, "etth2_consensus_precision_hierarchy.jl"),
)

const LINEAR2 = Module(gensym(:ETThLinear2Study), true, true)
Core.eval(LINEAR2, :(include(path) = Base.include($LINEAR2, path)))
Base.include(LINEAR2, joinpath(@__DIR__, "dynamic_deep_kernel_precision.jl"))

struct ValidationOnlyCache
    targets::Vector{Float64}
    predictions::Matrix{Float64}
    features::Vector{Vector{Float64}}
    source::String
end

struct RefitCache
    validation::ValidationOnlyCache
    test_targets::Vector{Float64}
    test_predictions::Matrix{Float64}
    test_features::Vector{Vector{Float64}}
end

function cache_path(dataset, horizon)
    candidates = CONSENSUS.cache_candidates(dataset, horizon)
    index = findfirst(isfile, candidates)
    isnothing(index) && error("missing prepared cache for $dataset H=$horizon")
    return candidates[index]
end

"""Load only validation arrays.  This is the sole loader used by `select`."""
function load_validation_only(dataset, horizon)
    path = cache_path(dataset, horizon)
    return jldopen(path, "r") do file
        ValidationOnlyCache(
            Float64.(file["y_val"]),
            Float64.(file["predictions_val"]),
            [Float64.(value) for value in file["features_val"]],
            path,
        )
    end
end

"""Load validation and test arrays, reachable only from the post-selection refit."""
function load_refit_data(dataset, horizon)
    validation = load_validation_only(dataset, horizon)
    path = validation.source
    return jldopen(path, "r") do file
        RefitCache(
            validation,
            Float64.(file["y_test"]),
            Float64.(file["predictions_test"]),
            [Float64.(value) for value in file["features_test"]],
        )
    end
end

function chronological_validation_split(cache::ValidationOnlyCache)
    n = length(cache.targets)
    n_fit = floor(Int, 0.8n)
    fit = ValidationOnlyCache(
        cache.targets[1:n_fit],
        cache.predictions[:, 1:n_fit],
        cache.features[1:n_fit],
        cache.source,
    )
    selection = ValidationOnlyCache(
        cache.targets[(n_fit + 1):end],
        cache.predictions[:, (n_fit + 1):end],
        cache.features[(n_fit + 1):end],
        cache.source,
    )
    isempty(selection.targets) && error("empty chronological selection partition")
    return fit, selection
end

const STUDY_CANDIDATES = let candidates = NamedTuple[]
    push!(candidates, (
        id = "linear2_ngmp",
        kind = :linear2,
        setup = :linear_all,
        gain = 1.0,
        carrier = :beta,
        beta_rate = 1e3,
    ))
    for setup in (:linear_all, :rff_all), gain in (1.5, 3.0, 5.0)
        push!(candidates, (
            id = "consensus_$(setup)_g$(replace(string(gain), "." => "p"))_no_beta",
            kind = :consensus,
            setup,
            gain,
            carrier = :nobeta,
            beta_rate = 1e3,
        ))
        for rate in (1.0, 1e3)
            rate_id = rate == 1.0 ? "1" : "1000"
            push!(candidates, (
                id = "consensus_$(setup)_g$(replace(string(gain), "." => "p"))_beta_rate_$rate_id",
                kind = :consensus,
                setup,
                gain,
                carrier = :beta,
                beta_rate = rate,
            ))
        end
    end
    Tuple(candidates)
end

candidate_by_id(id) = only(filter(candidate -> candidate.id == id, STUDY_CANDIDATES))

function normal_metrics(means, sigmas, targets)
    n = length(targets)
    length(means) == n == length(sigmas) || error("prediction/target length mismatch")
    all(isfinite, means) || error("non-finite predictive mean")
    all(value -> isfinite(value) && value > 0, sigmas) ||
        error("improper predictive distribution")
    squared = (means .- targets) .^ 2
    nll_terms = [-logpdf(Normal(means[j], sigmas[j]), targets[j]) for j in 1:n]
    covered = (targets .>= means .- STUDY_Z95 .* sigmas) .&
        (targets .<= means .+ STUDY_Z95 .* sigmas)
    widths = 2STUDY_Z95 .* sigmas
    return (
        mse = mean(squared),
        mse_ci = STUDY_Z95 * std(squared) / sqrt(n),
        nll = mean(nll_terms),
        nll_ci = STUDY_Z95 * std(nll_terms) / sqrt(n),
        cov95 = mean(covered),
        cov_ci = STUDY_Z95 * sqrt(mean(covered) * (1 - mean(covered)) / n),
        interval_width = mean(widths),
        interval_width_ci = STUDY_Z95 * std(widths) / sqrt(n),
        n = n,
    )
end

function linear_result_path(stage, dataset, horizon)
    joinpath(
        STUDY_OUTPUT,
        "raw_linear2",
        stage,
        "$(dataset)_h$(horizon)_linear2_ngmp.jls",
    )
end

function run_linear2(stage, dataset, horizon, train, evaluation)
    path = linear_result_path(stage, dataset, horizon)
    if isfile(path) && get(ENV, "ETTH_STUDY_FORCE", "0") != "1"
        @printf("skip (exists): %s\n", path)
        return deserialize(path)
    end
    mkpath(dirname(path))
    record = Dict{String,Any}(
        "stage" => stage,
        "dataset" => dataset,
        "horizon" => horizon,
        "configuration_id" => "linear2_ngmp",
        "status" => "ok",
        "error" => "",
        "n_fit" => length(train.targets),
    )
    elapsed = @elapsed try
        config = merge(
            LINEAR2.DDK_CONFIG,
            (; n_obs = length(train.targets), alpha = 0.2),
        )
        arm = LINEAR2.fit_arm(
            train.features,
            train.targets,
            train.predictions,
            config,
        )
        means, sigmas = LINEAR2.predict_arm(
            arm,
            evaluation.features,
            evaluation.predictions,
            config,
        )
        metrics = normal_metrics(means, sigmas, evaluation.targets)
        record["metrics"] = Dict(String(key) => Float64(value) for
            (key, value) in pairs(metrics))
        record["mu"] = means
        record["sigma"] = sigmas
        record["parameter_count"] = 7 * (arm.dimension + 2)
        record["dimension"] = arm.dimension
    catch exception
        record["status"] = "unstable"
        record["error"] = sprint(showerror, exception)
    end
    record["elapsed"] = elapsed
    serialize(path, record)
    return record
end

function consensus_tag(stage, dataset, horizon)
    "$(stage)_$(lowercase(dataset))_h$(horizon)"
end

function consensus_result_path(stage, dataset, horizon, candidate)
    tag = consensus_tag(stage, dataset, horizon)
    base = CONSENSUS.arm_name(candidate.setup, 2, candidate.carrier, :scalar)
    joinpath(
        CONSENSUS.RESULTS_DIR,
        tag,
        base * "__" * candidate.id * "_tx.jld2",
    )
end

function run_consensus(stage, dataset, horizon, candidate, train, evaluation)
    CONSENSUS.HYPER[] = (
        alpha = 0.2,
        momentum = 0.0,
        method = :damped,
        gain = candidate.gain,
        beta_rate0 = candidate.beta_rate,
        anchor_var = 1.0,
    )
    bases = CONSENSUS.base_designs(
        train.features,
        evaluation.features,
        length(train.targets),
        12345;
        need_rff = candidate.setup == :rff_all,
    )
    generic = Dict{String,Any}(
        "y_val" => train.targets,
        "predictions_val" => train.predictions,
        "features_val" => train.features,
        # These are evaluation arrays.  During selection they are sliced only
        # from the original validation partition.
        "y_test" => evaluation.targets,
        "predictions_test" => evaluation.predictions,
        "features_test" => evaluation.features,
    )
    CONSENSUS.run_arm(
        generic,
        (dataset, horizon),
        consensus_tag(stage, dataset, horizon),
        candidate.setup,
        2,
        candidate.carrier,
        :scalar,
        length(train.targets),
        bases;
        suffix = "__" * candidate.id,
    )
    path = consensus_result_path(stage, dataset, horizon, candidate)
    result = JLD2.load(path)["result"]
    result["stage"] = stage
    result["configuration_id"] = candidate.id
    p = size(candidate.setup == :rff_all ? bases.rff.train : bases.linear.train, 2)
    result["parameter_count"] = 7 * 2 * p +
        (candidate.carrier == :beta ? 7 : 0) + 1
    JLD2.jldsave(path; result)
    return result
end

function result_for(stage, dataset, horizon, candidate)
    if candidate.kind == :linear2
        path = linear_result_path(stage, dataset, horizon)
        return isfile(path) ? deserialize(path) : nothing
    end
    path = consensus_result_path(stage, dataset, horizon, candidate)
    return isfile(path) ? JLD2.load(path)["result"] : nothing
end

function run_selection_cell(dataset, horizon)
    dataset in STUDY_DATASETS || error("unknown dataset: $dataset")
    horizon in STUDY_HORIZONS || error("unknown horizon: $horizon")
    validation = load_validation_only(dataset, horizon)
    fit, selection = chronological_validation_split(validation)
    @printf(
        "%s H=%d validation selection: first %d fit, final %d score\n",
        dataset,
        horizon,
        length(fit.targets),
        length(selection.targets),
    )
    for candidate in STUDY_CANDIDATES
        @printf("\n[%s]\n", candidate.id)
        if candidate.kind == :linear2
            run_linear2("selection", dataset, horizon, fit, selection)
        else
            run_consensus("selection", dataset, horizon, candidate, fit, selection)
        end
    end
    write_selection_csv()
    return nothing
end

function metric_value(result, name)
    result === nothing && return missing
    get(result, "status", "unstable") == "ok" || return missing
    name == "n" && return Float64(length(result["mu"]))
    metrics = result["metrics"]
    haskey(metrics, name) || return missing
    value = metrics[name]
    return value isa Real && isfinite(value) ? Float64(value) : missing
end

function selection_rows(; selected = Dict{String,String}())
    rows = NamedTuple[]
    for dataset in STUDY_DATASETS, candidate in STUDY_CANDIDATES
        results = [result_for("selection", dataset, horizon, candidate)
                   for horizon in STUDY_HORIZONS]
        complete = all(!isnothing, results)
        proper = complete && all(result ->
            get(result, "status", "unstable") == "ok" &&
            !ismissing(metric_value(result, "nll")) &&
            !ismissing(metric_value(result, "mse")), results)
        mean_nll = proper ? mean(metric_value(result, "nll") for result in results) : missing
        mean_mse = proper ? mean(metric_value(result, "mse") for result in results) : missing
        for (horizon, result) in zip(STUDY_HORIZONS, results)
            source = candidate.kind == :linear2 ?
                linear_result_path("selection", dataset, horizon) :
                consensus_result_path("selection", dataset, horizon, candidate)
            status = isnothing(result) ? "missing" : get(result, "status", "unstable")
            count_value = metric_value(result, "n")
            push!(rows, (
                method = "Precision-gated ensemble",
                optimizer = "NGMP",
                architecture = candidate.kind == :linear2 ?
                    "affine independent-expert" :
                    "consensus-x depth-2 $(candidate.setup)",
                dataset,
                horizon,
                mse = metric_value(result, "mse"),
                mse_ci = metric_value(result, "mse_ci"),
                nll = metric_value(result, "nll"),
                nll_ci = metric_value(result, "nll_ci"),
                coverage95 = metric_value(result, "cov95"),
                coverage95_ci = metric_value(result, "cov_ci"),
                interval_width = metric_value(result, "interval_width"),
                interval_width_ci = metric_value(result, "interval_width_ci"),
                test_count = ismissing(count_value) ? missing : Int(round(count_value)),
                configuration_id = candidate.id,
                parameter_count = isnothing(result) ? missing :
                    get(result, "parameter_count", missing),
                proper,
                status,
                selected = get(selected, dataset, "") == candidate.id,
                selection_mean_nll = mean_nll,
                selection_mean_mse = mean_mse,
                source_provenance = relpath(source, STUDY_ROOT),
            ))
        end
    end
    return rows
end

function write_selection_csv(; selected = Dict{String,String}())
    mkpath(STUDY_OUTPUT)
    path = joinpath(STUDY_OUTPUT, "selection.csv")
    CSV.write(path, DataFrame(selection_rows(; selected)))
    return path
end

function choose_configurations()
    selected = Dict{String,String}()
    selected_rows = NamedTuple[]
    for dataset in STUDY_DATASETS
        ranking = NamedTuple[]
        for candidate in STUDY_CANDIDATES
            results = [result_for("selection", dataset, horizon, candidate)
                       for horizon in STUDY_HORIZONS]
            any(isnothing, results) && error(
                "selection is incomplete for $dataset / $(candidate.id)",
            )
            proper = all(result ->
                get(result, "status", "unstable") == "ok" &&
                !ismissing(metric_value(result, "nll")) &&
                !ismissing(metric_value(result, "mse")), results)
            proper || continue
            counts = unique(get(result, "parameter_count", typemax(Int)) for result in results)
            length(counts) == 1 || error("parameter count changes across horizons")
            push!(ranking, (
                configuration_id = candidate.id,
                mean_nll = mean(metric_value(result, "nll") for result in results),
                mean_mse = mean(metric_value(result, "mse") for result in results),
                parameter_count = only(counts),
            ))
        end
        isempty(ranking) && error("no proper candidate remains for $dataset")
        sort!(ranking; by = row -> (
            row.mean_nll,
            row.mean_mse,
            row.parameter_count,
            row.configuration_id,
        ))
        winner = first(ranking)
        selected[dataset] = winner.configuration_id
        push!(selected_rows, (; dataset, winner...))
        @printf(
            "%s selected %s: mean NLL %.6f, mean MSE %.6f, %d parameters\n",
            dataset,
            winner.configuration_id,
            winner.mean_nll,
            winner.mean_mse,
            winner.parameter_count,
        )
    end
    mkpath(STUDY_OUTPUT)
    CSV.write(
        joinpath(STUDY_OUTPUT, "selected_configurations.csv"),
        DataFrame(selected_rows),
    )
    write_selection_csv(; selected)
    return selected
end

function read_selected_configurations()
    path = joinpath(STUDY_OUTPUT, "selected_configurations.csv")
    isfile(path) || error("run `choose` before refitting")
    frame = CSV.read(path, DataFrame)
    nrow(frame) == length(STUDY_DATASETS) || error("expected one selection per dataset")
    length(unique(frame.dataset)) == length(STUDY_DATASETS) ||
        error("duplicate selected dataset")
    return Dict(String(row.dataset) => String(row.configuration_id) for row in eachrow(frame))
end

function run_refit_cell(dataset, horizon)
    selected = read_selected_configurations()
    haskey(selected, dataset) || error("no selected configuration for $dataset")
    candidate = candidate_by_id(selected[dataset])
    data = load_refit_data(dataset, horizon)
    length(data.test_targets) == STUDY_TEST_COUNTS[horizon] || error(
        "unexpected test count for $dataset H=$horizon",
    )
    evaluation = ValidationOnlyCache(
        data.test_targets,
        data.test_predictions,
        data.test_features,
        data.validation.source,
    )
    if candidate.kind == :linear2
        run_linear2("test", dataset, horizon, data.validation, evaluation)
    else
        run_consensus("test", dataset, horizon, candidate, data.validation, evaluation)
    end
    return nothing
end

function array_digest(values)
    bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(values)))))
end

function verify_protocol()
    selected = read_selected_configurations()
    length(selected) == 2 || error("expected exactly two dataset selections")
    for dataset in STUDY_DATASETS
        id = selected[dataset]
        candidate_by_id(id)
        for horizon in STUDY_HORIZONS
            selection = result_for("selection", dataset, horizon, candidate_by_id(id))
            isnothing(selection) && error("missing selected validation result")
            test = result_for("test", dataset, horizon, candidate_by_id(id))
            isnothing(test) && error("missing selected test result")
            get(test, "configuration_id", "") == id ||
                error("selected configuration drifted at $dataset H=$horizon")
            Int(round(metric_value(test, "n"))) == STUDY_TEST_COUNTS[horizon] ||
                error("test-count mismatch at $dataset H=$horizon")
        end
    end

    # The IVON checkpoints preserve exact target traces.  Equality here verifies
    # that neural and NGMP summaries refer to identical ordered test origins.
    for dataset in STUDY_DATASETS, horizon in STUDY_HORIZONS
        cache = load_refit_data(dataset, horizon)
        ivon_path = joinpath(
            STUDY_ROOT,
            "paper_materials",
            "ivon",
            "moe",
            "checkpoints",
            "$(dataset)_h$(horizon)_moe_ivon.jld2",
        )
        traces = JLD2.load(ivon_path, "standardized_traces")
        traces.target == cache.test_targets ||
            error("target trace mismatch for $dataset H=$horizon")
    end
    println("verified: validation-only selection, two dataset selections, eight test cells, identical IVON/NGMP targets")
    return true
end

function write_protocol_config()
    mkpath(STUDY_OUTPUT)
    protocol = Dict(
        "benchmark" => "ETTh precision-gated ensemble study",
        "datasets" => collect(STUDY_DATASETS),
        "horizons" => collect(STUDY_HORIZONS),
        "selection_fit_fraction" => 0.8,
        "selection_score_fraction" => 0.2,
        "selection_metric" => "mean standardized NLL, horizons equally weighted",
        "tie_breakers" => ["mean MSE", "parameter count", "configuration ID"],
        "damping_alpha" => 0.2,
        "momentum" => 0.0,
        "anchor_variance" => 1.0,
        "rff_lengthscale" => 1.5,
        "rff_count" => 400,
        "seed" => 12345,
        "x_message" => "Student-t tangent projection",
        "excluded" => ["depth-three", "functional-tau-y", "calibration-link"],
        "standardized_target" => "OT",
        "test_counts" => Dict(string(key) => value for (key, value) in STUDY_TEST_COUNTS),
    )
    open(joinpath(STUDY_OUTPUT, "config.toml"), "w") do io
        TOML.print(io, protocol; sorted = true)
    end
end

function main(args)
    write_protocol_config()
    isempty(args) && error("usage: select|refit <dataset> <horizon> | choose | verify")
    mode = args[1]
    if mode in ("select", "refit")
        length(args) == 3 || error("usage: $mode <dataset> <horizon>")
        dataset, horizon = args[2], parse(Int, args[3])
        mode == "select" ? run_selection_cell(dataset, horizon) :
            run_refit_cell(dataset, horizon)
    elseif mode == "choose"
        length(args) == 1 || error("usage: choose")
        choose_configurations()
    elseif mode == "verify"
        length(args) == 1 || error("usage: verify")
        verify_protocol()
    else
        error("unknown mode: $mode")
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
