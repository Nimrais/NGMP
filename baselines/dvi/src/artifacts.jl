const CONFIG_RUNTIME_ONLY_FIELDS = Set([
    "output_dir",
    "resume",
    "make_plot",
    "show_progress",
    "selection_only",
])
const DVI_POSTERIOR_SCHEMA_VERSION = "dvi-uci-posterior-v3"

function configuration_stem(
    dataset::AbstractString,
    split_id::Int,
    likelihood::AbstractString,
    propagation::AbstractString,
)
    return @sprintf(
        "%s_split%02d_%s_%s",
        dataset,
        split_id,
        likelihood,
        propagation,
    )
end

function posterior_checkpoint_path(
    config::DVIConfig,
    dataset::String,
    split_id::Int,
    likelihood::String,
)
    return joinpath(
        config.output_dir,
        "checkpoints",
        configuration_stem(
            dataset, split_id, likelihood, config.propagation,
        ) * ".jld2",
    )
end

function split_spec_record(spec::UCISplitSpec)
    return (
        protocol_version = spec.protocol_version,
        split_id = spec.split_id,
        n_observations = spec.n_observations,
        outer_seed = spec.outer_seed,
        inner_seed = spec.inner_seed,
        test_fraction = spec.test_fraction,
        validation_fraction = spec.validation_fraction,
        train_indices = spec.train_indices,
        test_indices = spec.test_indices,
        inner_train_indices = spec.inner_train_indices,
        validation_indices = spec.validation_indices,
    )
end

function atomic_jld2_write(path::AbstractString, record::NamedTuple)
    mkpath(dirname(path))
    temporary = path * ".tmp-" * string(getpid())
    try
        JLD2.jldsave(temporary; record...)
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return path
end

function ensure_output_directories(config::DVIConfig)
    mkpath(config.output_dir)
    mkpath(joinpath(config.output_dir, "histories"))
    mkpath(joinpath(config.output_dir, "failures"))
    config.save_checkpoints &&
        mkpath(joinpath(config.output_dir, "checkpoints"))
    return config.output_dir
end

function validate_resume_config(path::AbstractString, config::DVIConfig)
    existing = TOML.parsefile(path)
    requested = config_dictionary(config)
    for key in keys(requested)
        key in CONFIG_RUNTIME_ONLY_FIELDS && continue
        haskey(existing, key) ||
            throw(ArgumentError("existing config is missing '$key': $path"))
        isequal(existing[key], requested[key]) || throw(ArgumentError(
            "result-affecting config '$key' differs in $path",
        ))
    end
    return nothing
end

function save_config(config::DVIConfig)
    path = joinpath(config.output_dir, "config.toml")
    isfile(path) && validate_resume_config(path, config)
    temporary = path * ".tmp-" * string(getpid())
    open(temporary, "w") do io
        TOML.print(io, config_dictionary(config); sorted = true)
    end
    mv(temporary, path; force = true)
    return path
end

function write_split_manifest(config::DVIConfig)
    datasets = Dict{String, Any}()
    for dataset_key in config.datasets
        dataset = load_dataset(dataset_key)
        datasets[dataset_key] = [
            split_spec_record(spec)
            for spec in uci_regression_splits(
                size(dataset.features, 1),
                config.n_splits;
                base_seed = config.split_seed,
                test_fraction = config.test_fraction,
                validation_fraction = config.validation_fraction,
            )
        ]
    end
    return atomic_jld2_write(
        joinpath(config.output_dir, "split_manifest.jld2"),
        (
            schema_version = "uci-split-manifest-v1",
            protocol_version = UCI_SPLIT_PROTOCOL_VERSION,
            base_seed = config.split_seed,
            n_splits = config.n_splits,
            test_fraction = config.test_fraction,
            validation_fraction = config.validation_fraction,
            datasets = datasets,
        ),
    )
end

function atomic_csv_write(path::AbstractString, table)
    temporary = path * ".tmp"
    CSV.write(temporary, table)
    mv(temporary, path; force = true)
    return path
end

function load_run_table(config::DVIConfig)
    path = joinpath(config.output_dir, "runs.csv")
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame)
end

function configuration_succeeded(
    runs::DataFrame,
    dataset::String,
    split_id::Int,
    likelihood::String,
    propagation::String,
    config::DVIConfig,
)
    isempty(runs) && return false
    required = (:dataset, :split, :likelihood, :propagation, :status)
    all(name -> name in propertynames(runs), required) || return false
    expected_status = config.selection_only ? "selection_success" : "success"
    row_succeeded = any(eachrow(runs)) do row
        string(row.dataset) == dataset &&
            Int(row.split) == split_id &&
            string(row.likelihood) == likelihood &&
            string(row.propagation) == propagation &&
            string(row.status) == expected_status
    end
    row_succeeded || return false
    (config.save_checkpoints && !config.selection_only) || return true
    path = posterior_checkpoint_path(
        config, dataset, split_id, likelihood,
    )
    return try
        JLD2.jldopen(path, "r") do file
            all(
                key -> haskey(file, key),
                (
                    "schema_version",
                    "dataset",
                    "split_id",
                    "likelihood",
                    "propagation",
                    "posterior_params",
                    "split_spec",
                ),
            ) &&
                file["schema_version"] == DVI_POSTERIOR_SCHEMA_VERSION &&
                file["dataset"] == dataset &&
                file["split_id"] == split_id &&
                file["likelihood"] == likelihood &&
                file["propagation"] == propagation
        end
    catch
        false
    end
end

function append_run!(runs::DataFrame, row::NamedTuple, config::DVIConfig)
    incoming = DataFrame([row])
    updated = if isempty(runs)
        incoming
    else
        required = (:dataset, :split, :likelihood, :propagation)
        all(name -> name in propertynames(runs), required) || throw(
            ArgumentError("existing runs.csv lacks configuration identity"),
        )
        keep = map(eachrow(runs)) do existing
            !(string(existing.dataset) == string(row.dataset) &&
              Int(existing.split) == Int(row.split) &&
              string(existing.likelihood) == string(row.likelihood) &&
              string(existing.propagation) == string(row.propagation))
        end
        vcat(runs[keep, :], incoming; cols = :union)
    end
    sort!(updated, [:dataset, :split, :likelihood, :propagation])
    atomic_csv_write(joinpath(config.output_dir, "runs.csv"), updated)
    return updated
end

function validate_complete_runs(
    runs::DataFrame,
    datasets::Vector{String},
    split_ids::Vector{Int},
    likelihoods::Vector{String},
    propagation::String,
)
    required = (:dataset, :split, :likelihood, :propagation, :status)
    all(name -> name in propertynames(runs), required) || throw(
        ArgumentError("runs table lacks completeness columns"),
    )
    identity_columns = [:dataset, :split, :likelihood, :propagation]
    nrow(unique(runs[:, identity_columns])) == nrow(runs) || throw(
        ArgumentError("runs table contains duplicate configurations"),
    )
    all(string.(runs.status) .== "success") || throw(ArgumentError(
        "runs table contains failed or selection-only configurations",
    ))
    expected = Set(
        (dataset, split_id, likelihood, propagation)
        for dataset in datasets
        for split_id in split_ids
        for likelihood in likelihoods
    )
    observed = Set(
        (
            string(row.dataset),
            Int(row.split),
            string(row.likelihood),
            string(row.propagation),
        )
        for row in eachrow(runs)
    )
    observed == expected || throw(ArgumentError(
        "runs table does not exactly match the expected configurations",
    ))
    return runs
end

function summary_table(runs::DataFrame)
    isempty(runs) && return DataFrame()
    successful = filter(row -> row.status == "success", runs)
    isempty(successful) && return DataFrame()
    rows = NamedTuple[]
    for group in groupby(
        successful,
        [:dataset, :dataset_name, :likelihood, :propagation],
    )
        metric_summary = Pair{Symbol, Any}[]
        for metric in DVI_SCALAR_METRIC_NAMES
            values = Float64.(group[!, metric])
            push!(metric_summary, Symbol(metric, "_mean") => mean(values))
            push!(
                metric_summary,
                Symbol(metric, "_std") =>
                    (length(values) > 1 ?
                     std(values; corrected = true) : 0.0),
            )
            push!(
                metric_summary,
                Symbol(metric, "_se") =>
                    (length(values) > 1 ?
                     std(values; corrected = true) /
                        sqrt(length(values)) : 0.0),
            )
        end
        push!(rows, (; (
            dataset = string(first(group.dataset)),
            dataset_name = string(first(group.dataset_name)),
            likelihood = string(first(group.likelihood)),
            propagation = string(first(group.propagation)),
            n = nrow(group),
            metric_summary...,
        )...))
    end
    return sort!(
        DataFrame(rows),
        [:dataset, :propagation, :likelihood],
    )
end

format_pm(mean_value, std_value) =
    @sprintf("%.4f ± %.4f", mean_value, std_value)

function write_tables(summary::DataFrame, config::DVIConfig)
    isempty(summary) && return nothing
    atomic_csv_write(joinpath(config.output_dir, "summary.csv"), summary)
    markdown_path = joinpath(config.output_dir, "table.md")
    open(markdown_path, "w") do io
        println(
            io,
            "| Dataset | Method | Likelihood | Runs | LPD (original) | RMSE | LPD (standardized) |",
        )
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        for row in eachrow(summary)
            println(
                io,
                "| $(row.dataset_name) | $(row.propagation) | " *
                "$(row.likelihood) | $(row.n) | " *
                "$(format_pm(row.lpd_original_mean, row.lpd_original_std)) | " *
                "$(format_pm(row.rmse_original_mean, row.rmse_original_std)) | " *
                "$(format_pm(row.lpd_standardized_mean, row.lpd_standardized_std)) |",
            )
        end
    end

    latex_path = joinpath(config.output_dir, "table.tex")
    open(latex_path, "w") do io
        println(io, "\\begin{tabular}{lllrrr}")
        println(io, "\\toprule")
        println(
            io,
            "Dataset & Method & Likelihood & Runs & LPD (original) & RMSE \\\\",
        )
        println(io, "\\midrule")
        for row in eachrow(summary)
            dataset_name = replace(string(row.dataset_name), "_" => "\\_")
            println(
                io,
                "$dataset_name & $(row.propagation) & $(row.likelihood) & " *
                "$(row.n) & " *
                @sprintf(
                    "%.4f \$\\pm\$ %.4f & %.4f \$\\pm\$ %.4f \\\\",
                    row.lpd_original_mean,
                    row.lpd_original_std,
                    row.rmse_original_mean,
                    row.rmse_original_std,
                ),
            )
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
    return (markdown = markdown_path, latex = latex_path)
end

function make_summary_plot(summary::DataFrame, config::DVIConfig)
    config.make_plot || return nothing
    isempty(summary) && return nothing
    labels = [
        "$(row.dataset_name)\n$(row.propagation)"
        for row in eachrow(summary)
    ]
    positions = collect(eachindex(labels))
    plot_lpd = Plots.scatter(
        positions,
        summary.lpd_original_mean;
        yerror = summary.lpd_original_std,
        ylabel = "LPD (original units)",
        xticks = (positions, labels),
        xrotation = 30,
        legend = false,
        color = :darkgreen,
        markersize = 6,
    )
    plot_rmse = Plots.scatter(
        positions,
        summary.rmse_original_mean;
        yerror = summary.rmse_original_std,
        ylabel = "RMSE (original units)",
        xticks = (positions, labels),
        xrotation = 30,
        legend = false,
        color = :darkgreen,
        markersize = 6,
    )
    combined = Plots.plot(
        plot_lpd,
        plot_rmse;
        layout = (2, 1),
        size = (1050, 950),
    )
    path = joinpath(config.output_dir, "dvi_uci_summary.png")
    Plots.savefig(combined, path)
    return path
end

function refresh_artifacts(runs::DataFrame, config::DVIConfig)
    summary = summary_table(runs)
    isempty(summary) && return summary
    write_tables(summary, config)
    make_summary_plot(summary, config)
    return summary
end
