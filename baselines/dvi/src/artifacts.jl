function ensure_output_directories(config::DVIConfig)
    mkpath(config.output_dir)
    mkpath(joinpath(config.output_dir, "histories"))
    config.save_checkpoints &&
        mkpath(joinpath(config.output_dir, "checkpoints"))
    return config.output_dir
end

function save_config(config::DVIConfig)
    path = joinpath(config.output_dir, "config.toml")
    open(path, "w") do io
        TOML.print(io, config_dictionary(config); sorted = true)
    end
    return path
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
)
    isempty(runs) && return false
    required = (:dataset, :split, :likelihood, :propagation, :status)
    all(name -> name in propertynames(runs), required) || return false
    return any(eachrow(runs)) do row
        string(row.dataset) == dataset &&
            Int(row.split) == split_id &&
            string(row.likelihood) == likelihood &&
            string(row.propagation) == propagation &&
            string(row.status) == "success"
    end
end

function append_run!(runs::DataFrame, row::NamedTuple, config::DVIConfig)
    incoming = DataFrame([row])
    updated = isempty(runs) ? incoming :
        vcat(runs, incoming; cols = :union)
    atomic_csv_write(joinpath(config.output_dir, "runs.csv"), updated)
    return updated
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
            "| Dataset | Method | Likelihood | Runs | LPD (original) | RMSE |",
        )
        println(io, "|---|---:|---:|---:|---:|---:|")
        for row in eachrow(summary)
            println(
                io,
                "| $(row.dataset_name) | $(row.propagation) | " *
                "$(row.likelihood) | $(row.n) | " *
                "$(format_pm(row.lpd_original_mean, row.lpd_original_std)) | " *
                "$(format_pm(row.rmse_original_mean, row.rmse_original_std)) |",
            )
        end
    end
    return markdown_path
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
