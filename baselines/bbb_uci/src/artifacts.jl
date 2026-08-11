const SCALAR_METRIC_NAMES = (
    :lpd_standardized,
    :nll_standardized,
    :expected_log_likelihood_standardized,
    :lpd_original,
    :nll_original,
    :expected_log_likelihood_original,
    :rmse_original,
    :coverage_50,
    :coverage_80,
    :coverage_95,
    :mean_total_variance_original,
    :mean_epistemic_variance_original,
    :mean_aleatoric_variance_original,
)

const CONFIG_RUNTIME_ONLY_FIELDS = Set([
    "output_dir",
    "resume",
    "make_plot",
    "show_progress",
])

configuration_stem(dataset::String, split_id::Int, likelihood::String) =
    @sprintf("%s_split%02d_%s", dataset, split_id, likelihood)

function posterior_checkpoint_path(
    config::BBBConfig,
    dataset::String,
    split_id::Int,
    likelihood::String,
)
    return joinpath(
        config.output_dir,
        "checkpoints",
        configuration_stem(dataset, split_id, likelihood) * ".jld2",
    )
end

function ensure_output_directories(config::BBBConfig)
    mkpath(config.output_dir)
    mkpath(joinpath(config.output_dir, "histories"))
    config.save_checkpoints && mkpath(joinpath(config.output_dir, "checkpoints"))
    return config.output_dir
end

function validate_resume_config(path::AbstractString, config::BBBConfig)
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

function save_config(config::BBBConfig)
    path = joinpath(config.output_dir, "config.toml")
    isfile(path) && validate_resume_config(path, config)
    temporary = path * ".tmp-" * string(getpid())
    open(temporary, "w") do io
        TOML.print(io, config_dictionary(config); sorted = true)
    end
    mv(temporary, path; force = true)
    return path
end

function write_split_manifest(config::BBBConfig)
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
    record = (
        schema_version = "uci-split-manifest-v1",
        protocol_version = UCI_SPLIT_PROTOCOL_VERSION,
        base_seed = config.split_seed,
        n_splits = config.n_splits,
        test_fraction = config.test_fraction,
        validation_fraction = config.validation_fraction,
        datasets = datasets,
    )
    return atomic_jld2_write(
        joinpath(config.output_dir, "split_manifest.jld2"),
        record,
    )
end

function atomic_csv_write(path::AbstractString, table)
    temporary = path * ".tmp"
    CSV.write(temporary, table)
    mv(temporary, path; force = true)
    return path
end

function load_run_table(config::BBBConfig)
    path = joinpath(config.output_dir, "runs.csv")
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame)
end

function configuration_succeeded(
    runs::DataFrame,
    dataset::String,
    split_id::Int,
    likelihood::String,
    config::BBBConfig,
)
    isempty(runs) && return false
    required = (:dataset, :split, :likelihood, :status)
    all(name -> name in propertynames(runs), required) || return false
    row_succeeded = any(eachrow(runs)) do row
        string(row.dataset) == dataset &&
            Int(row.split) == split_id &&
            string(row.likelihood) == likelihood &&
            string(row.status) == "success"
    end
    row_succeeded || return false
    config.save_checkpoints || return true
    path = posterior_checkpoint_path(
        config, dataset, split_id, likelihood,
    )
    return try
        checkpoint = load_posterior_checkpoint(path)
        checkpoint.dataset == dataset &&
            checkpoint.split_id == split_id &&
            checkpoint.likelihood == likelihood
    catch
        false
    end
end

function append_run!(runs::DataFrame, row::NamedTuple, config::BBBConfig)
    incoming = DataFrame([row])
    updated = isempty(runs) ? incoming : vcat(runs, incoming; cols = :union)
    atomic_csv_write(joinpath(config.output_dir, "runs.csv"), updated)
    return updated
end

function summary_table(runs::DataFrame)
    isempty(runs) && return DataFrame()
    successful = filter(row -> row.status == "success", runs)
    isempty(successful) && return DataFrame()
    rows = NamedTuple[]
    for group in groupby(successful, [:dataset, :dataset_name, :likelihood])
        values_for(metric) = Float64.(group[!, metric])
        metric_summary = Pair{Symbol, Any}[]
        for metric in SCALAR_METRIC_NAMES
            values = values_for(metric)
            push!(metric_summary, Symbol(metric, "_mean") => mean(values))
            push!(
                metric_summary,
                Symbol(metric, "_std") =>
                    (length(values) > 1 ? std(values; corrected = true) : 0.0),
            )
            push!(
                metric_summary,
                Symbol(metric, "_se") =>
                    (length(values) > 1 ?
                        std(values; corrected = true) / sqrt(length(values)) : 0.0),
            )
        end
        push!(rows, (; (
            dataset = string(first(group.dataset)),
            dataset_name = string(first(group.dataset_name)),
            likelihood = string(first(group.likelihood)),
            n = nrow(group),
            metric_summary...,
        )...))
    end
    return sort!(DataFrame(rows), [:dataset, :likelihood])
end

"""Format a mean with its normal-approximation 95% CI half-width."""
format_ci95(mean_value, standard_error) =
    @sprintf("%.4f ± %.4f", mean_value, 1.96 * standard_error)

function write_tables(summary::DataFrame, config::BBBConfig)
    isempty(summary) && return nothing
    atomic_csv_write(joinpath(config.output_dir, "summary.csv"), summary)

    markdown_path = joinpath(config.output_dir, "table.md")
    open(markdown_path, "w") do io
        println(io, "| Dataset | Likelihood | Runs | LPD (original, 95% CI) | RMSE (95% CI) | LPD (standardized, 95% CI) |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        for row in eachrow(summary)
            println(
                io,
                "| $(row.dataset_name) | $(row.likelihood) | $(row.n) | " *
                "$(format_ci95(row.lpd_original_mean, row.lpd_original_se)) | " *
                "$(format_ci95(row.rmse_original_mean, row.rmse_original_se)) | " *
                "$(format_ci95(row.lpd_standardized_mean, row.lpd_standardized_se)) |",
            )
        end
    end

    latex_path = joinpath(config.output_dir, "table.tex")
    open(latex_path, "w") do io
        println(io, "\\begin{tabular}{llrrr}")
        println(io, "\\toprule")
        println(io, "Dataset & Likelihood & Runs & LPD (original, 95\\% CI) & RMSE (95\\% CI) \\\\")
        println(io, "\\midrule")
        for row in eachrow(summary)
            dataset_name = replace(string(row.dataset_name), "_" => "\\_")
            println(
                io,
                "$dataset_name & $(row.likelihood) & $(row.n) & " *
                @sprintf("%.4f \$\\pm\$ %.4f & %.4f \$\\pm\$ %.4f \\\\",
                    row.lpd_original_mean,
                    1.96 * row.lpd_original_se,
                    row.rmse_original_mean,
                    1.96 * row.rmse_original_se,
                ),
            )
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
    return (markdown = markdown_path, latex = latex_path)
end

function make_summary_plot(summary::DataFrame, config::BBBConfig)
    config.make_plot || return nothing
    isempty(summary) && return nothing
    datasets = unique(summary.dataset_name)
    modes = ["homoscedastic", "heteroscedastic"]
    offsets = Dict("homoscedastic" => -0.12, "heteroscedastic" => 0.12)
    colors = Dict("homoscedastic" => :steelblue, "heteroscedastic" => :darkorange)
    plot_lpd = Plots.plot(
        ylabel = "LPD (original units)",
        xticks = (1:length(datasets), datasets),
        xrotation = 30,
        legend = :bottomright,
        size = (1050, 700),
    )
    plot_rmse = Plots.plot(
        ylabel = "RMSE (original units)",
        xticks = (1:length(datasets), datasets),
        xrotation = 30,
        legend = false,
    )
    for mode in modes
        xs = Float64[]
        lpds = Float64[]
        lpd_errors = Float64[]
        rmses = Float64[]
        rmse_errors = Float64[]
        for (index, dataset) in enumerate(datasets)
            matches = filter(
                row -> row.dataset_name == dataset && row.likelihood == mode,
                summary,
            )
            isempty(matches) && continue
            row = first(eachrow(matches))
            push!(xs, index + offsets[mode])
            push!(lpds, row.lpd_original_mean)
            push!(lpd_errors, row.lpd_original_std)
            push!(rmses, row.rmse_original_mean)
            push!(rmse_errors, row.rmse_original_std)
        end
        isempty(xs) && continue
        Plots.scatter!(
            plot_lpd,
            xs,
            lpds;
            yerror = lpd_errors,
            label = mode,
            color = colors[mode],
            markersize = 6,
        )
        Plots.scatter!(
            plot_rmse,
            xs,
            rmses;
            yerror = rmse_errors,
            color = colors[mode],
            markersize = 6,
        )
    end
    combined = Plots.plot(
        plot_lpd, plot_rmse; layout = (2, 1), size = (1050, 950),
    )
    path = joinpath(config.output_dir, "bbb_uci_summary.png")
    Plots.savefig(combined, path)
    return path
end

function refresh_artifacts(runs::DataFrame, config::BBBConfig)
    summary = summary_table(runs)
    isempty(summary) && return summary
    write_tables(summary, config)
    make_summary_plot(summary, config)
    return summary
end
