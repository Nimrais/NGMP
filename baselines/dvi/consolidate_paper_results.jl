#!/usr/bin/env julia

using CSV
using DataFrames
using DVIUCI
using JLD2
using TOML

length(ARGS) >= 5 || error(
    "usage: consolidate_paper_results.jl DESTINATION NON_YACHT_V2 " *
    "YACHT_SPLIT20_V2 LEGACY_YACHT_SHARD...",
)

destination = abspath(ARGS[1])
non_yacht_directory = abspath(ARGS[2])
yacht_v2_directory = abspath(ARGS[3])
legacy_yacht_directories = abspath.(ARGS[4:end])

ispath(destination) && error("destination already exists: $destination")

function read_runs(directory::AbstractString)
    path = joinpath(directory, "runs.csv")
    isfile(path) || error("missing runs table: $path")
    return CSV.read(path, DataFrame)
end

function successful_rows(runs::DataFrame)
    return filter(row -> string(row.status) == "success", runs)
end

function label_source!(runs::DataFrame, label::AbstractString)
    runs[!, :result_source] = fill(string(label), nrow(runs))
    return runs
end

non_yacht_runs = filter(
    row -> string(row.dataset) != "yacht",
    successful_rows(read_runs(non_yacht_directory)),
)
label_source!(non_yacht_runs, "selection-optimized-v2-reactant-gpu")

legacy_runs = reduce(
    (left, right) -> vcat(left, right; cols = :union),
    read_runs(directory) for directory in legacy_yacht_directories
)
legacy_yacht_runs = filter(successful_rows(legacy_runs)) do row
    string(row.dataset) == "yacht" && 1 <= Int(row.split) <= 19
end
label_source!(legacy_yacht_runs, "legacy-v1-zygote-cpu")

yacht_v2_runs = filter(successful_rows(read_runs(yacht_v2_directory))) do row
    string(row.dataset) == "yacht" && Int(row.split) == 20
end
nrow(yacht_v2_runs) == 1 || error(
    "expected exactly one successful v2 Yacht split-20 row",
)
label_source!(yacht_v2_runs, "selection-optimized-v2-reactant-gpu")

runs = vcat(
    non_yacht_runs,
    legacy_yacht_runs,
    yacht_v2_runs;
    cols = :union,
)
sort!(runs, [:dataset, :split, :likelihood, :propagation])

datasets = ["yacht", "concrete", "energy", "housing", "power", "wine"]
split_ids = collect(1:20)
DVIUCI.validate_complete_runs(
    runs,
    datasets,
    split_ids,
    ["heteroscedastic"],
    "full",
)
nrow(runs) == 120 || error("expected 120 consolidated DVI rows")
for dataset in datasets
    count(==(dataset), string.(runs.dataset)) == 20 || error(
        "expected 20 consolidated rows for $dataset",
    )
end

function artifact_source(row)
    dataset = string(row.dataset)
    split_id = Int(row.split)
    if dataset != "yacht"
        return non_yacht_directory
    elseif split_id == 20
        return yacht_v2_directory
    end

    stem = DVIUCI.configuration_stem(
        dataset,
        split_id,
        string(row.likelihood),
        string(row.propagation),
    )
    matches = filter(legacy_yacht_directories) do directory
        isfile(joinpath(directory, "histories", "$stem.csv"))
    end
    length(matches) == 1 || error(
        "expected exactly one legacy history source for $stem",
    )
    return only(matches)
end

reference_manifest = JLD2.load(joinpath(
    first(legacy_yacht_directories),
    "split_manifest.jld2",
))
non_yacht_manifest = JLD2.load(joinpath(
    non_yacht_directory,
    "split_manifest.jld2",
))
yacht_v2_manifest = JLD2.load(joinpath(
    yacht_v2_directory,
    "split_manifest.jld2",
))

manifest_scalar_keys = [
    "base_seed",
    "n_splits",
    "protocol_version",
    "schema_version",
    "test_fraction",
    "validation_fraction",
]
for manifest in (non_yacht_manifest, yacht_v2_manifest)
    all(
        key -> isequal(reference_manifest[key], manifest[key]),
        manifest_scalar_keys,
    ) || error("source split-manifest protocols differ")
end
for dataset in setdiff(datasets, ["yacht"])
    isequal(
        reference_manifest["datasets"][dataset],
        non_yacht_manifest["datasets"][dataset],
    ) || error("split manifest differs for $dataset")
end
isequal(
    reference_manifest["datasets"]["yacht"],
    yacht_v2_manifest["datasets"]["yacht"],
) || error("split manifest differs for yacht")

parent = dirname(destination)
mkpath(parent)
staging = mktempdir(parent; prefix = ".dvi-consolidation-")
completed = Ref(false)
try
    for subdirectory in ("histories", "checkpoints", "failures")
        mkpath(joinpath(staging, subdirectory))
    end
    DVIUCI.atomic_csv_write(joinpath(staging, "runs.csv"), runs)

    for row in eachrow(runs)
        stem = DVIUCI.configuration_stem(
            string(row.dataset),
            Int(row.split),
            string(row.likelihood),
            string(row.propagation),
        )
        source = artifact_source(row)
        for (subdirectory, extension) in (
            ("histories", "csv"),
            ("checkpoints", "jld2"),
        )
            source_path = joinpath(
                source,
                subdirectory,
                "$stem.$extension",
            )
            isfile(source_path) || error("missing artifact: $source_path")
            cp(
                source_path,
                joinpath(staging, subdirectory, basename(source_path)),
            )
        end
    end
    cp(
        joinpath(
            first(legacy_yacht_directories),
            "split_manifest.jld2",
        ),
        joinpath(staging, "split_manifest.jld2"),
    )

    merged_config = TOML.parsefile(joinpath(
        non_yacht_directory,
        "config.toml",
    ))
    merged_config["datasets"] = datasets
    merged_config["split_ids"] = split_ids
    merged_config["output_dir"] = destination
    merged_config["resume"] = false
    merged_config["execution_backend"] = "mixed"
    merged_config["execution_device"] = "mixed"
    merged_config["implementation_version"] =
        "consolidated-legacy-v1-and-selection-optimized-v2"
    merged_config["consolidation_schema"] = "dvi-paper-consolidation-v1"
    merged_config["consolidation_policy"] =
        "non-yacht=v2; yacht-1:19=legacy-v1; yacht-20=v2"
    open(joinpath(staging, "config.toml"), "w") do io
        TOML.print(io, merged_config; sorted = true)
    end

    provenance = Dict(
        "schema_version" => "dvi-paper-consolidation-v1",
        "policy" =>
            "non-yacht=v2; yacht-1:19=legacy-v1; yacht-20=v2",
        "rows" => nrow(runs),
        "components" => Dict(
            "non_yacht_v2" => Dict(
                "path" => relpath(non_yacht_directory, parent),
                "rows" => nrow(non_yacht_runs),
            ),
            "yacht_legacy_v1" => Dict(
                "paths" => relpath.(legacy_yacht_directories, parent),
                "rows" => nrow(legacy_yacht_runs),
                "splits" => collect(1:19),
            ),
            "yacht_v2" => Dict(
                "path" => relpath(yacht_v2_directory, parent),
                "rows" => nrow(yacht_v2_runs),
                "splits" => [20],
            ),
        ),
    )
    open(joinpath(staging, "provenance.toml"), "w") do io
        TOML.print(io, provenance; sorted = true)
    end

    artifact_config = DVIUCI.DVIConfig(
        output_dir = staging,
        datasets = datasets,
        split_ids = split_ids,
        propagation = "full",
        make_plot = false,
        show_progress = false,
    )
    DVIUCI.refresh_artifacts(runs, artifact_config)

    ispath(destination) && error(
        "destination was created during consolidation: $destination",
    )
    mv(staging, destination)
    completed[] = true
finally
    !completed[] && isdir(staging) && rm(staging; recursive = true)
end

println("Consolidated $(nrow(runs)) DVI rows into $destination")
