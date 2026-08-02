#!/usr/bin/env julia

using CSV
using DataFrames
using DVIUCI
using TOML

length(ARGS) >= 2 || error(
    "usage: merge_shards.jl DESTINATION SHARD_DIRECTORY...",
)

destination = abspath(first(ARGS))
shard_directories = abspath.(ARGS[2:end])
runtime_fields = Set(["output_dir", "resume", "make_plot", "show_progress"])
shard_fields = Set(["split_ids"])

configs = [TOML.parsefile(joinpath(path, "config.toml"))
           for path in shard_directories]
reference = first(configs)
for (path, config) in zip(shard_directories[2:end], configs[2:end])
    for key in union(keys(reference), keys(config))
        key in runtime_fields && continue
        key in shard_fields && continue
        get(reference, key, nothing) == get(config, key, nothing) ||
            error("result-affecting config '$key' differs in $path")
    end
end

runs = reduce(
    (left, right) -> vcat(left, right; cols = :union),
    CSV.read(joinpath(path, "runs.csv"), DataFrame)
    for path in shard_directories
)
configured_split_ids = [Int.(config["split_ids"]) for config in configs]
sum(length, configured_split_ids) == length(unique(reduce(
    vcat, configured_split_ids,
))) || error("shard split_ids overlap")
merged_split_ids = sort!(unique(reduce(vcat, configured_split_ids)))
n_splits = Int(reference["n_splits"])
merged_split_ids == collect(1:n_splits) || error(
    "shards must cover every split in 1:$n_splits exactly once",
)

datasets = string.(reference["datasets"])
likelihoods = string.(reference["likelihoods"])
propagation = string(reference["propagation"])
DVIUCI.validate_complete_runs(
    runs, datasets, merged_split_ids, likelihoods, propagation,
)

function find_required_artifact(subdirectory, filename)
    matches = filter(
        path -> isfile(joinpath(path, subdirectory, filename)),
        shard_directories,
    )
    length(matches) == 1 || error(
        "expected exactly one $subdirectory/$filename across shards",
    )
    return joinpath(only(matches), subdirectory, filename)
end

for row in eachrow(runs)
    stem = DVIUCI.configuration_stem(
        string(row.dataset),
        Int(row.split),
        string(row.likelihood),
        string(row.propagation),
    )
    find_required_artifact("histories", "$stem.csv")
    if Bool(reference["save_checkpoints"])
        find_required_artifact("checkpoints", "$stem.jld2")
    end
end
sort!(runs, [:dataset, :split, :likelihood, :propagation])

mkpath(destination)
mkpath(joinpath(destination, "histories"))
mkpath(joinpath(destination, "checkpoints"))
mkpath(joinpath(destination, "failures"))
DVIUCI.atomic_csv_write(joinpath(destination, "runs.csv"), runs)

for shard in shard_directories
    for subdirectory in ("histories", "checkpoints", "failures")
        source = joinpath(shard, subdirectory)
        isdir(source) || continue
        for filename in readdir(source)
            cp(
                joinpath(source, filename),
                joinpath(destination, subdirectory, filename);
                force = true,
            )
        end
    end
end
cp(
    joinpath(first(shard_directories), "split_manifest.jld2"),
    joinpath(destination, "split_manifest.jld2");
    force = true,
)

merged_config = copy(reference)
merged_config["output_dir"] = destination
merged_config["split_ids"] = merged_split_ids
merged_config["resume"] = true
merged_config["make_plot"] = false
merged_config["show_progress"] = false
open(joinpath(destination, "config.toml"), "w") do io
    TOML.print(io, merged_config; sorted = true)
end

artifact_config = DVIConfig(
    output_dir = destination,
    propagation = string(reference["propagation"]),
    make_plot = false,
    show_progress = false,
)
DVIUCI.refresh_artifacts(runs, artifact_config)
println(
    "Merged $(length(shard_directories)) shards and $(nrow(runs)) runs into " *
    destination,
)
