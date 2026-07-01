# ETTh1 dataset - hourly Electricity Transformer Temperature data.
#
# This follows the same DataDeps-backed pattern as `Sunspots`: register the
# download URL and checksum at module init time, then materialise the CSV on
# first access into `~/.julia/datadeps/ETTh1/`.

const ETTH1_DEPNAME = "ETTh1"
const ETTH1_LINK = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/"
const ETTH1_DOCS = "https://github.com/zhouhaoyi/ETDataset"
const ETTH1_FILE = "ETTh1.csv"
const ETTH1_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
const ETTH1_FEATURE_COLUMNS = [:HUFL, :HULL, :MUFL, :MULL, :LUFL, :LULL]
const ETTH1_TARGET_COLUMNS = [:OT]

function __init__etth1()
    register(DataDep(ETTH1_DEPNAME,
                     """
                     Dataset: ETTh1 hourly Electricity Transformer Temperature data.
                     Website: $ETTH1_DOCS
                     """,
                     ETTH1_LINK * ETTH1_FILE,
                     ETTH1_SHA256))
end

"""
    ETTh1(; as_df = true, dir = nothing)

Hourly Electricity Transformer Temperature data from the ETT-small benchmark.

Each row reports one hourly observation from station/region `h1`, with six
external power-load variables and the oil temperature target `OT`. The data is
downloaded on first use from the
[ETDataset repository]($(ETTH1_DOCS)) and cached locally via DataDeps.jl.

Columns:

- `date`: recorded timestamp.
- `HUFL`: High UseFul Load.
- `HULL`: High UseLess Load.
- `MUFL`: Middle UseFul Load.
- `MULL`: Middle UseLess Load.
- `LUFL`: Low UseFul Load.
- `LULL`: Low UseLess Load.
- `OT`: Oil Temperature target.

NOTE: this is a time series with no pre-defined train-test split.

# Arguments

- If `as_df = true` (default), `features`/`targets` are `DataFrame`s, and
  `dataframe` holds the full table; otherwise they are plain `Matrix`es and
  `dataframe` is `nothing`.
- `dir`: directory to load/download into. Defaults to the DataDeps cache.

# Fields

- `metadata`: dictionary with `path`, `n_observations`, `feature_names`,
  `target_names`, `date_name`, `date_range`, `frequency`, `source`.
- `features`: the `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL` columns.
- `targets`: the `OT` column.
- `dataframe`: the full table, or `nothing` when `as_df = false`.

# Examples

```julia-repl
julia> using SurrogateModelling: ETTh1

julia> ds = ETTh1();

julia> ds.dataframe[1:3, :]
3x8 DataFrame with columns:
date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT

julia> X, y = ETTh1(as_df = false)[:];
```
"""
struct ETTh1
    metadata::Dict{String, Any}
    features::Any
    targets::Any
    dataframe::Any
end

function ETTh1(; dir = nothing, as_df = true)
    path = if dir === nothing
        joinpath(datadep"ETTh1", ETTH1_FILE)
    else
        joinpath(dir, ETTH1_FILE)
    end
    df = CSV.read(path, DataFrames.DataFrame)

    features = df[!, ETTH1_FEATURE_COLUMNS]
    targets = df[!, ETTH1_TARGET_COLUMNS]

    metadata = Dict{String, Any}()
    metadata["path"] = path
    metadata["n_observations"] = DataFrames.nrow(df)
    metadata["feature_names"] = DataFrames.names(features)
    metadata["target_names"] = DataFrames.names(targets)
    metadata["date_name"] = "date"
    metadata["date_range"] = (String(df.date[begin]), String(df.date[end]))
    metadata["frequency"] = "hourly"
    metadata["source"] = ETTH1_DOCS

    if !as_df
        features = Matrix(features)
        targets = Matrix(targets)
        df = nothing
    end

    return ETTh1(metadata, features, targets, df)
end

# `ds[:]` / `ds[i]` return (features, targets), matching the MLDatasets
# tabular convention.
Base.getindex(d::ETTh1, ::Colon) = (d.features, d.targets)
function Base.getindex(d::ETTh1, i)
    if d.dataframe === nothing
        return (d.features[i, :], d.targets[i, :])
    else
        return (d.features[i, :], d.targets[i, :])
    end
end
Base.length(d::ETTh1) = d.metadata["n_observations"]

function Base.show(io::IO, ::MIME"text/plain", d::ETTh1)
    print(io, "ETTh1 dataset:")
    print(io, "\n  metadata   =>  Dict with $(length(d.metadata)) entries")
    print(io, "\n  features   =>  $(summary(d.features))")
    print(io, "\n  targets    =>  $(summary(d.targets))")
    print(io, "\n  dataframe  =>  $(d.dataframe === nothing ? "nothing" : summary(d.dataframe))")
end
