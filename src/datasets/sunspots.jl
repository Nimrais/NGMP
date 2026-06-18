# Sunspots dataset — monthly mean relative sunspot numbers from 1749 onwards.
#
# This mirrors the download pattern used by MLDatasets.jl: a DataDep is
# registered at module init time (mapping a name to a download URL + SHA256
# checksum), and the file is materialised on first access into
# `~/.julia/datadeps/Sunspots/`. Subsequent calls reuse the cached file.

const SUNSPOTS_DEPNAME = "Sunspots"
const SUNSPOTS_LINK = "https://raw.githubusercontent.com/STAT-JET-ASU/Datasets/master/Instructor/"
const SUNSPOTS_DOCS = "https://stat-jet-asu.github.io/Datasets/InstructorDescriptions/sunspots.html"
const SUNSPOTS_FILE = "sunspots.csv"
const SUNSPOTS_SHA256 = "657169832a67571b46385046a2966a6056890ad22da82c26f7a26e615adde840"

function __init__sunspots()
    register(DataDep(SUNSPOTS_DEPNAME,
                     """
                     Dataset: Monthly mean relative sunspot numbers (1749-).
                     Website: $SUNSPOTS_DOCS
                     """,
                     SUNSPOTS_LINK * SUNSPOTS_FILE,
                     SUNSPOTS_SHA256))
end

"""
    Sunspots(; as_df = true, dir = nothing)

Monthly mean relative sunspot numbers from 1749 onwards, a classic
univariate time-series dataset.

Each row reports, for one month, the average daily sunspot number together
with its standard deviation. The data is downloaded on first use from the
[STAT-JET-ASU datasets collection]($(SUNSPOTS_DOCS)) and cached locally
via DataDeps.jl.

Columns:

- `year`: calendar year of the observation.
- `month`: month of the year (1-12).
- `date`: observation date as a fractional year.
- `average`: monthly mean relative sunspot number — used as the target.
- `sd`: standard deviation of the daily values within the month.

NOTE: this is a time series with no pre-defined train-test split.

# Arguments

- If `as_df = true` (default), `features`/`targets` are `DataFrame`s, and
  `dataframe` holds the full table; otherwise they are plain `Matrix`es and
  `dataframe` is `nothing`.
- `dir`: directory to load/download into. Defaults to the DataDeps cache.

# Fields

- `metadata`: dictionary with `path`, `n_observations`, `feature_names`,
  `target_names`.
- `features`: the `year`, `month`, `date`, `sd` columns.
- `targets`: the `average` column.
- `dataframe`: the full table, or `nothing` when `as_df = false`.

# Examples

```julia-repl
julia> using SurrogateModelling: Sunspots

julia> ds = Sunspots();

julia> ds.dataframe[1:3, :]
3×5 DataFrame
 Row │ year   month  date     average  sd
     │ Int64  Int64  Float64  Float64  Float64
─────┼──────────────────────────────────────────
   1 │  1749      1  1749.04     58.0     24.1
   2 │  1749      2  1749.12     62.6     25.1
   3 │  1749      3  1749.2      70.0     26.6

julia> X, y = Sunspots(as_df = false)[:];
```
"""
struct Sunspots
    metadata::Dict{String, Any}
    features::Any
    targets::Any
    dataframe::Any
end

function Sunspots(; dir = nothing, as_df = true)
    path = if dir === nothing
        joinpath(datadep"Sunspots", SUNSPOTS_FILE)
    else
        joinpath(dir, SUNSPOTS_FILE)
    end
    df = CSV.read(path, DataFrames.DataFrame)

    features = df[!, DataFrames.Not(:average)]
    targets = df[!, [:average]]

    metadata = Dict{String, Any}()
    metadata["path"] = path
    metadata["n_observations"] = DataFrames.nrow(df)
    metadata["feature_names"] = DataFrames.names(features)
    metadata["target_names"] = DataFrames.names(targets)

    if !as_df
        features = Matrix(features)
        targets = Matrix(targets)
        df = nothing
    end

    return Sunspots(metadata, features, targets, df)
end

# `ds[:]` / `ds[i]` return (features, targets), matching the MLDatasets
# tabular convention.
Base.getindex(d::Sunspots, ::Colon) = (d.features, d.targets)
function Base.getindex(d::Sunspots, i)
    if d.dataframe === nothing
        return (d.features[i, :], d.targets[i, :])
    else
        return (d.features[i, :], d.targets[i, :])
    end
end
Base.length(d::Sunspots) = d.metadata["n_observations"]

function Base.show(io::IO, ::MIME"text/plain", d::Sunspots)
    print(io, "Sunspots dataset:")
    print(io, "\n  metadata   =>  Dict with $(length(d.metadata)) entries")
    print(io, "\n  features   =>  $(summary(d.features))")
    print(io, "\n  targets    =>  $(summary(d.targets))")
    print(io, "\n  dataframe  =>  $(d.dataframe === nothing ? "nothing" : summary(d.dataframe))")
end
