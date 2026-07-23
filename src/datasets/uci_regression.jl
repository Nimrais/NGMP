# UCI regression benchmarks — the six tabular datasets used throughout the
# BNN literature (Hernández-Lobato & Adams 2015 and follow-ups such as the
# Bayes-by-Backprop / Bayesian-Predictive-Coding RMSE-LPD comparisons):
# yacht, concrete, energy, housing, power, wine.
#
# Files are fetched from Yarin Gal's `DropoutUncertaintyExps` mirror, which
# redistributes the original UCI tables as uniform whitespace-delimited
# `data.txt` matrices with the target in the last column(s) — this is the
# exact data behind the published benchmark numbers (the yacht file is
# byte-identical to the UCI original). Same DataDeps-backed pattern as
# `Sunspots`/`ETTh1`: register URL + checksum at module init, materialise on
# first access into `~/.julia/datadeps/<depname>/`.

const UCI_REGRESSION_LINK = "https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/"
const UCI_REGRESSION_FILE = "data.txt"
const UCI_REGRESSION_MIRROR = "https://github.com/yaringal/DropoutUncertaintyExps"

const UCI_REGRESSION_SPECS = (
    (type = :Yacht,
     depname = "UCI-Yacht",
     remote_dir = "yacht",
     sha256 = "00dfecc0fc01ddd4c90b558a3ac11b246df8ebcfea130724223475a9a67f0ea1",
     docs = "https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics",
     title = "Yacht Hydrodynamics",
     summary = "Residuary resistance of sailing yachts from hull geometry and the Froude number (308 observations, 6 features).",
     features = [:longitudinal_position, :prismatic_coefficient, :length_displacement_ratio,
                 :beam_draught_ratio, :length_beam_ratio, :froude_number],
     targets = [:residuary_resistance]),
    (type = :Concrete,
     depname = "UCI-Concrete",
     remote_dir = "concrete",
     sha256 = "43d290fd2c2a399ad7e62c45cab5337ba94ece69ebdcf5875aa668cb886243de",
     docs = "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength",
     title = "Concrete Compressive Strength",
     summary = "Compressive strength of concrete from mixture composition and age (1030 observations, 8 features).",
     features = [:cement, :blast_furnace_slag, :fly_ash, :water, :superplasticizer,
                 :coarse_aggregate, :fine_aggregate, :age],
     targets = [:compressive_strength]),
    (type = :EnergyEfficiency,
     depname = "UCI-Energy",
     remote_dir = "energy",
     sha256 = "7f8bf024cea437267d56be99b7af7bb3faebec8a7bccfbb20d06d52d9372a950",
     docs = "https://archive.ics.uci.edu/dataset/242/energy+efficiency",
     title = "Energy Efficiency",
     summary = "Heating load of simulated buildings from geometric shape parameters (768 observations, 8 features).",
     features = [:relative_compactness, :surface_area, :wall_area, :roof_area,
                 :overall_height, :orientation, :glazing_area, :glazing_area_distribution],
     targets = [:heating_load]),
    (type = :BostonHousing,
     depname = "UCI-BostonHousing",
     remote_dir = "bostonHousing",
     sha256 = "baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a",
     docs = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/",
     title = "Boston Housing",
     summary = "Median home value in Boston suburbs from socio-economic and structural features (506 observations, 13 features).",
     features = [:crim, :zn, :indus, :chas, :nox, :rm, :age, :dis, :rad, :tax,
                 :ptratio, :b, :lstat],
     targets = [:medv]),
    (type = :PowerPlant,
     depname = "UCI-PowerPlant",
     remote_dir = "power-plant",
     sha256 = "daebd20c408dfc5c4979604f240e891be162c3a5d00d662380aa669044a1fb31",
     docs = "https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant",
     title = "Combined Cycle Power Plant",
     summary = "Net hourly electrical energy output of a combined-cycle power plant from ambient conditions (9568 observations, 4 features).",
     features = [:temperature, :exhaust_vacuum, :ambient_pressure, :relative_humidity],
     targets = [:net_energy_output]),
    (type = :WineQualityRed,
     depname = "UCI-WineQualityRed",
     remote_dir = "wine-quality-red",
     sha256 = "3ba41eaf4ab562088cf47b5c9dca2b4de2a330c3ee38a7920b4527a09574d5c1",
     docs = "https://archive.ics.uci.edu/dataset/186/wine+quality",
     title = "Wine Quality (red)",
     summary = "Sensory quality score of red vinho verde wines from physicochemical tests (1599 observations, 11 features).",
     features = [:fixed_acidity, :volatile_acidity, :citric_acid, :residual_sugar,
                 :chlorides, :free_sulfur_dioxide, :total_sulfur_dioxide, :density,
                 :pH, :sulphates, :alcohol],
     targets = [:quality]),
)

function __init__uci_regression()
    for spec in UCI_REGRESSION_SPECS
        register(DataDep(spec.depname,
                         """
                         Dataset: UCI $(spec.title) regression benchmark.
                         Website: $(spec.docs)
                         Mirror: $(UCI_REGRESSION_MIRROR)
                         """,
                         UCI_REGRESSION_LINK * spec.remote_dir * "/data/" * UCI_REGRESSION_FILE,
                         spec.sha256))
    end
end

# The mirror mixes delimiters across files (tabs, single spaces, repeated
# spaces — sometimes within one file), so split rows on generic whitespace
# instead of parsing with a fixed CSV delimiter.
function _read_uci_regression(path, colnames)
    rows = [parse.(Float64, split(line)) for line in eachline(path) if !isempty(strip(line))]
    return DataFrames.DataFrame(permutedims(reduce(hcat, rows)), colnames)
end

"""
    UCIRegressionDataset

Abstract supertype of the six UCI regression benchmarks ([`Yacht`](@ref),
[`Concrete`](@ref), [`EnergyEfficiency`](@ref), [`BostonHousing`](@ref),
[`PowerPlant`](@ref), [`WineQualityRed`](@ref)). All share the
`Sunspots`/`ETTh1` interface: `metadata`, `features`, `targets`, `dataframe`
fields, `ds[:]`/`ds[i]` indexing, and `as_df`/`dir` keyword arguments.
"""
abstract type UCIRegressionDataset end

function _load_uci_regression(T, spec, path, as_df)
    df = _read_uci_regression(path, vcat(spec.features, spec.targets))

    features = df[!, spec.features]
    targets = df[!, spec.targets]

    metadata = Dict{String, Any}()
    metadata["path"] = path
    metadata["n_observations"] = DataFrames.nrow(df)
    metadata["feature_names"] = DataFrames.names(features)
    metadata["target_names"] = DataFrames.names(targets)
    metadata["source"] = spec.docs
    metadata["mirror"] = UCI_REGRESSION_MIRROR

    if !as_df
        features = Matrix(features)
        targets = Matrix(targets)
        df = nothing
    end

    return T(metadata, features, targets, df)
end

for spec in UCI_REGRESSION_SPECS
    T = spec.type
    depname = spec.depname
    docstring = """
        $(T)(; as_df = true, dir = nothing)

    UCI $(spec.title) regression benchmark.

    $(spec.summary) Downloaded on first use from the
    [DropoutUncertaintyExps mirror]($(UCI_REGRESSION_MIRROR)) of the
    [UCI repository]($(spec.docs)) and cached locally via DataDeps.jl.

    Features: $(join(("`$(f)`" for f in spec.features), ", ")).
    Target: $(join(("`$(t)`" for t in spec.targets), ", ")).

    NOTE: no pre-defined train-test split; the BNN benchmark papers
    (Bayes by Backprop, PBP, BPC, ...) average over random 90/10 splits.

    # Arguments

    - If `as_df = true` (default), `features`/`targets` are `DataFrame`s, and
      `dataframe` holds the full table; otherwise they are plain `Matrix`es
      and `dataframe` is `nothing`.
    - `dir`: directory holding `$(UCI_REGRESSION_FILE)` to load from.
      Defaults to the DataDeps cache.

    # Examples

    ```julia-repl
    julia> using SurrogateModelling: $(T)

    julia> ds = $(T)();

    julia> X, y = $(T)(as_df = false)[:];
    ```
    """
    @eval begin
        struct $T <: UCIRegressionDataset
            metadata::Dict{String, Any}
            features::Any
            targets::Any
            dataframe::Any
        end

        function $T(; dir = nothing, as_df = true)
            path = if dir === nothing
                joinpath(@datadep_str($depname), UCI_REGRESSION_FILE)
            else
                joinpath(dir, UCI_REGRESSION_FILE)
            end
            return _load_uci_regression($T, $spec, path, as_df)
        end

        @doc $docstring $T
    end
end

# `ds[:]` / `ds[i]` return (features, targets), matching the MLDatasets
# tabular convention used by `Sunspots`/`ETTh1`.
Base.getindex(d::UCIRegressionDataset, ::Colon) = (d.features, d.targets)
Base.getindex(d::UCIRegressionDataset, i) = (d.features[i, :], d.targets[i, :])
Base.length(d::UCIRegressionDataset) = d.metadata["n_observations"]

function Base.show(io::IO, ::MIME"text/plain", d::UCIRegressionDataset)
    print(io, nameof(typeof(d)), " dataset:")
    print(io, "\n  metadata   =>  Dict with $(length(d.metadata)) entries")
    print(io, "\n  features   =>  $(summary(d.features))")
    print(io, "\n  targets    =>  $(summary(d.targets))")
    print(io, "\n  dataframe  =>  $(d.dataframe === nothing ? "nothing" : summary(d.dataframe))")
end
