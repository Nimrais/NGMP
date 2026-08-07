"""
Version identifier for the shared UCI repeated-holdout benchmark protocol.

Changing the RNG, seed schedule, rounding rule, or index semantics requires a
new version so saved baseline artifacts remain auditable.
"""
const UCI_SPLIT_PROTOCOL_VERSION = "repeated-holdout-v1"
const UCI_DEFAULT_N_SPLITS = 20
const UCI_DEFAULT_SPLIT_SEED = 20_260_726
const UCI_DEFAULT_TEST_FRACTION = 0.1
const UCI_DEFAULT_VALIDATION_FRACTION = 0.1
const UCI_INNER_SEED_OFFSET = 1_000_000

"""
    UCISplitSpec

One nested repeated-holdout split. Every index is a sorted, one-based index in
the original dataset row coordinates. `inner_train_indices` and
`validation_indices` partition `train_indices`; neither can contain an outer
holdout row.
"""
struct UCISplitSpec
    protocol_version::String
    split_id::Int
    n_observations::Int
    outer_seed::Int
    inner_seed::Int
    test_fraction::Float64
    validation_fraction::Float64
    train_indices::Vector{Int}
    test_indices::Vector{Int}
    inner_train_indices::Vector{Int}
    validation_indices::Vector{Int}
end

# The default struct `==` compares the Vector fields by identity, so two
# independently generated but protocol-identical specs would never be equal.
Base.:(==)(a::UCISplitSpec, b::UCISplitSpec) =
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(UCISplitSpec))
Base.hash(spec::UCISplitSpec, h::UInt) =
    foldl((acc, f) -> hash(getfield(spec, f), acc), fieldnames(UCISplitSpec); init = hash(UCISplitSpec, h))

function _uci_partition_indices(
    source_indices::AbstractVector{<:Integer};
    seed::Int,
    holdout_fraction::Real,
)
    n = length(source_indices)
    n >= 3 || throw(ArgumentError("at least three observations are required"))
    0 < holdout_fraction < 1 ||
        throw(ArgumentError("holdout_fraction must be in (0, 1)"))

    n_holdout = clamp(round(Int, holdout_fraction * n), 1, n - 2)
    order = randperm(StableRNG(seed), n)
    holdout_indices = sort!(Int.(source_indices[order[1:n_holdout]]))
    train_indices = sort!(Int.(source_indices[order[(n_holdout + 1):end]]))

    isempty(intersect(train_indices, holdout_indices)) ||
        error("internal UCI split error: train and holdout overlap")
    length(train_indices) + length(holdout_indices) == n ||
        error("internal UCI split error: partition is incomplete")
    return train_indices, holdout_indices
end

"""
    uci_regression_split(n_observations, split_id; kwargs...)

Construct split `split_id` from the shared versioned UCI protocol. Split IDs
are prefix-stable: split 1 is identical whether a caller asks for one, five,
or twenty total splits.
"""
function uci_regression_split(
    n_observations::Int,
    split_id::Int;
    base_seed::Int = UCI_DEFAULT_SPLIT_SEED,
    test_fraction::Real = UCI_DEFAULT_TEST_FRACTION,
    validation_fraction::Real = UCI_DEFAULT_VALIDATION_FRACTION,
)
    n_observations >= 4 ||
        throw(ArgumentError("at least four observations are required"))
    split_id >= 1 || throw(ArgumentError("split_id must be positive"))
    0 < test_fraction < 1 ||
        throw(ArgumentError("test_fraction must be in (0, 1)"))
    0 < validation_fraction < 1 ||
        throw(ArgumentError("validation_fraction must be in (0, 1)"))

    outer_seed = Base.checked_add(base_seed, split_id - 1)
    inner_seed = Base.checked_add(outer_seed, UCI_INNER_SEED_OFFSET)
    train_indices, test_indices = _uci_partition_indices(
        collect(1:n_observations);
        seed = outer_seed,
        holdout_fraction = test_fraction,
    )
    inner_train_indices, validation_indices = _uci_partition_indices(
        train_indices;
        seed = inner_seed,
        holdout_fraction = validation_fraction,
    )

    return UCISplitSpec(
        UCI_SPLIT_PROTOCOL_VERSION,
        split_id,
        n_observations,
        outer_seed,
        inner_seed,
        Float64(test_fraction),
        Float64(validation_fraction),
        train_indices,
        test_indices,
        inner_train_indices,
        validation_indices,
    )
end

"""
    uci_regression_splits(n_observations, n_splits; kwargs...)

Return the first `n_splits` shared UCI repeated-holdout specifications.
"""
function uci_regression_splits(
    n_observations::Int,
    n_splits::Int;
    kwargs...,
)
    n_splits >= 1 || throw(ArgumentError("n_splits must be positive"))
    return [
        uci_regression_split(n_observations, split_id; kwargs...)
        for split_id in 1:n_splits
    ]
end

"""
    uci_regression_splits(dataset, n_splits = UCI_DEFAULT_N_SPLITS; kwargs...)
    uci_regression_splits(DatasetType, n_splits = UCI_DEFAULT_N_SPLITS; dir = nothing, kwargs...)

Construct the shared repeated-holdout splits for a specific UCI regression
dataset. Passing a dataset type, such as `Yacht`, loads the repository's
checksummed copy of that dataset before determining its number of rows.

The returned indices are one-based original-row indices and are identical to
those produced by `uci_regression_splits(size(features, 1), n_splits; kwargs...)`.
The type form makes the dataset association explicit:

```julia
splits = uci_regression_splits(Yacht, 20)
first_yacht_test_indices = splits[1].test_indices
```

Use `dir` with the type form to load a local `data.txt` in the same canonical
row order. All remaining keyword arguments configure the split protocol, for
example `base_seed`, `test_fraction`, and `validation_fraction`.
"""
function uci_regression_splits(
    dataset::UCIRegressionDataset,
    n_splits::Int = UCI_DEFAULT_N_SPLITS;
    kwargs...,
)
    return uci_regression_splits(length(dataset), n_splits; kwargs...)
end

function uci_regression_splits(
    ::Type{T},
    n_splits::Int = UCI_DEFAULT_N_SPLITS;
    dir = nothing,
    kwargs...,
) where {T <: UCIRegressionDataset}
    dataset = T(; dir = dir, as_df = false)
    return uci_regression_splits(dataset, n_splits; kwargs...)
end

function _validate_uci_partition(
    features::AbstractMatrix,
    targets::AbstractVector,
    train_indices::AbstractVector{<:Integer},
    holdout_indices::AbstractVector{<:Integer},
)
    n = size(features, 1)
    n == length(targets) ||
        throw(DimensionMismatch("features and targets disagree"))
    isempty(train_indices) && throw(ArgumentError("training indices are empty"))
    isempty(holdout_indices) && throw(ArgumentError("holdout indices are empty"))
    all(index -> 1 <= index <= n, train_indices) ||
        throw(BoundsError(features, train_indices))
    all(index -> 1 <= index <= n, holdout_indices) ||
        throw(BoundsError(features, holdout_indices))
    isempty(intersect(train_indices, holdout_indices)) ||
        throw(ArgumentError("training and holdout indices overlap"))
    return nothing
end

"""
    fit_uci_standardizer(features, targets, train_indices)

Fit feature and target standardization from training rows only.
"""
function fit_uci_standardizer(
    features::AbstractMatrix,
    targets::AbstractVector,
    train_indices::AbstractVector{<:Integer},
)
    isempty(train_indices) && throw(ArgumentError("training indices are empty"))
    x_train = @view features[train_indices, :]
    y_train = @view targets[train_indices]
    x_center = vec(mean(x_train; dims = 1))
    x_scale = vec(std(x_train; dims = 1, corrected = true))
    x_scale = map(
        scale -> isfinite(scale) && scale > sqrt(eps(Float64)) ? scale : 1.0,
        x_scale,
    )
    y_center = mean(y_train)
    y_scale = std(y_train; corrected = true)
    isfinite(y_scale) && y_scale > sqrt(eps(Float64)) ||
        throw(ArgumentError("training targets have zero or invalid scale"))
    return (
        x_center = Float64.(x_center),
        x_scale = Float64.(x_scale),
        y_center = Float64(y_center),
        y_scale = Float64(y_scale),
    )
end

function transform_uci_features(
    features::AbstractMatrix,
    standardizer,
)
    size(features, 2) == length(standardizer.x_center) ||
        throw(DimensionMismatch("features and standardizer disagree"))
    return Matrix{Float32}(
        (features .- standardizer.x_center') ./ standardizer.x_scale',
    )
end

function transform_uci_targets(
    targets::AbstractVector,
    standardizer,
)
    return Float32.(
        (targets .- standardizer.y_center) ./ standardizer.y_scale,
    )
end

"""
    prepare_uci_regression_partition(
        features, targets, train_indices, holdout_indices,
    )

Slice a UCI partition and standardize it using statistics fitted only on its
training rows.
"""
function prepare_uci_regression_partition(
    features::AbstractMatrix,
    targets::AbstractVector,
    train_indices::AbstractVector{<:Integer},
    holdout_indices::AbstractVector{<:Integer},
)
    vector_targets = vec(targets)
    _validate_uci_partition(
        features, vector_targets, train_indices, holdout_indices,
    )
    standardizer = fit_uci_standardizer(
        features, vector_targets, train_indices,
    )
    x_train = Matrix{Float64}(features[train_indices, :])
    x_test = Matrix{Float64}(features[holdout_indices, :])
    y_train = Float64.(vector_targets[train_indices])
    y_test = Float64.(vector_targets[holdout_indices])
    return (
        train_indices = Int.(train_indices),
        test_indices = Int.(holdout_indices),
        x_train = x_train,
        x_test = x_test,
        y_train = y_train,
        y_test = y_test,
        standardizer = standardizer,
        x_train_standardized =
            transform_uci_features(x_train, standardizer),
        x_test_standardized =
            transform_uci_features(x_test, standardizer),
        y_train_standardized =
            transform_uci_targets(y_train, standardizer),
        y_test_standardized =
            transform_uci_targets(y_test, standardizer),
    )
end
