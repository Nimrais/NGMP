function dataset_spec(key::String)
    index = findfirst(spec -> spec[1] == key, DATASET_REGISTRY)
    isnothing(index) && throw(ArgumentError("unknown dataset '$key'"))
    return DATASET_REGISTRY[index], index
end

function load_dataset(key::String)
    (canonical, display_name, constructor), index = dataset_spec(key)
    dataset = constructor(as_df = false)
    features, targets = dataset[:]
    return (
        key = canonical,
        display_name = display_name,
        dataset_index = index,
        features = Matrix{Float64}(features),
        targets = vec(Float64.(targets)),
    )
end

"""
    deterministic_split(features, targets; seed, test_fraction)

Seeded row split identical in semantics to `scripts/uci_hierarchy_deep_kernel.jl`.
Indices are sorted after sampling so row order is stable and train/test never
overlap.
"""
function deterministic_split(
    features::AbstractMatrix,
    targets::AbstractVector;
    seed::Int,
    test_fraction::Real = 0.1,
)
    n = size(features, 1)
    n == length(targets) ||
        throw(DimensionMismatch("features and targets disagree"))
    n >= 3 || throw(ArgumentError("at least three observations are required"))
    0 < test_fraction < 1 ||
        throw(ArgumentError("test_fraction must be in (0, 1)"))

    n_test = clamp(round(Int, test_fraction * n), 1, n - 2)
    order = randperm(StableRNG(seed), n)
    test_indices = sort!(order[1:n_test])
    train_indices = sort!(order[(n_test + 1):end])
    isempty(intersect(train_indices, test_indices)) ||
        error("internal split error: train and test overlap")

    return (
        train_indices = train_indices,
        test_indices = test_indices,
        x_train = Matrix{Float64}(features[train_indices, :]),
        x_test = Matrix{Float64}(features[test_indices, :]),
        y_train = Float64.(targets[train_indices]),
        y_test = Float64.(targets[test_indices]),
    )
end

function fit_standardizer(x_train::AbstractMatrix, y_train::AbstractVector)
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
        x_center = x_center,
        x_scale = x_scale,
        y_center = y_center,
        y_scale = y_scale,
    )
end

function transform_features(x::AbstractMatrix, standardizer)
    return Matrix{Float32}((x .- standardizer.x_center') ./ standardizer.x_scale')
end

function transform_targets(y::AbstractVector, standardizer)
    return Float32.((y .- standardizer.y_center) ./ standardizer.y_scale)
end

function prepare_split(split)
    standardizer = fit_standardizer(split.x_train, split.y_train)
    return merge(split, (
        standardizer = standardizer,
        x_train_standardized = transform_features(split.x_train, standardizer),
        x_test_standardized = transform_features(split.x_test, standardizer),
        y_train_standardized = transform_targets(split.y_train, standardizer),
        y_test_standardized = transform_targets(split.y_test, standardizer),
    ))
end

function outer_and_inner_splits(dataset, split_id::Int, config::BBBConfig)
    outer_seed = config.split_seed + split_id - 1
    outer = deterministic_split(
        dataset.features,
        dataset.targets;
        seed = outer_seed,
        test_fraction = config.test_fraction,
    )

    inner = deterministic_split(
        outer.x_train,
        outer.y_train;
        seed = outer_seed + 1_000_000,
        test_fraction = config.validation_fraction,
    )
    return (
        outer = prepare_split(outer),
        inner = prepare_split(inner),
        outer_seed = outer_seed,
    )
end
