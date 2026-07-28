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

function outer_and_inner_splits(
    dataset,
    split_id::Int,
    config::DVIConfig,
)
    spec = uci_regression_split(
        size(dataset.features, 1),
        split_id;
        base_seed = config.split_seed,
        test_fraction = config.test_fraction,
        validation_fraction = config.validation_fraction,
    )
    return (
        outer = prepare_uci_regression_partition(
            dataset.features,
            dataset.targets,
            spec.train_indices,
            spec.test_indices,
        ),
        inner = prepare_uci_regression_partition(
            dataset.features,
            dataset.targets,
            spec.inner_train_indices,
            spec.validation_indices,
        ),
        outer_seed = spec.outer_seed,
        split_spec = spec,
    )
end
