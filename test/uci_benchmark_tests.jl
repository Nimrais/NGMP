@testset "shared UCI repeated-holdout protocol" begin
    split = uci_regression_split(
        20,
        1;
        base_seed = 12,
        test_fraction = 0.2,
        validation_fraction = 0.25,
    )
    repeated = uci_regression_split(
        20,
        1;
        base_seed = 12,
        test_fraction = 0.2,
        validation_fraction = 0.25,
    )
    @test split.protocol_version == UCI_SPLIT_PROTOCOL_VERSION
    @test split.test_indices == [7, 8, 9, 16]
    @test split.test_indices == repeated.test_indices
    @test split.inner_train_indices == repeated.inner_train_indices
    @test sort(vcat(split.train_indices, split.test_indices)) == 1:20
    @test isempty(intersect(split.train_indices, split.test_indices))
    @test sort(vcat(
        split.inner_train_indices, split.validation_indices,
    )) == split.train_indices
    @test isempty(intersect(
        split.validation_indices, split.test_indices,
    ))

    first_five = uci_regression_splits(40, 5; base_seed = 99)
    first_twenty = uci_regression_splits(40, 20; base_seed = 99)
    @test getfield.(first_five, :test_indices) ==
        getfield.(first_twenty[1:5], :test_indices)
end

@testset "shared UCI train-only preprocessing" begin
    features = reshape(collect(1.0:40.0), 20, 2)
    targets = collect(1.0:20.0)
    split = uci_regression_split(
        20,
        1;
        base_seed = 12,
        test_fraction = 0.2,
    )
    prepared = prepare_uci_regression_partition(
        features, targets, split.train_indices, split.test_indices,
    )
    @test maximum(abs, vec(mean(
        prepared.x_train_standardized; dims = 1,
    ))) < 1e-6
    @test abs(mean(prepared.y_train_standardized)) < 1e-6

    altered = copy(features)
    altered[split.test_indices, :] .+= 1e6
    altered_prepared = prepare_uci_regression_partition(
        altered, targets, split.train_indices, split.test_indices,
    )
    @test altered_prepared.standardizer == prepared.standardizer
end
