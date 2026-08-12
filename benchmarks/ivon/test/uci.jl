using DataFrames
using IVONRepro: posterior_variance
using StableRNGs

include(joinpath(@__DIR__, "..", "run_uci.jl"))
using .IVONUCI

function synthetic_uci_dataset(; n = 48, d = 4, seed = 91)
    rng = StableRNG(seed)
    features = randn(rng, n, d)
    targets = 0.7 .* features[:, 1] .- 0.25 .* features[:, 2] .+
        0.05 .* randn(rng, n)
    return (
        key = "yacht",
        name = "Synthetic Yacht",
        index = 1,
        features = Matrix{Float64}(features),
        targets = Float64.(targets),
    )
end

function synthetic_partition(; n_train = 18, n_test = 6, d = 3, seed = 83)
    rng = StableRNG(seed)
    features = randn(rng, Float32, n_train + n_test, d)
    targets = 0.4f0 .* features[:, 1] .- 0.15f0 .* features[:, 2]
    return (
        x_train_standardized = features[1:n_train, :],
        y_train_standardized = targets[1:n_train],
        x_test_standardized = features[(n_train + 1):end, :],
        y_test_standardized = targets[(n_train + 1):end],
        y_test = Float64.(targets[(n_train + 1):end]),
        standardizer = (y_center = 0.0, y_scale = 1.0),
    )
end

@testset "IVON UCI BBB-compatible seed schedule" begin
    @test IVONUCI.SPLIT_BASE_SEED == 20_260_726
    @test IVONUCI.MODEL_BASE_SEED == 20_260_727
    for dataset_index in eachindex(DATASET_REGISTRY), split_id in (1, 7, 20)
        seeds = model_and_evaluation_seeds(dataset_index, split_id)
        expected_model = 20_260_727 + 10_000 * dataset_index +
            100 * split_id + 1
        @test seeds == (
            model = expected_model,
            batches = expected_model + 10,
            validation = expected_model + 20,
            test = expected_model + 30,
        )
    end
end

@testset "IVON UCI deterministic replay and posterior variances" begin
    partition = synthetic_partition()
    config = UCIConfig(
        datasets = ["yacht"],
        n_splits = 1,
        hidden_units = 4,
        max_epochs = 1,
        min_epochs = 1,
        validation_every = 1,
        patience = 1,
        eval_samples = 3,
        output_root = "/tmp/ivon-uci-unit",
        show_progress = false,
    )
    seeds = model_and_evaluation_seeds(1, 1)
    first_fit = train_with_validation(
        partition, config; learning_rate = 0.01, ess_multiplier = 1, seeds,
    )
    second_fit = train_with_validation(
        partition, config; learning_rate = 0.01, ess_multiplier = 1, seeds,
    )
    @test first_fit.state.parameters == second_fit.state.parameters
    @test first_fit.state.optimizer_state == second_fit.state.optimizer_state
    @test first_fit.best_components == second_fit.best_components
    @test IVONUCI.posterior_variances_valid(
        posterior_variance(first_fit.optimizer, first_fit.state.optimizer_state),
    )
    @test first_fit.best_metrics.mean_total_variance_standardized > 0
end

@testset "IVON UCI pilot selection" begin
    rows = NamedTuple[]
    for dataset in IVONUCI.DATASET_KEYS,
        learning_rate in LEARNING_RATES,
        ess_multiplier in ESS_MULTIPLIERS
        winner = learning_rate == 0.01 && ess_multiplier == 100
        push!(rows, (
            dataset = dataset,
            learning_rate = learning_rate,
            ess_multiplier = ess_multiplier,
            status = "success",
            validation_lpd_standardized = winner ? -0.8 : -1.0,
            validation_rmse_standardized = winner ? 0.9 : 1.0,
        ))
    end
    selected = select_pilot_configuration(DataFrame(rows)).selected
    @test selected.learning_rate == 0.01
    @test selected.ess_multiplier == 100

    tied = DataFrame(rows)
    tied.validation_lpd_standardized .= -1.0
    tied.validation_rmse_standardized .= 1.0
    tie_selected = select_pilot_configuration(tied).selected
    @test tie_selected.learning_rate == minimum(LEARNING_RATES)
    @test tie_selected.ess_multiplier == minimum(ESS_MULTIPLIERS)
end

@testset "IVON UCI finite-mixture metrics" begin
    components = (
        means = [0.0 1.0; 2.0 3.0],
        variances = ones(2, 2),
    )
    metrics = predictive_metrics(
        components,
        [1.0, 2.0],
        [12.0, 14.0],
        (y_center = 10.0, y_scale = 2.0),
    )
    @test isfinite(metrics.lpd_standardized)
    @test metrics.lpd_original ≈ metrics.lpd_standardized - log(2.0)
    @test metrics.mean_epistemic_variance_standardized ≈ 1.0
    @test metrics.mean_aleatoric_variance_standardized ≈ 1.0
    @test metrics.mean_total_variance_standardized ≈ 2.0
    @test metrics.mean_total_variance_original ≈ 8.0
    @test all(isfinite, (metrics.coverage_50, metrics.coverage_80,
        metrics.coverage_95, metrics.rmse_original))

    extreme = predictive_metrics(
        (means = [1.0e5 -1.0e5; -1.0e5 1.0e5], variances = fill(1.0e4, 2, 2)),
        [0.0, 0.0],
        [0.0, 0.0],
        (y_center = 0.0, y_scale = 1.0),
    )
    @test isfinite(extreme.lpd_standardized)
end

@testset "IVON UCI checkpoint replay" begin
    dataset = synthetic_uci_dataset()
    mktempdir() do output_root
        config = UCIConfig(
            datasets = ["yacht"],
            n_splits = 1,
            hidden_units = 4,
            max_epochs = 1,
            min_epochs = 1,
            validation_every = 1,
            patience = 1,
            eval_samples = 3,
            output_root = output_root,
            show_progress = false,
        )
        selected = (learning_rate = 0.01, ess_multiplier = 1)
        row = IVONUCI.run_final_configuration(dataset, 1, config, selected)
        path = IVONUCI.final_checkpoint_path(config, "yacht", 1)
        replay = replay_checkpoint(path)
        @test row.status == "success"
        @test replay.digest == IVONUCI.component_digest(replay.components)
        @test replay.metrics.lpd_original == row.lpd_original
        @test replay.metrics.rmse_original == row.rmse_original
        @test replay.metrics.mean_total_variance_original > 0

        runs = DataFrame([row])
        @test validate_final_rows(runs, config; require_complete = true)
        first_summary = IVONUCI.summary_table(runs, config)
        second_summary = IVONUCI.summary_table(runs, config)
        @test isequal(first_summary, second_summary)
    end
end
