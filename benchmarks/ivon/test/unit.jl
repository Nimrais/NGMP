using JLD2
using Lux
using Optimisers
using StableRNGs

function synthetic_data(; n = 6, seed = 73)
    rng = StableRNG(seed)
    features = [randn(rng, Float32, 65) for _ = 1:n]
    targets = [[Float32(0.25 * j)] for j = 1:n]
    predictions = Array{Vector{Float32}}(undef, 7, n)
    for i = 1:7, j = 1:n
        predictions[i, j] = [targets[j][1] + Float32(0.03 * (i - 4))]
    end
    return (; predictions, features, targets)
end

@testset "dependency and frozen-model provenance" begin
    @test IVONPGE.verify_dependency_pins()
    hashes = verify_frozen_hashes("ETTh1", 96)
    @test Set(keys(hashes)) == Set(["CNN", "NLinear", "LSTM", "DLinear", "NConv", "VAE"])
end

@testset "gate architectures" begin
    rng = StableRNG(1)
    for architecture in (:moe, :moe_big)
        gate = build_gate(architecture)
        ps, st = Lux.setup(rng, gate)
        @test parameter_count(ps) == expected_parameter_count(architecture)
        logits, _ = gate(ones(Float32, 65), ps, st)
        @test size(logits) == (7,)
    end
    big = build_gate(:moe_big)
    ps, st = Lux.setup(StableRNG(2), big)
    probe = fill(-1.0f0, 65)
    hidden_pre = ps.layer_1.weight * probe .+ ps.layer_1.bias
    full, _ = big(probe, ps, st)
    expected = ps.layer_2.weight * max.(hidden_pre, 0.0f0) .+ ps.layer_2.bias
    @test full ≈ expected
end

@testset "IVON update, deterministic replay, and posterior variance" begin
    data = synthetic_data()
    first_run = train_ivon_gate(data, data, :moe;
        learning_rate = 0.01, ess_multiplier = 1, seed = 12345,
        n_epochs = 1, patience = 2, min_delta = 0.0)
    second_run = train_ivon_gate(data, data, :moe;
        learning_rate = 0.01, ess_multiplier = 1, seed = 12345,
        n_epochs = 1, patience = 2, min_delta = 0.0)
    @test first_run.step == length(data.features)
    @test first_run.parameters == second_run.parameters
    @test first_run.optimizer_state == second_run.optimizer_state
    @test posterior_variances_valid(
        IVONPGE.posterior_variance(first_run.optimizer, first_run.optimizer_state))

    first_components = IVONPGE.posterior_components(first_run, data;
        nsamples = 8, seed = 44)
    second_components = IVONPGE.posterior_components(first_run, data;
        nsamples = 8, seed = 44)
    @test first_components.means == second_components.means
    @test first_components.variances == second_components.variances
end

@testset "stable mixture metrics and uncertainty decomposition" begin
    means = [0.0 1.0; 2.0 3.0]
    variances = [1.0 2.0; 1.0 2.0]
    components = (;
        means,
        variances,
        gate_diagnostics = (
            probability_mean = fill(1 / 7, 7),
            probability_std = zeros(7),
            entropy = log(7),
            posterior_mean_entropy = log(7),
            top_expert_share = [1.0; zeros(6)],
            max_top_expert_share = 1.0,
            switching_rate = 0.0,
            posterior_mean_switching_rate = 0.0,
        ),
    )
    result = mixture_metrics(components, [[1.0], [2.0]]; interval_seed = 9)
    @test result.metrics.epistemic_variance ≈ 1.0
    @test result.metrics.aleatoric_variance ≈ 1.5
    @test result.metrics.total_variance ≈ 2.5
    @test isfinite(result.metrics.nll)
    @test isfinite(result.metrics.crps)

    extreme = IVONPGE.stable_mixture_logpdf([1.0e5, -1.0e5], [1.0e-4, 1.0e4], 0.0)
    @test isfinite(extreme)

    original = IVONPGE.original_unit_metrics(result.metrics, 2.5)
    @test original.nll ≈ result.metrics.nll + log(2.5)
    @test original.log_predictive_density ≈
        result.metrics.log_predictive_density - log(2.5)
    @test original.total_variance ≈ result.metrics.total_variance * 2.5^2
end

@testset "zero posterior variance limit" begin
    data = synthetic_data(; n = 3)
    training = train_ivon_gate(data, data, :moe;
        learning_rate = 0.01, ess_multiplier = 1, seed = 4,
        n_epochs = 1, patience = 2, min_delta = 0.0)
    mean_components = IVONPGE.posterior_components(training, data;
        nsamples = 1, seed = 8, sample_posterior = false)
    repeated_mean = IVONPGE.posterior_components(training, data;
        nsamples = 10, seed = 8, sample_posterior = false)
    @test all(repeated_mean.means .== repeat(mean_components.means; outer = (10, 1)))
    @test all(repeated_mean.variances .== repeat(mean_components.variances; outer = (10, 1)))
end

@testset "checkpoint round trip" begin
    data = synthetic_data(; n = 3)
    training = train_ivon_gate(data, data, :moe;
        learning_rate = 0.01, ess_multiplier = 1, seed = 12,
        n_epochs = 1, patience = 2, min_delta = 0.0)
    components = IVONPGE.posterior_components(training, data; nsamples = 2, seed = 13)
    standardized = mixture_metrics(components, data.targets; interval_seed = 14)
    mean_components = IVONPGE.posterior_components(training, data;
        nsamples = 1, seed = 13, sample_posterior = false)
    mean_standardized = mixture_metrics(mean_components, data.targets; interval_seed = 14)
    config = IVONPGE.effective_config(; phase = :test, dataset = "ETTh1",
        horizon = 96, architecture = :moe, learning_rate = 0.01,
        ess_multiplier = 1, n_epochs = 1, posterior_samples = 2)
    mktempdir() do directory
        path = joinpath(directory, "roundtrip.jld2")
        IVONPGE.save_checkpoint(path;
            config,
            training,
            frozen_hashes = Dict("synthetic" => "unchanged"),
            quantile_hashes_before = Dict("q10" => "a", "q90" => "b"),
            quantile_hashes_after = Dict("q10" => "a", "q90" => "b"),
            split_metadata = (test_targets_read_during_training = false,),
            scaler = nothing,
            standardized,
            original_units = IVONPGE.original_unit_metrics(standardized.metrics, 2.0),
            mean_standardized,
            mean_original_units = IVONPGE.original_unit_metrics(
                mean_standardized.metrics, 2.0),
            posterior_components = components,
            mean_components,
            prediction_digest = IVONPGE.component_digest(components),
        )
        loaded = IVONPGE.load_checkpoint(path)
        @test loaded["gate_parameters_mean"] == training.parameters
        @test loaded["optimizer_state"] == training.optimizer_state
        @test loaded["prediction_digest"] == IVONPGE.component_digest(components)
    end
end
