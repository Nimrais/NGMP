using Test
using LinearAlgebra
using Statistics
using BPCUCI

function tiny_config(; kwargs...)
    defaults = (
        datasets = ["yacht"],
        n_splits = 1,
        hidden_units = 3,
        batch_size = 8,
        max_epochs = 2,
        min_epochs = 1,
        validation_every = 1,
        patience = 1,
        latent_steps = 5,
        latent_learning_rate = 1e-3,
        eval_samples = 8,
        backend = "cpu",
        save_checkpoints = false,
        show_progress = false,
    )
    return BPCConfig(; merge(defaults, (; kwargs...))...)
end

@testset "configuration and model shape" begin
    config = validate_config(tiny_config())
    model = initialize_model(2, config; seed = 11)
    @test length(model.layers) == 3
    @test size(model.layers[1].M) == (3, 3)
    @test size(model.layers[2].M) == (3, 4)
    @test size(model.layers[3].M) == (1, 4)
    @test all(layer -> isposdef(Symmetric(layer.V)), model.layers)
    @test all(layer -> isposdef(Symmetric(layer.Psi)), model.layers)
    @test_throws ArgumentError validate_config(tiny_config(backend = "invalid"))
end

@testset "closed-form MNW sufficient statistics" begin
    config = tiny_config(hidden_units = 2, posterior_jitter = 1e-7)
    model = initialize_model(2, config; seed = 12)
    x = Float32[-1 2 3; 4 -5 6]
    z1 = Float32[1 2 3; -2 1 4]
    z2 = Float32[0.5 1 1.5; 2 1 0]
    y = reshape(Float32[1, -1, 2], 1, :)
    states = [x, z1, z2, y]
    first_layer = model.layers[1]
    prior_K = copy(first_layer.prior_K)
    prior_H = copy(first_layer.prior_H)
    prior_G = copy(first_layer.prior_G)
    prior_eta4 = first_layer.prior_eta4
    inputs = vcat(max.(x, 0f0), ones(Float32, 1, 3))

    posterior_update!(model, states, 1.0, 1.0)
    updated = model.layers[1]
    @test updated.K ≈ prior_K + inputs * inputs' rtol = 2e-6
    @test updated.H ≈ prior_H + z1 * inputs' rtol = 2e-6
    @test updated.G ≈ prior_G + z1 * z1' rtol = 2e-6
    @test updated.eta4 ≈ prior_eta4 + 3
    @test updated.M ≈ updated.H * inv(updated.K +
        Float32(config.posterior_jitter) * I) rtol = 2e-4
    @test isposdef(Symmetric(updated.V))
    @test isposdef(Symmetric(updated.Psi))
end

@testset "latent inference lowers expected energy" begin
    base = tiny_config(hidden_units = 4, latent_steps = 0)
    inferred = tiny_config(
        hidden_units = 4,
        latent_steps = 40,
        latent_learning_rate = 5e-4,
    )
    model = initialize_model(2, base; seed = 13)
    x = randn(Float32, 2, 12)
    y = reshape(Float32.(0.7 .* x[1, :] .- 0.2 .* x[2, :]), 1, :)
    initial_states = infer_latent_states(model, x, y, base)
    optimized_states = infer_latent_states(model, x, y, inferred)
    @test expected_energy(model, optimized_states) <=
        expected_energy(model, initial_states) + 1e-3
end

@testset "posterior prediction and metric contract" begin
    config = tiny_config(hidden_units = 3, eval_samples = 10)
    model = initialize_model(2, config; seed = 14)
    x_columns = randn(Float32, 2, 16)
    y_columns = reshape(Float32.(x_columns[1, :] .+ 0.1 .* x_columns[2, :]), 1, :)
    states = infer_latent_states(model, x_columns, y_columns, config)
    posterior_update!(model, states, 1.0, 1.0)

    features = Matrix{Float64}(transpose(x_columns))
    samples = predictive_samples(model, features, config; seed = 99)
    @test size(samples.means) == (10, 16)
    @test size(samples.variances) == (10, 16)
    @test all(isfinite, samples.means)
    @test all(>(0), samples.variances)

    y = vec(Float64.(y_columns))
    standardizer = (y_center = 2.0, y_scale = 3.0)
    metrics = predictive_metrics(samples, y, 2 .+ 3 .* y, standardizer)
    @test isfinite(metrics.lpd_original)
    @test isfinite(metrics.rmse_original)
    @test metrics.mean_total_variance_original > 0
    @test metrics.mean_total_variance_original ≈
        metrics.mean_epistemic_variance_original +
        metrics.mean_aleatoric_variance_original
end

@testset "adaptive predictive Cholesky stabilization" begin
    nearly_positive = Float32[1 1; 1 1 - 1f-4]
    @test !isposdef(Symmetric(nearly_positive + 1f-7 * I))
    factor = BPCUCI.stable_cholesky(nearly_positive, 1e-7)
    @test issuccess(factor)
    @test all(isfinite, factor.L)
    inverse = BPCUCI.stable_spd_inverse(nearly_positive, 1e-7)
    @test isposdef(Symmetric(inverse))
end

@testset "shared split protocol" begin
    config = tiny_config()
    dataset = (features = randn(100, 2), targets = randn(100))
    split = BPCUCI.uci_regression_split(
        100, 1;
        base_seed = config.split_seed,
        test_fraction = config.test_fraction,
        validation_fraction = config.validation_fraction,
    )
    @test split.protocol_version == BPCUCI.UCI_SPLIT_PROTOCOL_VERSION
    @test length(split.test_indices) == 10
    @test isempty(intersect(split.train_indices, split.test_indices))
end

@testset "optional CUDA parity smoke" begin
    if backend_available("cuda")
        cpu_config = tiny_config(hidden_units = 3, latent_steps = 2)
        gpu_config = tiny_config(
            hidden_units = 3, latent_steps = 2, backend = "cuda",
        )
        cpu_model = initialize_model(2, cpu_config; seed = 15)
        gpu_model = initialize_model(2, gpu_config; seed = 15)
        x = randn(Float32, 2, 8)
        y = reshape(randn(Float32, 8), 1, :)
        cpu_states = infer_latent_states(cpu_model, x, y, cpu_config)
        gpu_states = infer_latent_states(
            gpu_model,
            BPCUCI.to_device(x, gpu_config),
            BPCUCI.to_device(y, gpu_config),
            gpu_config,
        )
        posterior_update!(cpu_model, cpu_states, 1.0)
        posterior_update!(gpu_model, gpu_states, 1.0)
        gpu_host = BPCUCI.host_model(gpu_model)
        @test gpu_host.layers[1].M ≈ cpu_model.layers[1].M rtol = 2e-4 atol = 2e-5
    else
        @test true
    end
end
