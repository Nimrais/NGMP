using BBBUCI
using Optimisers
using StableRNGs
using Statistics
using Test
using Zygote

function tiny_config(; kwargs...)
    defaults = (
        datasets = ["yacht"],
        n_splits = 1,
        likelihoods = ["heteroscedastic"],
        hidden_units = 4,
        batch_size = 16,
        learning_rate = 1e-3,
        max_epochs = 4,
        min_epochs = 1,
        validation_every = 1,
        patience = 4,
        train_samples = 1,
        eval_samples = 3,
        show_progress = false,
        save_checkpoints = false,
        make_plot = false,
    )
    return BBBConfig(; merge(defaults, (; kwargs...))...)
end

@testset "softplus and posterior parameterization" begin
    for scale in (1e-3, 0.05, 1.0, 30.0)
        @test stable_softplus(inverse_softplus(scale)) ≈ scale rtol = 1e-10
    end
    config = tiny_config()
    params = initialize_model(3, "heteroscedastic", config; seed = 1)
    @test all(stable_softplus.(params.layer1.weight_rho) .> 0)
    @test mean(stable_softplus.(params.layer1.weight_rho)) ≈
        config.initial_posterior_std rtol = 1e-5
end

@testset "sampling, forward pass, and KL" begin
    config = tiny_config()
    params = initialize_model(3, "heteroscedastic", config; seed = 2)
    rng1 = StableRNG(9)
    rng2 = StableRNG(9)
    epsilon1 = sample_epsilon(params, rng1)
    epsilon2 = sample_epsilon(params, rng2)
    @test epsilon1.layer1.weight == epsilon2.layer1.weight
    x = rand(StableRNG(4), Float32, 7, 3)
    means, scales = forward_sample(
        params, epsilon1, x, "heteroscedastic", config.noise_floor,
    )
    @test size(means) == (7,)
    @test size(scales) == (7,)
    @test all(scales .>= config.noise_floor)
    @test isfinite(gaussian_kl(params, 1.0))
    @test gaussian_kl(params, 1.0) > 0

    scalar_layer = (
        weight_mu = reshape(Float32[0.3], 1, 1),
        weight_rho = reshape(Float32[inverse_softplus(0.2)], 1, 1),
        bias_mu = Float32[-0.4],
        bias_rho = Float32[inverse_softplus(0.7)],
    )
    expected =
        0.5 * (0.2^2 + 0.3^2 - 1 - log(0.2^2)) +
        0.5 * (0.7^2 + 0.4^2 - 1 - log(0.7^2))
    @test BBBUCI.layer_gaussian_kl(scalar_layer, 1.0) ≈ expected rtol = 1e-5
end

@testset "mixture density and uncertainty decomposition" begin
    samples = (
        means = [0.0 1.0; 2.0 3.0],
        variances = [1.0 4.0; 1.0 4.0],
    )
    targets = [0.5, 2.0]
    densities = BBBUCI.mixture_logdensities(samples, targets)
    component1 = [
        -0.5 * (log(2pi) + (0.5 - 0.0)^2),
        -0.5 * (log(2pi) + (0.5 - 2.0)^2),
    ]
    expected1 = log(sum(exp, component1) / 2)
    @test densities.log_predictive_density[1] ≈ expected1

    moments = BBBUCI.predictive_moments(samples)
    @test moments.mean == [1.0, 2.0]
    @test moments.epistemic_variance == [1.0, 1.0]
    @test moments.aleatoric_variance == [1.0, 4.0]
    @test moments.total_variance ≈
        moments.epistemic_variance + moments.aleatoric_variance

    standardizer = (
        x_center = [0.0],
        x_scale = [1.0],
        y_center = 10.0,
        y_scale = 3.0,
    )
    metrics = predictive_metrics(
        samples,
        targets,
        standardizer.y_center .+ standardizer.y_scale .* targets,
        standardizer,
    )
    @test metrics.lpd_original ≈
        metrics.lpd_standardized - log(standardizer.y_scale)
    @test metrics.mean_total_variance_original ≈
        9 * mean(moments.total_variance)
end

@testset "train-only preprocessing and deterministic splits" begin
    x = reshape(collect(1.0:40.0), 20, 2)
    y = collect(1.0:20.0)
    split1 = deterministic_split(x, y; seed = 12, test_fraction = 0.2)
    split2 = deterministic_split(x, y; seed = 12, test_fraction = 0.2)
    @test split1.train_indices == split2.train_indices
    @test split1.test_indices == split2.test_indices
    @test isempty(intersect(split1.train_indices, split1.test_indices))

    prepared = prepare_split(split1)
    @test maximum(abs, vec(mean(prepared.x_train_standardized; dims = 1))) < 1e-6
    @test abs(mean(prepared.y_train_standardized)) < 1e-6

    altered_test = merge(split1, (x_test = split1.x_test .+ 1e6,))
    altered = prepare_split(altered_test)
    @test altered.standardizer == prepared.standardizer
end

@testset "finite ELBO gradient and fixed-sample update" begin
    config = tiny_config()
    params = initialize_model(2, "heteroscedastic", config; seed = 5)
    rng = StableRNG(6)
    x = randn(rng, Float32, 12, 2)
    y = Float32.(0.5 .* x[:, 1] .- x[:, 2])
    epsilons = [sample_epsilon(params, rng)]
    loss_before, gradients = Zygote.withgradient(params) do candidate
        BBBUCI.elbo_loss(
            candidate, epsilons, x, y, "heteroscedastic", config, length(y),
        )
    end
    @test isfinite(loss_before)
    @test isfinite(BBBUCI.gradient_sqnorm(gradients[1]))
    optimizer_state = Optimisers.setup(Optimisers.Adam(1e-4), params)
    _, updated = Optimisers.update(optimizer_state, params, gradients[1])
    loss_after = BBBUCI.elbo_loss(
        updated, epsilons, x, y, "heteroscedastic", config, length(y),
    )
    @test isfinite(loss_after)
    @test loss_after <= loss_before + 1e-5
end

@testset "offline synthetic integration" begin
    rng = StableRNG(22)
    x = randn(rng, 90, 3)
    noise_scale = 0.05 .+ 0.2 .* abs.(x[:, 1])
    y = sin.(x[:, 1]) .+ 0.5 .* x[:, 2] .+
        noise_scale .* randn(rng, 90)
    inner = prepare_split(
        deterministic_split(x, y; seed = 23, test_fraction = 0.2),
    )
    config = tiny_config(max_epochs = 3, patience = 3)
    selection = train_with_validation(
        inner, 3, "heteroscedastic", config;
        seed = 24, validation_seed = 25,
    )
    @test 1 <= selection.best_epoch <= config.max_epochs
    refit = refit_model(
        inner, 3, "heteroscedastic", config;
        seed = 24, epochs = selection.best_epoch,
    )
    samples = predictive_samples(
        refit.params,
        inner.x_test_standardized,
        "heteroscedastic",
        config;
        seed = 26,
    )
    metrics = predictive_metrics(
        samples,
        inner.y_test_standardized,
        inner.y_test,
        inner.standardizer,
    )
    @test isfinite(metrics.lpd_original)
    @test isfinite(metrics.rmse_original)
    @test metrics.mean_total_variance_original > 0
end
