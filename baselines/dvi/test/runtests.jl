using DVIUCI
using LinearAlgebra
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
        propagation = "full",
        hidden_units = 50,
        batch_size = 16,
        learning_rate = 1e-3,
        max_epochs = 4,
        min_epochs = 1,
        validation_every = 1,
        patience = 4,
        kl_warmup_epochs = 0,
        kl_anneal_epochs = 0,
        show_progress = false,
        save_checkpoints = false,
        make_plot = false,
    )
    return DVIConfig(; merge(defaults, (; kwargs...))...)
end

@testset "Gaussian and soft-ReLU helpers" begin
    @test standard_gaussian(0.0) ≈ inv(sqrt(2pi)) rtol = 1e-6
    @test gaussian_cdf(0.0) ≈ 0.5
    @test softrelu(0.0) ≈ inv(sqrt(2pi)) rtol = 1e-6
    @test gaussian_cdf(-1.2) ≈ 1 - gaussian_cdf(1.2) rtol = 1e-12
end

@testset "full and diagonal ReLU moments" begin
    rho = 0.5f0
    means = zeros(Float32, 1, 2)
    covariance = reshape(
        Float32[1 rho; rho 1],
        1,
        2,
        2,
    )
    full = relu_moments_full(means, covariance)
    diagonal = relu_moments_diagonal(
        means, Float32[1 1],
    )
    expected_mean = inv(sqrt(2pi))
    expected_variance = 0.5 - inv(2pi)
    expected_product =
        (
            sqrt(1 - rho^2) +
            (pi - acos(rho)) * rho
        ) / (2pi)
    expected_covariance = expected_product - inv(2pi)

    @test full.mean[1, 1] ≈ expected_mean rtol = 1e-5
    @test full.covariance[1, 1, 1] ≈ expected_variance rtol = 2e-5
    @test full.covariance[1, 1, 2] ≈ expected_covariance rtol = 2e-5
    @test diagonal.mean[1, 1] ≈ expected_mean rtol = 1e-5
    @test diagonal.variance[1, 1] ≈ expected_variance rtol = 2e-5
end

@testset "certain-input affine moments" begin
    layer = (
        weight_mu = Float32[0.2 -0.3; 0.5 0.1],
        weight_log_std = log.(Float32[0.4 0.2; 0.3 0.5]),
        bias_mu = Float32[0.1, -0.2],
        bias_log_std = log.(Float32[0.25, 0.35]),
    )
    features = Float32[1.5 -0.7; -0.2 0.9]
    observed = DVIUCI.linear_certain_full(features, layer)
    expected_mean =
        features * transpose(layer.weight_mu) .+
        transpose(layer.bias_mu)
    expected_variance =
        features .^ 2 * transpose(exp.(2 .* layer.weight_log_std)) .+
        transpose(exp.(2 .* layer.bias_log_std))
    @test observed.mean ≈ expected_mean
    @test DVIUCI.covariance_diagonal(observed.covariance) ≈
        expected_variance
    @test observed.covariance[:, 1, 2] == zeros(Float32, 2)
end

@testset "DVI moments agree with network Monte Carlo" begin
    config = tiny_config(hidden_units = 50)
    params = initialize_model(3, "heteroscedastic", config; seed = 11)
    features = Float32[0.3 -0.7 1.1]
    deterministic = propagate_dvi(params, features, config)
    rng = StableRNG(12)
    draws = reduce(
        hcat,
        (
            vec(DVIUCI.sample_forward(params, features, rng))
            for _ in 1:8_000
        ),
    )
    monte_carlo_mean = vec(mean(draws; dims = 2))
    monte_carlo_covariance = cov(transpose(draws); corrected = true)
    @test vec(deterministic.mean) ≈ monte_carlo_mean atol = 0.12
    @test deterministic.covariance[1, 1, 1] ≈
        monte_carlo_covariance[1, 1] rtol = 0.15
    @test deterministic.covariance[1, 2, 2] ≈
        monte_carlo_covariance[2, 2] rtol = 0.15
end

@testset "empirical-Bayes prior and KL" begin
    means = Float32[0.3, -0.4]
    log_stds = log.(Float32[0.2, 0.7])
    alpha = 1.0
    beta = 10.0
    second_moment = sum(exp.(2 .* log_stds) .+ means .^ 2)
    expected_variance =
        (second_moment + 2beta) / (length(means) + 2alpha + 2)
    expected_kl = 0.5 * (
        length(means) * log(expected_variance) +
        second_moment / expected_variance -
        (length(means) + 2sum(log_stds))
    )
    @test DVIUCI.empirical_bayes_group_prior_variance(
        means, log_stds, alpha, beta,
    ) ≈ expected_variance rtol = 1e-6
    @test DVIUCI.empirical_bayes_group_kl(
        means, log_stds, alpha, beta,
    ) ≈ expected_kl rtol = 1e-6

    layer = (
        weight_mu = reshape(means, 1, 2),
        weight_log_std = reshape(log_stds, 1, 2),
        bias_mu = Float32[0.1],
        bias_log_std = log.(Float32[0.3]),
    )
    layer_second_moment =
        second_moment + exp(2 * layer.bias_log_std[1]) +
        layer.bias_mu[1]^2
    layer_size = 3
    layer_variance =
        (layer_second_moment + 2beta) / (layer_size + 2alpha + 2)
    layer_kl = 0.5 * (
        layer_size * log(layer_variance) +
        layer_second_moment / layer_variance -
        (
            layer_size +
            2 * sum(log_stds) +
            2 * layer.bias_log_std[1]
        )
    )
    @test DVIUCI.empirical_bayes_layer_prior_variance(
        layer, alpha, beta,
    ) ≈ layer_variance rtol = 1e-6
    @test DVIUCI.empirical_bayes_layer_kl(
        layer, alpha, beta,
    ) ≈ layer_kl rtol = 1e-6
end

@testset "heteroscedastic expected likelihood equation 8" begin
    config = tiny_config()
    output = (
        mean = Float32[0.4 -0.7],
        covariance = reshape(
            Float32[0.3 0.08; 0.08 0.2],
            1,
            2,
            2,
        ),
    )
    target = Float32[0.1]
    expected = -0.5 * (
        log(2pi) - 0.7 +
        exp(0.7 + 0.1) * (0.3 + (0.4 - 0.08 - 0.1)^2)
    )
    @test only(expected_log_likelihood(
        output, target, "heteroscedastic", config,
    )) ≈ expected rtol = 1e-6
end

@testset "diagonal-DVI propagation" begin
    config = tiny_config(
        propagation = "diagonal",
        hidden_units = 8,
        initialization_scale = 0.1,
    )
    params = initialize_model(3, "heteroscedastic", config; seed = 15)
    features = randn(StableRNG(16), Float32, 5, 3)
    targets = randn(StableRNG(17), Float32, 5)
    output = propagate_dvi(params, features, config)
    @test size(output.mean) == (5, 2)
    @test size(output.variance) == (5, 2)
    @test all(output.variance .>= 0)
    @test isfinite(dvi_loss(
        params,
        features,
        targets,
        "heteroscedastic",
        config,
        length(targets),
    ))
end

@testset "preprocessing and deterministic splits" begin
    x = reshape(collect(1.0:40.0), 20, 2)
    y = collect(1.0:20.0)
    split1 = uci_regression_split(
        20, 1; base_seed = 12, test_fraction = 0.2,
    )
    split2 = uci_regression_split(
        20, 1; base_seed = 12, test_fraction = 0.2,
    )
    @test split1.train_indices == split2.train_indices
    @test split1.test_indices == split2.test_indices
    @test isempty(intersect(split1.train_indices, split1.test_indices))
    prepared = prepare_uci_regression_partition(
        x, y, split1.train_indices, split1.test_indices,
    )
    @test maximum(abs, vec(mean(
        prepared.x_train_standardized; dims = 1,
    ))) < 1e-6
    @test abs(mean(prepared.y_train_standardized)) < 1e-6
    @test DVIUCI.earliest_selection_epoch(tiny_config(
        min_epochs = 2,
        kl_warmup_epochs = 5,
        kl_anneal_epochs = 3,
        max_epochs = 10,
    )) == 8
end

@testset "finite deterministic ELBO gradient" begin
    config = tiny_config()
    params = initialize_model(2, "heteroscedastic", config; seed = 20)
    rng = StableRNG(21)
    features = randn(rng, Float32, 12, 2)
    targets = Float32.(0.5 .* features[:, 1] .- features[:, 2])
    loss, gradients = Zygote.withgradient(params) do candidate
        dvi_loss(
            candidate,
            features,
            targets,
            "heteroscedastic",
            config,
            length(targets),
        )
    end
    @test isfinite(loss)
    @test DVIUCI.parameters_are_finite(gradients[1])
    clipped = DVIUCI.clip_gradient(gradients[1], config.gradient_clip)
    @test DVIUCI.gradient_maximum_absolute(clipped) <=
        config.gradient_clip
    optimizer_state = Optimisers.setup(Optimisers.Adam(1e-4), params)
    _, updated = Optimisers.update(
        optimizer_state, params, clipped,
    )
    @test isfinite(dvi_loss(
        updated,
        features,
        targets,
        "heteroscedastic",
        config,
        length(targets),
    ))
end

@testset "offline synthetic integration" begin
    rng = StableRNG(30)
    features = randn(rng, 90, 3)
    noise_scale = 0.05 .+ 0.2 .* abs.(features[:, 1])
    targets =
        sin.(features[:, 1]) .+
        0.5 .* features[:, 2] .+
        noise_scale .* randn(rng, 90)
    split = uci_regression_split(
        length(targets), 1; base_seed = 31, test_fraction = 0.2,
    )
    prepared = prepare_uci_regression_partition(
        features,
        targets,
        split.train_indices,
        split.test_indices,
    )
    config = tiny_config(
        hidden_units = 6,
        initialization_scale = 0.1,
        max_epochs = 3,
        patience = 3,
    )
    selection = train_with_validation(
        prepared, 3, "heteroscedastic", config; seed = 32,
    )
    @test 1 <= selection.best_epoch <= config.max_epochs
    refit = refit_model(
        prepared,
        3,
        "heteroscedastic",
        config;
        seed = 32,
        epochs = selection.best_epoch,
    )
    metrics = predictive_metrics(
        refit.params,
        prepared.x_test_standardized,
        prepared.y_test_standardized,
        prepared.y_test,
        prepared.standardizer,
        "heteroscedastic",
        config,
    )
    @test isfinite(metrics.lpd_original)
    @test isfinite(metrics.rmse_original)
    @test metrics.mean_total_variance_original > 0
end
