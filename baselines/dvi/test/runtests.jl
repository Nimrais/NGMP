using DVIUCI
using CSV
using DataFrames
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
        split_ids = [1],
        likelihoods = ["heteroscedastic"],
        propagation = "full",
        hidden_units = 50,
        batch_size = 16,
        learning_rate = 3e-4,
        max_epochs = 4,
        min_epochs = 1,
        validation_every = 1,
        patience = 4,
        kl_schedule_unit = "steps",
        kl_warmup_steps = 0,
        kl_anneal_steps = 0,
        kl_warmup_epochs = 0,
        kl_anneal_epochs = 0,
        show_progress = false,
        save_checkpoints = false,
        make_plot = false,
    )
    return DVIConfig(; merge(defaults, (; kwargs...))...)
end

@testset "split selection" begin
    @test parse_split_ids("all", 4) == [1, 2, 3, 4]
    @test parse_split_ids("4, 2, 2", 4) == [2, 4]
    @test_throws ArgumentError parse_split_ids("0", 4)
    @test_throws ArgumentError parse_split_ids("5", 4)
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
    observed = DVIUCI.linear_certain_full(features, layer, tiny_config())
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

@testset "allocation-efficient full-covariance helpers" begin
    rng = StableRNG(101)
    covariance = randn(rng, Float32, 7, 5, 5)
    covariance = covariance .* permutedims(covariance, (1, 3, 2))
    weights = randn(rng, Float32, 2, 5)

    expected_diagonal = hcat([
        covariance[:, index, index] for index in axes(covariance, 2)
    ]...)
    @test DVIUCI.covariance_diagonal(covariance) == expected_diagonal

    expected_quadratic = permutedims(cat([
        weights * covariance[batch, :, :] * transpose(weights)
        for batch in axes(covariance, 1)
    ]...; dims = 3), (3, 1, 2))
    @test DVIUCI.batch_quadratic(weights, covariance) ≈
        expected_quadratic rtol = 2e-6 atol = 2e-6

    objective(candidate) = sum(DVIUCI.batch_quadratic(
        candidate, covariance,
    ))
    gradient = only(Zygote.gradient(objective, weights))
    reference_objective(candidate) = sum(permutedims(cat([
        candidate * covariance[batch, :, :] * transpose(candidate)
        for batch in axes(covariance, 1)
    ]...; dims = 3), (3, 1, 2)))
    reference_gradient = only(Zygote.gradient(
        reference_objective, weights,
    ))
    @test all(isfinite, gradient)
    @test gradient ≈ reference_gradient rtol = 1e-5 atol = 1e-6
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
            vec(DVIUCI.sample_forward(params, features, rng, config))
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
        batch_size = 10,
        kl_warmup_steps = 5,
        kl_anneal_steps = 3,
        max_epochs = 10,
    ), 20) == 4
end

@testset "bounded numerical protocol" begin
    config = tiny_config(safe_exp_min = -20.0, safe_exp_max = 20.0)
    tracker = NumericalTracker()
    values = Float32[-100, -1, 0, 1, 100]
    observed = safe_exp(values, config, tracker)
    @test all(isfinite, observed)
    @test observed[1] ≈ exp(-20f0)
    @test observed[end] ≈ exp(20f0)
    counts = tracker_record(tracker)
    @test counts.lower_clamps == 1
    @test counts.upper_clamps == 1
    @test counts.clamp_count == 2
    @test counts.clamp_rate == 0.4

    @test kl_weight(14_000, DVIConfig()) == 0.0
    @test kl_weight(14_500, DVIConfig()) == 0.5
    @test kl_weight(15_000, DVIConfig()) == 1.0
    @test_throws ArgumentError validate_config(tiny_config(
        kl_warmup_steps = 1,
        kl_warmup_epochs = 1,
    ))

    @test_throws ArgumentError validate_config(tiny_config(
        execution_backend = "zygote",
        execution_device = "gpu",
    ))
    @test validate_config(tiny_config(
        execution_backend = "reactant",
        execution_device = "gpu",
    )).implementation_version == DVIUCI.DVI_IMPLEMENTATION_VERSION
    @test occursin(
        "selection-optimized-v2", DVIUCI.DVI_IMPLEMENTATION_VERSION,
    )
    @test_throws ArgumentError validate_config(tiny_config(
        full_selection_patience_steps = 0,
    ))
    @test_throws ArgumentError validate_config(tiny_config(
        selection_max_optimizer_steps = -1,
    ))
    @test_throws ArgumentError validate_config(tiny_config(
        refit_max_optimizer_steps = -1,
    ))
    @test_throws ArgumentError validate_config(tiny_config(
        kl_warmup_steps = 10,
        kl_anneal_steps = 5,
        selection_max_optimizer_steps = 14,
    ))
    @test validate_config(tiny_config(
        kl_warmup_steps = 10,
        kl_anneal_steps = 5,
        selection_max_optimizer_steps = 15,
        refit_max_optimizer_steps = 20,
        training_budget_protocol = "selection-refit-max-50k-v1",
    )).selection_max_optimizer_steps == 15
    full_selection_config = tiny_config(
        propagation = "full",
        validation_every = 5,
        max_epochs = 100,
        full_selection_patience_steps = 50,
    )
    diagonal_selection_config = tiny_config(
        propagation = "diagonal",
        validation_every = 5,
        max_epochs = 100,
        full_selection_patience_steps = 50,
    )
    @test !DVIUCI.should_validate_selection_epoch(
        5, 20, full_selection_config,
    )
    @test DVIUCI.should_validate_selection_epoch(
        20, 20, full_selection_config,
    )
    @test DVIUCI.should_validate_selection_epoch(
        5, 20, diagonal_selection_config,
    )
    @test !DVIUCI.selection_patience_exhausted(
        full_selection_config, 1, 149, 100,
    )
    @test DVIUCI.selection_patience_exhausted(
        full_selection_config, 1, 150, 100,
    )
    @test !DVIUCI.selection_patience_exhausted(
        diagonal_selection_config, 1, 150, 100,
    )
    @test DVIUCI.selection_patience_exhausted(
        diagonal_selection_config,
        diagonal_selection_config.patience,
        150,
        100,
    )

    loss_config = tiny_config()
    params = initialize_model(
        2, "heteroscedastic", loss_config; seed = 102,
    )
    features = randn(StableRNG(103), Float32, 5, 2)
    targets = randn(StableRNG(104), Float32, 5)
    loss_tracker = NumericalTracker()
    dvi_loss(
        params,
        features,
        targets,
        "heteroscedastic",
        loss_config,
        length(targets);
        tracker = loss_tracker,
    )
    expected_counts = DVIUCI.dvi_loss_clamp_statistics(
        params, features, "heteroscedastic", loss_config,
    )
    observed_counts = tracker_record(loss_tracker)
    @test observed_counts.calls == expected_counts.calls
    @test observed_counts.elements == expected_counts.elements
    @test observed_counts.lower_clamps == expected_counts.lower_clamps
    @test observed_counts.upper_clamps == expected_counts.upper_clamps
end

@testset "heteroscedastic variance-head initialization" begin
    config = tiny_config(
        hidden_units = 5,
        log_variance_head_posterior_std = 0.05,
        log_variance_head_initial_bias = -0.25,
    )
    params = initialize_model(3, "heteroscedastic", config; seed = 17)
    @test params.output.weight_mu[2, :] == zeros(Float32, 5)
    @test all(
        params.output.weight_log_std[2, :] .≈ Float32(log(0.05)),
    )
    @test params.output.bias_mu[2] == -0.25f0
    @test params.output.bias_log_std[2] ≈ Float32(log(0.05))
end

@testset "extreme posterior scales remain finite" begin
    config = tiny_config(propagation = "diagonal", hidden_units = 4)
    params = initialize_model(2, "heteroscedastic", config; seed = 18)
    extreme_params = (
        hidden = merge(params.hidden, (
            weight_log_std = fill(60f0, size(params.hidden.weight_log_std)),
            bias_log_std = fill(-60f0, size(params.hidden.bias_log_std)),
        )),
        output = merge(params.output, (
            weight_log_std = fill(60f0, size(params.output.weight_log_std)),
            bias_log_std = fill(-60f0, size(params.output.bias_log_std)),
        )),
    )
    features = Float32[0.2 -0.3; 0.4 0.1]
    targets = Float32[0.1, -0.2]
    loss, gradient = Zygote.withgradient(extreme_params) do candidate
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
    @test DVIUCI.parameters_are_finite(gradient[1])
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

@testset "optional Reactant training parity" begin
    if lowercase(get(ENV, "DVI_TEST_REACTANT", "false")) == "true"
        features = randn(StableRNG(105), Float32, 12, 3)
        targets = Float32.(
            0.4 .* features[:, 1] .-
            0.2 .* features[:, 2] .+
            0.1 .* features[:, 3]
        )
        split = (
            x_train_standardized = features,
            y_train_standardized = targets,
        )
        reference_config = tiny_config(
            hidden_units = 4,
            batch_size = 4,
            max_epochs = 2,
            execution_backend = "zygote",
            execution_device = "cpu",
        )
        reactant_config = tiny_config(
            hidden_units = 4,
            batch_size = 4,
            max_epochs = 2,
            execution_backend = "reactant",
            execution_device = lowercase(get(
                ENV, "DVI_TEST_REACTANT_DEVICE", "cpu",
            )),
        )
        reference = refit_model(
            split, 3, "heteroscedastic", reference_config;
            seed = 106, epochs = 2,
        )
        observed = refit_model(
            split, 3, "heteroscedastic", reactant_config;
            seed = 106, epochs = 2,
        )
        @test observed.losses ≈ reference.losses rtol = 1e-4 atol = 1e-5
        for layer in (:hidden, :output), field in (
            :weight_mu, :weight_log_std, :bias_mu, :bias_log_std,
        )
            @test isapprox(
                getfield(getfield(observed.params, layer), field),
                getfield(getfield(reference.params, layer), field);
                rtol = 1e-4,
                atol = 1e-5,
            )
        end
        @test observed.numerical.clamp_count ==
            reference.numerical.clamp_count

        validation_params = initialize_model(
            3, "heteroscedastic", reference_config; seed = 107,
        )
        validation_backend = DVIUCI.initialize_training_backend(
            validation_params,
            features,
            targets,
            "heteroscedastic",
            reactant_config,
        )
        observed_tracker = NumericalTracker()
        observed_output = DVIUCI.training_backend_validation_output(
            validation_backend,
            features;
            tracker = observed_tracker,
        )
        reference_tracker = NumericalTracker()
        reference_output = propagate_dvi(
            validation_params,
            features,
            reference_config;
            tracker = reference_tracker,
        )
        @test observed_output.mean ≈
            reference_output.mean rtol = 2e-4 atol = 2e-5
        @test observed_output.covariance ≈
            reference_output.covariance rtol = 2e-4 atol = 2e-5
        @test tracker_record(observed_tracker) ==
            tracker_record(reference_tracker)
    else
        @test true
    end
end

@testset "non-finite loss aborts before an optimizer update" begin
    config = tiny_config(hidden_units = 4)
    params = initialize_model(2, "heteroscedastic", config; seed = 22)
    optimizer_state = Optimisers.setup(
        Optimisers.Adam(config.learning_rate), params,
    )
    features = Float32[0.1 -0.2]
    targets = Float32[Inf]
    error = try
        DVIUCI.gradient_step(
            params,
            optimizer_state,
            features,
            targets,
            "heteroscedastic",
            config,
            1,
            1,
            0;
            batch = 1,
            phase = "selection",
            tracker = NumericalTracker(),
        )
        nothing
    catch caught
        caught
    end
    @test error isa DVINumericalError
    @test error.kind == "non-finite DVI loss"
    @test error.diagnostics.phase == "selection"
    @test error.diagnostics.optimizer_step == 1
    @test DVIUCI.parameters_are_finite(params)
end


@testset "run rows are upserted by configuration identity" begin
    mktempdir() do output_dir
        config = tiny_config(output_dir = output_dir)
        first_row = (
            dataset = "yacht",
            split = 1,
            likelihood = "heteroscedastic",
            propagation = "diagonal",
            status = "failure",
        )
        replacement = merge(first_row, (status = "success",))
        runs = DVIUCI.append_run!(DataFrame(), first_row, config)
        runs = DVIUCI.append_run!(runs, replacement, config)
        @test nrow(runs) == 1
        @test only(runs.status) == "success"
    end
end

@testset "complete run validation" begin
    rows = DataFrame([
        (
            dataset = "yacht",
            split = split_id,
            likelihood = "heteroscedastic",
            propagation = "diagonal",
            status = "success",
        )
        for split_id in 1:2
    ])
    @test DVIUCI.validate_complete_runs(
        rows,
        ["yacht"],
        [1, 2],
        ["heteroscedastic"],
        "diagonal",
    ) === rows
    @test_throws ArgumentError DVIUCI.validate_complete_runs(
        rows[1:1, :],
        ["yacht"],
        [1, 2],
        ["heteroscedastic"],
        "diagonal",
    )
    failed = copy(rows)
    failed.status[2] = "failure"
    @test_throws ArgumentError DVIUCI.validate_complete_runs(
        failed,
        ["yacht"],
        [1, 2],
        ["heteroscedastic"],
        "diagonal",
    )
end

@testset "configuration stems accept CSV string subtypes" begin
    dataset = SubString("xconcrete", 2)
    likelihood = SubString("xheteroscedastic", 2)
    propagation = SubString("xdiagonal", 2)
    @test DVIUCI.configuration_stem(
        dataset,
        3,
        likelihood,
        propagation,
    ) == "concrete_split03_heteroscedastic_diagonal"
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
    history_directory = mktempdir()
    history_path = joinpath(history_directory, "history.csv")
    selection = train_with_validation(
        prepared,
        3,
        "heteroscedastic",
        config;
        seed = 32,
        history_path = history_path,
    )
    @test 1 <= selection.best_epoch <= config.max_epochs
    @test isfile(history_path)
    @test nrow(CSV.read(history_path, DataFrame)) == nrow(selection.history)
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
    propagated = propagate_dvi(
        refit.params, prepared.x_test_standardized, config,
    )
    metrics_from_output = DVIUCI.predictive_metrics_from_output(
        propagated,
        prepared.y_test_standardized,
        prepared.y_test,
        prepared.standardizer,
        "heteroscedastic",
        config,
    )
    @test isfinite(metrics.lpd_original)
    @test isfinite(metrics.rmse_original)
    @test metrics.mean_total_variance_original > 0
    @test metrics_from_output.lpd_original ≈ metrics.lpd_original
    @test metrics_from_output.rmse_original ≈ metrics.rmse_original

    budget_config = tiny_config(
        hidden_units = 6,
        initialization_scale = 0.1,
        max_epochs = 10,
        validation_every = 10,
        patience = 10,
        selection_max_optimizer_steps = 6,
        refit_max_optimizer_steps = 6,
        training_budget_protocol = "test-budget-v1",
    )
    budget_selection = train_with_validation(
        prepared,
        3,
        "heteroscedastic",
        budget_config;
        seed = 32,
    )
    @test budget_selection.budget_limited
    @test nrow(budget_selection.history) == 1
    @test budget_selection.optimizer_steps >= 6
    @test budget_selection.optimizer_steps <
        6 + cld(size(prepared.x_train_standardized, 1), 16)
    budget_refit = refit_model(
        prepared,
        3,
        "heteroscedastic",
        budget_config;
        seed = 32,
        epochs = 10,
    )
    @test budget_refit.budget_limited
    @test budget_refit.optimizer_steps >= 6
    @test budget_refit.optimizer_steps <
        6 + cld(size(prepared.x_train_standardized, 1), 16)
    @test length(budget_refit.losses) == 2
end

@testset "BBB-compatible paper tables" begin
    metric_values = (; (
        name => 1.0 for name in DVIUCI.DVI_SCALAR_METRIC_NAMES
    )...)
    runs = DataFrame([(
        dataset = "yacht",
        dataset_name = "Yacht",
        likelihood = "heteroscedastic",
        propagation = "full",
        status = "success",
        metric_values...,
    )])
    summary = DVIUCI.summary_table(runs)
    mktempdir() do output_dir
        paths = DVIUCI.write_tables(
            summary,
            tiny_config(output_dir = output_dir),
        )
        @test isfile(joinpath(output_dir, "summary.csv"))
        @test isfile(paths.markdown)
        @test isfile(paths.latex)
        @test occursin("LPD (standardized)", read(paths.markdown, String))
        @test occursin("\\begin{tabular}", read(paths.latex, String))
    end
end
