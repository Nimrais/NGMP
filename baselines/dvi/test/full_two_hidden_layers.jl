@testset "full two-hidden covariance and gradients" begin
    rng = StableRNG(607)
    weights = randn(rng, Float64, 4, 3)
    covariance = randn(rng, Float64, 5, 3, 3)
    # Deliberately nonsymmetric to detect transposed output indexing.
    expected = cat([reshape(weights * covariance[b, :, :] * weights', 1, 4, 4)
        for b in 1:5]...; dims = 1)
    @test DVIUCI.batch_quadratic_matmul(weights, covariance) ≈ expected
    objective(w, c) = sum(abs2, DVIUCI.batch_quadratic_matmul(w, c))
    reference(w, c) = sum(sum(abs2, w * c[b, :, :] * w') for b in axes(c, 1))
    @test all(isapprox.(Zygote.gradient(objective, weights, covariance),
        Zygote.gradient(reference, weights, covariance); rtol = 1e-10))
    config = tiny_config(hidden_layers = 2, hidden_units = 4, initialization_scale = 0.2)
    @test config.implementation_version == DVIUCI.DVI_FULL_TWO_HIDDEN_VERSION
    @test tiny_config(propagation = "diagonal", hidden_layers = 2).implementation_version ==
        DVIUCI.DVI_IMPLEMENTATION_VERSION
    params = initialize_model(3, "heteroscedastic", config; seed = 608)
    params.output.weight_mu[2, :] .= [0.1, -0.2, 0.3, 0.15]
    features, targets = randn(rng, Float32, 8, 3), randn(rng, Float32, 8)
    first_pre = DVIUCI.linear_certain_full(features, params.hidden, config)
    first_post = relu_moments_full(first_pre.mean, first_pre.covariance)
    second_pre = DVIUCI.linear_full(first_post, params.hidden2, config)
    second_post = relu_moments_full(second_pre.mean, second_pre.covariance)
    output = propagate_dvi(params, features, config)
    expected_output = DVIUCI.linear_full(second_post, params.output, config)
    @test output.mean ≈ expected_output.mean
    @test output.covariance ≈ expected_output.covariance
    @test maximum(abs, output.covariance[:, 1, 2]) > 1e-5
    @test output.covariance ≈ permutedims(output.covariance, (1, 3, 2)) rtol = 1e-5
    loss, gradient = Zygote.withgradient(p -> dvi_loss(p, features, targets,
        "heteroscedastic", config, 8), params)
    @test isfinite(loss)
    @test DVIUCI.parameters_are_finite(only(gradient))
    @test all(DVIUCI.gradient_maximum_absolute(layer) > 0 for layer in values(only(gradient)))
end

@testset "full two-hidden artifact round trip" begin
    mktempdir() do directory
        config = tiny_config(hidden_layers = 2, hidden_units = 4,
            initialization_scale = 0.1, output_dir = directory,
            save_checkpoints = true, max_epochs = 2)
        DVIUCI.ensure_output_directories(config)
        DVIUCI.save_config(config)
        rng = StableRNG(609)
        dataset = (key = "yacht", display_name = "Yacht", dataset_index = 1,
            features = randn(rng, 50, 3), targets = randn(rng, 50))
        row = run_configuration(dataset, 1, "heteroscedastic", config)
        runs = DVIUCI.append_run!(DataFrame(), row, config)
        DVIUCI.refresh_artifacts(runs, config)
        @test row.status == "success"
        @test occursin("DVI-2HL", row.method)
        @test !occursin("dDVI-2HL", row.method)
        checkpoint = JLD2.load(DVIUCI.posterior_checkpoint_path(config, "yacht", 1, "heteroscedastic"))
        partition = DVIUCI.outer_and_inner_splits(dataset, 1, config).outer
        replay = predictive_metrics(checkpoint["posterior_params"],
            partition.x_test_standardized, partition.y_test_standardized,
            partition.y_test, checkpoint["standardizer"], "heteroscedastic", config)
        @test replay.nll_original ≈ row.nll_original rtol = 1e-12
        @test DVIUCI.configuration_succeeded(runs, "yacht", 1, "heteroscedastic", "full", config)
        wrong = tiny_config(hidden_layers = 2, hidden_units = 4, propagation = "diagonal",
            output_dir = directory, save_checkpoints = true)
        @test_throws ArgumentError DVIUCI.save_config(wrong)
        @test !DVIUCI.configuration_succeeded(runs, "yacht", 1, "heteroscedastic", "diagonal", wrong)
    end
    @test length(FinalTableTests.table_sources()) == 9
    sources = FinalTableTests.table_sources(include_dvi_big = true)
    @test length(sources) == 10
    @test only(filter(s -> s.method == "DVI-2HL-Hetero", sources)).propagation == "full"
end
