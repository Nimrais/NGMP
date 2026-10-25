using JLD2
using TOML

@testset "two-hidden-layer configuration and legacy initialization" begin
    @test_throws ArgumentError validate_config(tiny_config(hidden_layers = 2))
    @test_throws ArgumentError validate_config(tiny_config(hidden_layers = 3))
    config = tiny_config(propagation = "diagonal", hidden_layers = 2)
    @test validate_config(config) === config
    for d in (4, 6, 8, 11, 13)
        params = initialize_model(d, "heteroscedastic", config; seed = 101)
        @test size(params.hidden.weight_mu) == (50, d)
        @test size(params.hidden2.weight_mu) == (50, 50)
        @test size(params.output.weight_mu) == (2, 50)
        count = sum(length(layer.weight_mu) + length(layer.bias_mu)
            for layer in values(params))
        @test count == 50d + 2702 == DVIUCI.model_coefficient_count(d, "heteroscedastic", config)
        @test DVIUCI.model_architecture_matches(params, d, "heteroscedastic", config)
        malformed = merge(params, (hidden2 = merge(params.hidden2,
            (weight_mu = zeros(Float32, 49, 50),)),))
        @test !DVIUCI.model_architecture_matches(malformed, d, "heteroscedastic", config)
    end
    legacy_config = tiny_config()
    rng = StableRNG(101)
    hidden = DVIUCI.initialize_dvi_layer(rng, 3, 50, legacy_config)
    output = DVIUCI.initialize_dvi_layer(rng, 50, 1, legacy_config)
    legacy = initialize_model(3, "homoscedastic", legacy_config; seed = 101)
    @test legacy == (hidden = hidden, output = output)
    @test_throws ArgumentError propagate_dvi(legacy, zeros(Float32, 2, 3), config)
end

@testset "second-layer independent moments, KL and gradients" begin
    config = tiny_config(propagation = "diagonal", hidden_layers = 2,
        hidden_units = 4, initialization_scale = 0.1)
    rng = StableRNG(222)
    params = initialize_model(3, "heteroscedastic", config; seed = 223)
    features = randn(rng, Float32, 8, 3)
    targets = randn(rng, Float32, 8)
    first_pre = DVIUCI.linear_certain_diagonal(features, params.hidden, config)
    first_post = relu_moments_diagonal(first_pre.mean, first_pre.variance)
    layer = params.hidden2
    expected_mean = zeros(Float32, 8, 4)
    expected_var = zeros(Float32, 8, 4)
    for i in 1:8, j in 1:4
        expected_mean[i,j] = layer.bias_mu[j] + sum(
            first_post.mean[i,k] * layer.weight_mu[j,k] for k in 1:4)
        expected_var[i,j] = exp(2layer.bias_log_std[j]) + sum(
            (first_post.variance[i,k] + first_post.mean[i,k]^2) * exp(2layer.weight_log_std[j,k]) +
            first_post.variance[i,k] * layer.weight_mu[j,k]^2 for k in 1:4)
    end
    observed = DVIUCI.linear_diagonal(first_post, layer, config)
    @test observed.mean ≈ expected_mean
    @test observed.variance ≈ expected_var
    second_post = relu_moments_diagonal(expected_mean, expected_var)
    expected_output = DVIUCI.linear_diagonal(second_post, params.output, config)
    actual = propagate_dvi(params, features, config)
    @test actual.mean ≈ expected_output.mean
    @test actual.variance ≈ expected_output.variance
    expected_kl = sum(DVIUCI.empirical_bayes_layer_kl(l, config.eb_alpha, config.eb_beta, config)
        for l in values(params))
    @test empirical_bayes_kl(params, config) ≈ expected_kl
    loss, gradient = Zygote.withgradient(p -> dvi_loss(p, features, targets,
        "heteroscedastic", config, 8), params)
    @test isfinite(loss)
    @test DVIUCI.parameters_are_finite(only(gradient))
    for name in keys(params)
        @test DVIUCI.gradient_maximum_absolute(getproperty(only(gradient), name)) > 0
    end
    _, updated = Optimisers.update(Optimisers.setup(Optimisers.Adam(3e-4), params), params, only(gradient))
    @test updated.hidden2.weight_mu != params.hidden2.weight_mu
    @test DVIUCI.parameter_variance_clamp_statistics(params, config).elements ==
        sum(length(l.weight_mu) + length(l.bias_mu) for l in values(params))
end

@testset "two-layer saved artifacts, checkpoint replay and resume" begin
    mktempdir() do directory
        config = tiny_config(propagation = "diagonal", hidden_layers = 2,
            hidden_units = 4, initialization_scale = 0.1,
            output_dir = directory, save_checkpoints = true, max_epochs = 2)
        DVIUCI.ensure_output_directories(config)
        DVIUCI.save_config(config)
        rng = StableRNG(224)
        dataset = (key = "yacht", display_name = "Yacht", dataset_index = 1,
            features = randn(rng, 90, 3), targets = randn(rng, 90))
        row = run_configuration(dataset, 1, "heteroscedastic", config)
        @test row.status == "success"
        @test row.hidden_layers == 2
        runs = DVIUCI.append_run!(DataFrame(), row, config)
        DVIUCI.refresh_artifacts(runs, config)
        path = DVIUCI.posterior_checkpoint_path(config, "yacht", 1, "heteroscedastic")
        checkpoint = JLD2.load(path)
        @test checkpoint["prediction_config"].hidden_layers == 2
        @test length(checkpoint["refit_losses"]) == row.best_epoch
        prepared = DVIUCI.outer_and_inner_splits(dataset, 1, config).outer
        replay = predictive_metrics(checkpoint["posterior_params"],
            prepared.x_test_standardized, prepared.y_test_standardized, prepared.y_test,
            checkpoint["standardizer"], "heteroscedastic", config)
        @test replay.lpd_original ≈ row.lpd_original rtol = 1e-12
        @test replay.rmse_original ≈ row.rmse_original rtol = 1e-12
        @test replay.mean_total_variance_original > 0
        @test DVIUCI.configuration_succeeded(runs, "yacht", 1, "heteroscedastic", "diagonal", config)
        wrong = tiny_config(propagation = "diagonal", hidden_units = 4,
            output_dir = directory, save_checkpoints = true)
        @test_throws ArgumentError DVIUCI.save_config(wrong)
        @test !DVIUCI.configuration_succeeded(runs, "yacht", 1, "heteroscedastic", "diagonal", wrong)
        for name in ("runs.csv", "summary.csv", "table.md", "table.tex", "config.toml")
            @test isfile(joinpath(directory, name))
        end
        @test only(CSV.read(joinpath(directory, "summary.csv"), DataFrame).hidden_layers) == 2
        @test occursin("dDVI-2HL", read(joinpath(directory, "table.md"), String))
        rm(path)
        @test !DVIUCI.configuration_succeeded(runs, "yacht", 1, "heteroscedastic", "diagonal", config)
    end
end

module FinalTableTests
include(joinpath(@__DIR__, "..", "..", "..", "paper_materials", "make_final_baseline_table.jl"))
end

@testset "final table new source and architecture validation" begin
    source = only(filter(s -> s.method == "dDVI-2HL-Hetero", FinalTableTests.SOURCES))
    @test source.hidden_layers == 2
    @test length(FinalTableTests.SOURCES) == 9
    @test any(s -> s.method == "dDVI-Hetero", FinalTableTests.SOURCES)
    mktempdir() do directory
        candidate = FinalTableTests.ResultSource("dDVI-2HL-Hetero", "heteroscedastic",
            "diagonal", directory; hidden_layers = 2)
        open(joinpath(directory, "config.toml"), "w") do io
            TOML.print(io, Dict("hidden_layers" => 2, "hidden_units" => 50,
                "propagation" => "diagonal", "likelihoods" => ["heteroscedastic"]))
        end
        rows = DataFrame(dataset = ["yacht"], hidden_layers = [2],
            hidden_units = [50], model_coefficients = [3002])
        @test FinalTableTests.validate_source_architecture(candidate, rows, rows) === nothing
        rows.hidden_layers .= 1
        @test_throws ErrorException FinalTableTests.validate_source_architecture(candidate, rows, rows)
        @test_throws ErrorException FinalTableTests.load_results([candidate])
    end
end
