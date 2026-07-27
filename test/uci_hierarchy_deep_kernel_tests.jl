using StableRNGs

include(joinpath(@__DIR__, "..", "scripts", "uci_hierarchy_deep_kernel.jl"))

function tiny_uci_config()
    base = withenv(
        "UCI_DATASETS" => "yacht",
        "UCI_DEPTHS" => "1-3",
        "UCI_CONSTRAINTS" => "factorized",
        "UCI_ARCHITECTURES" => "shared,separate",
    ) do
        load_config()
    end
    return replace_config(
        base;
        n_splits = 1,
        mean_rffs = 4,
        noise_rffs = 3,
        fit_iterations = 2,
        prediction_iterations = 2,
        output_dir = mktempdir(),
        show_progress = false,
    )
end

@testset "UCI hierarchy preprocessing and features" begin
    rng = StableRNG(91)
    features = randn(rng, 30, 3)
    targets = collect(range(-2.0, 3.0; length = 30))
    first_split = deterministic_split(features, targets; seed = 17)
    second_split = deterministic_split(features, targets; seed = 17)

    @test first_split.train_indices == second_split.train_indices
    @test first_split.test_indices == second_split.test_indices
    @test isempty(intersect(
        first_split.train_indices, first_split.test_indices,
    ))

    prepared = standardize_from_training(first_split)
    @test vec(mean(
        prepared.x_train_standardized; dims = 1,
    )) ≈ zeros(3) atol = 1e-12
    @test std(prepared.y_train_standardized) ≈ 1.0

    lengthscale = median_distance_lengthscale(
        prepared.x_train_standardized; seed = 19,
    )
    map1 = build_rff_map(3, 7, lengthscale, 23; include_linear = true)
    map2 = build_rff_map(3, 7, lengthscale, 23; include_linear = true)
    design1 = map1(prepared.x_train_standardized)
    design2 = map2(prepared.x_train_standardized)
    @test design1 == design2
    @test size(design1) == (length(first_split.train_indices), 11)
    @test all(design1[:, end] .== 1.0)
end

@testset "UCI hierarchy direct RxInfer predictive variance" begin
    config = tiny_uci_config()
    rng = StableRNG(29)
    mean_design = hcat(randn(rng, 12, 3), ones(12))
    targets = 0.4 .* mean_design[:, 1] .- 0.2 .* mean_design[:, 2] .+
        0.1 .* randn(rng, 12)
    test_design = hcat(randn(rng, 4, 3), ones(4))

    fit = fit_layers(
        2,
        mean_design,
        mean_design,
        targets,
        config,
        "factorized",
    )
    prediction = validate_prediction(predict_layers(
        fit, test_design, test_design, config,
    ))

    @test prediction.predictive_variance_source == "rxinfer_q_y"
    @test prediction.total_variance == prediction.rxinfer_predictive_variance
    @test maximum(prediction.epistemic_variance) > 0
    @test any(abs.(
        prediction.total_variance .-
        (prediction.rxinfer_predictive_variance .+
         prediction.epistemic_variance)
    ) .> 1e-10)

    metrics = gaussian_metrics(
        targets[1:4],
        prediction.mean,
        prediction.total_variance,
    )
    @test isfinite(metrics.mean_log_predictive_density)
    @test -metrics.mean_log_predictive_density > 0
end

@testset "UCI hierarchy model enumeration" begin
    specs = model_specs(
        [1, 2, 3], ["factorized", "coupled"], ["shared", "separate"],
    )
    @test length(specs) == 9
    @test count(spec -> spec.depth == 1, specs) == 1
    @test count(spec -> spec.constraint_variant == "coupled", specs) == 4
end
