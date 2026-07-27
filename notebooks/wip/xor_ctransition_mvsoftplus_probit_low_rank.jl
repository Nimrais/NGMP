# Width-4 Probit experiment with a fixed 14-coefficient transition dictionary.
# It uses the promoted score-precision setup and final seed 1 from
# xor_ctransition_mvsoftplus_probit_tuning.jl.
#
#   OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_ctransition_mvsoftplus_probit_low_rank.jl

ENV["SAVE_OUTPUTS"] = "false"
ENV["SHOW_PROGRESS"] = "false"
ENV["REQUIRE_CLEAN_BASELINE"] = "false"

include(joinpath(@__DIR__, "xor_ctransition_mvsoftplus_probit.jl"))

using LinearAlgebra: Diagonal, svd

function probit_low_rank_offset(output_dim, input_dim, scale)
    diagonal = fill(Float64(scale), min(output_dim, input_dim))
    matrix = zeros(Float64, output_dim, input_dim)
    @inbounds for index in eachindex(diagonal)
        matrix[index, index] = diagonal[index]
    end
    return diagonal, matrix
end

function probit_orthogonal_low_rank_meta(
    reference_mean,
    parameter_count,
    offset_scale,
    prior_variance;
    pair_seed = 0,
    atom_scale_multiplier = 1.0,
)
    output_dim, input_dim = size(reference_mean)
    full_parameter_count = output_dim * input_dim
    0 < parameter_count <= full_parameter_count || throw(ArgumentError(
        "parameter_count must be in 1:$full_parameter_count",
    ))

    a0_diagonal, A0 = probit_low_rank_offset(output_dim, input_dim, offset_scale)
    decomposition = svd(reference_mean - A0; full = true)
    svd_count = min(parameter_count, length(decomposition.S))
    atom_scale_multiplier > 0 || throw(ArgumentError(
        "atom_scale_multiplier must be positive",
    ))
    atom_scale = atom_scale_multiplier * sqrt(full_parameter_count / parameter_count)

    pairs = [(index, index) for index in 1:min(output_dim, input_dim)]
    cross_pairs = [
        (output_index, input_index)
        for output_index in 1:output_dim for input_index in 1:input_dim
        if output_index != input_index
    ]
    pair_seed == 0 || Random.shuffle!(StableRNG(pair_seed), cross_pairs)
    append!(pairs, cross_pairs)

    U = zeros(Float64, output_dim, parameter_count)
    V = zeros(Float64, input_dim, parameter_count)
    coefficient_mean = zeros(Float64, parameter_count)
    for (index, (output_index, input_index)) in
        enumerate(Iterators.take(pairs, parameter_count))
        U[:, index] .= atom_scale .* decomposition.U[:, output_index]
        V[:, index] .= decomposition.V[:, input_index]
        if output_index == input_index && index <= svd_count
            coefficient_mean[index] = decomposition.S[index] / atom_scale
        end
    end

    meta = LinearLowRankMeta(a0_diagonal, U, V)
    prior = MvNormalMeanCovariance(
        coefficient_mean,
        prior_variance .* Diagonal(ones(parameter_count)),
    )
    reconstructed = A0 + U * Diagonal(coefficient_mean) * V'
    return (
        meta = meta,
        prior = prior,
        parameter_count = parameter_count,
        svd_count = svd_count,
        reconstruction_error = maximum(abs, reconstructed - reference_mean),
    )
end

function make_probit_low_rank_setup(
    config;
    map_parameters = 6,
    pred_parameters = 8,
    offset_scale = 1.0,
    basis_seed = 0,
    atom_scale_multiplier = 1.0,
    d_f = 3,
)
    priors = make_priors(config; d_f = d_f)
    reference_map = reshape(mean(priors[:a_map]), config.d_hidden, d_f)
    reference_pred = reshape(
        mean(priors[:a_pred]), config.d_hidden, config.d_hidden
    )
    map = probit_orthogonal_low_rank_meta(
        reference_map,
        map_parameters,
        offset_scale,
        config.a_prior_variance;
        pair_seed = basis_seed == 0 ? 0 : basis_seed + 101,
        atom_scale_multiplier = atom_scale_multiplier,
    )
    pred = probit_orthogonal_low_rank_meta(
        reference_pred,
        pred_parameters,
        offset_scale,
        config.a_prior_variance;
        pair_seed = basis_seed == 0 ? 0 : basis_seed + 202,
        atom_scale_multiplier = atom_scale_multiplier,
    )
    priors[:a_map] = map.prior
    priors[:a_pred] = pred.prior
    return (; priors, meta_map = map.meta, meta_pred = pred.meta, map, pred)
end

function run_probit_low_rank_clean(config = CONFIG)
    output_prefix = get(
        ENV,
        "OUTPUT_PREFIX",
        joinpath(
            @__DIR__,
            "..",
            "viz",
            "xor_ctransition_mvsoftplus_probit_low_rank_14",
        ),
    )
    config = merge(config, (;
        n_samples = env_int("N_SAMPLES", 2_000),
        iterations = env_int("N_ITERATIONS", 160),
        prediction_iterations = env_int("PREDICTION_ITERATIONS", 10),
        score_precision_mean = env_float("SCORE_PRECISION_MEAN", 10.0),
        score_precision_concentration = env_float("SCORE_PRECISION_CONCENTRATION", 10.0),
        data_seed = env_int("DATA_SEED", 2_030),
        split_seed = env_int("SPLIT_SEED", 2_031),
        prior_seed = env_int("PRIOR_SEED", 44),
        save_outputs = true,
        output_prefix = output_prefix,
        show_progress = false,
        require_clean_baseline = false,
    ))
    validate_config(config)

    map_parameters = env_int("LOW_RANK_MAP_PARAMETERS", 6)
    pred_parameters = env_int("LOW_RANK_PRED_PARAMETERS", 8)
    basis_seed = env_int("LOW_RANK_BASIS_SEED", 0)
    atom_scale_multiplier = env_float("LOW_RANK_ATOM_SCALE_MULTIPLIER", 1.0)
    setup = make_probit_low_rank_setup(
        config;
        map_parameters,
        pred_parameters,
        basis_seed,
        atom_scale_multiplier,
    )
    println(
        "Low-rank Probit setup: ",
        map_parameters + pred_parameters,
        " = ",
        map_parameters,
        " map + ",
        pred_parameters,
        " predictor coefficients",
    )
    println("basis seed / atom scale multiplier: ", basis_seed, " / ", atom_scale_multiplier)
    println(
        "prior-mean max errors: ",
        setup.map.reconstruction_error,
        " / ",
        setup.pred.reconstruction_error,
    )

    dataset = make_clean_xor_dataset(n = config.n_samples, seed = config.data_seed)
    train_data, test_data = split_dataset(
        dataset;
        train_fraction = config.train_fraction,
        seed = config.split_seed,
    )
    train_features = build_features(train_data)
    test_features = build_features(test_data)

    fit = fit_probit_arm(
        config,
        train_features,
        Float64.(train_data.label);
        priors = setup.priors,
        meta_map = setup.meta_map,
        meta_pred = setup.meta_pred,
    )

    prediction_elapsed = @elapsed test_statistics = score_statistics(
        predict_score_marginals(fit, test_features; config = config)
    )
    labels = Float64.(test_data.label)
    metrics = classification_metrics(labels, test_statistics.probability)
    brier = mean(abs2, test_statistics.probability .- labels)
    baseline = class_prior_baseline(Float64.(train_data.label), labels)

    grid = make_prediction_grid(config.grid_size)
    grid_statistics = reshape_grid_statistics(score_statistics(
        predict_score_marginals(fit, grid.features; config = config);
        verify_native = false,
    ), grid)
    score_mean_limits = nondegenerate_limits(
        minimum(grid_statistics.score_mean),
        maximum(grid_statistics.score_mean),
    )
    score_variance_limits = nondegenerate_limits(
        minimum(grid_statistics.score_variance),
        maximum(grid_statistics.score_variance),
    )
    surface_path = save_five_panel_surface(
        :clean,
        grid,
        grid_statistics,
        config.output_prefix;
        score_mean_limits = score_mean_limits,
        score_variance_limits = score_variance_limits,
    )

    println()
    println("=== width-4 / 14-coefficient CT-MvSoftplus-Probit ===")
    println("accuracy / NLL / Brier : ", metrics.accuracy, " / ", metrics.nll, " / ", brier)
    println("baseline accuracy / NLL: ", baseline.accuracy, " / ", baseline.nll)
    println("training / prediction  : ", fit.elapsed, "s / ", prediction_elapsed, "s")
    println(
        "test latent variance   : ",
        minimum(test_statistics.score_variance),
        " / ",
        mean(test_statistics.score_variance),
        " / ",
        maximum(test_statistics.score_variance),
    )
    println("surface                 : ", surface_path)

    return (;
        config,
        setup,
        fit,
        test_statistics,
        metrics,
        brier,
        baseline,
        prediction_elapsed,
        surface_path,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    low_rank_probit_report = run_probit_low_rank_clean(CONFIG)
end
