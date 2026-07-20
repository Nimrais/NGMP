# Matched full-reshape versus fixed-basis low-rank ContinuousTransition run.
#
# Full promoted run:
#   OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_ctransition_mvsoftplus_low_rank_comparison.jl
#
# Fast development run (keeps hidden width 4):
#   N_SAMPLES=80 N_ITERATIONS=2 PREDICTION_ITERATIONS=1 \
#     OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_ctransition_mvsoftplus_low_rank_comparison.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Distributions
using LinearAlgebra: Diagonal, svd
using Statistics

include(joinpath(@__DIR__, "xor_ctransition_mvsoftplus.jl"))

"Return a dense rectangular diagonal matrix for constructing the residual SVD."
function low_rank_offset(output_dim, input_dim, scale)
    diagonal = fill(Float64(scale), min(output_dim, input_dim))
    matrix = zeros(Float64, output_dim, input_dim)
    @inbounds for index in eachindex(diagonal)
        matrix[index, index] = diagonal[index]
    end
    return diagonal, matrix
end

"Build a Frobenius-orthogonal rank-one dictionary from the full SVD bases."
function svd_low_rank_meta(
    reference_mean,
    parameter_count,
    offset_scale,
    prior_variance,
)
    output_dim, input_dim = size(reference_mean)
    matrix_parameter_count = output_dim * input_dim
    0 < parameter_count <= matrix_parameter_count || throw(ArgumentError(
        "the parameter count must be in 1:$matrix_parameter_count, got $parameter_count",
    ))

    a0_diagonal, A0 = low_rank_offset(output_dim, input_dim, offset_scale)
    decomposition = svd(reference_mean - A0; full = true)
    svd_count = min(parameter_count, length(decomposition.S))

    # Unit-Frobenius atoms would give E[||Delta A||_F^2] = r * Var(z).
    # Scaling U makes that energy equal to the full entrywise Gaussian prior's
    # output_dim * input_dim * Var(a), while retaining the same z variance.
    atom_scale = sqrt(matrix_parameter_count / parameter_count)
    U = zeros(Float64, output_dim, parameter_count)
    V = zeros(Float64, input_dim, parameter_count)
    z_mean = zeros(Float64, parameter_count)

    # The diagonal pairs reconstruct the truncated SVD. Off-diagonal pairs then
    # add new directions. Every p_i*q_j' is Frobenius-orthogonal to every other
    # selected atom, which avoids the ill-conditioned random overcomplete basis.
    pairs = [(index, index) for index in 1:min(output_dim, input_dim)]
    append!(pairs, [
        (output_index, input_index)
        for output_index in 1:output_dim for input_index in 1:input_dim
        if output_index != input_index
    ])
    for (index, (output_index, input_index)) in
        enumerate(Iterators.take(pairs, parameter_count))
        U[:, index] .= atom_scale .* decomposition.U[:, output_index]
        V[:, index] .= decomposition.V[:, input_index]
        if output_index == input_index && index <= svd_count
            z_mean[index] = decomposition.S[index] / atom_scale
        end
    end

    meta = LinearLowRankMeta(a0_diagonal, U, V)
    prior = MvNormalMeanCovariance(
        z_mean,
        prior_variance .* Diagonal(ones(parameter_count)),
    )

    reconstructed = A0 + U * Diagonal(z_mean) * V'
    reconstruction_error = maximum(abs, reconstructed - reference_mean)
    return (; meta, prior, rank = parameter_count, svd_count, reconstruction_error, atom_scale)
end

function comparison_base_priors(config, d_f = 3)
    return make_priors(
        d_h = config.d_hidden,
        d_f = d_f,
        seed = config.prior_seed,
        ct_precision_mean = config.ct_precision_mean,
        a_prior_mean_scale = config.a_prior_mean_scale,
        a_prior_variance = config.a_prior_variance,
        theta_prior_mean_scale = config.theta_prior_mean_scale,
        theta_prior_variance = config.theta_prior_variance,
        gamma_obs_mean = config.gamma_obs_mean,
        gamma_obs_concentration = config.gamma_obs_concentration,
    )
end

function low_rank_setup(config; map_parameters, pred_parameters, offset_scale, d_f = 3)
    d_h = config.d_hidden
    priors = comparison_base_priors(config, d_f)
    reference_map = reshape(mean(priors[:a_map]), d_h, d_f)
    reference_pred = reshape(mean(priors[:a_pred]), d_h, d_h)

    map = svd_low_rank_meta(
        reference_map,
        map_parameters,
        offset_scale,
        config.a_prior_variance,
    )
    pred = svd_low_rank_meta(
        reference_pred,
        pred_parameters,
        offset_scale,
        config.a_prior_variance,
    )
    priors[:a_map] = map.prior
    priors[:a_pred] = pred.prior
    return (; priors, meta_map = map.meta, meta_pred = pred.meta, map, pred)
end

function run_comparison_arm(config, label; priors = nothing, meta_map = nothing, meta_pred = nothing)
    result = nothing
    metrics = nothing
    wall_elapsed = @elapsed begin
        result, metrics = run_experiment(
            config;
            priors = priors,
            meta_map = meta_map,
            meta_pred = meta_pred,
            experiment_label = label,
            compact_posteriors = true,
        )
    end
    result = nothing
    GC.gc()
    return merge(metrics, (; wall_elapsed))
end

function print_comparison(baseline, low_rank, full_parameter_count, low_rank_parameter_count)
    training_speedup = baseline.training_elapsed / low_rank.training_elapsed
    prediction_speedup = baseline.prediction_elapsed / low_rank.prediction_elapsed
    wall_speedup = baseline.wall_elapsed / low_rank.wall_elapsed
    println()
    println("=== matched comparison ===")
    println("arm,gaussian_parameters,train_s,predict_s,wall_s,train_mse,test_mse,normalized_test_mse")
    println(join((
        "reshape",
        full_parameter_count,
        baseline.training_elapsed,
        baseline.prediction_elapsed,
        baseline.wall_elapsed,
        baseline.train_mse,
        baseline.test_mse,
        baseline.test_mse / baseline.baseline,
    ), ','))
    println(join((
        "low_rank",
        low_rank_parameter_count,
        low_rank.training_elapsed,
        low_rank.prediction_elapsed,
        low_rank.wall_elapsed,
        low_rank.train_mse,
        low_rank.test_mse,
        low_rank.test_mse / low_rank.baseline,
    ), ','))
    println("parameter reduction     : ", full_parameter_count / low_rank_parameter_count, "x")
    println("training speedup        : ", round(training_speedup, digits = 3), "x")
    println("prediction speedup      : ", round(prediction_speedup, digits = 3), "x")
    println("end-to-end speedup      : ", round(wall_speedup, digits = 3), "x")
end

function run_low_rank_comparison(config = CONFIG)
    config = merge(config, (;
        save_outputs = env_bool("COMPARISON_SAVE_OUTPUTS", false),
        show_progress = false,
    ))
    d_h, d_f = config.d_hidden, 3
    default_budget = min(d_h, d_f) + d_h
    parameter_budget = env_int("LOW_RANK_PARAMETER_BUDGET", default_budget)
    map_parameters = env_int(
        "LOW_RANK_MAP_PARAMETERS",
        clamp(round(Int, 3parameter_budget / 7), 1, parameter_budget - 1),
    )
    pred_parameters = parameter_budget - map_parameters
    offset_scale = env_float("LOW_RANK_A0_SCALE", 1.0)
    setup = low_rank_setup(config; map_parameters, pred_parameters, offset_scale, d_f)
    run_reshape = env_bool("RUN_RESHAPE", true)

    println("Low-rank setup: A = A0 + U*Diag(z)*V'")
    println("parameter budget map/pred: ", parameter_budget, " = ", setup.map.rank, " + ", setup.pred.rank)
    println("SVD-seeded atoms        : ", setup.map.svd_count, " / ", setup.pred.svd_count)
    println("A0 diagonal scale       : ", offset_scale)
    println("prior-mean max errors   : ", setup.map.reconstruction_error, " / ", setup.pred.reconstruction_error)
    println("map U/V sizes           : ", size(setup.meta_map.U), " / ", size(setup.meta_map.V))
    println("pred U/V sizes          : ", size(setup.meta_pred.U), " / ", size(setup.meta_pred.V))

    if env_bool("COMPARISON_WARMUP", true)
        warmup_config = merge(config, (;
            n_samples = min(config.n_samples, 24),
            train_fraction = 0.75,
            iterations = 1,
            prediction_iterations = 1,
            prediction_batch_size = 24,
        ))
        println("Warming selected rule paths before measurement...")
        if run_reshape
            run_comparison_arm(warmup_config, "reshape warmup")
        end
        run_comparison_arm(
            warmup_config,
            "fixed-basis low-rank warmup";
            priors = setup.priors,
            meta_map = setup.meta_map,
            meta_pred = setup.meta_pred,
        )
    end

    baseline = run_reshape ? run_comparison_arm(config, "reshape baseline") : nothing
    low_rank = run_comparison_arm(
        config,
        "fixed-basis low rank";
        priors = setup.priors,
        meta_map = setup.meta_map,
        meta_pred = setup.meta_pred,
    )

    full_parameter_count = d_h * d_f + d_h * d_h
    low_rank_parameter_count = setup.map.rank + setup.pred.rank
    if run_reshape
        print_comparison(baseline, low_rank, full_parameter_count, low_rank_parameter_count)
    else
        println()
        println("=== low-rank result ===")
        println("gaussian_parameters,train_s,predict_s,wall_s,train_mse,test_mse,normalized_test_mse")
        println(join((
            low_rank_parameter_count,
            low_rank.training_elapsed,
            low_rank.prediction_elapsed,
            low_rank.wall_elapsed,
            low_rank.train_mse,
            low_rank.test_mse,
            low_rank.test_mse / low_rank.baseline,
        ), ','))
    end
    return (; baseline, low_rank, setup)
end

if abspath(PROGRAM_FILE) == @__FILE__
    comparison = run_low_rank_comparison(CONFIG)
end
