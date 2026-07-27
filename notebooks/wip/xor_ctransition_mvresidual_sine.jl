# XOR (2x2 checkerboard) with a ContinuousTransition / MvResidualSine sandwich:
#
#   x_f -> ContinuousTransition(a_map, P) -> h1 -> MvResidualSine -> s
#       -> ContinuousTransition(a_pred, Gamma2) -> h2
#       -> softdot(theta, gamma_obs) -> y
#
# Two forward-message arms share the exact same graph and exact analytic
# backward Gaussian Fisher projection:
#
#   PHI_FORWARD_MODE=delta
#       implicit-function delta tangent projection of the exact pushforward
#       log-density at mean(q(s));
#
#   PHI_FORWARD_MODE=moments
#       exact analytic pushforward mean/covariance followed by a Gaussian
#       assumed-density replacement (no sigma points).
#
# Direct smoke comparison:
#   XOR_RS_SMOKE=true SAVE_OUTPUTS=false OPENBLAS_NUM_THREADS=1 \
#     julia --project=. experiments/xor_ctransition_mvresidual_sine.jl
#
# Full width-4 comparison:
#   OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_ctransition_mvresidual_sine.jl
#
# Width-14 fixed-basis low-rank comparison:
#   D_HIDDEN=14 USE_LOW_RANK=true OPENBLAS_NUM_THREADS=1 julia --project=. \
#     experiments/xor_ctransition_mvresidual_sine.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# This includes the original experiment and its low-rank construction helpers.
# Neither included file executes its main block because PROGRAM_FILE remains
# this residual-sine entry point.
include(joinpath(@__DIR__, "xor_ctransition_mvsoftplus_low_rank_comparison.jl"))

using DataFrames
using Distributions
using ExponentialFamily
using LinearAlgebra: Diagonal, Symmetric
using RxInfer
using Statistics

import BayesBase: mean_cov
import ClosedFormExpectations: Logpdf
import ExponentialFamily: MvNormalWeightedMeanPrecision, weightedmean_precision

const RESIDUAL_SINE_SMOKE = env_bool("XOR_RS_SMOKE")

function residual_sine_config()
    config = if RESIDUAL_SINE_SMOKE
        merge(CONFIG, (;
            n_samples = env_int("N_SAMPLES", 60),
            d_hidden = env_int("D_HIDDEN", 2),
            iterations = env_int("N_ITERATIONS", 3),
            prediction_iterations = env_int("PREDICTION_ITERATIONS", 3),
            prediction_batch_size = env_int("PREDICTION_BATCH_SIZE", 64),
            grid_size = env_int("GRID_SIZE", 16),
            save_outputs = env_bool("SAVE_OUTPUTS", false),
            show_progress = env_bool("SHOW_PROGRESS", false),
        ))
    else
        CONFIG
    end
    return merge(config, (;
        phi_rho = env_float("PHI_RHO", 0.9),
        phi_omega = env_float("PHI_OMEGA", 1.0),
        run_softplus_baseline = env_bool("RUN_SOFTPLUS_BASELINE", true),
        use_low_rank = env_bool("USE_LOW_RANK", false),
        output_prefix = get(
            ENV,
            "OUTPUT_PREFIX",
            joinpath(@__DIR__, "..", "viz", "xor_ctransition_mvresidual_sine"),
        ),
    ))
end

const RESIDUAL_SINE_CONFIG = residual_sine_config()

function residual_sine_forward_modes()
    raw = if haskey(ENV, "PHI_FORWARD_MODE")
        ENV["PHI_FORWARD_MODE"]
    else
        get(ENV, "PHI_FORWARD_MODES", "delta,moments")
    end
    modes = Symbol.(strip.(split(lowercase(raw), ',')))
    isempty(modes) && throw(ArgumentError("at least one PHI_FORWARD_MODE is required"))
    all(mode -> mode in (:delta, :moments), modes) || throw(ArgumentError(
        "PHI_FORWARD_MODE(S) must contain only delta and/or moments",
    ))
    return unique(modes)
end

@model function xor_ct_mvresidual_sine(
    y,
    features,
    priors,
    feature_cov,
    meta_map,
    meta_pred,
    ct_a_deps,
    ct2_deps,
    activation_deps,
    activation,
)
    a_map ~ priors[:a_map]
    a_pred ~ priors[:a_pred]
    theta ~ priors[:theta]
    P ~ priors[:P]
    Gamma2 ~ priors[:Gamma2]
    gamma_obs ~ priors[:gamma_obs]
    for index in eachindex(y)
        x_f[index] ~ MvNormalMeanCovariance(features[index], feature_cov)
        h1[index] ~ ContinuousTransition(x_f[index], a_map, P) where {
            dependencies = ct_a_deps,
            meta = meta_map
        }
        s[index] ~ MvResidualSine(h1[index]) where {
            dependencies = activation_deps,
            meta = activation
        }
        h2[index] ~ ContinuousTransition(s[index], a_pred, Gamma2) where {
            dependencies = ct2_deps,
            meta = meta_pred
        }
        y[index] ~ softdot(theta, h2[index], gamma_obs)
    end
end

@model function xor_ct_mvresidual_sine_prediction(
    features,
    priors,
    feature_cov,
    meta_map,
    meta_pred,
    activation_deps,
    activation,
    y_prior_variance,
)
    local x_f, h1, s, h2, y

    a_map ~ priors[:a_map]
    a_pred ~ priors[:a_pred]
    theta ~ priors[:theta]
    P ~ priors[:P]
    Gamma2 ~ priors[:Gamma2]
    gamma_obs ~ priors[:gamma_obs]
    for index in eachindex(features)
        x_f[index] ~ MvNormalMeanCovariance(features[index], feature_cov)
        h1[index] ~ ContinuousTransition(x_f[index], a_map, P) where {meta = meta_map}
        s[index] ~ MvResidualSine(h1[index]) where {
            dependencies = activation_deps,
            meta = activation
        }
        h2[index] ~ ContinuousTransition(s[index], a_pred, Gamma2) where {meta = meta_pred}
        y[index] ~ softdot(theta, h2[index], gamma_obs)
        y[index] ~ NormalMeanVariance(0.0, y_prior_variance)
    end
end

function make_residual_sine_activation(config)
    return ResidualSineMeta(
        rho = config.phi_rho,
        omega = config.phi_omega,
    )
end

residual_sine_forward_projection(::Val{:delta}) =
    TangentProjection(type = DeltaApproximation)
residual_sine_forward_projection(::Val{:moments}) =
    TangentProjection(type = ClosedForm)
function residual_sine_forward_projection(::Val{mode}) where {mode}
    throw(ArgumentError("unsupported residual-sine forward mode: $mode"))
end

function make_residual_sine_dependencies(config, forward_projection)
    return NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = forward_projection,
        damping = DampingMeta(
            alpha = config.ngmp_alpha,
            beta = config.ngmp_beta,
            max_step = config.ngmp_max_step,
        ),
    )
end

function make_residual_sine_initial(config, hidden_dimension, activation)
    initial_mean, initial_covariance = SurrogateModelling._mv_residual_sine_mean_cov(
        zeros(hidden_dimension),
        Matrix(Diagonal(ones(hidden_dimension))),
        activation,
    )
    return MvNormalMeanCovariance(initial_mean, initial_covariance)
end

function run_residual_sine_prediction_batch(
    priors,
    features;
    config,
    activation,
    hidden_dimension,
    feature_dimension,
    output_mean,
    meta_map,
    meta_pred,
    forward_projection,
)
    isempty(features) && return Any[]
    activation_dependencies =
        make_residual_sine_dependencies(config, forward_projection)
    model = xor_ct_mvresidual_sine_prediction(
        priors = priors,
        feature_cov = Matrix(Diagonal(fill(config.feature_jitter, feature_dimension))),
        meta_map = meta_map,
        meta_pred = meta_pred,
        activation_deps = activation_dependencies,
        activation = activation,
        y_prior_variance = config.prediction_prior_variance,
    )
    result = infer(
        model = model,
        data = (features = features,),
        constraints = xor_ct_prediction_constraints(priors),
        initialization = make_prediction_initialization(
            priors,
            hidden_dimension,
            output_mean,
            config.prediction_prior_variance,
            make_residual_sine_initial(config, hidden_dimension, activation),
        ),
        iterations = config.prediction_iterations,
        free_energy = false,
        showprogress = false,
        returnvars = (y = KeepLast(),),
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    marginals = collect(vec(result.posteriors[:y]))
    length(marginals) == length(features) ||
        error("prediction graph returned the wrong number of y marginals")
    return marginals
end

function predict_residual_sine_marginals(
    priors,
    features;
    batch_size,
    kwargs...,
)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    marginals = Vector{Any}(undef, length(features))
    for first_index in 1:batch_size:length(features)
        indices = first_index:min(first_index + batch_size - 1, length(features))
        marginals[indices] = run_residual_sine_prediction_batch(
            priors,
            features[indices];
            kwargs...,
        )
    end
    return marginals
end

function save_residual_sine_predictive_surface(
    grid,
    prediction,
    output_prefix,
    forward_mode,
)
    variance_lower = minimum(prediction.variance)
    variance_upper = maximum(prediction.variance)
    variance_limits = variance_lower == variance_upper ?
        (variance_lower, nextfloat(variance_upper)) : (variance_lower, variance_upper)
    mean_panel = contourf(
        grid.x,
        grid.y,
        prediction.mean;
        color = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Predictive mean",
        linewidth = 0,
        aspect_ratio = :equal,
    )
    variance_panel = contourf(
        grid.x,
        grid.y,
        prediction.variance;
        color = :viridis,
        levels = 20,
        clims = variance_limits,
        xlabel = "x1",
        ylabel = "x2",
        title = "Predictive variance q(y)",
        linewidth = 0,
        aspect_ratio = :equal,
    )
    target_panel = heatmap(
        grid.x,
        grid.y,
        grid.actual;
        color = :RdBu,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Clean 2x2 target",
        aspect_ratio = :equal,
    )
    figure = plot(
        mean_panel,
        variance_panel,
        target_panel;
        layout = (1, 3),
        size = (1_350, 420),
        plot_title = "ContinuousTransition / MvResidualSine ($forward_mode) posterior prediction",
    )
    filename = output_prefix * "_$(forward_mode)_predictive.png"
    mkpath(dirname(filename))
    savefig(figure, filename)
    return filename
end

function benchmark_residual_sine_messages(
    activation,
    forward_projection,
    hidden_dimension;
    repetitions = 20,
)
    covariance = Matrix(Diagonal(fill(0.8, hidden_dimension)))
    covariance .+= 0.05 .* ones(hidden_dimension, hidden_dimension)
    input_message = MvNormalMeanCovariance(
        collect(range(-0.4, 0.4; length = hidden_dimension)),
        covariance,
    )
    output_belief = MvNormalMeanCovariance(
        collect(range(0.3, -0.3; length = hidden_dimension)),
        covariance,
    )
    output_message = MvNormalWeightedMeanPrecision(
        collect(range(0.5, -0.2; length = hidden_dimension)),
        Matrix(Diagonal(fill(1.3, hidden_dimension))),
    )
    xi, Lambda = weightedmean_precision(output_message)
    backward_exact = SurrogateModelling.MvResidualSineGaussianBackwardMessage(
        xi,
        Lambda,
        activation,
    )

    forward_call = () -> SurrogateModelling._mv_residual_sine_forward_site(
        activation,
        forward_projection,
        input_message,
        output_belief,
    )
    backward_call = () -> project(
        TangentProjection(type = ClosedForm),
        input_message,
        Logpdf(backward_exact),
    )

    forward_call()
    backward_call()
    forward_times = [@elapsed forward_call() for _ in 1:repetitions]
    backward_times = [@elapsed backward_call() for _ in 1:repetitions]
    return (
        forward_median_seconds = median(forward_times),
        backward_median_seconds = median(backward_times),
    )
end

function run_residual_sine_experiment(
    config;
    forward_mode,
    priors = nothing,
    meta_map = nothing,
    meta_pred = nothing,
    compact_posteriors = true,
)
    hidden_dimension = config.d_hidden
    feature_dimension = 3
    activation = make_residual_sine_activation(config)
    forward_projection = residual_sine_forward_projection(Val(forward_mode))
    meta_map = isnothing(meta_map) ?
        LinearReshapeMeta(hidden_dimension, feature_dimension) : meta_map
    meta_pred = isnothing(meta_pred) ?
        LinearReshapeMeta(hidden_dimension, hidden_dimension) : meta_pred

    dataset = make_checkerboard_dataset(
        n = config.n_samples,
        noise_std = config.noise_std,
        seed = config.data_seed,
    )
    train_data, test_data = split_dataset(
        dataset;
        train_fraction = config.train_fraction,
        seed = config.split_seed,
    )
    train_features = build_features(train_data)
    test_features = build_features(test_data)
    priors = isnothing(priors) ? make_priors(
        d_h = hidden_dimension,
        d_f = feature_dimension,
        seed = config.prior_seed,
        ct_precision_mean = config.ct_precision_mean,
        a_prior_mean_scale = config.a_prior_mean_scale,
        a_prior_variance = config.a_prior_variance,
        theta_prior_mean_scale = config.theta_prior_mean_scale,
        theta_prior_variance = config.theta_prior_variance,
        gamma_obs_mean = config.gamma_obs_mean,
        gamma_obs_concentration = config.gamma_obs_concentration,
    ) : priors

    activation_dependencies =
        make_residual_sine_dependencies(config, forward_projection)
    ct_a_dependencies = NGMPDependencies(
        a = nothing,
        damping = DampingMeta(
            alpha = config.ct_a_alpha,
            beta = config.ct_a_beta,
            max_step = config.ct_a_max_step,
        ),
    )
    ct2_dependencies = NGMPDependencies(
        a = nothing,
        damping = DampingMeta(
            alpha = config.ct_a_alpha,
            beta = config.ct_a_beta,
            max_step = config.ct_a_max_step,
        ),
    )
    training_returnvars = compact_posteriors && !config.diagnostics ?
        (
            a_map = KeepLast(),
            a_pred = KeepLast(),
            theta = KeepLast(),
            P = KeepLast(),
            Gamma2 = KeepLast(),
            gamma_obs = KeepLast(),
        ) : nothing

    training_elapsed = @elapsed result = infer(
        model = xor_ct_mvresidual_sine(
            priors = priors,
            feature_cov = Matrix(Diagonal(fill(config.feature_jitter, feature_dimension))),
            meta_map = meta_map,
            meta_pred = meta_pred,
            ct_a_deps = ct_a_dependencies,
            ct2_deps = ct2_dependencies,
            activation_deps = activation_dependencies,
            activation = activation,
        ),
        data = (y = train_data.OT, features = train_features),
        constraints = xor_ct_constraints(),
        initialization = make_initialization(
            priors,
            hidden_dimension,
            make_residual_sine_initial(config, hidden_dimension, activation),
        ),
        iterations = config.iterations,
        free_energy = true,
        showprogress = config.show_progress,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
        returnvars = training_returnvars,
    )

    prediction_priors_for_run = prediction_priors(result)
    output_mean = mean(train_data.OT)
    prediction_arguments = (
        batch_size = config.prediction_batch_size,
        config = config,
        activation = activation,
        hidden_dimension = hidden_dimension,
        feature_dimension = feature_dimension,
        output_mean = output_mean,
        meta_map = meta_map,
        meta_pred = meta_pred,
        forward_projection = forward_projection,
    )
    prediction_elapsed = @elapsed begin
        train_prediction = predictive_statistics(predict_residual_sine_marginals(
            prediction_priors_for_run,
            train_features;
            prediction_arguments...,
        ))
        test_prediction = predictive_statistics(predict_residual_sine_marginals(
            prediction_priors_for_run,
            test_features;
            prediction_arguments...,
        ))
    end
    train_mse = mean(abs2, train_prediction.mean .- train_data.OT)
    test_mse = mean(abs2, test_prediction.mean .- test_data.OT)
    baseline_mse = mean(abs2, mean(train_data.OT) .- test_data.OT)

    surface_path = nothing
    if config.save_outputs
        grid = make_prediction_grid(config.grid_size)
        grid_statistics = predictive_statistics(predict_residual_sine_marginals(
            prediction_priors_for_run,
            grid.features;
            prediction_arguments...,
        ))
        grid_prediction = (
            mean = reshape(grid_statistics.mean, length(grid.y), length(grid.x)),
            variance = reshape(grid_statistics.variance, length(grid.y), length(grid.x)),
        )
        surface_path = save_residual_sine_predictive_surface(
            grid,
            grid_prediction,
            config.output_prefix,
            forward_mode,
        )
    end

    message_benchmark = benchmark_residual_sine_messages(
        activation,
        forward_projection,
        hidden_dimension,
    )
    free_energy_values = result.free_energy
    activation_firings = getproperty.(activation_dependencies.states, :nfired)
    println()
    println("=== xor_ctransition_mvresidual_sine ($forward_mode, d_h = $hidden_dimension, " *
            "iterations = $(config.iterations), n_train = $(nrow(train_data)), " *
            "n_test = $(nrow(test_data)))")
    println("training elapsed       : ", round(training_elapsed, digits = 3), "s")
    println("prediction elapsed     : ", round(prediction_elapsed, digits = 3), "s")
    println("message forward median : ", 1e6 * message_benchmark.forward_median_seconds, " us")
    println("message backward median: ", 1e6 * message_benchmark.backward_median_seconds, " us")
    println("free energy first/last : ", first(free_energy_values), " / ", last(free_energy_values))
    println("free energy finite     : ", all(isfinite, free_energy_values))
    println("train MSE              : ", round(train_mse, digits = 5))
    println("test MSE               : ", round(test_mse, digits = 5))
    println("normalized test MSE    : ", round(test_mse / baseline_mse, digits = 5))
    println("activation states/fire : ", length(activation_firings), " / ",
            isempty(activation_firings) ? "none" :
            "$(minimum(activation_firings))..$(maximum(activation_firings))")
    println("test q(y) variance     : ",
            round(minimum(test_prediction.variance), digits = 5), " / ",
            round(mean(test_prediction.variance), digits = 5), " / ",
            round(maximum(test_prediction.variance), digits = 5))
    isnothing(surface_path) || println("predictive surface     : ", surface_path)

    return result, (
        arm = String(forward_mode),
        hidden_dimension = hidden_dimension,
        training_elapsed = training_elapsed,
        prediction_elapsed = prediction_elapsed,
        forward_message_median_seconds = message_benchmark.forward_median_seconds,
        backward_message_median_seconds = message_benchmark.backward_median_seconds,
        train_mse = train_mse,
        test_mse = test_mse,
        baseline = baseline_mse,
        normalized_test_mse = test_mse / baseline_mse,
        test_predictive_variance = (
            minimum = minimum(test_prediction.variance),
            mean = mean(test_prediction.variance),
            maximum = maximum(test_prediction.variance),
        ),
        free_energy = free_energy_values,
        surface_path = surface_path,
    )
end

function residual_sine_shared_setup(config)
    if !config.use_low_rank
        priors = make_priors(
            d_h = config.d_hidden,
            d_f = 3,
            seed = config.prior_seed,
            ct_precision_mean = config.ct_precision_mean,
            a_prior_mean_scale = config.a_prior_mean_scale,
            a_prior_variance = config.a_prior_variance,
            theta_prior_mean_scale = config.theta_prior_mean_scale,
            theta_prior_variance = config.theta_prior_variance,
            gamma_obs_mean = config.gamma_obs_mean,
            gamma_obs_concentration = config.gamma_obs_concentration,
        )
        return (
            priors = priors,
            meta_map = LinearReshapeMeta(config.d_hidden, 3),
            meta_pred = LinearReshapeMeta(config.d_hidden, config.d_hidden),
            parameter_count = config.d_hidden * 3 + config.d_hidden^2,
        )
    end

    default_budget = min(config.d_hidden, 3) + config.d_hidden
    parameter_budget = env_int("LOW_RANK_PARAMETER_BUDGET", default_budget)
    map_parameters = env_int(
        "LOW_RANK_MAP_PARAMETERS",
        clamp(round(Int, 3parameter_budget / 7), 1, parameter_budget - 1),
    )
    prediction_parameters = parameter_budget - map_parameters
    setup = low_rank_setup(
        config;
        map_parameters = map_parameters,
        pred_parameters = prediction_parameters,
        offset_scale = env_float("LOW_RANK_A0_SCALE", 1.0),
        d_f = 3,
    )
    return (
        priors = setup.priors,
        meta_map = setup.meta_map,
        meta_pred = setup.meta_pred,
        parameter_count = map_parameters + prediction_parameters,
    )
end

function run_residual_sine_comparison(config = RESIDUAL_SINE_CONFIG)
    modes = residual_sine_forward_modes()
    setup = residual_sine_shared_setup(config)
    println("Residual-sine comparison: rho=$(config.phi_rho), omega=$(config.phi_omega), " *
            "d_h=$(config.d_hidden), low_rank=$(config.use_low_rank), " *
            "parameters=$(setup.parameter_count)")

    # Compile the same graph shapes used by the measured runs. GraphPPL's
    # generated graph type depends on plate/batch sizes, so a tiny warmup does
    # not remove all compilation from a full-size timing.
    if env_bool("COMPARISON_WARMUP", true)
        full_shape_warmup = env_bool("COMPARISON_FULL_SHAPE_WARMUP", true)
        warmup_config = merge(config, (;
            n_samples = full_shape_warmup ? config.n_samples : min(config.n_samples, 24),
            train_fraction = full_shape_warmup ? config.train_fraction : 0.75,
            iterations = 1,
            prediction_iterations = 1,
            prediction_batch_size = full_shape_warmup ? config.prediction_batch_size : 24,
            save_outputs = false,
            show_progress = false,
        ))
        for mode in modes
            run_residual_sine_experiment(
                warmup_config;
                forward_mode = mode,
                priors = deepcopy(setup.priors),
                meta_map = setup.meta_map,
                meta_pred = setup.meta_pred,
            )
            GC.gc()
        end
        if config.run_softplus_baseline
            baseline_warmup_config = merge(warmup_config, (;
                mvsoftplus_projection = "unscented",
                output_prefix = config.output_prefix * "_softplus_warmup",
            ))
            run_experiment(
                baseline_warmup_config;
                priors = deepcopy(setup.priors),
                meta_map = setup.meta_map,
                meta_pred = setup.meta_pred,
                experiment_label = "matched MvSoftplus warmup",
                compact_posteriors = true,
            )
            GC.gc()
        end
    end

    softplus_metrics = nothing
    if config.run_softplus_baseline
        baseline_config = merge(config, (;
            mvsoftplus_projection = "unscented",
            output_prefix = config.output_prefix * "_softplus",
        ))
        _, softplus_metrics = run_experiment(
            baseline_config;
            priors = deepcopy(setup.priors),
            meta_map = setup.meta_map,
            meta_pred = setup.meta_pred,
            experiment_label = "matched MvSoftplus baseline",
            compact_posteriors = true,
        )
        GC.gc()
    end

    residual_metrics = map(modes) do mode
        mode_config = merge(config, (;
            output_prefix = config.output_prefix,
        ))
        _, metrics = run_residual_sine_experiment(
            mode_config;
            forward_mode = mode,
            priors = deepcopy(setup.priors),
            meta_map = setup.meta_map,
            meta_pred = setup.meta_pred,
        )
        GC.gc()
        return metrics
    end

    println()
    println("=== matched activation comparison ===")
    println("arm,parameters,train_s,predict_s,train_mse,test_mse,normalized_test_mse")
    if !isnothing(softplus_metrics)
        println(join((
            "softplus_unscented",
            setup.parameter_count,
            softplus_metrics.training_elapsed,
            softplus_metrics.prediction_elapsed,
            softplus_metrics.train_mse,
            softplus_metrics.test_mse,
            softplus_metrics.test_mse / softplus_metrics.baseline,
        ), ','))
    end
    for metrics in residual_metrics
        println(join((
            "residual_sine_$(metrics.arm)",
            setup.parameter_count,
            metrics.training_elapsed,
            metrics.prediction_elapsed,
            metrics.train_mse,
            metrics.test_mse,
            metrics.normalized_test_mse,
        ), ','))
    end
    return (; softplus = softplus_metrics, residual_sine = residual_metrics, setup)
end

if abspath(PROGRAM_FILE) == @__FILE__
    comparison = run_residual_sine_comparison()
end
