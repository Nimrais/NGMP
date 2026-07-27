# Cubic epistemic-gap benchmark for the current separate mean/variance model.
#
#   y = x^3 + Normal(0, 9)
#   x ~ 0.5 Uniform(-5, -3) + 0.5 Uniform(3, 5)
#
# The interval (-3, 3) contains no training observations. Training is one
# full-data infer call capped at 100 iterations. Prediction keeps w joint with
# its forward variables and uses the independently validated 100-step
# prediction settings.

cubic_n_string = get(ENV, "JOINT_CUBIC_N", "80")
cubic_link = lowercase(get(ENV, "JOINT_CUBIC_LINK", "squareplus"))
cubic_link in ("exp", "squareplus") ||
    throw(ArgumentError("JOINT_CUBIC_LINK must be exp or squareplus"))
ENV["JOINT_HETERO_N"] = cubic_n_string
ENV["JOINT_HETERO_SEPARATE_LINK"] = cubic_link
ENV["JOINT_HETERO_USE_PRECISION_PRIOR"] = "false"
ENV["JOINT_HETERO_ACTIVATION_ALPHA"] = "0.05"
ENV["JOINT_HETERO_LOG_ALPHA"] = "0.05"
ENV["JOINT_HETERO_G_INITIAL_VARIANCE"] = "0.0001"
ENV["JOINT_HETERO_SEPARATE_DEFINITIONS_ONLY"] = "true"

include(joinpath(
    @__DIR__,
    "manyplus_residual_sine_joint_heteroscedastic_separate_variance_squareplus.jl",
))

cubic_n = parse(Int, cubic_n_string)
iseven(cubic_n) ||
    throw(ArgumentError("JOINT_CUBIC_N must be even"))

cubic_output = get(
    ENV,
    "JOINT_CUBIC_OUTPUT",
    "/tmp/manyplus_joint_cubic_epistemic.png",
)
cubic_posterior_output = get(
    ENV,
    "JOINT_CUBIC_POSTERIORS",
    "/tmp/manyplus_joint_cubic_epistemic_posteriors.jls",
)
cubic_config = merge(
    separate_config,
    (
        n_observations = cubic_n,
        definitions_only = false,
        use_precision_prior = false,
        output_path = cubic_output,
        posterior_path = cubic_posterior_output,
        grid_points = env_int(
            "JOINT_CUBIC_GRID",
            separate_config.smoke_mode ? 31 : 241,
        ),
    ),
)

cubic_rng = StableRNG(cubic_config.data_seed + 1)
cubic_x = vcat(
    -5 .+ 2 .* rand(cubic_rng, cubic_n ÷ 2),
    3 .+ 2 .* rand(cubic_rng, cubic_n ÷ 2),
)
cubic_x = cubic_x[randperm(cubic_rng, cubic_n)]
cubic_true_mean_train = cubic_x .^ 3
cubic_y = cubic_true_mean_train .+ 3 .* randn(cubic_rng, cubic_n)

cubic_x_scale = 4.0
cubic_y_center = mean(cubic_y)
cubic_y_scale = std(cubic_y)
cubic_y_scale > 0 || error("cubic response scale must be positive")
cubic_features(x_values) = [
    [1.0, Float64(x) / cubic_x_scale]
    for x in x_values
]
cubic_targets =
    (Float64.(cubic_y) .- cubic_y_center) ./ cubic_y_scale
cubic_mean_to_data(value) =
    cubic_y_center + cubic_y_scale * value
cubic_variance_to_data(value) =
    abs2(cubic_y_scale) * value

cubic_fit = train_separate_variance_model(
    cubic_targets,
    cubic_features(cubic_x),
    cubic_config,
)
cubic_fitted_priors = learned_separate_priors(cubic_fit)

cubic_grid = collect(range(
    -6.0,
    6.0;
    length = cubic_config.grid_points,
))
cubic_grid_features = cubic_features(cubic_grid)
cubic_prediction_config = merge(
    cubic_config,
    (
        activation_alpha =
            cubic_config.prediction_activation_alpha,
        log_alpha = cubic_config.prediction_log_alpha,
        ngmp_max_step = cubic_config.prediction_max_step,
    ),
)
cubic_prediction_inits = separate_pushforward_inits(
    cubic_fitted_priors,
    cubic_grid_features,
    cubic_prediction_config,
)

cubic_prediction_result = infer(
    model = separate_variance_qy_prediction(
        n_neurons = cubic_config.n_neurons,
        priors = cubic_fitted_priors,
        mean_activation = activation_meta(cubic_prediction_config),
        mean_activation_deps =
            activation_dependencies(cubic_prediction_config),
        noise_activation = activation_meta(cubic_prediction_config),
        noise_activation_deps =
            activation_dependencies(cubic_prediction_config),
        link_deps = link_dependencies(cubic_prediction_config),
        link_meta = link_damping(cubic_prediction_config),
        positive_link = cubic_config.positive_link,
        precision_kappa = cubic_config.precision_kappa,
        prediction_prior_variance =
            cubic_config.prediction_prior_variance,
        use_precision_prior = false,
    ),
    data = (features = cubic_grid_features,),
    constraints = separate_variance_prediction_constraints(),
    initialization = separate_variance_prediction_initialization(
        cubic_fitted_priors,
        cubic_prediction_inits,
    ),
    returnvars = (
        y = KeepLast(),
        mean_output = KeepLast(),
        precision = KeepLast(),
    ),
    iterations = cubic_config.prediction_iterations,
    free_energy = false,
    showprogress = false,
    options = (limit_stack_depth = 100,),
    disable_inference_error_hint = true,
)

cubic_q_y = collect(vec(
    cubic_prediction_result.posteriors[:y],
))
cubic_q_mean = collect(vec(
    cubic_prediction_result.posteriors[:mean_output],
))
cubic_q_precision = collect(vec(
    cubic_prediction_result.posteriors[:precision],
))
cubic_predicted_mean =
    cubic_mean_to_data.(mean.(cubic_q_y))
cubic_predictive_variance =
    cubic_variance_to_data.(var.(cubic_q_y))
cubic_mean_variance =
    cubic_variance_to_data.(var.(cubic_q_mean))
cubic_aleatoric_variance = cubic_variance_to_data.(
    inv.(mean.(cubic_q_precision)),
)
cubic_decomposition_residual =
    cubic_predictive_variance .-
    cubic_mean_variance .-
    cubic_aleatoric_variance
cubic_true_mean = cubic_grid .^ 3
cubic_true_variance = fill(9.0, length(cubic_grid))

cubic_observed_domain =
    (abs.(cubic_grid) .>= 3.0) .&
    (abs.(cubic_grid) .<= 5.0)
cubic_gap = abs.(cubic_grid) .<= 2.5
cubic_outer = abs.(cubic_grid) .>= 5.5

cubic_mean_rmse_observed = sqrt(mean(abs2.(
    cubic_predicted_mean[cubic_observed_domain] .-
    cubic_true_mean[cubic_observed_domain],
)))
cubic_mean_rmse_gap = sqrt(mean(abs2.(
    cubic_predicted_mean[cubic_gap] .-
    cubic_true_mean[cubic_gap],
)))
cubic_epistemic_observed =
    mean(cubic_mean_variance[cubic_observed_domain])
cubic_epistemic_gap =
    mean(cubic_mean_variance[cubic_gap])
cubic_epistemic_outer =
    mean(cubic_mean_variance[cubic_outer])
cubic_total_observed =
    mean(cubic_predictive_variance[cubic_observed_domain])
cubic_total_gap =
    mean(cubic_predictive_variance[cubic_gap])
cubic_aleatoric_observed =
    mean(cubic_aleatoric_variance[cubic_observed_domain])
cubic_aleatoric_gap =
    mean(cubic_aleatoric_variance[cubic_gap])
cubic_gap_function_coverage = mean(
    abs.(
        cubic_predicted_mean[cubic_gap] .-
        cubic_true_mean[cubic_gap]
    ) .<=
    1.96 .* sqrt.(cubic_mean_variance[cubic_gap]),
)
cubic_gap_total_coverage = mean(
    abs.(
        cubic_predicted_mean[cubic_gap] .-
        cubic_true_mean[cubic_gap]
    ) .<=
    1.96 .* sqrt.(cubic_predictive_variance[cubic_gap]),
)

cubic_metrics = (
    positive_link = cubic_config.positive_link,
    observations = cubic_n,
    neurons_per_network = cubic_config.n_neurons,
    output_weight_reference_neurons =
        cubic_config.output_weight_reference_neurons,
    output_weight_width_scale =
        cubic_config.output_weight_width_scale,
    training_iterations = cubic_fit.iterations,
    prediction_iterations = cubic_config.prediction_iterations,
    training_seconds = cubic_fit.elapsed_seconds,
    mean_rmse_observed = cubic_mean_rmse_observed,
    mean_rmse_gap = cubic_mean_rmse_gap,
    epistemic_observed_mean = cubic_epistemic_observed,
    epistemic_gap_mean = cubic_epistemic_gap,
    epistemic_gap_ratio =
        cubic_epistemic_gap / cubic_epistemic_observed,
    epistemic_outer_mean = cubic_epistemic_outer,
    epistemic_outer_ratio =
        cubic_epistemic_outer / cubic_epistemic_observed,
    aleatoric_observed_mean = cubic_aleatoric_observed,
    aleatoric_gap_mean = cubic_aleatoric_gap,
    true_aleatoric = 9.0,
    total_observed_mean = cubic_total_observed,
    total_gap_mean = cubic_total_gap,
    total_gap_ratio = cubic_total_gap / cubic_total_observed,
    gap_function_coverage = cubic_gap_function_coverage,
    gap_total_coverage = cubic_gap_total_coverage,
    decomposition_residual_mean = mean(
        cubic_decomposition_residual,
    ),
    decomposition_residual_max_abs = maximum(
        abs,
        cubic_decomposition_residual,
    ),
)
println("CUBIC_EPISTEMIC_RESULT = ", cubic_metrics)

cubic_artifact = (
    config = cubic_config,
    metrics = cubic_metrics,
    learned_parameter_posteriors = cubic_fitted_priors,
    x_train = cubic_x,
    y_train = cubic_y,
    x_grid = cubic_grid,
    q_y = cubic_q_y,
    q_mean = cubic_q_mean,
    q_precision = cubic_q_precision,
    predicted_mean = cubic_predicted_mean,
    predictive_variance = cubic_predictive_variance,
    mean_function_variance = cubic_mean_variance,
    aleatoric_variance = cubic_aleatoric_variance,
    true_mean = cubic_true_mean,
    true_variance = cubic_true_variance,
    decomposition_residual = cubic_decomposition_residual,
)
serialize(cubic_posterior_output, cubic_artifact)

cubic_interval =
    1.96 .* sqrt.(max.(cubic_predictive_variance, 0.0))
cubic_fit_panel = plot(
    cubic_grid,
    cubic_predicted_mean;
    ribbon = cubic_interval,
    fillalpha = 0.18,
    color = :royalblue,
    linewidth = 2,
    label = "mean(q(y*)) ± 1.96 SD",
    xlabel = "x",
    ylabel = "y",
    title = "Cubic benchmark: $(cubic_config.n_neurons) neurons/head",
    legend = :topleft,
)
plot!(
    cubic_fit_panel,
    cubic_grid,
    cubic_true_mean;
    color = :black,
    linewidth = 2,
    label = "true x³",
)
scatter!(
    cubic_fit_panel,
    cubic_x,
    cubic_y;
    color = :gray50,
    markersize = 3,
    markeralpha = 0.55,
    markerstrokewidth = 0,
    label = "training observations",
)
vspan!(
    cubic_fit_panel,
    [-3.0, 3.0];
    color = :gray85,
    alpha = 0.18,
    label = "no training data",
)

cubic_variance_panel = plot(
    cubic_grid,
    cubic_predictive_variance;
    color = :darkorange,
    linewidth = 2,
    label = "Var(q(y*))",
    xlabel = "x",
    ylabel = "variance (log scale)",
    yscale = :log10,
    title = "$(cubic_config.positive_link) predictive decomposition",
    legend = :topleft,
)
plot!(
    cubic_variance_panel,
    cubic_grid,
    cubic_mean_variance;
    color = :royalblue,
    linestyle = :dot,
    linewidth = 2,
    label = "Var(q(mean))",
)
plot!(
    cubic_variance_panel,
    cubic_grid,
    cubic_aleatoric_variance;
    color = :purple,
    linestyle = :dash,
    linewidth = 2,
    label = "1/E[q(precision)]",
)
hline!(
    cubic_variance_panel,
    [9.0];
    color = :black,
    linestyle = :dashdot,
    linewidth = 1.7,
    label = "true aleatoric = 9",
)
vspan!(
    cubic_variance_panel,
    [-3.0, 3.0];
    color = :gray85,
    alpha = 0.18,
    label = "no training data",
)

cubic_epistemic_ratio =
    cubic_mean_variance ./ cubic_epistemic_observed
cubic_total_ratio =
    cubic_predictive_variance ./ cubic_total_observed
cubic_ratio_panel = plot(
    cubic_grid,
    cubic_epistemic_ratio;
    color = :royalblue,
    linewidth = 2,
    label = "epistemic / observed-domain mean",
    xlabel = "x",
    ylabel = "relative variance",
    title = "Does uncertainty detect the gap?",
    legend = :topleft,
)
plot!(
    cubic_ratio_panel,
    cubic_grid,
    cubic_total_ratio;
    color = :darkorange,
    linestyle = :dash,
    linewidth = 2,
    label = "total / observed-domain mean",
)
hline!(
    cubic_ratio_panel,
    [1.0];
    color = :black,
    linestyle = :dot,
    label = "observed-domain reference",
)
vspan!(
    cubic_ratio_panel,
    [-3.0, 3.0];
    color = :gray85,
    alpha = 0.18,
    label = "no training data",
)

cubic_figure = plot(
    cubic_fit_panel,
    cubic_variance_panel,
    cubic_ratio_panel;
    layout = (1, 3),
    size = (1_650, 430),
)
savefig(cubic_figure, cubic_output)
println("saved cubic epistemic plot to ", cubic_output)
println("saved cubic learned posteriors to ", cubic_posterior_output)
