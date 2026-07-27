# Prediction-only convergence audit for the best combined training run.
#
# This does not retrain any parameter. It reloads the learned posteriors and
# runs the explicit q(y*) graph for 200 iterations while retaining checkpoints.

ENV["JOINT_HETERO_LINK"] = "squareplus"
ENV["JOINT_HETERO_USE_PRECISION_PRIOR"] = "false"
ENV["JOINT_HETERO_ACTIVATION_ALPHA"] = "0.05"
ENV["JOINT_HETERO_LOG_ALPHA"] = "0.05"
ENV["JOINT_HETERO_SEPARATE_DEFINITIONS_ONLY"] = "true"

include(joinpath(
    @__DIR__,
    "manyplus_residual_sine_joint_heteroscedastic_separate_variance_squareplus.jl",
))

audit_source = get(
    ENV,
    "JOINT_HETERO_AUDIT_POSTERIORS",
    "/tmp/manyplus_joint_separate_variance_no_gamma_fast_damping_qy_posteriors.jls",
)
audit_output = get(
    ENV,
    "JOINT_HETERO_AUDIT_OUTPUT",
    "/tmp/manyplus_joint_prediction_convergence_audit.png",
)
audit_artifact_output = get(
    ENV,
    "JOINT_HETERO_AUDIT_ARTIFACT",
    "/tmp/manyplus_joint_prediction_convergence_audit.jls",
)
audit_iterations = env_int(
    "JOINT_HETERO_AUDIT_ITERATIONS",
    200,
)
audit_training_artifact = deserialize(audit_source)
audit_activation_alpha = env_float(
    "JOINT_HETERO_AUDIT_ACTIVATION_ALPHA",
    audit_training_artifact.config.activation_alpha,
)
audit_log_alpha = env_float(
    "JOINT_HETERO_AUDIT_LOG_ALPHA",
    audit_training_artifact.config.log_alpha,
)
audit_max_step = env_float(
    "JOINT_HETERO_AUDIT_MAX_STEP",
    audit_training_artifact.config.ngmp_max_step,
)

audit_config = merge(
    audit_training_artifact.config,
    (
        prediction_iterations = audit_iterations,
        definitions_only = false,
        activation_alpha = audit_activation_alpha,
        log_alpha = audit_log_alpha,
        ngmp_max_step = audit_max_step,
    ),
)
audit_priors =
    audit_training_artifact.learned_parameter_posteriors
audit_x_grid = collect(audit_training_artifact.x_grid)
audit_features = make_features(audit_x_grid)
audit_inits = separate_pushforward_inits(
    audit_priors,
    audit_features,
    audit_config,
)

audit_result = infer(
    model = separate_variance_qy_prediction(
        n_neurons = audit_config.n_neurons,
        priors = audit_priors,
        mean_activation = activation_meta(audit_config),
        mean_activation_deps = activation_dependencies(audit_config),
        noise_activation = activation_meta(audit_config),
        noise_activation_deps = activation_dependencies(audit_config),
        link_deps = link_dependencies(audit_config),
        link_meta = link_damping(audit_config),
        positive_link = audit_config.positive_link,
        precision_kappa = audit_config.precision_kappa,
        prediction_prior_variance =
            audit_config.prediction_prior_variance,
        use_precision_prior =
            audit_config.use_precision_prior,
    ),
    data = (features = audit_features,),
    constraints = separate_variance_prediction_constraints(),
    initialization = separate_variance_prediction_initialization(
        audit_priors,
        audit_inits,
    ),
    returnvars = (
        y = KeepEach(),
        mean_output = KeepEach(),
        precision = KeepEach(),
    ),
    iterations = audit_iterations,
    free_energy = false,
    showprogress = false,
    options = (limit_stack_depth = 100,),
    disable_inference_error_hint = true,
)

audit_checkpoints = unique(sort(filter(
    iteration -> iteration <= audit_iterations,
    [
        1, 2, 4, 8, 16, 32, 64, 100, 150, 200,
        300, 400, 500, 750, 1_000,
    ],
)))
audit_supported = abs.(audit_x_grid) .<= 1.5
audit_true_mean = clean_mean.(audit_x_grid)
audit_true_variance = true_variance.(audit_x_grid)

function audit_metrics_at(iteration)
    q_y = collect(vec(audit_result.posteriors[:y][iteration]))
    q_mean =
        collect(vec(audit_result.posteriors[:mean_output][iteration]))
    q_precision =
        collect(vec(audit_result.posteriors[:precision][iteration]))

    predicted_mean = mean_to_data.(mean.(q_y))
    predictive_variance = variance_to_data.(var.(q_y))
    mean_variance = variance_to_data.(var.(q_mean))
    aleatoric_variance = variance_to_data.(
        inv.(mean.(q_precision)),
    )
    residual =
        predictive_variance .- mean_variance .- aleatoric_variance

    return (
        iteration = iteration,
        q_y = q_y,
        q_mean = q_mean,
        q_precision = q_precision,
        predicted_mean = predicted_mean,
        predictive_variance = predictive_variance,
        mean_variance = mean_variance,
        aleatoric_variance = aleatoric_variance,
        decomposition_residual = residual,
        mean_rmse = sqrt(mean(abs2.(
            predicted_mean[audit_supported] .-
            audit_true_mean[audit_supported],
        ))),
        qy_variance_mean = mean(
            predictive_variance[audit_supported],
        ),
        mean_variance_mean = mean(
            mean_variance[audit_supported],
        ),
        aleatoric_variance_mean = mean(
            aleatoric_variance[audit_supported],
        ),
        aleatoric_correlation = cor(
            aleatoric_variance[audit_supported],
            audit_true_variance[audit_supported],
        ),
        residual_mean = mean(residual[audit_supported]),
        residual_max_abs = maximum(abs, residual[audit_supported]),
    )
end

audit_checkpoint_results = [
    audit_metrics_at(iteration)
    for iteration in audit_checkpoints
]
for checkpoint in audit_checkpoint_results
    println(
        "PREDICTION_AUDIT_CHECKPOINT = ",
        (
            iteration = checkpoint.iteration,
            mean_rmse = checkpoint.mean_rmse,
            qy_variance_mean = checkpoint.qy_variance_mean,
            mean_variance_mean = checkpoint.mean_variance_mean,
            aleatoric_variance_mean =
                checkpoint.aleatoric_variance_mean,
            aleatoric_correlation =
                checkpoint.aleatoric_correlation,
            residual_mean = checkpoint.residual_mean,
            residual_max_abs = checkpoint.residual_max_abs,
        ),
    )
end

audit_final = last(audit_checkpoint_results)
audit_summary = (
    source = audit_source,
    iterations = audit_iterations,
    activation_alpha = audit_activation_alpha,
    log_alpha = audit_log_alpha,
    max_step = audit_max_step,
    checkpoints = audit_checkpoints,
    checkpoint_metrics = [
        (
            iteration = checkpoint.iteration,
            mean_rmse = checkpoint.mean_rmse,
            qy_variance_mean = checkpoint.qy_variance_mean,
            mean_variance_mean = checkpoint.mean_variance_mean,
            aleatoric_variance_mean =
                checkpoint.aleatoric_variance_mean,
            aleatoric_correlation =
                checkpoint.aleatoric_correlation,
            residual_mean = checkpoint.residual_mean,
            residual_max_abs = checkpoint.residual_max_abs,
        )
        for checkpoint in audit_checkpoint_results
    ],
    final_q_y = audit_final.q_y,
    final_q_mean = audit_final.q_mean,
    final_q_precision = audit_final.q_precision,
)
serialize(audit_artifact_output, audit_summary)

iteration_axis = getindex.(
    audit_checkpoint_results,
    :iteration,
)
convergence_panel = plot(
    iteration_axis,
    getindex.(audit_checkpoint_results, :qy_variance_mean);
    xscale = :log10,
    color = :darkorange,
    linewidth = 2,
    marker = :circle,
    label = "mean Var(q(y*))",
    xlabel = "prediction iteration",
    ylabel = "mean variance on |x| ≤ 1.5",
    title = "Prediction convergence (training fixed)",
)
plot!(
    convergence_panel,
    iteration_axis,
    getindex.(audit_checkpoint_results, :mean_variance_mean);
    color = :royalblue,
    linewidth = 2,
    marker = :circle,
    label = "mean Var(q(mean))",
)
plot!(
    convergence_panel,
    iteration_axis,
    getindex.(audit_checkpoint_results, :aleatoric_variance_mean);
    color = :purple,
    linewidth = 2,
    marker = :circle,
    label = "mean 1/E[q(precision)]",
)
hline!(
    convergence_panel,
    [mean(audit_true_variance[audit_supported])];
    color = :black,
    linestyle = :dash,
    label = "mean true variance",
)

audit_final_uncertainty =
    1.96 .* sqrt.(max.(audit_final.predictive_variance, 0.0))
final_fit_panel = scatter(
    x_train,
    y_train;
    color = :gray60,
    markersize = 3,
    markeralpha = 0.55,
    markerstrokewidth = 0,
    label = "observations",
    xlabel = "x",
    ylabel = "y",
    title = "After $audit_iterations prediction iterations",
    legend = :bottomleft,
)
plot!(
    final_fit_panel,
    audit_x_grid,
    audit_final.predicted_mean;
    ribbon = audit_final_uncertainty,
    fillalpha = 0.18,
    color = :royalblue,
    linewidth = 2,
    label = "mean(q(y*)) ± 1.96 SD",
)
plot!(
    final_fit_panel,
    audit_x_grid,
    audit_true_mean;
    color = :black,
    linewidth = 2,
    label = "true mean",
)

final_variance_panel = plot(
    audit_x_grid,
    audit_true_variance;
    color = :black,
    linewidth = 2,
    label = "true Var(y | x)",
    xlabel = "x",
    ylabel = "variance",
    title = "Final predictive decomposition",
    legend = :topleft,
)
plot!(
    final_variance_panel,
    audit_x_grid,
    audit_final.predictive_variance;
    color = :darkorange,
    linewidth = 2,
    label = "Var(q(y*))",
)
plot!(
    final_variance_panel,
    audit_x_grid,
    audit_final.mean_variance;
    color = :royalblue,
    linestyle = :dot,
    linewidth = 2,
    label = "Var(q(mean))",
)
plot!(
    final_variance_panel,
    audit_x_grid,
    audit_final.aleatoric_variance;
    color = :purple,
    linestyle = :dash,
    linewidth = 2,
    label = "1/E[q(precision)]",
)

audit_figure = plot(
    convergence_panel,
    final_fit_panel,
    final_variance_panel;
    layout = (1, 3),
    size = (1_650, 430),
)
savefig(audit_figure, audit_output)
println("saved prediction audit to ", audit_output)
println("saved prediction audit artifact to ", audit_artifact_output)
