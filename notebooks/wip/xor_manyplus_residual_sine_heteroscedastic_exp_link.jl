# Fixed-mean heteroscedastic XOR ablation with an exponential precision link.
#
# This intentionally changes only the positive link relative to
# `xor_manyplus_residual_sine_heteroscedastic_ngmp.jl`:
#
#     Squareplus(score)  ->  Exp(log_precision)
#
# The constant-noise mean model, shared Gamma hierarchy, paired residual-sine
# head, one-pass batches, priors, and cold-start variances remain unchanged.
# If gamma_bar is the baseline's learned precision, the deviation head starts
# at log(gamma_bar), so Exp maps its initial mean back to gamma_bar exactly.
#
# Smoke:
#   OPENBLAS_NUM_THREADS=1 XOR_HETERO_STAGE=smoke \
#     julia --project=. experiments/xor_manyplus_residual_sine_heteroscedastic_exp_link.jl
#
# Full fixed-mean comparison:
#   OPENBLAS_NUM_THREADS=1 XOR_HETERO_STAGE=full \
#     julia --project=. experiments/xor_manyplus_residual_sine_heteroscedastic_exp_link.jl

include(joinpath(
    @__DIR__,
    "xor_manyplus_residual_sine_heteroscedastic_ngmp.jl",
))

import ProbabilisticEnsembling: Exp

function exp_link_config()
    base = heteroscedastic_config()
    base.stage in (:smoke, :full) || throw(ArgumentError(
        "the focused Exp-link ablation supports XOR_HETERO_STAGE=smoke or full",
    ))
    output_dir = get(
        ENV,
        "OUTPUT_DIR",
        joinpath(@__DIR__, "xor_manyplus_residual_sine_heteroscedastic_exp_output"),
    )
    return validate_config(merge(base, (output_dir = output_dir,)))
end

function exp_link_dependencies(config, alpha, max_step)
    return NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = TangentProjection(type = ClosedForm),
        damping = DampingMeta(
            alpha = alpha,
            beta = 0.0,
            max_step = max_step,
        ),
    )
end

function make_exp_noise_priors(
    config,
    n_noise_neurons,
    kappa,
    baseline_precision,
)
    priors = make_noise_priors(
        config, n_noise_neurons, kappa, baseline_precision,
    )
    log_precision = log(baseline_precision)
    priors[:noise_intercept] = NormalMeanVariance(
        log_precision, config.noise_intercept_prior_variance,
    )
    priors[:score_intercept] = log_precision
    exp(priors[:score_intercept]) ≈ baseline_precision ||
        error("Exp-link intercept does not reproduce baseline precision")
    return priors
end

function exp_noise_pushforward_inits(
    priors,
    features,
    n_noise_neurons,
    config;
    cold_start,
)
    inits = noise_pushforward_inits(
        priors, features, n_noise_neurons, config; cold_start,
    )
    cold_start && return inits

    log_precision_means = Float64.(mean.(inits.score))
    precision_means = exp.(log_precision_means)
    all(value -> isfinite(value) && value > 0, precision_means) ||
        error("Exp-link initialization produced invalid precisions")
    gamma = [
        GammaShapeRate(priors[:kappa], priors[:kappa] / precision)
        for precision in precision_means
    ]
    return merge(inits, (; gamma))
end

@model function hetero_fixed_mean_exp_model(
    n_noise_neurons,
    kappa,
    features,
    fixed_mean,
    y,
    priors,
    activation,
    noise_activation_deps,
    exp_deps,
)
    local noise_w, noise_v, noise_za, noise_h, noise_c, noise_score, gamma

    noise_tau ~ priors[:noise_tau]
    noise_tau_c ~ priors[:noise_tau_c]
    noise_intercept ~ priors[:noise_intercept]
    noise_beta ~ priors[:noise_beta]
    for neuron in 1:n_noise_neurons
        noise_w[neuron] ~ priors[:noise_w][neuron]
        noise_v[neuron] ~ priors[:noise_v][neuron]
    end
    for observation in eachindex(y)
        for neuron in 1:n_noise_neurons
            noise_za[neuron, observation] ~ softdot(
                features[observation], noise_w[neuron], noise_tau,
            )
            noise_h[neuron, observation] ~ ResidualSine(
                noise_za[neuron, observation],
            ) where {
                dependencies = noise_activation_deps,
                meta = activation,
            }
            noise_c[neuron, observation] ~ softdot(
                noise_v[neuron], noise_h[neuron, observation], noise_tau_c,
            )
        end
        noise_c[n_noise_neurons + 1, observation] ~ NormalMeanPrecision(
            noise_intercept, 1e12,
        )
        noise_score[observation] ~ ManyPlus(inputs = [
            noise_c[input, observation] for input in 1:(n_noise_neurons + 1)
        ])
        gamma[observation] ~ GammaShapeRate(kappa, noise_beta)
        gamma[observation] ~ Exp(noise_score[observation]) where {
            dependencies = exp_deps,
        }
        y[observation] ~ NormalMeanPrecision(
            fixed_mean[observation], gamma[observation],
        )
    end
end

function run_exp_fixed_batch(
    priors,
    observations,
    fixed_mean,
    features,
    config,
    spec;
    cold_start,
)
    noise_activation_deps = activation_dependencies(
        config; alpha = spec.link_alpha, max_step = spec.link_max_step,
    )
    exp_deps = exp_link_dependencies(
        config, spec.link_alpha, spec.link_max_step,
    )
    noise_inits = exp_noise_pushforward_inits(
        priors, features, spec.n_noise_neurons, config; cold_start,
    )
    local result
    elapsed = @elapsed result = infer(
        model = hetero_fixed_mean_exp_model(
            n_noise_neurons = spec.n_noise_neurons,
            kappa = spec.kappa,
            priors = priors,
            activation = activation_meta(config),
            noise_activation_deps = noise_activation_deps,
            exp_deps = exp_deps,
        ),
        data = (
            y = observations,
            fixed_mean = fixed_mean,
            features = features,
        ),
        constraints = hetero_fixed_mean_constraints(),
        initialization = hetero_fixed_mean_initialization(
            priors, noise_inits,
        ),
        returnvars = (
            noise_w = KeepLast(),
            noise_v = KeepLast(),
            noise_tau = KeepLast(),
            noise_tau_c = KeepLast(),
            noise_intercept = KeepLast(),
            noise_beta = KeepLast(),
        ),
        iterations = config.max_batch_iterations,
        free_energy = true,
        callbacks = (after_iteration = make_delayed_stopper(config),),
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    all(isfinite, result.free_energy) ||
        error("Exp-link batch has non-finite free energy")
    updated = hierarchy_posterior_priors(
        result, priors; include_mean = false,
    )
    validate_noise_priors(updated)
    expected_states = 2length(observations)
    length(exp_deps.states) == expected_states || error(
        "Exp state count $(length(exp_deps.states)) != $expected_states",
    )
    return (
        priors = updated,
        iterations = length(result.free_energy),
        elapsed_seconds = elapsed,
        free_energy = Float64.(result.free_energy),
        link_states = length(exp_deps.states),
    )
end

function run_exp_fixed_training(
    train,
    features,
    fixed_means,
    batches,
    initial_priors,
    config,
    spec,
)
    carried = deepcopy(initial_priors)
    reports = NamedTuple[]
    for (batch, indices) in enumerate(batches)
        fitted = run_exp_fixed_batch(
            carried,
            train.y[indices],
            fixed_means[indices],
            features[indices],
            config,
            spec;
            cold_start = batch == 1,
        )
        carried = fitted.priors
        push!(reports, (
            batch = batch,
            observations = length(indices),
            iterations = fitted.iterations,
            elapsed_seconds = fitted.elapsed_seconds,
            free_energy = fitted.free_energy,
            final_free_energy = last(fitted.free_energy),
            link_states = fitted.link_states,
        ))
    end
    return (; priors = carried, reports)
end

@model function hetero_noise_exp_prediction_model(
    n_noise_neurons,
    kappa,
    features,
    priors,
    activation,
    activation_deps,
    exp_deps,
)
    local noise_w, noise_v, noise_za, noise_h, noise_c, noise_score, gamma

    noise_tau ~ priors[:noise_tau]
    noise_tau_c ~ priors[:noise_tau_c]
    noise_intercept ~ priors[:noise_intercept]
    noise_beta ~ priors[:noise_beta]
    for neuron in 1:n_noise_neurons
        noise_w[neuron] ~ priors[:noise_w][neuron]
        noise_v[neuron] ~ priors[:noise_v][neuron]
    end
    for observation in eachindex(features)
        for neuron in 1:n_noise_neurons
            noise_za[neuron, observation] ~ softdot(
                features[observation], noise_w[neuron], noise_tau,
            )
            noise_h[neuron, observation] ~ ResidualSine(
                noise_za[neuron, observation],
            ) where {
                dependencies = activation_deps,
                meta = activation,
            }
            noise_c[neuron, observation] ~ softdot(
                noise_v[neuron], noise_h[neuron, observation], noise_tau_c,
            )
        end
        noise_c[n_noise_neurons + 1, observation] ~ NormalMeanPrecision(
            noise_intercept, 1e12,
        )
        noise_score[observation] ~ ManyPlus(inputs = [
            noise_c[input, observation] for input in 1:(n_noise_neurons + 1)
        ])
        gamma[observation] ~ GammaShapeRate(kappa, noise_beta)
        gamma[observation] ~ Exp(noise_score[observation]) where {
            dependencies = exp_deps,
        }
    end
end

function predict_exp_noise_batch(priors, features, config, spec)
    isempty(features) && return Any[]
    exp_deps = exp_link_dependencies(
        config, spec.link_alpha, spec.link_max_step,
    )
    activation_deps = activation_dependencies(
        config; alpha = spec.link_alpha, max_step = spec.link_max_step,
    )
    inits = exp_noise_pushforward_inits(
        priors, features, spec.n_noise_neurons, config; cold_start = false,
    )
    result = infer(
        model = hetero_noise_exp_prediction_model(
            n_noise_neurons = spec.n_noise_neurons,
            kappa = spec.kappa,
            priors = priors,
            activation = activation_meta(config),
            activation_deps = activation_deps,
            exp_deps = exp_deps,
        ),
        data = (features = features,),
        constraints = hetero_noise_prediction_constraints(priors),
        initialization = hetero_noise_prediction_initialization(priors, inits),
        returnvars = (gamma = KeepLast(),),
        iterations = max(1, config.prediction_iterations),
        free_energy = false,
        showprogress = false,
        options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true,
    )
    length(exp_deps.states) == 2length(features) ||
        error("Exp prediction state count is wrong")
    return collect(vec(result.posteriors[:gamma]))
end

function predict_exp_noise(priors, features, config, spec)
    marginals = Vector{Any}(undef, length(features))
    for first_index in 1:config.prediction_batch_size:length(features)
        indices = first_index:min(
            first_index + config.prediction_batch_size - 1,
            length(features),
        )
        marginals[indices] = predict_exp_noise_batch(
            priors, features[indices], config, spec,
        )
    end
    precision_mean = Float64.(mean.(marginals))
    aleatoric_variance = gamma_inverse_mean.(marginals)
    all(value -> isfinite(value) && value > 0, precision_mean) ||
        error("Exp prediction contains invalid precision means")
    return (; precision_mean, aleatoric_variance, marginals)
end

function exp_hierarchy_prediction(priors, features, config, spec)
    mean_prediction = predict_mean(priors, features, config)
    noise_prediction = predict_exp_noise(priors, features, config, spec)
    return (
        mean = mean_prediction.mean,
        epistemic_variance = mean_prediction.epistemic_variance,
        aleatoric_variance = noise_prediction.aleatoric_variance,
        total_variance = mean_prediction.epistemic_variance .+
                         noise_prediction.aleatoric_variance,
        precision_mean = noise_prediction.precision_mean,
    )
end

function exp_result_row(config; kwargs...)
    base = (
        stage = String(config.stage),
        link = "exp",
        task = "",
        seed = 0,
        model = "",
        attempt = 0,
        status = "failure",
        error = "",
        n_samples = config.n_samples,
        n_train = 0,
        n_validation = 0,
        n_test = 0,
        n_batches = config.n_training_batches,
        total_iterations = 0,
        elapsed_seconds = NaN,
        final_free_energy = NaN,
        initial_precision = NaN,
        initial_log_precision = NaN,
        final_precision_mean = NaN,
        mean_alpha = NaN,
        link_alpha = NaN,
        link_max_step = NaN,
        batch_iterations = "",
        batch_seconds = "",
        mean_coefficients = "",
        mean_intercept_mean = NaN,
        noise_coefficients = "",
        noise_intercept_mean = NaN,
    )
    return merge(base, empty_metrics(), (; kwargs...))
end

function persist_exp_rows(rows, config)
    config.save_outputs || return nothing
    ensure_output_directory(config)
    CSV.write(joinpath(config.output_dir, "attempts.csv"), DataFrame(rows))
    return nothing
end

function record_exp_row!(rows, row, config)
    push!(rows, row)
    persist_exp_rows(rows, config)
    return row
end

function exp_retry_specs(config)
    raw = (
        (config.squareplus_alpha, config.squareplus_max_step),
        (min(config.squareplus_alpha, 0.02), 0.5),
        (min(config.squareplus_alpha, 0.005), 0.1),
        (min(config.squareplus_alpha, 0.001), 0.05),
    )
    unique_settings = NamedTuple[]
    for (alpha, max_step) in raw
        setting = (
            n_noise_neurons = config.full_noise_neurons,
            kappa = config.full_kappa,
            link_alpha = alpha,
            link_max_step = max_step,
        )
        setting in unique_settings || push!(unique_settings, setting)
    end
    return unique_settings
end

function fit_exp_baseline!(
    rows,
    task,
    seed,
    split,
    train,
    train_features,
    evaluation,
    evaluation_features,
    batches,
    config,
)
    for (attempt, alpha) in enumerate(unique([
        config.mean_ngmp_alpha,
        min(config.mean_ngmp_alpha, 0.02),
        min(config.mean_ngmp_alpha, 0.005),
    ]))
        try
            fitted = run_baseline_training(
                train, train_features, batches, config; alpha,
            )
            prediction = baseline_prediction(
                fitted.priors, evaluation_features, config,
            )
            metrics = evaluate_prediction(prediction, evaluation, config)
            summary = batch_summary(fitted.reports)
            record_exp_row!(rows, exp_result_row(
                config;
                task = String(task),
                seed,
                model = "baseline",
                attempt,
                status = "success",
                split_metadata(split)...,
                summary...,
                metrics...,
                initial_precision = mean(make_mean_priors(config)[:obs_noise]),
                final_precision_mean = mean(fitted.priors[:obs_noise]),
                mean_alpha = alpha,
                mean_coefficients = compact_means(fitted.priors[:v]),
                mean_intercept_mean = mean(
                    fitted.priors[:mean_intercept],
                ),
            ), config)
            println(
                "baseline task=$task seed=$seed MSE=",
                round(metrics.mean_mse; digits = 5),
                " NLL=", round(metrics.nll; digits = 5),
            )
            return (; fitted, prediction)
        catch exception
            message = concise_error(exception)
            record_exp_row!(rows, exp_result_row(
                config;
                task = String(task),
                seed,
                model = "baseline",
                attempt,
                status = "failure",
                error = message,
                split_metadata(split)...,
                mean_alpha = alpha,
            ), config)
            println("baseline retry $attempt failed: $message")
        end
    end
    return nothing
end

function fit_exp_hierarchy!(
    rows,
    task,
    seed,
    split,
    baseline,
    train,
    train_features,
    fixed_means,
    evaluation,
    evaluation_features,
    batches,
    config,
)
    baseline_obs_noise = baseline.fitted.priors[:obs_noise]
    baseline_precision = mean(baseline_obs_noise)
    for (attempt, spec) in enumerate(exp_retry_specs(config))
        noise_priors = make_exp_noise_priors(
            config,
            spec.n_noise_neurons,
            spec.kappa,
            baseline_precision,
        )
        initial_priors = merge_hierarchy_priors(
            baseline.fitted.priors,
            noise_priors,
            baseline_obs_noise,
        )
        try
            fitted = run_exp_fixed_training(
                train,
                train_features,
                fixed_means,
                batches,
                initial_priors,
                config,
                spec,
            )
            prediction = exp_hierarchy_prediction(
                fitted.priors, evaluation_features, config, spec,
            )
            metrics = evaluate_prediction(prediction, evaluation, config)
            summary = batch_summary(fitted.reports)
            row = exp_result_row(
                config;
                task = String(task),
                seed,
                model = "fixed_mean_exp",
                attempt,
                status = "success",
                split_metadata(split)...,
                summary...,
                metrics...,
                initial_precision = baseline_precision,
                initial_log_precision = log(baseline_precision),
                final_precision_mean = mean(prediction.precision_mean),
                mean_alpha = config.mean_ngmp_alpha,
                link_alpha = spec.link_alpha,
                link_max_step = spec.link_max_step,
                mean_coefficients = compact_means(fitted.priors[:v]),
                mean_intercept_mean = mean(
                    fitted.priors[:mean_intercept],
                ),
                noise_coefficients = compact_means(fitted.priors[:noise_v]),
                noise_intercept_mean = mean(fitted.priors[:noise_intercept]),
            )
            record_exp_row!(rows, row, config)
            println(
                "fixed_mean/exp task=$task seed=$seed attempt=$attempt ",
                "MSE=$(round(metrics.mean_mse; digits=5)) ",
                "NLL=$(round(metrics.nll; digits=5)) ",
                "noise-corr=$(round(metrics.aleatoric_correlation; digits=4))",
            )
            return (; fitted, prediction, spec, row)
        catch exception
            message = concise_error(exception)
            record_exp_row!(rows, exp_result_row(
                config;
                task = String(task),
                seed,
                model = "fixed_mean_exp",
                attempt,
                status = "failure",
                error = message,
                split_metadata(split)...,
                initial_precision = baseline_precision,
                initial_log_precision = log(baseline_precision),
                mean_alpha = config.mean_ngmp_alpha,
                link_alpha = spec.link_alpha,
                link_max_step = spec.link_max_step,
            ), config)
            println(
                "fixed_mean/exp retry $attempt failed for $task/$seed: ",
                message,
            )
        end
    end
    return nothing
end

function save_exp_diagnostics(fitted, task, seed, config, spec)
    (config.save_outputs || config.save_plots) || return nothing
    ensure_output_directory(config)
    axes, grid = grid_data(task, config)
    prediction = exp_hierarchy_prediction(
        fitted.priors, build_features(grid), config, spec,
    )
    prefix = join((
        String(config.stage),
        String(task),
        string(seed),
        "fixed_mean_exp",
        "n$(spec.n_noise_neurons)",
        "k$(replace(string(spec.kappa), '.' => '_'))",
    ), "_")
    if config.save_outputs
        CSV.write(joinpath(config.output_dir, prefix * "_grid.csv"), DataFrame(
            x1 = grid.x1,
            x2 = grid.x2,
            true_mean = grid.clean_mean,
            true_aleatoric_variance = grid.true_variance,
            predictive_mean = prediction.mean,
            epistemic_variance = prediction.epistemic_variance,
            aleatoric_variance = prediction.aleatoric_variance,
            total_variance = prediction.total_variance,
        ))
    end
    if config.save_plots
        maximum_total_variance = maximum(vcat(
            prediction.epistemic_variance,
            prediction.total_variance,
        ))
        maximum_aleatoric_variance = maximum(vcat(
            prediction.aleatoric_variance,
            grid.true_variance,
        ))
        mean_radius = max(0.5, maximum(abs.(prediction.mean .- 0.5)))
        panels = [
            heatmap_panel(
                axes, prediction.mean, config, "predictive mean";
                color = :balance,
                clims = (0.5 - mean_radius, 0.5 + mean_radius),
            ),
            heatmap_panel(
                axes,
                prediction.epistemic_variance,
                config,
                "epistemic variance";
                clims = (0, maximum_total_variance),
            ),
            heatmap_panel(
                axes,
                prediction.aleatoric_variance,
                config,
                "learned aleatoric variance (Exp)";
                clims = (0, maximum_aleatoric_variance),
            ),
            heatmap_panel(
                axes,
                prediction.total_variance,
                config,
                "total predictive variance";
                clims = (0, maximum_total_variance),
            ),
            heatmap_panel(
                axes,
                grid.true_variance,
                config,
                "true aleatoric variance";
                clims = (0, maximum_aleatoric_variance),
            ),
        ]
        savefig(
            plot(panels...; layout = (1, 5), size = (1900, 380)),
            joinpath(config.output_dir, prefix * "_uncertainty.png"),
        )
        save_free_energy_plot(
            fitted.reports,
            joinpath(config.output_dir, prefix * "_free_energy.png"),
        )
    end
    return prediction
end

function write_exp_note(config)
    config.save_outputs || return nothing
    ensure_output_directory(config)
    open(joinpath(config.output_dir, "README.md"), "w") do io
        println(io, "# Exp/log-precision heteroscedastic ablation")
        println(io)
        println(io, "This is the fixed-mean staged model selected from the Squareplus experiment. The only architectural change is `gamma = Exp(log_precision)` with the cold-start intercept `log(mean(q(obs_noise)))`. The shared Gamma hierarchy, κ, priors, initialization variances, batches, and residual-sine head are unchanged.")
        println(io)
        println(io, "The `link_alpha` columns in `attempts.csv` report Exp-node NGMP damping. The experiment is designed to reveal whether changing link scale alone is sufficient to unlock aleatoric learning.")
    end
    return nothing
end

function run_exp_link_study(config = exp_link_config())
    write_exp_note(config)
    rows = NamedTuple[]
    evaluation_split = config.stage === :full ? :test : :validation
    for seed in config.seeds
        paired = make_paired_datasets(config, seed)
        split = deterministic_split(
            config.n_samples, config, config.split_seed + seed,
        )
        for task in config.tasks
            println("\n=== Exp link: task=$task seed=$seed ===")
            data = paired[task]
            train = subset_data(data, split.train)
            evaluation = subset_data(
                data, getproperty(split, evaluation_split),
            )
            train_features = build_features(train)
            evaluation_features = build_features(evaluation)
            batches = deterministic_batch_ranges(
                length(train.y), config.n_training_batches,
            )
            baseline = fit_exp_baseline!(
                rows,
                task,
                seed,
                split,
                train,
                train_features,
                evaluation,
                evaluation_features,
                batches,
                config,
            )
            isnothing(baseline) && continue
            fixed_means = try
                predict_mean(
                    baseline.fitted.priors, train_features, config,
                ).mean
            catch exception
                println(
                    "training-point prediction failed: ",
                    concise_error(exception),
                )
                continue
            end
            fitted = fit_exp_hierarchy!(
                rows,
                task,
                seed,
                split,
                baseline,
                train,
                train_features,
                fixed_means,
                evaluation,
                evaluation_features,
                batches,
                config,
            )
            isnothing(fitted) && continue
            try
                save_exp_diagnostics(
                    fitted.fitted,
                    task,
                    seed,
                    config,
                    fitted.spec,
                )
            catch exception
                println(
                    "Exp diagnostic plot failed: ",
                    concise_error(exception),
                )
            end
        end
    end
    persist_exp_rows(rows, config)
    successes = count(
        row -> row.status == "success" &&
               row.model == "fixed_mean_exp",
        rows,
    )
    failures = count(row -> row.status == "failure", rows)
    println("\nExp-link completed: $successes successful fits, $failures failed attempts")
    successes > 0 || error("no Exp-link condition completed successfully")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_exp_link_study()
end
