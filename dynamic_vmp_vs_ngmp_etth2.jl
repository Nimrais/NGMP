using ClosedFormExpectations
using Distributions
using ExponentialFamily
using ExponentialFamilyProjection
using JLD2
using LinearAlgebra: dot
using Printf
using ProbabilisticEnsembling
using RxInfer
using Statistics
using SurrogateModelling
using YAML

const ROOT = @__DIR__
const SUPPORTED_HORIZONS = (96, 192, 336, 720)
const SUPPORTED_ARMS = (:vmp, :ngmp, :relaxed)
const ETTH2_COMPARISON_CONFIG = (
    training_observations=0, # 0 uses the complete validation split
    training_batch_size=nothing, # set to `nothing` for full-graph training
    repeat_batch=1, # resample once per variational iteration
    inference_iterations=20,
    prediction_batch_size=250,
    prediction_iterations=3,
    prediction_method=:rxinfer_fixed_marginals,
    alpha=0.2,
    beta=0.0,
    arms=SUPPORTED_ARMS,
    limit_stack_depth=500,
    show_progress=true,
)

@model function dynamic_vmp(n_forecasters, n_obs, y, features, predictions, priors)
    local w, z, gamma, tau, beta
    for i in 1:n_forecasters
        w[i] ~ priors[:w][i]
        tau[i] ~ priors[:τ][i]
        beta[i] ~ priors[:β][i]
    end
    for j in 1:n_obs, i in 1:n_forecasters
        z[i, j] ~ softdot(features[j], w[i], tau[i]) where { meta=LowRankMeta() }
        gamma[i, j] ~ GammaShapeRate(1.0, beta[i])
        z[i, j] ~ Log(gamma[i, j])
        y[j] ~ NormalMeanPrecision(predictions[i, j], gamma[i, j])
    end
end

@constraints function dynamic_vmp_constraints()
    q(w, z, gamma, tau, beta) = q(w)q(z, gamma)q(tau)q(beta)
    q(w)::MomentForm()
    q(z)::ProjectedTo(
        NormalMeanVariance,
        parameters=ProjectionParameters(strategy=ClosedFormStrategy()),
    )
    q(gamma)::ProjectedTo(
        Gamma,
        parameters=ProjectionParameters(strategy=ClosedFormStrategy()),
    )
end

@model function dynamic_ngmp(
    n_forecasters,
    n_obs,
    y,
    features,
    predictions,
    priors,
    dependencies,
    damping,
)
    local w, z, gamma, tau, beta
    for i in 1:n_forecasters
        w[i] ~ priors[:w][i]
        tau[i] ~ priors[:τ][i]
        beta[i] ~ priors[:β][i]
    end
    for j in 1:n_obs, i in 1:n_forecasters
        z[i, j] ~ softdot(features[j], w[i], tau[i]) where { meta=LowRankMeta() }
        gamma[i, j] ~ GammaShapeRate(1.0, beta[i])
        z[i, j] ~ Log(gamma[i, j]) where {
            dependencies=dependencies,
            meta=damping,
        }
        y[j] ~ NormalMeanPrecision(predictions[i, j], gamma[i, j])
    end
end

@constraints function dynamic_ngmp_constraints()
    q(w, z, gamma, tau, beta) = q(w)q(z, gamma)q(tau)q(beta)
    q(w)::MomentForm()
end

@model function dynamic_ngmp_relaxed(
    n_forecasters,
    n_obs,
    y,
    features,
    predictions,
    priors,
    dependencies,
    damping,
)
    local w, z, gamma, tau, beta
    for i in 1:n_forecasters
        w[i] ~ priors[:w][i]
        tau[i] ~ priors[:τ][i]
        beta[i] ~ priors[:β][i]
    end
    for j in 1:n_obs, i in 1:n_forecasters
        z[i, j] ~ softdot(features[j], w[i], tau[i])
        gamma[i, j] ~ GammaShapeRate(1.0, beta[i])
        z[i, j] ~ Log(gamma[i, j]) where {
            dependencies=dependencies,
            meta=damping,
        }
        y[j] ~ NormalMeanPrecision(predictions[i, j], gamma[i, j])
    end
end

@constraints function dynamic_relaxed_constraints()
    q(w, z, gamma, tau, beta) = q(w, z, gamma)q(tau)q(beta)
    q(w)::MomentForm()
end

@initialization function dynamic_initialization(priors)
    q(w) = deepcopy(priors[:w])
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(gamma) = GammaShapeScale(1.0, 1.0)
    q(tau) = priors[:τ]
    q(beta) = priors[:β]
end

@constraints function dynamic_prediction_constraints(priors)
    q(w, z, gamma, tau, beta) = q(w)q(z, gamma)q(tau)q(beta)
    q(z)::ProjectedTo(
        NormalMeanVariance,
        parameters=ProjectionParameters(strategy=ClosedFormStrategy()),
    )
    q(gamma)::ProjectedTo(
        Gamma,
        parameters=ProjectionParameters(strategy=ClosedFormStrategy()),
    )
    # Match PrecisionGatedExperts prediction: learned global marginals are
    # fixed, while z, gamma, and the missing target y are inferred per batch.
    for (i, prior) in enumerate(deepcopy(priors[:w]))
        q(w[i])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    for (i, prior) in enumerate(priors[:τ])
        q(tau[i])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    for (i, prior) in enumerate(priors[:β])
        q(beta[i])::RxInfer.FixedMarginalFormConstraint(prior)
    end
end

function dynamic_predict(
    features,
    predictions,
    w,
    tau,
    beta;
    iterations,
    limit_stack_depth,
)
    n_forecasters, n_obs = size(predictions)
    priors = Dict{Symbol,Any}(
        :w => deepcopy(w),
        :τ => deepcopy(tau),
        :β => deepcopy(beta),
    )
    result = infer(
        model=dynamic_vmp(; n_forecasters, n_obs, priors),
        data=(
            y=fill(missing, n_obs),
            features=features,
            predictions=predictions,
        ),
        constraints=dynamic_prediction_constraints(priors),
        initialization=dynamic_initialization(priors),
        iterations=iterations,
        free_energy=false,
        options=(limit_stack_depth=limit_stack_depth,),
    )
    predictive = last(result.predictions[:y])
    return mean.(predictive), std.(predictive)
end

function dynamic_predict_batched(
    features,
    predictions,
    w,
    tau,
    beta;
    iterations,
    limit_stack_depth,
    batch_size,
)
    n_obs = size(predictions, 2)
    n_obs > 0 || throw(ArgumentError("prediction data must not be empty"))
    batch_size > 0 || throw(ArgumentError("prediction batch size must be positive"))
    predictive_mean = Vector{Float64}(undef, n_obs)
    predictive_std = Vector{Float64}(undef, n_obs)
    for batch_start in 1:batch_size:n_obs
        batch_end = min(batch_start + batch_size - 1, n_obs)
        batch_range = batch_start:batch_end
        batch_mean, batch_std = dynamic_predict(
            features[batch_range],
            predictions[:, batch_range],
            w,
            tau,
            beta;
            iterations,
            limit_stack_depth,
        )
        predictive_mean[batch_range] = batch_mean
        predictive_std[batch_range] = batch_std
    end
    return predictive_mean, predictive_std
end

function mean_with_confidence95(values)
    value = mean(values)
    half_width = length(values) > 1 ? 1.959963984540054 * std(values) / sqrt(length(values)) : 0.0
    return (
        mean=value,
        half_width=half_width,
        lower=value - half_width,
        upper=value + half_width,
    )
end

function predictive_metrics(predicted_mean, predicted_std, target)
    squared_error = (predicted_mean .- target) .^ 2
    negative_log_likelihood = [
        -logpdf(Normal(predicted_mean[j], predicted_std[j]), target[j]) for
        j in eachindex(target)
    ]
    mse_summary = mean_with_confidence95(squared_error)
    nll_summary = mean_with_confidence95(negative_log_likelihood)
    z95 = 1.959963984540054
    coverage95 = mean(
        (target .>= predicted_mean .- z95 .* predicted_std) .&
        (target .<= predicted_mean .+ z95 .* predicted_std),
    )
    pinball = Float64[]
    for q in (0.1, 0.9)
        estimate = predicted_mean .+ quantile(Normal(), q) .* predicted_std
        push!(pinball, mean(max.(q .* (target .- estimate), (q - 1) .* (target .- estimate))))
    end
    return (
        mae=mean(abs.(predicted_mean .- target)),
        mse=mse_summary.mean,
        mse_ci95=mse_summary.half_width,
        mse_ci95_lower=mse_summary.lower,
        mse_ci95_upper=mse_summary.upper,
        negative_log_likelihood=nll_summary.mean,
        negative_log_likelihood_ci95=nll_summary.half_width,
        negative_log_likelihood_ci95_lower=nll_summary.lower,
        negative_log_likelihood_ci95_upper=nll_summary.upper,
        coverage95=coverage95,
        pinball=mean(pinball),
    )
end

function required_paths(session)
    raw = YAML.load_file(session)
    params = raw["params"]
    relative_paths = String[params["dataset_path"]]
    append!(relative_paths, String.(params["experts"]))
    push!(relative_paths, "models/ETTh2_s96_VAE_enzyme.jld2")
    return [joinpath(ROOT, path) for path in relative_paths]
end

function verify_prerequisites(session)
    isfile(session) || error("Session file not found: $session")
    raw = YAML.load_file(session)
    params = raw["params"]
    get(params, "dataset", nothing) == "ETTh2" ||
        error("Session must configure dataset ETTh2: $session")
    horizon = Int(params["horizon"])
    horizon in SUPPORTED_HORIZONS || throw(ArgumentError(
        "unsupported horizon $horizon; choose one of $(collect(SUPPORTED_HORIZONS))",
    ))
    paths = required_paths(session)
    missing = filter(path -> !isfile(path), paths)
    isempty(missing) || error("Missing ETTh2 prerequisites:\n" * join(missing, "\n"))
    return raw
end

function prepare_data(session; rebuild_cache=false)
    raw = verify_prerequisites(session)
    horizon = Int(raw["params"]["horizon"])
    cache_dir = joinpath(ROOT, "cache")
    mkpath(cache_dir)
    cache_path = joinpath(cache_dir, "dynamic_etth2_h$(horizon)_cache.jld2")
    if rebuild_cache || !isfile(cache_path)
        spec = ProbabilisticEnsembling._parse_spec(raw)
        prepared = cd(() -> ProbabilisticEnsembling.before_rxinfer(spec), ROOT)
        jldsave(
            cache_path;
            y_val=prepared[1],
            y_test=prepared[2],
            predictions_val=prepared[3],
            predictions_test=prepared[4],
            features_val=prepared[5],
            features_test=prepared[6],
        )
    end
    spec = ProbabilisticEnsembling._parse_spec(raw)
    return spec, raw, load(cache_path), cache_path
end

function run_arm(
    arm,
    priors,
    features,
    predictions,
    targets;
    iterations,
    training_batch_size,
    repeat_batch,
    alpha,
    beta,
    limit_stack_depth,
    showprogress,
)
    n_forecasters = size(predictions, 1)
    training_observations = length(targets)
    training_observations > 0 || throw(ArgumentError("training data must not be empty"))
    repeat_batch > 0 || throw(ArgumentError("repeat_batch must be positive"))
    n_obs, common_data = if isnothing(training_batch_size)
        training_observations, (y=targets, features=features, predictions=predictions)
    else
        training_batch_size > 0 ||
            throw(ArgumentError("training batch size must be positive or nothing"))
        batch_size = min(training_batch_size, training_observations)
        # The wrappers use identical deterministic RNG seeds, keeping targets,
        # features, and expert predictions synchronized while resampling batches.
        batch_data = (
            y=ProbabilisticEnsembling.SubsampledData(targets, batch_size, repeat_batch),
            features=ProbabilisticEnsembling.SubsampledData(features, batch_size, repeat_batch),
            predictions=ProbabilisticEnsembling.SubsampledData(
                predictions,
                batch_size,
                repeat_batch,
            ),
        )
        batch_size, batch_data
    end
    common_options = (limit_stack_depth=limit_stack_depth,)

    if arm === :vmp
        result = infer(
            model=dynamic_vmp(; n_forecasters, n_obs, priors),
            data=common_data,
            constraints=dynamic_vmp_constraints(),
            initialization=dynamic_initialization(priors),
            iterations=iterations,
            options=common_options,
            showprogress=showprogress,
        )
        return result, nothing
    end

    dependencies = NGMPDependencies(out=nothing, in=nothing)
    damping = DampingMeta(; alpha, beta)
    if arm === :ngmp
        result = infer(
            model=dynamic_ngmp(; n_forecasters, n_obs, priors, dependencies, damping),
            data=common_data,
            constraints=dynamic_ngmp_constraints(),
            initialization=dynamic_initialization(priors),
            iterations=iterations,
            options=common_options,
            showprogress=showprogress,
        )
    elseif arm === :relaxed
        result = infer(
            model=dynamic_ngmp_relaxed(;
                n_forecasters,
                n_obs,
                priors,
                dependencies,
                damping,
            ),
            data=common_data,
            constraints=dynamic_relaxed_constraints(),
            initialization=dynamic_initialization(priors),
            iterations=iterations,
            options=common_options,
            showprogress=showprogress,
        )
    else
        throw(ArgumentError("unsupported arm $arm"))
    end
    @assert length(dependencies.states) == 2 * n_forecasters * n_obs
    return result, dependencies
end

function write_report(path, config, metrics)
    open(path, "w") do io
        println(io, "# Dynamic VMP vs NGMP on ETTh2")
        println(io)
        println(io, "Horizon: **$(config.horizon)**  ")
        println(io, "Training observations: **$(config.n_obs)**  ")
        training_batch_label = isnothing(config.training_batch_size) ? "full graph" :
                               string(config.training_batch_size)
        println(io, "Training batch size: **$training_batch_label**  ")
        println(io, "Inference iterations: **$(config.iterations)**  ")
        if !isnothing(config.training_batch_size)
            println(io, "Batch reuse iterations: **$(config.repeat_batch)**  ")
        end
        println(io, "Prediction batch size: **$(config.prediction_batch_size)**  ")
        println(io, "Prediction iterations: **$(config.prediction_iterations)**  ")
        println(io, "NGMP damping: alpha=$(config.alpha), beta=$(config.beta)")
        println(io, "Prediction method: **$(config.prediction_method)**")
        println(io, "Confidence intervals: mean ± 1.96 × standard error across test observations")
        println(io, "Session: `$(config.session)`")
        println(io)
        println(io, "| Method | MAE | MSE ± 95% CI | NLL ± 95% CI | Coverage 95% | Pinball |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        labels = Dict(:vmp => "VMP", :ngmp => "NGMP", :relaxed => "NGMP relaxed")
        for arm in config.arms
            value = metrics[arm]
            println(
                io,
                "| $(labels[arm]) | $(@sprintf("%.4f", value.mae)) | " *
                "$(@sprintf("%.4f", value.mse)) ± $(@sprintf("%.4f", value.mse_ci95)) | " *
                "$(@sprintf("%.4f", value.negative_log_likelihood)) ± " *
                "$(@sprintf("%.4f", value.negative_log_likelihood_ci95)) | " *
                "$(@sprintf("%.4f", value.coverage95)) | $(@sprintf("%.4f", value.pinball)) |",
            )
        end
    end
end

function run_etth2_comparison(
    ;
    session,
    prepare_only=false,
    rebuild_cache=false,
)
    session_path = isabspath(session) ? normpath(session) : normpath(joinpath(ROOT, session))
    spec, raw, cache, cache_path = prepare_data(session_path; rebuild_cache)
    println("prepared_cache=$cache_path")
    prepare_only && return (cache_path=cache_path,)

    comparison = ETTH2_COMPARISON_CONFIG
    horizon = Int(raw["params"]["horizon"])
    requested_n_obs = comparison.training_observations
    configured_training_batch_size = comparison.training_batch_size
    repeat_batch = comparison.repeat_batch
    iterations = comparison.inference_iterations
    configured_prediction_batch_size = comparison.prediction_batch_size
    prediction_iterations = comparison.prediction_iterations
    prediction_method = comparison.prediction_method
    alpha = comparison.alpha
    beta = comparison.beta
    selected_arms = collect(comparison.arms)
    limit_stack_depth = comparison.limit_stack_depth
    showprogress = comparison.show_progress

    y_val = cache["y_val"]
    y_test = cache["y_test"]
    predictions_val = cache["predictions_val"]
    predictions_test = cache["predictions_test"]
    features_val = cache["features_val"]
    features_test = cache["features_test"]
    train_count = requested_n_obs == 0 ? length(y_val) : min(requested_n_obs, length(y_val))
    training_batch_size = isnothing(configured_training_batch_size) ? nothing :
                          min(configured_training_batch_size, train_count)
    prediction_batch_size = min(configured_prediction_batch_size, length(y_test))
    all(arm -> arm in SUPPORTED_ARMS, selected_arms) ||
        throw(ArgumentError("arms must be selected from $(collect(SUPPORTED_ARMS))"))

    config = (;
        session=relpath(session_path, ROOT),
        horizon,
        n_obs=train_count,
        training_batch_size,
        repeat_batch,
        iterations,
        prediction_batch_size,
        prediction_iterations,
        prediction_method,
        alpha,
        beta,
        arms=selected_arms,
        limit_stack_depth,
        showprogress,
    )
    metrics = Dict{Symbol,Any}()
    predictions = Dict{Symbol,Any}()
    posteriors = Dict{Symbol,Any}()
    for arm in selected_arms
        batch_label = isnothing(training_batch_size) ? "full" : string(training_batch_size)
        println("running arm=$arm horizon=$horizon observations=$train_count batch=$batch_label repeat_batch=$repeat_batch iterations=$iterations")
        result, _ = run_arm(
            arm,
            spec.priors,
            features_val[1:train_count],
            predictions_val[:, 1:train_count],
            y_val[1:train_count];
            iterations,
            training_batch_size,
            repeat_batch,
            alpha,
            beta,
            limit_stack_depth,
            showprogress,
        )
        w = last(result.posteriors[:w])
        tau = last(result.posteriors[:tau])
        beta_posterior = last(result.posteriors[:beta])
        predicted_mean, predicted_std = dynamic_predict_batched(
            features_test,
            predictions_test,
            w,
            tau,
            beta_posterior;
            iterations=prediction_iterations,
            limit_stack_depth,
            batch_size=prediction_batch_size,
        )
        metrics[arm] = predictive_metrics(predicted_mean, predicted_std, y_test)
        predictions[arm] = (mean=predicted_mean, std=predicted_std)
        posteriors[arm] = (w=w, tau=tau, beta=beta_posterior)
        println("arm=$arm metrics=$(metrics[arm])")
    end

    results_dir = joinpath(ROOT, "results")
    mkpath(results_dir)
    batch_token = isnothing(training_batch_size) ? "full" : string(training_batch_size)
    stem = "dynamic_etth2_h$(horizon)_n$(train_count)_b$(batch_token)_i$(iterations)_pb$(prediction_batch_size)_prx"
    jld2_path = joinpath(results_dir, "$stem.jld2")
    markdown_path = joinpath(results_dir, "$stem.md")
    jldsave(jld2_path; config, metrics, predictions, posteriors, y_test)
    write_report(markdown_path, config, metrics)
    println("results_jld2=$jld2_path")
    println("results_markdown=$markdown_path")
    return (; config, metrics, predictions, posteriors, jld2_path, markdown_path)
end

function parse_cli(args)
    options = Dict{String,String}()
    flags = Set{String}()
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument in ("--prepare-only", "--rebuild-cache")
            push!(flags, argument)
            index += 1
        elseif startswith(argument, "--")
            index == length(args) && error("missing value for $argument")
            options[argument] = args[index + 1]
            index += 2
        else
            error("unknown argument $argument")
        end
    end
    allowed_options = Set(["--session", "--horizon"])
    unknown = setdiff(Set(keys(options)), allowed_options)
    isempty(unknown) || error("unknown options: $(join(sort!(collect(unknown)), ", "))")
    horizon_values = split(get(options, "--horizon", "96"), ',')
    horizons = parse.(Int, strip.(horizon_values))
    isempty(horizons) && error("--horizon must contain at least one horizon")
    allunique(horizons) || error("--horizon contains duplicate values")
    unsupported = setdiff(horizons, collect(SUPPORTED_HORIZONS))
    isempty(unsupported) || error(
        "unsupported horizons: $(join(unsupported, ", ")); supported horizons: $(join(SUPPORTED_HORIZONS, ", "))",
    )
    haskey(options, "--session") && length(horizons) > 1 && error(
        "--session can only be combined with a single --horizon value",
    )
    sessions = if haskey(options, "--session")
        [options["--session"]]
    else
        [joinpath("sessions", "dynamic", "vae", "dynamic_ETTh2_$(horizon).yaml") for horizon in horizons]
    end
    return (
        sessions,
        prepare_only="--prepare-only" in flags,
        rebuild_cache="--rebuild-cache" in flags,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    cli = parse_cli(ARGS)
    for session in cli.sessions
        run_etth2_comparison(;
            session,
            prepare_only=cli.prepare_only,
            rebuild_cache=cli.rebuild_cache,
        )
    end
end
