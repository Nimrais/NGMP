#!/usr/bin/env julia

# Direct frozen-Gaussian backend for the UCI deep-kernel sweep.
#
# This includes the RxInfer benchmark only for its dataset protocol, feature
# maps, constants, and result helpers. No RxInfer graph is constructed by the
# training or prediction functions below.
include(joinpath(@__DIR__, "uci_deep_kernel_paper_benchmark.jl"))

const DIRECT_ITERATIONS =
    parse(Int, get(ENV, "UCI_DIRECT_ITERATIONS", "60"))
const DIRECT_TOLERANCE =
    parse(Float64, get(ENV, "UCI_DIRECT_TOLERANCE", "1e-5"))
const DIRECT_JITTER =
    parse(Float64, get(ENV, "UCI_DIRECT_JITTER", "1e-8"))
const DIRECT_PREPROCESSING = Symbol.(split(
    get(
        ENV,
        "UCI_DIRECT_PREPROCESSING",
        "multiscale_matern32_linear,multiscale_rbf_linear",
    ),
    ',',
))
const DIRECT_FEATURE_DIMENSIONS = parse.(Int, split(
    get(ENV, "UCI_DIRECT_FEATURE_DIMENSIONS", "300,400"),
    ',',
))
const DIRECT_VECTOR_TRANSPORT_ALPHA = parse(
    Float64,
    get(ENV, "UCI_DIRECT_VECTOR_TRANSPORT_ALPHA", "0.60"),
)
const DIRECT_VECTOR_TRANSPORT_NESTEROV_ALPHA = parse(
    Float64,
    get(ENV, "UCI_DIRECT_VECTOR_TRANSPORT_NESTEROV_ALPHA", "0.60"),
)

direct_optimizer_alpha(method) =
    method == :vector_transport ? DIRECT_VECTOR_TRANSPORT_ALPHA :
    method == :vector_transport_nesterov ?
    DIRECT_VECTOR_TRANSPORT_NESTEROV_ALPHA :
    ALPHA

function direct_rff_design(
    x_train,
    x_test,
    seed,
    preprocessing,
    feature_dimension,
    fixed_lengthscale = nothing,
)
    preprocessing in (
        :multiscale_matern32_linear,
        :multiscale_rbf_linear,
    ) ||
        throw(ArgumentError(
            "unknown direct UCI preprocessing: $preprocessing",
        ))
    feature_dimension > 0 ||
        throw(ArgumentError("feature dimension must be positive"))
    isnothing(fixed_lengthscale) || fixed_lengthscale > 0 ||
        throw(ArgumentError("fixed lengthscale must be positive"))

    d = size(x_train, 2)
    rng = MersenneTwister(seed)
    sample = x_train[
        rand(rng, 1:size(x_train, 1), min(512, size(x_train, 1))),
        :,
    ]
    distances = [
        norm(sample[index, :] - sample[index - 1, :])
        for index in 2:size(sample, 1)
    ]
    heuristic_lengthscale = max(median(distances), 0.25)
    lengthscale = isnothing(fixed_lengthscale) ?
        heuristic_lengthscale :
        Float64(fixed_lengthscale)
    counts = [
        div(feature_dimension, 3),
        div(feature_dimension, 3),
        feature_dimension - 2 * div(feature_dimension, 3),
    ]
    frequencies = reduce(
        vcat,
        map(zip(counts, (0.5, 1.0, 2.0))) do (count, scale)
            base = randn(rng, count, d)
            if preprocessing == :multiscale_matern32_linear
                # Matérn-3/2 spectral density: multivariate Student-t with
                # three degrees of freedom.
                base .*= reshape(
                    sqrt.(3 ./ rand(rng, Chisq(3), count)),
                    :,
                    1,
                )
            end
            base ./ (scale * lengthscale)
        end,
    )
    phases = 2pi .* rand(rng, feature_dimension)

    function transform(features)
        random = sqrt(2 / feature_dimension) .*
            cos.(features * frequencies' .+ phases')
        return hcat(random, features, ones(size(features, 1)))
    end

    return transform(x_train), transform(x_test)
end

function direct_summary_rows(
    rows,
    datasets;
    optimizer_configs_for_depth = optimizers_for_depth,
)
    summaries = NamedTuple[]
    for dataset in datasets,
        preprocessing in DIRECT_PREPROCESSING,
        feature_dimension in DIRECT_FEATURE_DIMENSIONS,
        depth in DEPTHS,
        (method, beta) in optimizer_configs_for_depth(depth)
        selected = filter(
            row ->
                row.dataset == dataset &&
                    row.preprocessing == preprocessing &&
                    row.feature_dimension == feature_dimension &&
                    row.depth == depth &&
                    row.optimizer == method &&
                    row.beta == beta,
            rows,
        )
        summary = summarize_rows(selected)
        push!(summaries, (;
            dataset,
            preprocessing,
            feature_dimension,
            depth,
            optimizer = method,
            beta,
            successful_splits = summary.successful,
            requested_splits = N_SPLITS,
            mean_logpdf = summary.mean_logpdf,
            std_logpdf = summary.std_logpdf,
            mean_rmse = summary.mean_rmse,
            paper_dvi = getproperty(DATASETS, dataset).paper_dvi,
        ))
    end
    return summaries
end

function direct_gaussian_update(
    Φ,
    targets,
    observation_precision,
    prior_mean,
    prior_precision,
)
    weighted_Φ = Φ .* observation_precision
    posterior_precision =
        prior_precision + Φ' * weighted_Φ +
        DIRECT_JITTER * I
    posterior_weighted_mean =
        prior_precision * prior_mean +
        Φ' * (observation_precision .* targets)
    factorization = cholesky(Symmetric(posterior_precision))
    posterior_mean = factorization \ posterior_weighted_mean
    posterior_covariance = Matrix(inv(factorization))
    return posterior_mean, posterior_covariance
end

row_quadratic_forms(Φ, covariance) =
    vec(sum((Φ * covariance) .* Φ; dims = 2))

function direct_prior_parameters(depth, p, targets)
    anchor = -log(max(mean(abs2, diff(sort(targets))) / 2, 1e-8))
    mean_prior_mean = zeros(p)
    mean_prior_precision = Matrix{Float64}(I, p, p)
    level_prior_means = [
        vcat(zeros(p - 1), level == 1 ? anchor : log(TOP_CARRIER))
        for level in 1:(depth - 1)
    ]
    level_prior_precisions = [
        Matrix(Diagonal(vcat(fill(inv(0.4^2), p - 1), 1.0)))
        for _ in 1:(depth - 1)
    ]
    return (;
        mean_prior_mean,
        mean_prior_precision,
        level_prior_means,
        level_prior_precisions,
    )
end

direct_prior_parameters(depth, Φ::AbstractMatrix, targets) =
    direct_prior_parameters(depth, size(Φ, 2), targets)

function new_exp_states(levels, observations, method, beta)
    meta = DampingMeta(
        alpha = direct_optimizer_alpha(method),
        beta = beta,
        max_step = MAX_STEP,
        method = method,
    )
    return [
        NGMPEdgeState(meta)
        for _ in 1:levels, _ in 1:observations
    ]
end

function damp_exp_sites!(
    states,
    level,
    gamma_shape,
    gamma_rate,
    score_mean,
    score_variance,
)
    observations = length(score_mean)
    site_weighted_mean = Vector{Float64}(undef, observations)
    site_precision = Vector{Float64}(undef, observations)
    for observation in 1:observations
        exponential_term = gamma_rate[observation] * exp(
            score_mean[observation] + score_variance[observation] / 2,
        )
        target_precision = exponential_term
        target_weighted_mean =
            (gamma_shape - 1) +
            (score_mean[observation] - 1) * exponential_term
        message = SurrogateModelling.NaturalGradientMP.apply_damping!(
            states[level, observation],
            target_weighted_mean,
            target_precision,
        )
        site_weighted_mean[observation] = weightedmean(message)
        site_precision[observation] = precision(message)
    end
    return site_weighted_mean, site_precision
end

function finite_direct_state(mean_weights, covariance, level_means, level_covariances)
    return all(isfinite, mean_weights) &&
        all(isfinite, covariance) &&
        all(weights -> all(isfinite, weights), level_means) &&
        all(matrix -> all(isfinite, matrix), level_covariances)
end

function fit_direct_depth_one(
    Φ,
    targets;
    prior_builder = direct_prior_parameters,
)
    n, p = size(Φ)
    prior = prior_builder(1, Φ, targets)
    noise_shape = 2.0
    noise_rate = 2.0 / TOP_CARRIER
    expected_precision = noise_shape / noise_rate
    mean_weights = zeros(p)
    mean_covariance = Matrix{Float64}(I, p, p)

    for _ in 1:min(DIRECT_ITERATIONS, 20)
        mean_weights, mean_covariance = direct_gaussian_update(
            Φ,
            targets,
            fill(expected_precision, n),
            prior.mean_prior_mean,
            prior.mean_prior_precision,
        )
        residual_second_moment =
            abs2.(targets - Φ * mean_weights) +
            row_quadratic_forms(Φ, mean_covariance)
        noise_shape = 2.0 + n / 2
        noise_rate =
            2.0 / TOP_CARRIER + sum(residual_second_moment) / 2
        updated_precision = noise_shape / noise_rate
        abs(updated_precision - expected_precision) <=
            DIRECT_TOLERANCE * max(1.0, expected_precision) && break
        expected_precision = updated_precision
    end

    return (;
        depth = 1,
        method = :damped,
        beta = 0.0,
        mean_weights,
        mean_covariance,
        noise_shape,
        noise_rate,
        level_weights = Vector{Vector{Float64}}(),
        level_covariances = Vector{Matrix{Float64}}(),
    )
end

function fit_direct_deep(
    depth,
    Φ,
    targets,
    method,
    beta;
    prior_builder = direct_prior_parameters,
)
    n, p = size(Φ)
    levels = depth - 1
    prior = prior_builder(depth, Φ, targets)
    mean_weights = copy(prior.mean_prior_mean)
    mean_covariance = inv(prior.mean_prior_precision)
    level_weights = copy.(prior.level_prior_means)
    level_covariances = inv.(prior.level_prior_precisions)
    score_means = [Φ * weights for weights in level_weights]
    score_variances = [
        row_quadratic_forms(Φ, covariance) .+ inv(TOP_CARRIER)
        for covariance in level_covariances
    ]
    expected_precisions = [
        exp.(score_means[level] .+ score_variances[level] ./ 2)
        for level in 1:levels
    ]
    exp_states = new_exp_states(levels, n, method, beta)

    previous_vector = vcat(mean_weights, level_weights...)
    for _ in 1:DIRECT_ITERATIONS
        mean_weights, mean_covariance = direct_gaussian_update(
            Φ,
            targets,
            max.(expected_precisions[1], DIRECT_JITTER),
            prior.mean_prior_mean,
            prior.mean_prior_precision,
        )

        gamma_rate =
            (
                abs2.(targets - Φ * mean_weights) +
                row_quadratic_forms(Φ, mean_covariance)
            ) ./ 2

        for level in 1:levels
            carrier_precision = level == levels ?
                fill(TOP_CARRIER, n) :
                max.(expected_precisions[level + 1], DIRECT_JITTER)
            conditional_mean = Φ * level_weights[level]
            conditional_variance =
                row_quadratic_forms(Φ, level_covariances[level]) +
                inv.(carrier_precision)

            site_xi, site_precision = damp_exp_sites!(
                exp_states,
                level,
                1.5,
                max.(gamma_rate, DIRECT_JITTER),
                score_means[level],
                score_variances[level],
            )

            positive_site_precision =
                max.(site_precision, DIRECT_JITTER)
            posterior_score_precision =
                carrier_precision + positive_site_precision
            effective_precision =
                carrier_precision .* positive_site_precision ./
                posterior_score_precision
            site_target = site_xi ./ positive_site_precision
            level_weights[level], level_covariances[level] =
                direct_gaussian_update(
                    Φ,
                    site_target,
                    effective_precision,
                    prior.level_prior_means[level],
                    prior.level_prior_precisions[level],
                )

            updated_conditional_mean = Φ * level_weights[level]
            updated_conditional_variance =
                row_quadratic_forms(Φ, level_covariances[level])
            carrier_fraction =
                carrier_precision ./ posterior_score_precision
            posterior_score_mean =
                (
                    carrier_precision .* updated_conditional_mean +
                    site_xi
                ) ./ posterior_score_precision
            posterior_score_variance =
                inv.(posterior_score_precision) +
                carrier_fraction .^ 2 .* updated_conditional_variance

            score_means[level] = posterior_score_mean
            score_variances[level] = posterior_score_variance
            expected_precisions[level] = exp.(
                posterior_score_mean + posterior_score_variance / 2,
            )

            residual_variance =
                posterior_score_variance +
                updated_conditional_variance -
                2 .* carrier_fraction .* updated_conditional_variance
            gamma_rate =
                (
                    abs2.(
                        posterior_score_mean - updated_conditional_mean,
                    ) +
                    max.(residual_variance, DIRECT_JITTER)
                ) ./ 2
        end

        finite_direct_state(
            mean_weights,
            mean_covariance,
            level_weights,
            level_covariances,
        ) || error("direct backend produced a non-finite posterior")
        all(level -> all(
            value -> isfinite(value) && value > 0,
            expected_precisions[level],
        ), 1:levels) ||
            error("direct backend produced a non-finite precision")

        current_vector = vcat(mean_weights, level_weights...)
        delta = norm(current_vector - previous_vector) /
            max(1.0, norm(previous_vector))
        delta < DIRECT_TOLERANCE && break
        previous_vector = current_vector
    end

    return (;
        depth,
        method,
        beta,
        mean_weights,
        mean_covariance,
        noise_shape = NaN,
        noise_rate = NaN,
        level_weights,
        level_covariances,
    )
end

function fit_direct_model(
    depth,
    Φ,
    targets,
    method,
    beta;
    prior_builder = direct_prior_parameters,
)
    depth == 1 && return fit_direct_depth_one(
        Φ,
        targets;
        prior_builder,
    )
    return fit_direct_deep(
        depth,
        Φ,
        targets,
        method,
        beta;
        prior_builder,
    )
end

function predict_direct_model(fit, Φ)
    predictive_mean = Φ * fit.mean_weights
    epistemic_variance =
        row_quadratic_forms(Φ, fit.mean_covariance)
    aleatoric_variance = if fit.depth == 1
        fill(
            fit.noise_rate / max(fit.noise_shape - 1, DIRECT_JITTER),
            size(Φ, 1),
        )
    else
        score_mean = Φ * fit.level_weights[1]
        score_variance =
            row_quadratic_forms(Φ, fit.level_covariances[1])
        exp.(-score_mean + score_variance / 2)
    end
    predictive_variance =
        max.(epistemic_variance + aleatoric_variance, DIRECT_JITTER)
    return (; mean = predictive_mean, variance = predictive_variance)
end

function direct_main(
    ;
    prior_builder = direct_prior_parameters,
    output_stem = "uci_deep_kernel_direct_paper",
    backend_label = "direct",
    fixed_lengthscale = nothing,
    optimizer_configs_for_depth = optimizers_for_depth,
)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    datasets =
        Symbol.(split(get(ENV, "UCI_DATASETS", "yacht,energy,concrete"), ','))
    rows = NamedTuple[]
    path = joinpath(
        dirname(@__DIR__),
        "results",
        "$output_stem.csv",
    )
    summary_path = joinpath(
        dirname(@__DIR__),
        "results",
        "$(output_stem)_summary.csv",
    )
    runs_per_preprocessing_dimension =
        N_SPLITS * sum(
            length(optimizer_configs_for_depth(depth))
            for depth in DEPTHS
        )
    total_runs =
        length(datasets) *
        length(DIRECT_PREPROCESSING) *
        length(DIRECT_FEATURE_DIMENSIONS) *
        runs_per_preprocessing_dimension
    run_index = 0

    cached_splits = Dict{Tuple{Symbol, Symbol, Int}, Any}()
    for dataset in datasets,
        preprocessing in DIRECT_PREPROCESSING,
        feature_dimension in DIRECT_FEATURE_DIMENSIONS
        cached_splits[(dataset, preprocessing, feature_dimension)] = map(
            paper_splits(dataset; count = N_SPLITS),
        ) do split
            prepared = prepare_split(split)
            Φtrain, Φtest = direct_rff_design(
                prepared.x_train_std,
                prepared.x_test_std,
                10_000 * split.split_id,
                preprocessing,
                feature_dimension,
                fixed_lengthscale,
            )
            (; split, prepared, Φtrain, Φtest)
        end
    end

    for dataset in datasets,
        preprocessing in DIRECT_PREPROCESSING,
        feature_dimension in DIRECT_FEATURE_DIMENSIONS,
        depth in DEPTHS,
        (method, beta) in optimizer_configs_for_depth(depth)
        first_split_unstable = false
        for (split_position, cached) in
            enumerate(cached_splits[(
                dataset,
                preprocessing,
                feature_dimension,
            )])
            split = cached.split
            prepared = cached.prepared
            run_index += 1
            @printf(
                "[%d/%d] backend=%s dataset=%s preprocessing=%s feature_dimension=%d split=%d/%d depth=%d optimizer=%s beta=%.2f alpha=%.2f\n",
                run_index,
                total_runs,
                backend_label,
                dataset,
                preprocessing,
                feature_dimension,
                split.split_id,
                N_SPLITS,
                depth,
                method,
                beta,
                direct_optimizer_alpha(method),
            )
            flush(stdout)

            if split_position > 1 && first_split_unstable
                println("  skipped: split 1 was unstable for this configuration")
                push!(rows, (;
                    dataset,
                    preprocessing,
                    feature_dimension,
                    split = split.split_id,
                    depth,
                    optimizer = method,
                    beta,
                    status = "skipped",
                    logpdf_standardized = NaN,
                    logpdf = NaN,
                    rmse = NaN,
                    paper_dvi = split.paper_dvi,
                    error = "skipped because split 1 was unstable",
                ))
            else
                try
                    fit = fit_direct_model(
                        depth,
                        cached.Φtrain,
                        prepared.y_train_std,
                        method,
                        beta,
                        ;
                        prior_builder,
                    )
                    prediction =
                        predict_direct_model(fit, cached.Φtest)
                    metrics = gaussian_logpdf_metrics(
                        prepared.y_test_std,
                        prediction.mean,
                        prediction.variance,
                        prepared.y_scale,
                    )
                    push!(rows, (;
                        dataset,
                        preprocessing,
                        feature_dimension,
                        split = split.split_id,
                        depth,
                        optimizer = method,
                        beta,
                        status = "ok",
                        metrics...,
                        paper_dvi = split.paper_dvi,
                        error = "",
                    ))
                catch error
                    first_split_unstable = split_position == 1
                    push!(rows, (;
                        dataset,
                        preprocessing,
                        feature_dimension,
                        split = split.split_id,
                        depth,
                        optimizer = method,
                        beta,
                        status = "unstable",
                        logpdf_standardized = NaN,
                        logpdf = NaN,
                        rmse = NaN,
                        paper_dvi = split.paper_dvi,
                        error = replace(
                            sprint(showerror, error),
                            '\n' => ' ',
                        ),
                    ))
                end
            end
            write_results(path, rows)
            write_results(
                summary_path,
                direct_summary_rows(
                    rows,
                    datasets;
                    optimizer_configs_for_depth,
                ),
            )
        end
    end

    println(
        "dataset preprocessing feature_dimension depth optimizer beta successful mean_logpdf std_logpdf paper_DVI",
    )
    for row in direct_summary_rows(
        rows,
        datasets;
        optimizer_configs_for_depth,
    )
        @printf(
            "%-9s %-28s %4d %5d %-27s %4.2f %3d/%-3d %10.3f %10.3f %9.2f\n",
            row.dataset,
            row.preprocessing,
            row.feature_dimension,
            row.depth,
            row.optimizer,
            row.beta,
            row.successful_splits,
            row.requested_splits,
            row.mean_logpdf,
            row.std_logpdf,
            row.paper_dvi,
        )
    end
    println("CSV: $path")
    println("Summary CSV: $summary_path")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && direct_main()
