const LOG2PI = log(2π)
const STANDARD_NORMAL = Normal()

function posterior_components(gate, optimizer, optimizer_state, parameters, states,
    gate_data; nsamples::Integer = 1000, seed::Integer = DEFAULT_SEED,
    sample_posterior::Bool = true)
    nsamples >= 1 || error("nsamples must be positive")
    n_observations = length(gate_data.features)
    n_experts = size(gate_data.predictions, 1)
    n_experts == 7 || error("Expected seven frozen forecasts")
    rng = StableRNG(seed)
    test_states = Lux.testmode(states)
    component_mean = Matrix{Float64}(undef, nsamples, n_observations)
    component_variance = Matrix{Float64}(undef, nsamples, n_observations)
    probability_sum = zeros(Float64, n_experts)
    probability_sum_squares = zeros(Float64, n_experts)
    posterior_mean_probability = zeros(Float64, n_experts, n_observations)
    top_counts = zeros(Int, n_experts)
    entropy_sum = 0.0
    switch_count = 0

    for sample = 1:nsamples
        sampled_parameters = sample_posterior ?
            rand(rng, optimizer, optimizer_state, parameters) : parameters
        previous_top = 0
        for j = 1:n_observations
            logits_raw, _ = gate(Float32.(gate_data.features[j]), sampled_parameters,
                test_states)
            logits = Float64.(vec(logits_raw))
            all(isfinite, logits) || error("Non-finite posterior gate logits")
            probabilities = stable_softmax(logits)
            forecasts = Float64[gate_data.predictions[i, j][1] for i = 1:n_experts]
            component_mean[sample, j] = dot(probabilities, forecasts)
            log_precision = logsumexp(logits)
            variance = exp(-log_precision)
            isfinite(variance) && variance > 0 || error("Invalid PGE Gaussian variance")
            component_variance[sample, j] = variance

            probability_sum .+= probabilities
            probability_sum_squares .+= probabilities .^ 2
            posterior_mean_probability[:, j] .+= probabilities
            entropy_sum -= sum(p -> p == 0 ? 0.0 : p * log(p), probabilities)
            top = argmax(probabilities)
            top_counts[top] += 1
            j > 1 && top != previous_top && (switch_count += 1)
            previous_top = top
        end
    end

    denominator = nsamples * n_observations
    probability_mean = probability_sum ./ denominator
    probability_variance = max.(
        probability_sum_squares ./ denominator .- probability_mean .^ 2, 0.0)
    posterior_mean_probability ./= nsamples
    consensus_top = [argmax(view(posterior_mean_probability, :, j)) for
                     j = 1:n_observations]
    consensus_switches = n_observations > 1 ?
        count(j -> consensus_top[j] != consensus_top[j + 1], 1:(n_observations - 1)) : 0
    switch_denominator = nsamples * max(n_observations - 1, 1)
    consensus_denominator = max(n_observations - 1, 1)
    posterior_mean_entropy = mean(1:n_observations) do j
        -sum(p -> p == 0 ? 0.0 : p * log(p),
            view(posterior_mean_probability, :, j))
    end

    gate_diagnostics = (
        probability_mean = probability_mean,
        probability_std = sqrt.(probability_variance),
        entropy = entropy_sum / denominator,
        posterior_mean_entropy = posterior_mean_entropy,
        top_expert_share = Float64.(top_counts) ./ denominator,
        max_top_expert_share = maximum(top_counts) / denominator,
        switching_rate = n_observations > 1 ? switch_count / switch_denominator : 0.0,
        posterior_mean_switching_rate = n_observations > 1 ?
            consensus_switches / consensus_denominator : 0.0,
    )
    return (; means = component_mean, variances = component_variance,
        gate_diagnostics)
end

function posterior_components(training_result, gate_data; nsamples::Integer = 1000,
    seed::Integer = DEFAULT_SEED, sample_posterior::Bool = true)
    return posterior_components(
        training_result.gate,
        training_result.optimizer,
        training_result.optimizer_state,
        training_result.parameters,
        training_result.states,
        gate_data;
        nsamples,
        seed,
        sample_posterior,
    )
end

function empirical_crps(draws::AbstractVector, target::Real)
    n = length(draws)
    sorted = sort(Float64.(draws))
    pair_term = sum((2i - n - 1) * sorted[i] for i = 1:n) / n^2
    return mean(abs.(draws .- target)) - pair_term
end

function normal_crps(mean_value::Real, variance::Real, target::Real)
    sigma = sqrt(variance)
    z = (target - mean_value) / sigma
    return sigma * (z * (2cdf(STANDARD_NORMAL, z) - 1) +
                      2pdf(STANDARD_NORMAL, z) - inv(sqrt(π)))
end

function stable_mixture_logpdf(means, variances, target)
    terms = @. -0.5 * (LOG2PI + log(variances) + (target - means)^2 / variances)
    return logsumexp(terms) - log(length(terms))
end

function mixture_selection_metrics(components, targets)
    y = Float64[t[1] for t in targets]
    predictive_mean = vec(mean(components.means; dims = 1))
    log_density = [stable_mixture_logpdf(
        view(components.means, :, j), view(components.variances, :, j), y[j])
                   for j in eachindex(y)]
    nll = -mean(log_density)
    mse = mean((predictive_mean .- y) .^ 2)
    all(isfinite, (nll, mse)) || error("Non-finite posterior-predictive selection metric")
    return (; nll, mse, log_predictive_density = -nll)
end

function mixture_metrics(components, targets; interval_seed::Integer = DEFAULT_SEED)
    y = Float64[t[1] for t in targets]
    n_samples, n_observations = size(components.means)
    length(y) == n_observations || error("Target/component length mismatch")
    rng = StableRNG(interval_seed)
    predictive_mean = Vector{Float64}(undef, n_observations)
    lower = similar(predictive_mean)
    upper = similar(predictive_mean)
    epistemic = similar(predictive_mean)
    aleatoric = similar(predictive_mean)
    total_variance = similar(predictive_mean)
    crps_terms = similar(predictive_mean)
    log_density = similar(predictive_mean)
    interval_score_terms = similar(predictive_mean)
    z95 = quantile(STANDARD_NORMAL, 0.975)

    for j = 1:n_observations
        means = view(components.means, :, j)
        variances = view(components.variances, :, j)
        predictive_mean[j] = mean(means)
        aleatoric[j] = mean(variances)
        epistemic[j] = mean(abs2, means .- predictive_mean[j])
        total_variance[j] = aleatoric[j] + epistemic[j]
        log_density[j] = stable_mixture_logpdf(means, variances, y[j])

        if n_samples == 1
            sigma = sqrt(only(variances))
            lower[j] = only(means) - z95 * sigma
            upper[j] = only(means) + z95 * sigma
            crps_terms[j] = normal_crps(only(means), only(variances), y[j])
        else
            draws = means .+ sqrt.(variances) .* randn(rng, n_samples)
            lower[j] = quantile(draws, 0.025)
            upper[j] = quantile(draws, 0.975)
            crps_terms[j] = empirical_crps(draws, y[j])
        end
        interval_score_terms[j] = (upper[j] - lower[j]) +
            40 * (lower[j] - y[j]) * (y[j] < lower[j]) +
            40 * (y[j] - upper[j]) * (y[j] > upper[j])
    end

    residual = predictive_mean .- y
    nll = -mean(log_density)
    metrics = (
        nll = nll,
        log_predictive_density = -nll,
        mse = mean(abs2, residual),
        rmse = sqrt(mean(abs2, residual)),
        mae = mean(abs, residual),
        crps = mean(crps_terms),
        coverage95 = mean((y .>= lower) .& (y .<= upper)),
        interval_width = mean(upper .- lower),
        interval_score = mean(interval_score_terms),
        epistemic_variance = mean(epistemic),
        aleatoric_variance = mean(aleatoric),
        total_variance = mean(total_variance),
        crps_method = n_samples == 1 ? "analytic_normal" :
                      "empirical_posterior_predictive_$n_samples",
        interval_method = n_samples == 1 ? "analytic_normal" :
                          "empirical_posterior_predictive_$n_samples",
    )
    numeric_metrics = filter(x -> x isa Real, collect(values(metrics)))
    all(isfinite, numeric_metrics) || error("Non-finite posterior-predictive metric")
    traces = (
        target = y,
        predictive_mean = predictive_mean,
        lower95 = lower,
        upper95 = upper,
        epistemic_variance = epistemic,
        aleatoric_variance = aleatoric,
        total_variance = total_variance,
        log_predictive_density = log_density,
    )
    return (; metrics, traces)
end

function original_unit_metrics(metrics, scale::Real)
    scale > 0 || error("Target scale must be positive")
    variance_scale = scale^2
    return (
        nll = metrics.nll + log(scale),
        log_predictive_density = metrics.log_predictive_density - log(scale),
        mse = metrics.mse * variance_scale,
        rmse = metrics.rmse * scale,
        mae = metrics.mae * scale,
        crps = metrics.crps * scale,
        coverage95 = metrics.coverage95,
        interval_width = metrics.interval_width * scale,
        interval_score = metrics.interval_score * scale,
        epistemic_variance = metrics.epistemic_variance * variance_scale,
        aleatoric_variance = metrics.aleatoric_variance * variance_scale,
        total_variance = metrics.total_variance * variance_scale,
        crps_method = metrics.crps_method,
        interval_method = metrics.interval_method,
    )
end

function component_digest(components)
    means_digest = bytes2hex(sha256(reinterpret(UInt8, vec(components.means))))
    variances_digest = bytes2hex(sha256(reinterpret(UInt8, vec(components.variances))))
    return bytes2hex(sha256(means_digest * variances_digest))
end
