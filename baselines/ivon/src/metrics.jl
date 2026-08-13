const LOG2PI = log(2π)
const STANDARD_NORMAL = Normal()

mutable struct GateDiagnosticAccumulator
    probability_sum::Vector{Float64}
    probability_sum_squares::Vector{Float64}
    posterior_probability_sum::Matrix{Float64}
    top_counts::Vector{Int}
    entropy_sum::Float64
    switch_count::Int
end

function GateDiagnosticAccumulator(n_experts::Integer, n_observations::Integer)
    return GateDiagnosticAccumulator(
        zeros(Float64, n_experts),
        zeros(Float64, n_experts),
        zeros(Float64, n_experts, n_observations),
        zeros(Int, n_experts),
        0.0,
        0,
    )
end

function posterior_subset_indices(nsamples::Integer;
    sample_counts = POSTERIOR_SAMPLE_COUNTS, seed::Integer = DEFAULT_SEED)
    nsamples >= 1 || error("nsamples must be positive")
    counts = sort!(unique(Int.(collect(sample_counts))))
    isempty(counts) && error("At least one posterior sample count is required")
    first(counts) >= 1 || error("Posterior sample counts must be positive")
    last(counts) == nsamples || error("The largest sample count must equal nsamples")
    permutation = randperm(StableRNG(seed), nsamples)
    groups = Dict{Int,Vector{Int}}()
    for count in counts
        groups[count] = count == nsamples ? collect(1:nsamples) : copy(permutation[1:count])
    end
    validate_posterior_subset_indices(groups, nsamples)
    return groups
end

function validate_posterior_subset_indices(sample_indices, nsamples::Integer)
    counts = sort!(Int.(collect(keys(sample_indices))))
    isempty(counts) && error("Posterior subset map is empty")
    last(counts) == nsamples || error("Posterior subset map does not contain the full bank")
    previous = Set{Int}()
    for count in counts
        indices = Int.(collect(sample_indices[count]))
        length(indices) == count || error("Posterior subset K=$count has the wrong length")
        length(unique(indices)) == count || error("Posterior subset K=$count contains duplicates")
        all(index -> 1 <= index <= nsamples, indices) ||
            error("Posterior subset K=$count contains an out-of-range index")
        issubset(previous, Set(indices)) || error("Posterior sample subsets are not nested")
        previous = Set(indices)
    end
    Set(Int.(collect(sample_indices[nsamples]))) == Set(1:nsamples) ||
        error("The largest posterior subset must contain the complete bank")
    return true
end

function posterior_subset_digest(sample_indices)
    pieces = String[]
    for count in sort!(Int.(collect(keys(sample_indices))))
        push!(pieces, "$count:" * join(Int.(collect(sample_indices[count])), ','))
    end
    return bytes2hex(sha256(join(pieces, '|')))
end

function update_gate_diagnostics!(accumulator::GateDiagnosticAccumulator,
    probabilities::AbstractMatrix, top_experts::AbstractVector{<:Integer})
    accumulator.probability_sum .+= vec(sum(probabilities; dims = 2))
    accumulator.probability_sum_squares .+= vec(sum(abs2, probabilities; dims = 2))
    accumulator.posterior_probability_sum .+= probabilities
    accumulator.entropy_sum -= sum(
        probability -> probability == 0 ? 0.0 : probability * log(probability),
        probabilities,
    )
    for top in top_experts
        accumulator.top_counts[top] += 1
    end
    accumulator.switch_count += length(top_experts) > 1 ? sum(
        left != right for (left, right) in zip(
            view(top_experts, 1:(length(top_experts) - 1)),
            view(top_experts, 2:length(top_experts)),
        )) : 0
    return accumulator
end

function gate_diagnostics_snapshot(accumulator::GateDiagnosticAccumulator,
    n_samples::Integer, n_observations::Integer)
    denominator = n_samples * n_observations
    probability_mean = accumulator.probability_sum ./ denominator
    probability_variance = max.(
        accumulator.probability_sum_squares ./ denominator .- probability_mean .^ 2,
        0.0,
    )
    posterior_mean_probability = accumulator.posterior_probability_sum ./ n_samples
    consensus_top = [argmax(view(posterior_mean_probability, :, j))
                     for j = 1:n_observations]
    consensus_switches = n_observations > 1 ? sum(
        left != right for (left, right) in zip(
            view(consensus_top, 1:(n_observations - 1)),
            view(consensus_top, 2:n_observations),
        )) : 0
    switch_denominator = n_samples * max(n_observations - 1, 1)
    consensus_denominator = max(n_observations - 1, 1)
    posterior_mean_entropy = mean(1:n_observations) do j
        -sum(probability -> probability == 0 ? 0.0 : probability * log(probability),
            view(posterior_mean_probability, :, j))
    end
    return (
        probability_mean = probability_mean,
        probability_std = sqrt.(probability_variance),
        entropy = accumulator.entropy_sum / denominator,
        posterior_mean_entropy = posterior_mean_entropy,
        top_expert_share = Float64.(accumulator.top_counts) ./ denominator,
        max_top_expert_share = maximum(accumulator.top_counts) / denominator,
        switching_rate = n_observations > 1 ?
            accumulator.switch_count / switch_denominator : 0.0,
        posterior_mean_switching_rate = n_observations > 1 ?
            consensus_switches / consensus_denominator : 0.0,
    )
end

function posterior_components(gate, optimizer, optimizer_state, parameters, states,
    gate_data; nsamples::Integer = 1000, seed::Integer = DEFAULT_SEED,
    sample_posterior::Bool = true, sample_indices = nothing)
    nsamples >= 1 || error("nsamples must be positive")
    n_observations = length(gate_data.features)
    n_observations >= 1 || error("Posterior evaluation data is empty")
    n_experts = size(gate_data.predictions, 1)
    n_experts == 7 || error("Expected seven frozen forecasts")
    groups = sample_indices === nothing ? Dict(nsamples => collect(1:nsamples)) :
        Dict(Int(count) => Int.(collect(indices)) for (count, indices) in sample_indices)
    validate_posterior_subset_indices(groups, nsamples)
    memberships = [Int[] for _ = 1:nsamples]
    for (count, indices) in groups, index in indices
        push!(memberships[index], count)
    end
    accumulators = Dict(count => GateDiagnosticAccumulator(n_experts, n_observations)
        for count in keys(groups))

    rng = StableRNG(seed)
    test_states = Lux.testmode(states)
    component_mean = Matrix{Float64}(undef, nsamples, n_observations)
    component_variance = Matrix{Float64}(undef, nsamples, n_observations)
    feature_matrix = Float32.(reduce(hcat, gate_data.features))
    size(feature_matrix) == (65, n_observations) ||
        error("Expected a 65×N posterior feature matrix")
    forecasts = Matrix{Float64}(undef, n_experts, n_observations)
    for expert = 1:n_experts, observation = 1:n_observations
        forecasts[expert, observation] = gate_data.predictions[expert, observation][1]
    end

    for sample = 1:nsamples
        sampled_parameters = sample_posterior ?
            rand(rng, optimizer, optimizer_state, parameters) : parameters
        logits_raw, _ = gate(feature_matrix, sampled_parameters, test_states)
        logits = Float64.(logits_raw)
        size(logits) == (n_experts, n_observations) ||
            error("Unexpected posterior gate output shape")
        all(isfinite, logits) || error("Non-finite posterior gate logits")
        maximum_logits = maximum(logits; dims = 1)
        exponentials = exp.(logits .- maximum_logits)
        exponential_sums = sum(exponentials; dims = 1)
        probabilities = exponentials ./ exponential_sums
        component_mean[sample, :] .= vec(sum(probabilities .* forecasts; dims = 1))
        log_precision = vec(maximum_logits .+ log.(exponential_sums))
        variances = exp.(-log_precision)
        all(variance -> isfinite(variance) && variance > 0, variances) ||
            error("Invalid PGE Gaussian variance")
        component_variance[sample, :] .= variances
        top_experts = [argmax(view(probabilities, :, j)) for j = 1:n_observations]
        for count in memberships[sample]
            update_gate_diagnostics!(accumulators[count], probabilities, top_experts)
        end
    end

    diagnostics_by_sample_count = Dict(count => gate_diagnostics_snapshot(
        accumulators[count], count, n_observations) for count in keys(groups))
    gate_diagnostics = diagnostics_by_sample_count[nsamples]
    return (; means = component_mean, variances = component_variance,
        gate_diagnostics, gate_diagnostics_by_sample_count = diagnostics_by_sample_count)
end

function posterior_components(training_result, gate_data; nsamples::Integer = 1000,
    seed::Integer = DEFAULT_SEED, sample_posterior::Bool = true,
    sample_indices = nothing)
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
        sample_indices,
    )
end

function posterior_component_subset(components, indices)
    selected = Int.(collect(indices))
    count = length(selected)
    diagnostics = hasproperty(components, :gate_diagnostics_by_sample_count) ?
        components.gate_diagnostics_by_sample_count[count] : components.gate_diagnostics
    return (
        means = view(components.means, selected, :),
        variances = view(components.variances, selected, :),
        gate_diagnostics = diagnostics,
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

function mixture_metrics(components, targets; interval_seed::Integer = DEFAULT_SEED,
    standard_normal_draws = nothing)
    y = Float64[t[1] for t in targets]
    n_samples, n_observations = size(components.means)
    length(y) == n_observations || error("Target/component length mismatch")
    rng = StableRNG(interval_seed)
    if standard_normal_draws !== nothing
        size(standard_normal_draws) == (n_samples, n_observations) ||
            error("Standard-normal draw bank shape does not match mixture components")
        all(isfinite, standard_normal_draws) || error("Non-finite standard-normal draw bank")
    end
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
            noise = standard_normal_draws === nothing ? randn(rng, n_samples) :
                view(standard_normal_draws, :, j)
            draws = means .+ sqrt.(variances) .* noise
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

function mixture_metrics_by_sample_count(components, targets, sample_indices;
    interval_seed::Integer = DEFAULT_SEED)
    n_samples, n_observations = size(components.means)
    validate_posterior_subset_indices(sample_indices, n_samples)
    normal_bank = randn(StableRNG(interval_seed), n_samples, n_observations)
    results = Dict{Int,Any}()
    for count in sort!(Int.(collect(keys(sample_indices))))
        indices = Int.(collect(sample_indices[count]))
        subset = posterior_component_subset(components, indices)
        results[count] = mixture_metrics(subset, targets;
            interval_seed, standard_normal_draws = view(normal_bank, indices, :))
    end
    return results
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
