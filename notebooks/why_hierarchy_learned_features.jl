### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 1019a7b1-250b-4d6f-93be-1b515b9f7e15
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 2f42b773-c616-47ed-8f42-a1d189a3d766
begin
    ENV["GKSwstype"] = "100"

    using LinearAlgebra
    using Plots
    using Printf
    using Random
    using RxInfer
    using StableRNGs
    using Statistics
    using SurrogateModelling

    import ProbabilisticEnsembling: Exp
end

# ╔═╡ 4327cabc-0bc4-418a-a9e7-e9c66b5a1915
md"""
# Hierarchical uncertainty with learned features

This notebook uses exactly the benchmark, split, and evaluation protocol from
`why_hierarchy_deep_kernel.jl`, but it does not supply random Fourier features.
Instead, two Bayesian `ResidualSine` networks learn features from the raw
``[1,x]`` input:

```math
\begin{aligned}
h_\mu(x) &= \phi(W_\mu[1,x]), &
\mu(x) &= v^\top h_\mu(x),\\
h_\lambda(x) &= \phi(W_\lambda[1,x]), &
s(x) &= g^\top h_\lambda(x) + a_0,\\
\lambda(x) &= \exp s(x), &
y\mid x &\sim \mathcal N\!\left(\mu(x),\lambda(x)^{-1}\right).
\end{aligned}
```

The two heads have separate learned features so that the mean cannot force the
noise function to use exactly the same representation.

The mean head differs critically from an additive `ManyPlus` network:
`MvStack` preserves the hidden vector and `v` is one multivariate Gaussian with
a **dense posterior covariance**. Consequently
``h_\mu(x)^\top\operatorname{Cov}(v)h_\mu(x)`` retains the covariance terms
needed for epistemic uncertainty to contract at observed inputs.
"""

# ╔═╡ f45fa3c6-5df6-4983-ae20-2d44185b3ce1
begin
    # Same benchmark defaults as why_hierarchy_deep_kernel.jl.
    const N_SAMPLES = 600
    const HOLDOUT_FRACTION = 1 / 3
    const DATA_SEED = 7

    # Learned feature maps. The dense covariance is only K x K; the input
    # weights remain independent by neuron, so the construction scales as O(Kd)
    # in input dimension instead of using a dense covariance over all Kd weights.
    const N_NEURONS = 16
    const ACTIVATION_PRECISION = 1.0e4
    const SCORE_CARRIER = 25.0
    const PHI_RHO = 0.9
    const PHI_OMEGA = 1.0

    # The benchmark defaults to one full-data graph, matching
    # why_hierarchy_deep_kernel.jl. More than one batch is an optional streaming
    # approximation; on this benchmark it is faster but tends to underestimate
    # the variance scale.
    const N_TRAINING_BATCHES = 1
    const MAX_BATCH_ITERATIONS = 60
    const ALPHA = 0.05
    const MAX_STEP = 0.25
    const METHOD = :damped
    const GRID_POINTS = 31
end

# ╔═╡ 61654e1b-fd0e-4fdc-ba7e-ce8cd73fb32a
md"""
## The unchanged benchmark

```math
x \sim \mathcal N(0,1),\qquad
y = -(x+\tfrac12)\sin(3\pi x)+\varepsilon,\qquad
\varepsilon\sim\mathcal N\!\left(0,[0.45(x+\tfrac12)]^2\right).
```

The same seeded permutation reserves one third of the observations for testing.
"""

# ╔═╡ 5ae83b6a-fb28-4493-bbf4-a2c42580348a
begin
    true_mean(x) = -(x + 0.5) * sin(3pi * x)
    true_noise_variance(x) = abs2(0.45 * (x + 0.5))

    data = let
        rng = StableRNG(DATA_SEED)
        x = randn(rng, N_SAMPLES)
        y = true_mean.(x) .+
            sqrt.(true_noise_variance.(x)) .* randn(rng, N_SAMPLES)
        order = randperm(rng, N_SAMPLES)
        n_test = round(Int, HOLDOUT_FRACTION * N_SAMPLES)
        test, train = order[1:n_test], order[(n_test + 1):end]
        (; x_train = x[train], y_train = y[train],
           x_test = x[test], y_test = y[test])
    end

    grid = collect(range(-3.0, 3.0; length = GRID_POINTS))
    raw_features(xs) = [[1.0, Float64(x)] for x in xs]

    @printf("%d training points, %d held out\n",
        length(data.y_train), length(data.y_test))
end

# ╔═╡ cc31c16e-7532-44f8-aa58-39b78f2e1bcd
begin
    activation_meta() = ResidualSineMeta(rho = PHI_RHO, omega = PHI_OMEGA)

    activation_dependencies(alpha = ALPHA) = NGMPDependencies(
        out = nothing, in = nothing;
        projection = TangentProjection(type = ClosedForm),
        damping = DampingMeta(
            alpha = alpha, beta = 0.0, max_step = MAX_STEP, method = METHOD,
        ),
    )

    exp_dependencies() = NGMPDependencies(
        out = nothing, in = nothing;
        projection = TangentProjection(type = ClosedForm),
    )

    exp_damping(alpha = ALPHA) =
        DampingMeta(alpha = alpha, beta = 0.0, max_step = MAX_STEP, method = METHOD)
end

# ╔═╡ 3ebcb381-250c-43fa-b963-4a3fb11a180d
md"""
## One precision, not one precision per neuron

The precision neurons produce contributions to one log-precision score. Placing
separate likelihood precisions on the neurons would identify only
``\sum_k\lambda_k(x)^{-1}``, not the individual values. Here there is one
statistically meaningful observation precision ``\lambda(x)``.

The constant `noise_anchor` shifts the score before `Exp`. It is the same
first-difference anchor used in the fixed-feature hierarchy and lets the neural
head learn deviations from a sensible initial noise level.
"""

# ╔═╡ b2c193c7-d874-4b8e-94f2-027cbbc19b69
begin
    noise_anchor(xs, ys) =
        -log(max(mean(abs2.(diff(ys[sortperm(xs)]))) / 2, 1e-8))

    gaussian(mean_vector, covariance) =
        MvNormalMeanCovariance(collect(mean_vector), Matrix(covariance))

    function paired_weight_priors(
        n_neurons, low_frequency, high_frequency;
        bias_sd = 0.35, relative_frequency_sd = 0.12, seed = 42,
    )
        iseven(n_neurons) ||
            throw(ArgumentError("paired priors require an even neuron count"))
        rng = StableRNG(seed)
        n_pairs = n_neurons ÷ 2
        frequencies = n_pairs == 1 ?
            [sqrt(low_frequency * high_frequency)] :
            exp.(range(log(low_frequency), log(high_frequency); length = n_pairs))

        return map(1:n_neurons) do neuron
            pair = cld(neuron, 2)
            sign = isodd(neuron) ? 1.0 : -1.0
            bias = sign * (pi / 4) * (1 + 0.04 * randn(rng))
            frequency = frequencies[pair] * (1 + 0.02 * randn(rng))
            frequency_sd = 0.15 + relative_frequency_sd * frequency
            gaussian(
                [bias, frequency],
                Diagonal([abs2(bias_sd), abs2(frequency_sd)]),
            )
        end
    end

    function make_priors(xs, ys)
        mean_w = paired_weight_priors(
            N_NEURONS, 0.75, 12.0; seed = 42,
        )
        noise_w = paired_weight_priors(
            N_NEURONS, 0.20, 4.0;
            bias_sd = 0.45, relative_frequency_sd = 0.18, seed = 1042,
        )

        # Non-zero alternating means break the sign symmetry and give the input
        # weights a learning signal from the first iteration.
        mean_v = [isodd(k) ? 0.20 : -0.20 for k in 1:N_NEURONS]
        mean_g = [isodd(k) ? 0.04 : -0.04 for k in 1:N_NEURONS]

        return (;
            w = mean_w,
            noise_w,
            v = gaussian(
                mean_v,
                Diagonal(fill(1 / N_NEURONS, N_NEURONS)),
            ),
            g = gaussian(
                mean_g,
                Diagonal(fill(abs2(0.5) / N_NEURONS, N_NEURONS)),
            ),
            anchor = noise_anchor(xs, ys),
        )
    end
end

# ╔═╡ d374accc-1151-45d6-b8c2-bb62320b894d
@model function learned_feature_hierarchy(
    y,
    features,
    n_neurons,
    priors,
    activation,
    activation_deps,
    link_deps,
    link_meta,
)
    local w, noise_w, za, h, noise_za, noise_h
    local h_vector, noise_h_vector
    local score, shifted_score, precision

    v ~ priors.v
    g ~ priors.g
    for neuron in 1:n_neurons
        w[neuron] ~ priors.w[neuron]
        noise_w[neuron] ~ priors.noise_w[neuron]
    end

    for observation in eachindex(y)
        for neuron in 1:n_neurons
            za[neuron, observation] ~ softdot(
                features[observation], w[neuron], ACTIVATION_PRECISION,
            )
            h[neuron, observation] ~ ResidualSine(
                za[neuron, observation],
            ) where {
                dependencies = activation_deps,
                meta = activation,
            }

            noise_za[neuron, observation] ~ softdot(
                features[observation], noise_w[neuron], ACTIVATION_PRECISION,
            )
            noise_h[neuron, observation] ~ ResidualSine(
                noise_za[neuron, observation],
            ) where {
                dependencies = activation_deps,
                meta = activation,
            }
        end

        h_vector[observation] ~ MvStack(inputs = [
            h[neuron, observation] for neuron in 1:n_neurons
        ])
        noise_h_vector[observation] ~ MvStack(inputs = [
            noise_h[neuron, observation] for neuron in 1:n_neurons
        ])

        score[observation] ~ softdot(
            noise_h_vector[observation], g, SCORE_CARRIER,
        )
        shifted_score[observation] :=
            score[observation] + priors.anchor
        precision[observation] ~ Exp(
            shifted_score[observation],
        ) where {
            dependencies = link_deps,
            meta = link_meta,
        }

        # softdot is the likelihood. There is no high-precision output carrier
        # between the hidden features and v.
        y[observation] ~ softdot(
            h_vector[observation], v, precision[observation],
        )
    end
end

# ╔═╡ 961232fc-5814-4764-915e-fe4e793f703e
@constraints function learned_feature_constraints()
    q(
        w, noise_w, v, g,
        za, h, h_vector,
        noise_za, noise_h, noise_h_vector,
        score, shifted_score, precision, y,
    ) =
        q(v, y) *
        q(w, za, h, h_vector) *
        q(g, score, shifted_score, precision) *
        q(noise_w, noise_za, noise_h, noise_h_vector)

    q(w)::MomentForm()
    q(noise_w)::MomentForm()
    q(v)::MomentForm()
    q(g)::MomentForm()
end

# ╔═╡ 3d2a1af1-950c-4634-9a7b-b0638b032f92
begin
    function hidden_moments(weight_distribution, feature, activation)
        weight_mean, weight_covariance = mean_cov(weight_distribution)
        preactivation_mean = dot(feature, weight_mean)
        preactivation_variance =
            dot(feature, weight_covariance * feature) + inv(ACTIVATION_PRECISION)
        return SurrogateModelling._residual_sine_mean_var_1d(
            preactivation_mean, preactivation_variance, activation,
        )
    end

    function bilinear_moments(weight_distribution, feature_mean, feature_covariance)
        weight_mean, weight_covariance = mean_cov(weight_distribution)
        output_mean = dot(weight_mean, feature_mean)
        output_variance =
            dot(weight_mean, feature_covariance * weight_mean) +
            dot(feature_mean, weight_covariance * feature_mean) +
            tr(weight_covariance * feature_covariance)
        return output_mean, output_variance
    end

    function initial_marginals(priors, features, activation)
        n_observations = length(features)
        za = Matrix{NormalMeanVariance{Float64}}(
            undef, N_NEURONS, n_observations,
        )
        h = similar(za)
        noise_za = similar(za)
        noise_h = similar(za)
        h_vector = Vector{MvNormalMeanCovariance{Float64, Vector{Float64}, Matrix{Float64}}}(
            undef, n_observations,
        )
        noise_h_vector = similar(h_vector)
        score = Vector{NormalMeanVariance{Float64}}(undef, n_observations)
        shifted_score = similar(score)
        precision = Vector{GammaShapeRate{Float64}}(undef, n_observations)

        for observation in eachindex(features)
            feature = features[observation]
            for neuron in 1:N_NEURONS
                for (weight, za_target, h_target) in (
                    (priors.w[neuron], za, h),
                    (priors.noise_w[neuron], noise_za, noise_h),
                )
                    weight_mean, weight_covariance = mean_cov(weight)
                    za_mean = dot(feature, weight_mean)
                    za_variance =
                        dot(feature, weight_covariance * feature) +
                        inv(ACTIVATION_PRECISION)
                    hidden_mean, hidden_variance =
                        SurrogateModelling._residual_sine_mean_var_1d(
                            za_mean, za_variance, activation,
                        )
                    za_target[neuron, observation] =
                        NormalMeanVariance(za_mean, za_variance)
                    h_target[neuron, observation] =
                        NormalMeanVariance(hidden_mean, hidden_variance)
                end
            end

            mean_hidden = mean.(h[:, observation])
            variance_hidden = var.(h[:, observation])
            mean_noise_hidden = mean.(noise_h[:, observation])
            variance_noise_hidden = var.(noise_h[:, observation])
            h_vector[observation] = MvNormalMeanCovariance(
                mean_hidden, Matrix(Diagonal(variance_hidden)),
            )
            noise_h_vector[observation] = MvNormalMeanCovariance(
                mean_noise_hidden, Matrix(Diagonal(variance_noise_hidden)),
            )

            score_mean, score_variance = bilinear_moments(
                priors.g,
                mean_noise_hidden,
                Matrix(Diagonal(variance_noise_hidden)),
            )
            score_variance += inv(SCORE_CARRIER)
            shifted_mean = score_mean + priors.anchor
            shape = 1 + inv(max(score_variance, 1e-6))

            score[observation] =
                NormalMeanVariance(score_mean, score_variance)
            shifted_score[observation] =
                NormalMeanVariance(shifted_mean, score_variance)
            precision[observation] =
                GammaShapeRate(shape, shape * exp(-shifted_mean))
        end

        return (;
            za, h, h_vector,
            noise_za, noise_h, noise_h_vector,
            score, shifted_score, precision,
        )
    end
end

# ╔═╡ 87b3a45a-7483-466f-a2c4-e7cf7399f5cc
@initialization function learned_feature_initialization(priors, states)
    q(v) = deepcopy(priors.v)
    q(g) = deepcopy(priors.g)
    q(za) = states.za
    q(h) = states.h
    q(h_vector) = states.h_vector
    q(noise_za) = states.noise_za
    q(noise_h) = states.noise_h
    q(noise_h_vector) = states.noise_h_vector
    q(score) = states.score
    q(shifted_score) = states.shifted_score
    q(precision) = states.precision
    μ(w) = deepcopy(priors.w)
    μ(noise_w) = deepcopy(priors.noise_w)
end

# ╔═╡ 34691611-3332-4bd1-88cc-dd17e9db031a
begin
    function batch_ranges(n_observations, n_batches)
        1 <= n_batches <= n_observations ||
            throw(ArgumentError(
                "number of batches must lie in 1:$n_observations; got $n_batches",
            ))
        base, extra = divrem(n_observations, n_batches)
        ranges = Vector{UnitRange{Int}}(undef, n_batches)
        first_index = 1
        for batch in 1:n_batches
            count = base + (batch <= extra ? 1 : 0)
            ranges[batch] = first_index:(first_index + count - 1)
            first_index += count
        end
        return ranges
    end

    function fit_training_batch(priors, features, observations)
        activation = activation_meta()
        states = initial_marginals(priors, features, activation)

        result = infer(
            model = learned_feature_hierarchy(
                n_neurons = N_NEURONS,
                priors = priors,
                activation = activation,
                activation_deps = activation_dependencies(),
                link_deps = exp_dependencies(),
                link_meta = exp_damping(),
            ),
            data = (y = observations, features = features),
            constraints = learned_feature_constraints(),
            initialization = learned_feature_initialization(priors, states),
            returnvars = (
                w = KeepLast(),
                noise_w = KeepLast(),
                v = KeepLast(),
                g = KeepLast(),
            ),
            iterations = MAX_BATCH_ITERATIONS,
            free_energy = false,
            showprogress = false,
            options = (limit_stack_depth = 100,),
            disable_inference_error_hint = true,
        )

        return merge(priors, (;
            w = deepcopy(collect(vec(result.posteriors[:w]))),
            noise_w = deepcopy(collect(vec(result.posteriors[:noise_w]))),
            v = deepcopy(result.posteriors[:v]),
            g = deepcopy(result.posteriors[:g]),
        ))
    end

    function fit_learned_hierarchy()
        priors = make_priors(data.x_train, data.y_train)
        ranges = batch_ranges(length(data.y_train), N_TRAINING_BATCHES)
        for (batch, indices) in enumerate(ranges)
            seconds = @elapsed priors = fit_training_batch(
                priors,
                raw_features(data.x_train[indices]),
                data.y_train[indices],
            )
            @printf(
                "batch %d/%d: %d points, %d iterations, %.1f s\n",
                batch,
                length(ranges),
                length(indices),
                MAX_BATCH_ITERATIONS,
                seconds,
            )
        end
        return priors
    end

    elapsed_seconds = @elapsed learned_fit = fit_learned_hierarchy()
    @printf("learned-feature hierarchy fitted in %.1f s\n", elapsed_seconds)

    initial_priors = make_priors(data.x_train, data.y_train)
    mean_feature_shift = mean(
        norm(mean(learned_fit.w[k]) - mean(initial_priors.w[k]))
        for k in 1:N_NEURONS
    )
    noise_feature_shift = mean(
        norm(mean(learned_fit.noise_w[k]) - mean(initial_priors.noise_w[k]))
        for k in 1:N_NEURONS
    )
    readout_covariance = cov(learned_fit.v)
    readout_off_diagonal =
        readout_covariance - Diagonal(diag(readout_covariance))
    @printf(
        "learned feature-weight shifts: mean head %.3e, precision head %.3e\n",
        mean_feature_shift,
        noise_feature_shift,
    )
    @printf(
        "q(v): variance %.3e..%.3e, max |off-diagonal| %.3e\n",
        minimum(diag(readout_covariance)),
        maximum(diag(readout_covariance)),
        maximum(abs, readout_off_diagonal),
    )
end

# ╔═╡ ef5bc21c-2b31-4548-aaf5-781f8411a280
md"""
## Closed-form moment prediction

Training is message passing. For prediction we propagate the learned Gaussian
marginals analytically instead of iterating an unobserved graph:

```math
\begin{aligned}
\text{epistemic}(x)
  &= \operatorname{Var}_{q(W_\mu,v)}[v^\top h_\mu(x)],\\
\text{aleatoric}(x)
  &= \mathbb E_q[\lambda(x)^{-1}]
   = \exp\{-m_s(x)+\tfrac12 V_s(x)\},\\
\operatorname{Var}(y_*\mid x,\mathcal D)
  &= \text{epistemic}(x)+\text{aleatoric}(x).
\end{aligned}
```

The last equality is the law of total variance. This also exposes the
difference between the scale-mixture value ``E[1/\lambda]`` and the VMP
effective value ``1/E[\lambda]``.
"""

# ╔═╡ f1d0d79c-d90c-440a-95cc-2351c720003a
begin
    safe_exp(value) = exp(clamp(value, -40.0, 40.0))

    function predictive_moments(fit, xs)
        activation = activation_meta()
        n_points = length(xs)
        predictive_mean = Vector{Float64}(undef, n_points)
        epistemic_variance = similar(predictive_mean)
        aleatoric_variance = similar(predictive_mean)
        vmp_aleatoric_variance = similar(predictive_mean)
        log_precision_mean = similar(predictive_mean)
        log_precision_variance = similar(predictive_mean)

        for (index, x) in enumerate(xs)
            feature = [1.0, Float64(x)]

            mean_hidden = Vector{Float64}(undef, N_NEURONS)
            variance_hidden = similar(mean_hidden)
            mean_noise_hidden = similar(mean_hidden)
            variance_noise_hidden = similar(mean_hidden)

            for neuron in 1:N_NEURONS
                mean_hidden[neuron], variance_hidden[neuron] =
                    hidden_moments(fit.w[neuron], feature, activation)
                mean_noise_hidden[neuron], variance_noise_hidden[neuron] =
                    hidden_moments(fit.noise_w[neuron], feature, activation)
            end

            predictive_mean[index], epistemic_variance[index] =
                bilinear_moments(
                    fit.v,
                    mean_hidden,
                    Matrix(Diagonal(variance_hidden)),
                )

            score_mean, score_variance = bilinear_moments(
                fit.g,
                mean_noise_hidden,
                Matrix(Diagonal(variance_noise_hidden)),
            )
            score_variance += inv(SCORE_CARRIER)
            score_mean += fit.anchor

            log_precision_mean[index] = score_mean
            log_precision_variance[index] = score_variance
            aleatoric_variance[index] =
                safe_exp(-score_mean + score_variance / 2)
            vmp_aleatoric_variance[index] =
                safe_exp(-score_mean - score_variance / 2)
        end

        return (;
            mean = predictive_mean,
            epistemic = max.(epistemic_variance, 0.0),
            aleatoric = aleatoric_variance,
            vmp_aleatoric = vmp_aleatoric_variance,
            variance = max.(epistemic_variance, 0.0) + aleatoric_variance,
            log_precision_mean,
            log_precision_variance,
        )
    end

    grid_prediction = predictive_moments(learned_fit, grid)
    test_prediction = predictive_moments(learned_fit, data.x_test)
end

# ╔═╡ 94ee0697-4ddd-45ca-a6d4-bd62b40b111e
begin
    prediction_band =
        1.96 .* sqrt.(max.(grid_prediction.variance, 0.0))

    mean_panel = plot(
        grid,
        grid_prediction.mean;
        ribbon = prediction_band,
        fillalpha = 0.20,
        color = :steelblue,
        linewidth = 2,
        label = "q(y*) ± 1.96 SD",
        xlabel = "x",
        ylabel = "y",
        title = "Learned features: predictive distribution",
        titlefontsize = 9,
        legend = :topleft,
    )
    plot!(
        mean_panel,
        grid,
        true_mean.(grid);
        color = :black,
        linewidth = 2,
        label = "true mean",
    )
    scatter!(
        mean_panel,
        data.x_train,
        data.y_train;
        color = :black,
        markersize = 2,
        markerstrokewidth = 0,
        alpha = 0.35,
        label = "train",
    )

    variance_panel = plot(
        grid,
        max.(grid_prediction.variance, 1e-5);
        color = :steelblue,
        linewidth = 2,
        label = "total",
        xlabel = "x",
        ylabel = "variance",
        yscale = :log10,
        title = "Epistemic + learned aleatoric variance",
        titlefontsize = 9,
        legend = :topleft,
    )
    plot!(
        variance_panel,
        grid,
        max.(grid_prediction.epistemic, 1e-5);
        color = :darkorange,
        linewidth = 2,
        label = "epistemic",
    )
    plot!(
        variance_panel,
        grid,
        max.(grid_prediction.aleatoric, 1e-5);
        color = :purple,
        linewidth = 2,
        label = "E[1/λ(x)]",
    )
    plot!(
        variance_panel,
        grid,
        max.(true_noise_variance.(grid), 1e-5);
        color = :black,
        linestyle = :dashdot,
        linewidth = 2,
        label = "true noise",
    )

    learned_feature_plot = plot(
        mean_panel,
        variance_panel;
        layout = (1, 2),
        size = (1120, 420),
        margin = 4Plots.mm,
    )
end

# ╔═╡ 82e7399c-cb96-47f5-b7ec-d704b0f23cf3
begin
    function predictive_scores(prediction, xs, ys)
        variance = max.(prediction.variance, 1e-10)
        residual = ys .- prediction.mean
        truth = true_noise_variance.(xs)
        return (;
            logpdf = mean(
                -0.5 .* (log.(2pi .* variance) .+ abs2.(residual) ./ variance),
            ),
            rmse = sqrt(mean(abs2.(residual))),
            latent_mean_rmse = sqrt(mean(abs2.(
                prediction.mean .- true_mean.(xs),
            ))),
            noise_correlation = cor(prediction.aleatoric, truth),
            mean_aleatoric = mean(prediction.aleatoric),
            true_mean_aleatoric = mean(truth),
            mean_epistemic = mean(prediction.epistemic),
        )
    end

    test_scores = predictive_scores(
        test_prediction, data.x_test, data.y_test,
    )

    @printf(
        "test logpdf %.4f | observed RMSE %.4f | latent-mean RMSE %.4f | noise corr %.4f | aleatoric %.4f (truth %.4f) | epistemic %.4f\n",
        test_scores.logpdf,
        test_scores.rmse,
        test_scores.latent_mean_rmse,
        test_scores.noise_correlation,
        test_scores.mean_aleatoric,
        test_scores.true_mean_aleatoric,
        test_scores.mean_epistemic,
    )
end

# ╔═╡ eb4af872-2877-46a0-b4cf-5921467eeed4
md"""
## What to inspect

The purple curve tests the new capability: the observation-noise variance is
now a learned function of ``x`` rather than one global scalar. The orange curve
is the epistemic contribution from both the dense readout covariance and the
uncertain learned features.

The principal held-out scores are:

| metric | value |
|---|---:|
| mean log predictive density | **$(round(test_scores.logpdf; digits = 4))** |
| observed-target RMSE | **$(round(test_scores.rmse; digits = 4))** |
| latent-mean RMSE | **$(round(test_scores.latent_mean_rmse; digits = 4))** |
| correlation with true noise | **$(round(test_scores.noise_correlation; digits = 4))** |
| learned mean aleatoric variance | **$(round(test_scores.mean_aleatoric; digits = 4))** |
| true mean aleatoric variance | **$(round(test_scores.true_mean_aleatoric; digits = 4))** |

This is the one-precision learned-feature analogue of `L = 2` in
`why_hierarchy_deep_kernel.jl`. A further hierarchy can replace the constant
`SCORE_CARRIER` with another learned `score → Exp → precision` chain without
changing the mean head.
"""

# ╔═╡ Cell order:
# ╠═1019a7b1-250b-4d6f-93be-1b515b9f7e15
# ╠═2f42b773-c616-47ed-8f42-a1d189a3d766
# ╟─4327cabc-0bc4-418a-a9e7-e9c66b5a1915
# ╠═f45fa3c6-5df6-4983-ae20-2d44185b3ce1
# ╟─61654e1b-fd0e-4fdc-ba7e-ce8cd73fb32a
# ╠═5ae83b6a-fb28-4493-bbf4-a2c42580348a
# ╠═cc31c16e-7532-44f8-aa58-39b78f2e1bcd
# ╟─3ebcb381-250c-43fa-b963-4a3fb11a180d
# ╠═b2c193c7-d874-4b8e-94f2-027cbbc19b69
# ╠═d374accc-1151-45d6-b8c2-bb62320b894d
# ╠═961232fc-5814-4764-915e-fe4e793f703e
# ╠═3d2a1af1-950c-4634-9a7b-b0638b032f92
# ╠═87b3a45a-7483-466f-a2c4-e7cf7399f5cc
# ╠═34691611-3332-4bd1-88cc-dd17e9db031a
# ╟─ef5bc21c-2b31-4548-aaf5-781f8411a280
# ╠═f1d0d79c-d90c-440a-95cc-2351c720003a
# ╠═94ee0697-4ddd-45ca-a6d4-bd62b40b111e
# ╠═82e7399c-cb96-47f5-b7ec-d704b0f23cf3
# ╟─eb4af872-2877-46a0-b4cf-5921467eeed4
