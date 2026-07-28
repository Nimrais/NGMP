### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 7ad34f84-f698-4c53-bf18-32a1529d1062
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 93685109-d147-40f7-9631-f78f4d79614b
begin
    ENV["GKSwstype"] = "100"

    using Distributions: InverseGamma
    using LinearAlgebra
    using Plots
    using Printf
    using Random
    using RxInfer
    using SpecialFunctions: digamma, trigamma
    using StableRNGs
    using Statistics
    using SurrogateModelling

    import ProbabilisticEnsembling: Exp
end

# ╔═╡ bd8f4104-c503-420d-9896-c55be3124ee9
md"""
# A vector readout for deep precision kernels

A serial precision hierarchy has only one route from an upper score to the
observations: the score must change the carrier precision of the level below it.
That route becomes progressively less informative with depth.

This notebook gives every score a direct route to the output log-precision. The
scores are assembled by `MvStack`, and one additional Gaussian vector learns their
joint readout:

```math
\begin{aligned}
\tilde s_{k,o} &\sim
\mathcal N\!\left(s_{k,o},\tau_{\mathrm{copy}}^{-1}\right),\\
z_o &= \operatorname{stack}
\left(\tilde s_{1,o},\ldots,\tilde s_{K,o},1\right),\\
\eta_o &\sim
\mathcal N\!\left(a^\top z_o,\tau_{\mathrm{readout}}^{-1}\right),\\
\lambda_{1,o} &= \exp(\eta_o).
\end{aligned}
```

The serial carrier hierarchy is unchanged. Only its final scalar aggregation is
replaced. The experiment asks two separate questions:

1. Does the vector readout keep upper-level message SNR away from zero?
2. Does that additional information improve held-out predictive quality?
"""

# ╔═╡ 46fb921f-8b36-4684-8a0a-8f8b6eaed02d
md"""
## Why the score-copy edge is finite

Messages inside the serial structured cluster are information-form cavity sites.
Such a site may be improper even when the corresponding posterior marginal is
proper. `MvStack` represents a Gaussian vector belief, so its scalar inputs must
have proper moments.

The finite Gaussian edge

```math
\tilde s_k \sim \mathcal N(s_k,\tau_{\mathrm{copy}}^{-1})
```

forms a variational boundary and gives the vector readout proper scalar inputs. It
also makes the bandwidth of the direct route explicit. Here both the copy and the
readout have standard deviation ``0.1``.

The first coefficient of ``a`` has prior mean one and a much smaller prior
standard deviation than the others. This preserves the identity path
``\eta\approx s_1``; otherwise the score scale and the readout scale are weakly
identified and can shrink each other.
"""

# ╔═╡ 29313b5b-6c9f-4fd7-9f0e-2f40f238a941
begin
    # ---- benchmark ---------------------------------------------------------
    const N_SAMPLES =
        parse(Int, get(ENV, "MVSTACK_N_SAMPLES", "600"))
    const HOLDOUT_FRACTION = 1 / 3
    const DATA_SEED = 7

    # ---- fixed kernel and hierarchy priors --------------------------------
    const N_BASIS =
        parse(Int, get(ENV, "MVSTACK_N_BASIS", "16"))
    const LENGTHSCALE = 0.25
    const SIGNAL_SD = 1.0
    const LEVEL_SD = 0.4
    const ANCHOR_SD = 1.0
    const TOP_CARRIER = 25.0

    # ---- vector readout ----------------------------------------------------
    const TOTAL_SKIP_WEIGHT =
        parse(Float64, get(ENV, "MVSTACK_TOTAL_SKIP_WEIGHT", "0.4"))
    const READOUT_WEIGHT_SD =
        parse(Float64, get(ENV, "MVSTACK_READOUT_WEIGHT_SD", "0.15"))
    const BASE_WEIGHT_SD =
        parse(Float64, get(ENV, "MVSTACK_BASE_WEIGHT_SD", "0.01"))
    const INTERCEPT_WEIGHT_SD =
        parse(Float64, get(ENV, "MVSTACK_INTERCEPT_WEIGHT_SD", "0.15"))
    const COPY_PRECISION =
        parse(Float64, get(ENV, "MVSTACK_COPY_PRECISION", "100.0"))
    const READOUT_PRECISION =
        parse(Float64, get(ENV, "MVSTACK_READOUT_PRECISION", "100.0"))
    const BIAS_VARIANCE =
        parse(Float64, get(ENV, "MVSTACK_BIAS_VARIANCE", "1e-4"))

    # ---- inference ---------------------------------------------------------
    const LAYER_COUNTS = parse.(
        Int,
        split(get(ENV, "MVSTACK_LAYER_COUNTS", "2,3,4,5"), ","),
    )
    const ITERATIONS =
        parse(Int, get(ENV, "MVSTACK_ITERATIONS", "240"))
    const PREDICT_ITERATIONS =
        parse(Int, get(ENV, "MVSTACK_PREDICT_ITERATIONS", "60"))
    const ALPHA = 0.6
    const PREDICT_ALPHA = 0.5
    const CARRIER_ALPHA = 0.3
    const MAX_STEP = 0.5
    const METHOD = :damped

    all(n_layers -> n_layers >= 2, LAYER_COUNTS) ||
        throw(ArgumentError("MVSTACK_LAYER_COUNTS must contain values >= 2"))
    all(
        value -> value > 0,
        (
            READOUT_WEIGHT_SD,
            BASE_WEIGHT_SD,
            INTERCEPT_WEIGHT_SD,
            COPY_PRECISION,
            READOUT_PRECISION,
            BIAS_VARIANCE,
        ),
    ) || throw(ArgumentError("all readout scales and precisions must be positive"))
end

# ╔═╡ 4920c6a5-83e6-44e0-9827-bf2b74f3997a
md"""
## Benchmark

```math
x\sim\mathcal N(0,1),\qquad
y=-(x+\tfrac12)\sin(3\pi x)+\varepsilon,\qquad
\varepsilon\sim
\mathcal N\!\left(0,[0.45(x+\tfrac12)]^2\right).
```

The conditional mean oscillates quickly, while the noise vanishes at
``x=-1/2`` and grows toward the edges. A third of the observations is held out
once and used for every depth.
"""

# ╔═╡ 34bcaee2-e18f-479a-8c10-1574e6b6c465
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
        test = order[1:n_test]
        train = order[(n_test + 1):end]
        (;
            x_train = x[train],
            y_train = y[train],
            x_test = x[test],
            y_test = y[test],
        )
    end

    grid = collect(range(-3.0, 3.0; length = 241))
    @printf(
        "%d training points, %d held out\n",
        length(data.y_train),
        length(data.y_test),
    )
end

# ╔═╡ ad68bb58-0134-403a-a0b7-2cb76824a4ce
begin
    feature_map = let rng = StableRNG(20260726)
        frequencies = randn(rng, N_BASIS) ./ LENGTHSCALE
        phases = 2pi .* rand(rng, N_BASIS)
        scale = sqrt(2 / N_BASIS)
        x -> vcat(
            scale .* cos.(frequencies .* x .+ phases),
            [1.0],
        )
    end

    const N_FEATURES = N_BASIS + 1
    design(xs) = [feature_map(x) for x in xs]

    noise_anchor(xs, ys) =
        -log(max(mean(abs2.(diff(ys[sortperm(xs)]))) / 2, 1e-8))

    gaussian(means, variances) =
        MvNormalMeanCovariance(
            collect(means),
            Matrix(Diagonal(collect(variances))),
        )
end

# ╔═╡ 933984b4-86b1-4424-9e28-caf650a23f6d
md"""
## One model, parameterized by depth

There are ``K=L-1`` precision scores. The top score has a constant carrier;
every lower score retains the serial carrier supplied by the score above it.

At the output boundary, all ``K`` scores are copied, stacked, and read by the
additional vector ``a``. Thus increasing `L` extends both the serial hierarchy
and the dimension of a single readout vector.
"""

# ╔═╡ d66719e4-5cd4-41de-bbdc-0cf44955a005
@model function mvstack_hierarchy_model(
    y,
    features,
    n_levels,
    v_prior,
    w_priors,
    readout_prior,
    top_carrier,
    copy_precision,
    readout_precision,
    bias_variance,
    exp_deps,
    exp_damping,
    carrier_deps,
    carrier_damping,
)
    local w, score, precision
    local readout_weight, readout_score, stack_bias, score_vector, eta

    v ~ v_prior
    readout_weight ~ readout_prior
    for level in 1:n_levels
        w[level] ~ w_priors[level]
    end

    for observation in eachindex(features)
        score[n_levels, observation] ~ softdot(
            features[observation],
            w[n_levels],
            top_carrier,
        )
        if n_levels > 1
            precision[n_levels, observation] ~ Exp(
                score[n_levels, observation],
            ) where {
                dependencies = exp_deps,
                meta = exp_damping,
            }

            for level in (n_levels - 1):-1:1
                score[level, observation] ~ softdot(
                    features[observation],
                    w[level],
                    precision[level + 1, observation],
                ) where {
                    dependencies = carrier_deps,
                    meta = carrier_damping,
                }
                if level > 1
                    precision[level, observation] ~ Exp(
                        score[level, observation],
                    ) where {
                        dependencies = exp_deps,
                        meta = exp_damping,
                    }
                end
            end
        end

        for level in 1:n_levels
            readout_score[level, observation] ~ NormalMeanPrecision(
                score[level, observation],
                copy_precision,
            )
        end
        stack_bias[observation] ~ NormalMeanVariance(
            1.0,
            bias_variance,
        )
        score_vector[observation] ~ MvStack(inputs = [
            level <= n_levels ?
                readout_score[level, observation] :
                stack_bias[observation]
            for level in 1:(n_levels + 1)
        ])
        eta[observation] ~ softdot(
            score_vector[observation],
            readout_weight,
            readout_precision,
        )

        precision[1, observation] ~ Exp(
            eta[observation],
        ) where {
            dependencies = exp_deps,
            meta = exp_damping,
        }
        y[observation] ~ softdot(
            features[observation],
            v,
            precision[1, observation],
        )
    end
end

# ╔═╡ f8f3a56a-394f-44be-912a-d5c9618cbef4
@constraints function mvstack_hierarchy_constraints()
    q(
        v,
        w,
        readout_weight,
        score,
        precision,
        readout_score,
        stack_bias,
        score_vector,
        eta,
        y,
    ) =
        q(v, y) *
        q(w, score) *
        q(readout_weight, eta, precision) *
        q(readout_score, stack_bias, score_vector)

    q(v)::MomentForm()
    q(w)::MomentForm()
    q(readout_weight)::MomentForm()
end

# ╔═╡ 240933b2-2cf5-45b2-9f20-80565f1d3a67
begin
    exp_deps() = NGMPDependencies(
        out = nothing,
        in = nothing;
        projection = TangentProjection(type = ClosedForm),
    )

    carrier_deps() = NGMPDependencies(
        γ = nothing;
        projection = TangentProjection(type = Unscented),
    )

    damping(alpha) = DampingMeta(
        alpha = alpha,
        beta = 0.0,
        max_step = MAX_STEP,
        method = METHOD,
    )

    gamma_initials(means, variances) = begin
        shapes = 1 .+ inv.(max.(variances, 1e-6))
        GammaShapeRate.(shapes, shapes .* exp.(-means))
    end
end

# ╔═╡ 9b990673-43e8-46c1-a0b2-949f8746d4d6
begin
    function make_priors(n_layers, rows, xs, ys)
        n_levels = n_layers - 1
        anchor = noise_anchor(xs, ys)
        level_centers = [
            level == 1 ? anchor : log(TOP_CARRIER)
            for level in 1:n_levels
        ]
        level_prior(level) = gaussian(
            vcat(zeros(N_BASIS), [level_centers[level]]),
            vcat(
                fill(abs2(LEVEL_SD), N_BASIS),
                [abs2(ANCHOR_SD)],
            ),
        )
        w = [level_prior(level) for level in 1:n_levels]
        level_scales = [
            let
                _, covariance = mean_cov(w[level])
                predictive_variance = [
                    dot(row, covariance * row) + inv(TOP_CARRIER)
                    for row in rows
                ]
                sqrt(mean(predictive_variance))
            end
            for level in 1:n_levels
        ]

        n_upper = n_levels - 1
        per_upper_weight =
            n_upper == 0 ? 0.0 : TOTAL_SKIP_WEIGHT / n_upper
        coefficients = zeros(n_levels)
        coefficients[1] = 1.0
        for level in 2:n_levels
            coefficients[level] =
                per_upper_weight / level_scales[level]
        end
        intercept = -sum(
            (
                coefficients[level] * level_centers[level]
                for level in 2:n_levels
            );
            init = 0.0,
        )
        readout_mean = vcat(coefficients, [intercept])
        readout_variance = vcat(
            [abs2(BASE_WEIGHT_SD)],
            fill(abs2(READOUT_WEIGHT_SD), n_levels - 1),
            [abs2(INTERCEPT_WEIGHT_SD)],
        )

        return (;
            v = gaussian(
                zeros(N_FEATURES),
                fill(abs2(SIGNAL_SD), N_FEATURES),
            ),
            w,
            readout = gaussian(readout_mean, readout_variance),
            readout_mean,
            level_centers,
            level_scales,
        )
    end

    function initial_states(priors, rows)
        feature_matrix = reduce(hcat, rows)'
        n_levels = length(priors.w)
        n_points = length(rows)

        score = Matrix{NormalMeanVariance{Float64}}(
            undef,
            n_levels,
            n_points,
        )
        score_means =
            Matrix{Float64}(undef, n_levels, n_points)
        score_variances =
            Matrix{Float64}(undef, n_levels, n_points)
        for level in 1:n_levels
            weight_mean, weight_covariance =
                mean_cov(priors.w[level])
            means = feature_matrix * weight_mean
            variances =
                vec(
                    sum(
                        (feature_matrix * weight_covariance) .*
                        feature_matrix;
                        dims = 2,
                    ),
                ) .+ inv(TOP_CARRIER)
            score[level, :] =
                NormalMeanVariance.(means, variances)
            score_means[level, :] = means
            score_variances[level, :] = variances
        end

        readout_score_variances =
            score_variances .+ inv(COPY_PRECISION)
        readout_score = NormalMeanVariance.(
            score_means,
            readout_score_variances,
        )
        stack_bias = fill(
            NormalMeanVariance(1.0, BIAS_VARIANCE),
            n_points,
        )
        readout_mean, readout_covariance =
            mean_cov(priors.readout)
        score_vector =
            Vector{MvNormalMeanCovariance{Float64}}(undef, n_points)
        eta =
            Vector{NormalMeanVariance{Float64}}(undef, n_points)
        eta_means = Vector{Float64}(undef, n_points)
        eta_variances = Vector{Float64}(undef, n_points)

        for observation in 1:n_points
            stacked_mean =
                vcat(score_means[:, observation], [1.0])
            stacked_covariance = Matrix(
                Diagonal(
                    vcat(
                        readout_score_variances[:, observation],
                        [BIAS_VARIANCE],
                    ),
                ),
            )
            score_vector[observation] =
                MvNormalMeanCovariance(
                    stacked_mean,
                    stacked_covariance,
                )

            eta_mean = dot(stacked_mean, readout_mean)
            eta_variance =
                tr(readout_covariance * stacked_covariance) +
                dot(
                    readout_mean,
                    stacked_covariance * readout_mean,
                ) +
                dot(
                    stacked_mean,
                    readout_covariance * stacked_mean,
                ) +
                inv(READOUT_PRECISION)
            eta[observation] =
                NormalMeanVariance(eta_mean, eta_variance)
            eta_means[observation] = eta_mean
            eta_variances[observation] = eta_variance
        end

        precision = Matrix{GammaShapeRate{Float64}}(
            undef,
            n_levels,
            n_points,
        )
        precision[1, :] =
            gamma_initials(eta_means, eta_variances)
        for level in 2:n_levels
            precision[level, :] = gamma_initials(
                score_means[level, :],
                score_variances[level, :],
            )
        end

        return (;
            score,
            precision,
            readout_score,
            stack_bias,
            score_vector,
            eta,
        )
    end
end

# ╔═╡ fa496821-fe24-4462-ae2c-d719d7d61bbd
begin
    function model_for(priors, n_levels, alpha)
        return mvstack_hierarchy_model(
            n_levels = n_levels,
            v_prior = priors.v,
            w_priors = priors.w,
            readout_prior = priors.readout,
            top_carrier = TOP_CARRIER,
            copy_precision = COPY_PRECISION,
            readout_precision = READOUT_PRECISION,
            bias_variance = BIAS_VARIANCE,
            exp_deps = exp_deps(),
            exp_damping = damping(alpha),
            carrier_deps = carrier_deps(),
            carrier_damping = damping(CARRIER_ALPHA),
        )
    end

    function model_initialization(priors, states)
        return @initialization(begin
            q(v) = deepcopy(priors.v)
            q(w) = deepcopy(priors.w)
            q(readout_weight) = deepcopy(priors.readout)
            q(score) = states.score
            q(precision) = states.precision
            q(readout_score) = states.readout_score
            q(stack_bias) = states.stack_bias
            q(score_vector) = states.score_vector
            q(eta) = states.eta
        end)
    end

    function fit_layers(n_layers)
        n_levels = n_layers - 1
        rows = design(data.x_train)
        priors = make_priors(
            n_layers,
            rows,
            data.x_train,
            data.y_train,
        )
        states = initial_states(priors, rows)

        result = infer(
            model = model_for(priors, n_levels, ALPHA),
            data = (y = data.y_train, features = rows),
            constraints = mvstack_hierarchy_constraints(),
            initialization =
                model_initialization(priors, states),
            returnvars = (
                v = KeepLast(),
                w = KeepLast(),
                readout_weight = KeepLast(),
                score = KeepLast(),
                precision = KeepLast(),
            ),
            iterations = ITERATIONS,
            free_energy = false,
            showprogress = false,
            allow_node_contraction = false,
            options = (limit_stack_depth = 100,),
        )

        return (;
            n_layers,
            v = result.posteriors[:v],
            w = collect(vec(result.posteriors[:w])),
            readout_weight =
                result.posteriors[:readout_weight],
            score = result.posteriors[:score],
            precision = result.posteriors[:precision],
            level_centers = priors.level_centers,
            readout_prior_mean = priors.readout_mean,
        )
    end

    function predict_layers(fit, xs)
        rows = design(xs)
        priors = (;
            v = fit.v,
            w = fit.w,
            readout = fit.readout_weight,
        )
        states = initial_states(priors, rows)

        result = infer(
            model = model_for(
                priors,
                fit.n_layers - 1,
                PREDICT_ALPHA,
            ),
            data = (features = rows,),
            constraints = mvstack_hierarchy_constraints(),
            initialization =
                model_initialization(priors, states),
            predictvars = (y = KeepLast(),),
            returnvars = (precision = KeepLast(),),
            iterations = PREDICT_ITERATIONS,
            free_energy = false,
            showprogress = false,
            allow_node_contraction = false,
            options = (limit_stack_depth = 100,),
        )

        marginals = collect(vec(result.predictions[:y]))
        precisions =
            collect(result.posteriors[:precision][1, :])

        _, mean_weight_covariance = mean_cov(fit.v)
        latent_variance = [
            dot(row, mean_weight_covariance * row)
            for row in rows
        ]
        inverse_precisions =
            InverseGamma.(shape.(precisions), rate.(precisions))
        noise_mean = mean.(inverse_precisions)
        variance_posterior = (;
            mean = latent_variance .+ noise_mean,
            median = latent_variance .+
                quantile.(inverse_precisions, 0.5),
            lower = latent_variance .+
                quantile.(inverse_precisions, 0.025),
            upper = latent_variance .+
                quantile.(inverse_precisions, 0.975),
            latent = latent_variance,
            noise_mean,
        )

        return (;
            mean = mean.(marginals),
            variance = var.(marginals),
            variance_posterior,
        )
    end
end

# ╔═╡ e4a4beb4-9ac0-49d2-b7e3-14aee9d63d64
begin
    fits = map(LAYER_COUNTS) do n_layers
        elapsed = @elapsed fit = fit_layers(n_layers)
        @printf("MvStack L=%d fitted in %5.1f s\n", n_layers, elapsed)
        fit
    end

    grid_predictions =
        map(fit -> predict_layers(fit, grid), fits)
    test_predictions =
        map(fit -> predict_layers(fit, data.x_test), fits)
    nothing
end

# ╔═╡ e3651518-5526-4c84-a226-452575be3a77
md"""
## Does every score receive usable evidence?

For each carrier, `signal` is the standard deviation across training inputs of
the posterior mean log-precision. `uncert` is its average pointwise posterior
standard deviation. Their ratio is the message SNR used in the serial hierarchy.

The additional `readout` column is the learned coefficient multiplying that
score. Level 1 is the base log-precision path; larger indices are the upper
uncertainty levels.
"""

# ╔═╡ 5a3f52d1-6fd4-4aec-8b59-99fa8aa258f5
begin
    function layer_message_diagnostic(fit, level)
        precision_marginals = fit.precision[level, :]
        log_means =
            digamma.(shape.(precision_marginals)) .-
            log.(rate.(precision_marginals))
        log_sds =
            sqrt.(trigamma.(shape.(precision_marginals)))
        weight_mean = mean(fit.w[level])
        readout_mean = mean(fit.readout_weight)
        signal = std(log_means)
        uncertainty = mean(log_sds)
        return (;
            signal,
            uncertainty,
            snr = signal / uncertainty,
            feature_weight_norm =
                norm(view(weight_mean, 1:N_BASIS)),
            intercept_shift =
                weight_mean[end] - fit.level_centers[level],
            readout = readout_mean[level],
            direct_eta_signal =
                abs(readout_mean[level]) *
                std(mean.(fit.score[level, :])),
        )
    end

    depth_diagnostics = [
        [
            layer_message_diagnostic(fit, level)
            for level in 1:(fit.n_layers - 1)
        ]
        for fit in fits
    ]

    @printf(
        "%-8s %5s %10s %10s %10s %10s %12s %9s %11s\n",
        "model",
        "level",
        "signal",
        "uncert",
        "SNR",
        "|w feat|",
        "Δ intercept",
        "readout",
        "η signal",
    )
    for (n_layers, diagnostics) in
        zip(LAYER_COUNTS, depth_diagnostics)
        for (level, diagnostic) in enumerate(diagnostics)
            @printf(
                "L=%-6d %5d %10.4f %10.4f %10.4f %10.4f %+12.4f %+9.4f %11.4f\n",
                n_layers,
                level,
                diagnostic.signal,
                diagnostic.uncertainty,
                diagnostic.snr,
                diagnostic.feature_weight_norm,
                diagnostic.intercept_shift,
                diagnostic.readout,
                diagnostic.direct_eta_signal,
            )
        end
    end
    depth_diagnostics
end

# ╔═╡ fcbe60f1-4d77-4b77-a4d5-cf8548a3f946
begin
    @printf("\n%-8s %-36s %-36s\n", "model", "readout prior mean", "posterior mean")
    for fit in fits
        @printf(
            "L=%-6d %-36s %-36s\n",
            fit.n_layers,
            string(round.(fit.readout_prior_mean; digits = 4)),
            string(round.(mean(fit.readout_weight); digits = 4)),
        )
    end
end

# ╔═╡ ba2e5db5-fc11-4df2-9186-57456a8e04b4
md"""
## Predictive mean and predictive variance

Prediction reruns the same graph with the learned posteriors used as priors and
`y` declared through `predictvars`. The left column shows the resulting
``q(y_*)``. The right column separates uncertainty in the mean weights from the
posterior over observation variance.

If ``q(\lambda_o)=\operatorname{Gamma}(\alpha_o,\beta_o)`` in shape-rate form,
then

```math
\operatorname{Var}(y_o\mid\lambda_o,\mathcal D)
=\phi_o^\top\Sigma_v\phi_o+\lambda_o^{-1},
\qquad
\lambda_o^{-1}\sim
\operatorname{InverseGamma}(\alpha_o,\beta_o).
```

The shaded region on the right is therefore an analytic pointwise 95% credible
interval for predictive variance; no sampling is used.
"""

# ╔═╡ a0616e38-be7b-4b91-8489-ebf68e74c17e
begin
    depth_label(n_layers) = "L = $n_layers layers"

    prediction_panels = []
    for (n_layers, prediction) in
        zip(LAYER_COUNTS, grid_predictions)
        band =
            1.96 .* sqrt.(max.(prediction.variance, 0.0))
        variance_summary = prediction.variance_posterior

        mean_panel = plot(
            grid,
            prediction.mean;
            ribbon = band,
            fillalpha = 0.2,
            color = :steelblue,
            linewidth = 2,
            label = "q(y*) ± 1.96 SD",
            xlabel = "x",
            ylabel = "y",
            title = depth_label(n_layers),
            titlefontsize = 9,
            legend = :topleft,
            legendfontsize = 6,
            ylims = (-4.5, 4.5),
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
            color = :gray55,
            markersize = 2,
            markeralpha = 0.35,
            markerstrokewidth = 0,
            label = "train",
        )

        variance_panel = plot(
            grid,
            max.(variance_summary.lower, 1e-4);
            fillrange =
                max.(variance_summary.upper, 1e-4),
            fillcolor = :steelblue,
            fillalpha = 0.18,
            color = :transparent,
            linewidth = 0,
            label = "95% CrI for variance",
            xlabel = "x",
            ylabel = "variance",
            yscale = :log10,
            title = "$(depth_label(n_layers)): variance",
            titlefontsize = 9,
            legend = :topleft,
            legendfontsize = 6,
            ylims = (1e-4, 1e2),
        )
        plot!(
            variance_panel,
            grid,
            max.(variance_summary.mean, 1e-4);
            color = :steelblue,
            linewidth = 2,
            label = "posterior mean variance",
        )
        plot!(
            variance_panel,
            grid,
            max.(variance_summary.latent, 1e-4);
            color = :darkorange,
            linestyle = :dot,
            linewidth = 1.5,
            label = "mean-weight contribution",
        )
        plot!(
            variance_panel,
            grid,
            max.(true_noise_variance.(grid), 1e-4);
            color = :black,
            linestyle = :dashdot,
            linewidth = 2,
            label = "true noise variance",
        )
        push!(prediction_panels, mean_panel, variance_panel)
    end

    prediction_figure = plot(
        prediction_panels...;
        layout = (length(LAYER_COUNTS), 2),
        size = (1_100, 1_500),
    )
    haskey(ENV, "MVSTACK_FIGURE") &&
        savefig(prediction_figure, ENV["MVSTACK_FIGURE"])
    prediction_figure
end

# ╔═╡ 5f34610b-bcc6-4cef-8088-e194ed0ad87d
md"""
## Held-out predictive quality

`logpdf` scores the complete predictive distribution and is the primary metric.
`RMSE` only scores the predictive mean. `cov95` is empirical coverage of the
pointwise 95% predictive interval. `noise corr` compares the posterior mean
noise variance with the known conditional variance at held-out inputs.

Because every model uses the same train/test split, basis, priors, and inference
budget, each row isolates the effect of changing depth.
"""

# ╔═╡ 7cc6c609-aac8-4748-8b19-144105ba685c
begin
    function score_prediction(prediction, xs, ys)
        variance = max.(prediction.variance, 1e-8)
        residual = ys .- prediction.mean
        estimated_noise =
            prediction.variance_posterior.noise_mean
        truth = true_noise_variance.(xs)
        effectively_constant =
            std(estimated_noise) <=
            sqrt(eps()) *
            max(mean(abs, estimated_noise), eps())
        noise_corr = effectively_constant ?
            0.0 : cor(estimated_noise, truth)
        return (;
            logpdf = mean(
                -0.5 .* (
                    log.(2pi .* variance) .+
                    abs2.(residual) ./ variance
                ),
            ),
            rmse = sqrt(mean(abs2.(residual))),
            coverage = mean(
                abs.(residual) .<=
                1.96 .* sqrt.(variance),
            ),
            noise_corr,
        )
    end

    holdout = [
        (;
            n_layers,
            score_prediction(
                prediction,
                data.x_test,
                data.y_test,
            )...,
        )
        for (n_layers, prediction) in
        zip(LAYER_COUNTS, test_predictions)
    ]

    @printf(
        "%-18s %10s %9s %9s %11s\n",
        "model",
        "logpdf",
        "RMSE",
        "cov95",
        "noise corr",
    )
    for row in holdout
        @printf(
            "%-18s %10.4f %9.4f %9.3f %11.3f\n",
            depth_label(row.n_layers),
            row.logpdf,
            row.rmse,
            row.coverage,
            row.noise_corr,
        )
    end
    holdout
end

# ╔═╡ 0913f532-fabf-4fe4-8c5f-cba59635c777
begin
    best_index =
        argmax(getproperty.(holdout, :logpdf))
    best_row = holdout[best_index]
    shallow_row = holdout[1]
    deepest_row = holdout[end]
    deepest_diagnostics = depth_diagnostics[end]
    upper_diagnostics = length(deepest_diagnostics) > 1 ?
        deepest_diagnostics[2:end] : []

    best_model_display = depth_label(best_row.n_layers)
    best_logpdf_display =
        round(best_row.logpdf; digits = 4)
    shallow_model_display =
        depth_label(shallow_row.n_layers)
    deepest_model_display =
        depth_label(LAYER_COUNTS[end])
    deepest_logpdf_change_display = @sprintf(
        "%+.4f",
        deepest_row.logpdf - shallow_row.logpdf,
    )
    deepest_rmse_change_display = @sprintf(
        "%+.4f",
        deepest_row.rmse - shallow_row.rmse,
    )
    deepest_coverage_change_display = @sprintf(
        "%+.3f",
        deepest_row.coverage - shallow_row.coverage,
    )
    deepest_upper_snr_display = isempty(upper_diagnostics) ?
        "no upper level" :
        join(
            round.(
                getproperty.(upper_diagnostics, :snr);
                digits = 4,
            ),
            ", ",
        )
    deepest_readout_display = join(
        round.(
            mean(fits[end].readout_weight)[
                2:(fits[end].n_layers - 1)
            ];
            digits = 4,
        ),
        ", ",
    )

    md"""
    ## What the depth comparison says

    The best held-out log predictive density is **$(best_logpdf_display)** for
    **$(best_model_display)**. From **$(shallow_model_display)** to
    **$(deepest_model_display)**, logpdf changes by
    **$(deepest_logpdf_change_display) nats per point**, RMSE by
    **$(deepest_rmse_change_display)**, and 95% coverage by
    **$(deepest_coverage_change_display)**.

    In **$(deepest_model_display)** the upper-level SNR values are
    $(deepest_upper_snr_display) and their learned readout coefficients are
    $(deepest_readout_display). The direct vector path therefore prevents the
    deeper scores from becoming numerically invisible. It does not imply that every
    additional level improves prediction: all scores still compete to explain the
    same scalar ``\eta``, so held-out logpdf decides whether the recovered structure
    is useful.

    The architecture changes the saturation pattern. The serial model loses upper
    information before it reaches the likelihood; the vector model exposes that
    information to the likelihood and shifts the remaining limitation to readout
    shrinkage and identifiability.
    """
end

# ╔═╡ Cell order:
# ╠═7ad34f84-f698-4c53-bf18-32a1529d1062
# ╠═93685109-d147-40f7-9631-f78f4d79614b
# ╟─bd8f4104-c503-420d-9896-c55be3124ee9
# ╟─46fb921f-8b36-4684-8a0a-8f8b6eaed02d
# ╠═29313b5b-6c9f-4fd7-9f0e-2f40f238a941
# ╟─4920c6a5-83e6-44e0-9827-bf2b74f3997a
# ╠═34bcaee2-e18f-479a-8c10-1574e6b6c465
# ╠═ad68bb58-0134-403a-a0b7-2cb76824a4ce
# ╟─933984b4-86b1-4424-9e28-caf650a23f6d
# ╠═d66719e4-5cd4-41de-bbdc-0cf44955a005
# ╠═f8f3a56a-394f-44be-912a-d5c9618cbef4
# ╠═240933b2-2cf5-45b2-9f20-80565f1d3a67
# ╠═9b990673-43e8-46c1-a0b2-949f8746d4d6
# ╠═fa496821-fe24-4462-ae2c-d719d7d61bbd
# ╠═e4a4beb4-9ac0-49d2-b7e3-14aee9d63d64
# ╟─e3651518-5526-4c84-a226-452575be3a77
# ╠═5a3f52d1-6fd4-4aec-8b59-99fa8aa258f5
# ╠═fcbe60f1-4d77-4b77-a4d5-cf8548a3f946
# ╟─ba2e5db5-fc11-4df2-9186-57456a8e04b4
# ╠═a0616e38-be7b-4b91-8489-ebf68e74c17e
# ╟─5f34610b-bcc6-4cef-8088-e194ed0ad87d
# ╠═7cc6c609-aac8-4748-8b19-144105ba685c
# ╠═0913f532-fabf-4fe4-8c5f-cba59635c777
