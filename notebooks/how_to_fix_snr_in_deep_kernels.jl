### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 2e8ec1ca-b936-4cb9-a0c7-43b49d3fa651
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 461debad-a740-4a9b-8052-5f02ac9b377d
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

# ╔═╡ 320ee55d-aa4b-493a-8151-dad80c24814a
md"""
# Can a residual readout restore SNR in a deep kernel?

In a serial uncertainty hierarchy, only the first score touches the observation
precision. An upper score influences the data through every carrier below it:

```text
s₂ → λ₂ → s₁ → λ₁ → y
```

That semantics is attractive — each level describes the reliability of the level below
it — but the information passed by a precision factor becomes flat once cavity
uncertainty dominates. On the one-dimensional benchmark, the first level has a
signal-to-uncertainty ratio near five while the next level is near zero.

This notebook tests one architectural change: retain the serial carrier and add a
centred residual path from the upper score to the single observation precision.
"""

# ╔═╡ 763eb91a-8bca-4963-bd94-49bd41db4180
md"""
## The residual readout

For the three-layer model, the serial graph is

```math
\begin{aligned}
s_2(x) &\sim \mathcal N\!\left(w_2^\top\phi(x),c_3^{-1}\right),
&\lambda_2(x)&=\exp s_2(x),\\
s_1(x) &\sim \mathcal N\!\left(w_1^\top\phi(x),\lambda_2(x)^{-1}\right),
&\lambda_y(x)&=\exp s_1(x).
\end{aligned}
```

The residual model replaces only the final line:

```math
\eta(x)
=
s_1(x)+a_2\,\frac{s_2(x)-b_2}{\sigma_2},
\qquad
\lambda_y(x)=\exp\eta(x).
```

Here ``b_2`` is the upper score's prior anchor and ``\sigma_2`` is its prior predictive
standard deviation. The centring is load-bearing: an upper level at its anchor
contributes exactly zero, so inactive depth recovers the serial model locally.

There is still only **one likelihood factor** for each observation. The skip changes
how that likelihood is parameterised; it does not count the observation twice.
For a Gaussian likelihood the Fisher information about log-precision is ``1/2``, so a
skip of strength ``a_2`` supplies direct information ``a_2^2/2`` to the upper score
instead of forcing all information through ``\lambda_2``.
"""

# ╔═╡ 0ce41e5e-cd87-47d9-a7a1-f6652fe0ec98
md"""
## Experimental protocol

The comparison changes only the readout. Data, random features, priors, initialization,
message projections, damping, and iteration counts are shared.

The data are split once into training, validation, and test sets. Validation selects a
skip strength from a short fixed list; the locked test set is used once for the final
serial-versus-residual comparison. Reporting the complete coefficient curve keeps the
architecture search visible.

A preliminary stability boundary at ``a_2=0.8`` was rejected before scoring because it
made a Gaussian weight cavity non-positive-definite. The reported grid therefore stops
at ``0.4``: a skip must first define stable inference before predictive metrics matter.
"""

# ╔═╡ 0e3ec438-9042-4775-b4ba-f5c27c9fe362
begin
    # ---- benchmark ---------------------------------------------------------
    const N_SAMPLES = parse(Int, get(ENV, "SNR_N_SAMPLES", "600"))
    const VALIDATION_FRACTION = 1 / 6
    const TEST_FRACTION = 1 / 6
    const DATA_SEED = 7

    # ---- fixed kernel and priors -------------------------------------------
    const N_BASIS = parse(Int, get(ENV, "SNR_N_BASIS", "16"))
    const LENGTHSCALE = 0.25
    const SIGNAL_SD = 1.0
    const LEVEL_SD = 0.4
    const ANCHOR_SD = 1.0
    const TOP_CARRIER = 25.0

    # ---- architecture search ----------------------------------------------
    const SKIP_WEIGHTS = parse.(
        Float64,
        split(get(ENV, "SNR_SKIP_WEIGHTS", "0.05,0.1,0.2,0.4"), ","),
    )

    # ---- inference ---------------------------------------------------------
    const ITERATIONS = parse(Int, get(ENV, "SNR_ITERATIONS", "240"))
    const ALPHA = 0.6
    const CARRIER_ALPHA = 0.3
    const MAX_STEP = 0.5
    const METHOD = :damped
    const PREDICT_ITERATIONS =
        parse(Int, get(ENV, "SNR_PREDICT_ITERATIONS", "60"))
    const PREDICT_ALPHA = 0.5
end

# ╔═╡ b8fce073-6351-44a9-9ea1-b46f6c3ac21e
begin
    true_mean(x) = -(x + 0.5) * sin(3pi * x)
    true_noise_variance(x) = abs2(0.45 * (x + 0.5))

    data = let
        rng = StableRNG(DATA_SEED)
        x = randn(rng, N_SAMPLES)
        y = true_mean.(x) .+
            sqrt.(true_noise_variance.(x)) .* randn(rng, N_SAMPLES)
        order = randperm(rng, N_SAMPLES)
        n_test = round(Int, TEST_FRACTION * N_SAMPLES)
        n_validation = round(Int, VALIDATION_FRACTION * N_SAMPLES)
        test = order[1:n_test]
        validation = order[(n_test + 1):(n_test + n_validation)]
        train = order[(n_test + n_validation + 1):end]
        (;
            x_train = x[train], y_train = y[train],
            x_validation = x[validation], y_validation = y[validation],
            x_test = x[test], y_test = y[test],
        )
    end

    grid = collect(range(-3.0, 3.0; length = 241))
    @printf(
        "%d training, %d validation, %d test points\n",
        length(data.y_train), length(data.y_validation), length(data.y_test),
    )
end

# ╔═╡ 664917d2-90b6-432f-abac-4b016cb205e8
begin
    feature_map = let rng = StableRNG(20260726)
        frequencies = randn(rng, N_BASIS) ./ LENGTHSCALE
        phases = 2pi .* rand(rng, N_BASIS)
        scale = sqrt(2 / N_BASIS)
        x -> vcat(scale .* cos.(frequencies .* x .+ phases), [1.0])
    end

    const N_FEATURES = N_BASIS + 1
    design(xs) = [feature_map(x) for x in xs]
end

# ╔═╡ b25c7791-53ef-4868-a85f-267a10a7b2f9
md"""
## Two graphs, one changed edge

`precision[2,o]` remains the carrier of `score[1,o]` in both models. In the residual
model, `precision[1,o]` is the exponential of the combined readout ``\eta`` rather than
the exponential of ``s_1`` alone.
"""

# ╔═╡ d16d24a8-dece-492f-9404-22cf1c92e8b7
@model function serial_l3_model(
    y, features, v_prior, w_priors, top_carrier,
    exp_deps, exp_damping, carrier_deps, carrier_damping,
)
    local w, score, precision
    v ~ v_prior
    w[1] ~ w_priors[1]
    w[2] ~ w_priors[2]

    for o in eachindex(features)
        score[2, o] ~ softdot(features[o], w[2], top_carrier)
        precision[2, o] ~ Exp(score[2, o]) where {
            dependencies = exp_deps, meta = exp_damping,
        }
        score[1, o] ~ softdot(
            features[o], w[1], precision[2, o],
        ) where {
            dependencies = carrier_deps, meta = carrier_damping,
        }
        precision[1, o] ~ Exp(score[1, o]) where {
            dependencies = exp_deps, meta = exp_damping,
        }
        y[o] ~ softdot(features[o], v, precision[1, o])
    end
end

# ╔═╡ f4cd8911-bff7-4cfb-9bca-375cbb21d46b
@model function residual_l3_model(
    y, features, v_prior, w_priors, top_carrier,
    skip_scale, skip_offset,
    exp_deps, exp_damping, carrier_deps, carrier_damping,
)
    local w, score, precision
    local scaled_upper, skip_bias, eta
    v ~ v_prior
    w[1] ~ w_priors[1]
    w[2] ~ w_priors[2]

    for o in eachindex(features)
        score[2, o] ~ softdot(features[o], w[2], top_carrier)
        precision[2, o] ~ Exp(score[2, o]) where {
            dependencies = exp_deps, meta = exp_damping,
        }
        score[1, o] ~ softdot(
            features[o], w[1], precision[2, o],
        ) where {
            dependencies = carrier_deps, meta = carrier_damping,
        }

        scaled_upper[o] := skip_scale * score[2, o]
        skip_bias[o] ~ NormalMeanVariance(skip_offset, 1e-12)
        eta[o] ~ ManyPlus(
            inputs = [score[1, o], scaled_upper[o], skip_bias[o]],
        )

        precision[1, o] ~ Exp(eta[o]) where {
            dependencies = exp_deps, meta = exp_damping,
        }
        y[o] ~ softdot(features[o], v, precision[1, o])
    end
end

# ╔═╡ 1715cb35-b364-48cb-a02f-b34e02959506
begin
    @constraints function l3_constraints()
        q(v, w, score, precision, y) = q(v, y)q(w, score)q(precision)
        q(v)::MomentForm()
        q(w)::MomentForm()
    end

    @constraints function residual_l3_constraints()
        q(
            v, w, score, precision,
            scaled_upper, skip_bias, eta, y,
        ) = q(v, y) *
            q(w, score, scaled_upper, skip_bias, eta) *
            q(precision)
        q(v)::MomentForm()
        q(w)::MomentForm()
    end
end

# ╔═╡ c8290a25-06d5-4ac7-9965-c729d687a391
begin
    noise_anchor(xs, ys) =
        -log(max(mean(abs2.(diff(ys[sortperm(xs)]))) / 2, 1e-8))

    gaussian(means, variances) =
        MvNormalMeanCovariance(collect(means), Matrix(Diagonal(collect(variances))))

    function make_priors(rows, xs, ys)
        anchor = noise_anchor(xs, ys)
        level_prior(k) = gaussian(
            vcat(zeros(N_BASIS), [k == 1 ? anchor : log(TOP_CARRIER)]),
            vcat(fill(abs2(LEVEL_SD), N_BASIS), [abs2(ANCHOR_SD)]),
        )
        w = [level_prior(1), level_prior(2)]
        upper_mean, upper_covariance = mean_cov(w[2])
        upper_variance = [
            dot(row, upper_covariance * row) + inv(TOP_CARRIER)
            for row in rows
        ]
        return (;
            v = gaussian(zeros(N_FEATURES), fill(abs2(SIGNAL_SD), N_FEATURES)),
            w,
            upper_center = log(TOP_CARRIER),
            upper_scale = sqrt(mean(upper_variance)),
        )
    end
end

# ╔═╡ 8a21ea56-b916-420d-ab52-c8f21738e345
begin
    exp_deps() = NGMPDependencies(
        out = nothing, in = nothing;
        projection = TangentProjection(type = ClosedForm),
    )
    carrier_deps() = NGMPDependencies(
        γ = nothing;
        projection = TangentProjection(type = Unscented),
    )
    damping(alpha) =
        DampingMeta(
            alpha = alpha, beta = 0.0, max_step = MAX_STEP, method = METHOD,
        )

    gamma_initials(means, variances) = begin
        shapes = 1 .+ inv.(max.(variances, 1e-6))
        GammaShapeRate.(shapes, shapes .* exp.(-means))
    end

    function initial_states(priors, rows, skip_weight)
        Φ = reduce(hcat, rows)'
        n_points = length(rows)
        score = Matrix{NormalMeanVariance{Float64}}(undef, 2, n_points)
        score_means = Matrix{Float64}(undef, 2, n_points)
        score_variances = Matrix{Float64}(undef, 2, n_points)

        for level in 1:2
            weight_mean, weight_covariance = mean_cov(priors.w[level])
            means = Φ * weight_mean
            variances =
                vec(sum((Φ * weight_covariance) .* Φ; dims = 2)) .+
                inv(TOP_CARRIER)
            score[level, :] = NormalMeanVariance.(means, variances)
            score_means[level, :] = means
            score_variances[level, :] = variances
        end

        skip_scale = skip_weight / priors.upper_scale
        skip_offset = -skip_scale * priors.upper_center
        scaled_upper_means = skip_scale .* score_means[2, :]
        scaled_upper_variances =
            abs2(skip_scale) .* score_variances[2, :]
        scaled_upper =
            NormalMeanVariance.(scaled_upper_means, scaled_upper_variances)
        skip_bias =
            fill(NormalMeanVariance(skip_offset, 1e-12), n_points)
        eta_means =
            score_means[1, :] .+ scaled_upper_means .+ skip_offset
        eta_variances =
            score_variances[1, :] .+ scaled_upper_variances
        eta = NormalMeanVariance.(eta_means, eta_variances)

        precision = Matrix{GammaShapeRate{Float64}}(undef, 2, n_points)
        precision[1, :] = gamma_initials(eta_means, eta_variances)
        precision[2, :] =
            gamma_initials(score_means[2, :], score_variances[2, :])

        return (;
            score,
            precision,
            scaled_upper,
            skip_bias,
            eta,
            skip_scale,
            skip_offset,
        )
    end
end

# ╔═╡ 64af094b-8158-47d5-98c7-8beeb743bd69
begin
    serial_spec = (; label = "serial", kind = :serial, skip_weight = 0.0)
    residual_specs = [
        (;
            label = @sprintf("residual a₂ = %.2f", weight),
            kind = :residual,
            skip_weight = weight,
        )
        for weight in SKIP_WEIGHTS
    ]
    model_specs = [serial_spec; residual_specs]

    function build_l3_model(spec, priors, states, alpha)
        common = (;
            v_prior = priors.v,
            w_priors = priors.w,
            top_carrier = TOP_CARRIER,
            exp_deps = exp_deps(),
            exp_damping = damping(alpha),
            carrier_deps = carrier_deps(),
            carrier_damping = damping(CARRIER_ALPHA),
        )
        return spec.kind === :serial ?
            serial_l3_model(; common...) :
            residual_l3_model(
                ;
                common...,
                skip_scale = states.skip_scale,
                skip_offset = states.skip_offset,
            )
    end
end

# ╔═╡ c1bc27bd-b28f-4a86-8e0b-cd05814d3e14
begin
    function fit_model(spec)
        rows = design(data.x_train)
        priors = make_priors(rows, data.x_train, data.y_train)
        states = initial_states(priors, rows, spec.skip_weight)

        init = spec.kind === :serial ?
            @initialization(begin
                q(v) = deepcopy(priors.v)
                q(w) = deepcopy(priors.w)
                q(score) = states.score
                q(precision) = states.precision
            end) :
            @initialization(begin
                q(v) = deepcopy(priors.v)
                q(w) = deepcopy(priors.w)
                q(score) = states.score
                q(precision) = states.precision
                q(scaled_upper) = states.scaled_upper
                q(skip_bias) = states.skip_bias
                q(eta) = states.eta
                μ(v) = deepcopy(priors.v)
                μ(w) = deepcopy(priors.w)
                μ(score) = states.score
                μ(precision) = states.precision
                μ(scaled_upper) = states.scaled_upper
                μ(skip_bias) = states.skip_bias
                μ(eta) = states.eta
            end)

        result = infer(
            model = build_l3_model(spec, priors, states, ALPHA),
            data = (y = data.y_train, features = rows),
            constraints = spec.kind === :serial ?
                l3_constraints() : residual_l3_constraints(),
            initialization = init,
            returnvars = spec.kind === :serial ?
                (
                    v = KeepLast(), w = KeepLast(),
                    score = KeepLast(), precision = KeepLast(),
                ) :
                (
                    v = KeepLast(), w = KeepLast(),
                    score = KeepLast(), precision = KeepLast(), eta = KeepLast(),
                ),
            iterations = ITERATIONS,
            free_energy = false,
            showprogress = false,
            allow_node_contraction = false,
            options = (limit_stack_depth = 100,),
        )

        return (;
            spec,
            v = result.posteriors[:v],
            w = collect(vec(result.posteriors[:w])),
            score = result.posteriors[:score],
            precision = result.posteriors[:precision],
            eta = spec.kind === :serial ? nothing : result.posteriors[:eta],
            upper_center = priors.upper_center,
            upper_scale = priors.upper_scale,
        )
    end

    function predict_model(fit, xs)
        rows = design(xs)
        priors = (;
            v = fit.v,
            w = fit.w,
            upper_center = fit.upper_center,
            upper_scale = fit.upper_scale,
        )
        states = initial_states(priors, rows, fit.spec.skip_weight)

        init = fit.spec.kind === :serial ?
            @initialization(begin
                q(v) = deepcopy(priors.v)
                q(w) = deepcopy(priors.w)
                q(score) = states.score
                q(precision) = states.precision
            end) :
            @initialization(begin
                q(v) = deepcopy(priors.v)
                q(w) = deepcopy(priors.w)
                q(score) = states.score
                q(precision) = states.precision
                q(scaled_upper) = states.scaled_upper
                q(skip_bias) = states.skip_bias
                q(eta) = states.eta
                μ(v) = deepcopy(priors.v)
                μ(w) = deepcopy(priors.w)
                μ(score) = states.score
                μ(precision) = states.precision
                μ(scaled_upper) = states.scaled_upper
                μ(skip_bias) = states.skip_bias
                μ(eta) = states.eta
            end)

        result = infer(
            model = build_l3_model(fit.spec, priors, states, PREDICT_ALPHA),
            data = (features = rows,),
            constraints = fit.spec.kind === :serial ?
                l3_constraints() : residual_l3_constraints(),
            initialization = init,
            predictvars = (y = KeepLast(),),
            returnvars = (precision = KeepLast(),),
            iterations = PREDICT_ITERATIONS,
            free_energy = false,
            showprogress = false,
            allow_node_contraction = false,
            options = (limit_stack_depth = 100,),
        )

        marginals = collect(vec(result.predictions[:y]))
        precisions = collect(result.posteriors[:precision][1, :])
        inverse_precisions =
            InverseGamma.(shape.(precisions), rate.(precisions))
        return (;
            mean = mean.(marginals),
            variance = var.(marginals),
            noise_variance = mean.(inverse_precisions),
        )
    end
end

# ╔═╡ a99195d7-c2ed-44e4-b88c-53d82d1929b4
begin
    function score_prediction(prediction, xs, ys)
        variance = max.(prediction.variance, 1e-8)
        residual = ys .- prediction.mean
        truth = true_noise_variance.(xs)
        return (;
            logpdf = mean(
                -0.5 .* (
                    log.(2pi .* variance) .+
                    abs2.(residual) ./ variance
                ),
            ),
            rmse = sqrt(mean(abs2.(residual))),
            coverage = mean(abs.(residual) .<= 1.96 .* sqrt.(variance)),
            noise_corr = cor(prediction.noise_variance, truth),
        )
    end

    function carrier_diagnostic(fit)
        upper = fit.precision[2, :]
        log_means = digamma.(shape.(upper)) .- log.(rate.(upper))
        log_sds = sqrt.(trigamma.(shape.(upper)))
        weight_mean = mean(fit.w[2])
        raw_signal = std(log_means)
        uncertainty = mean(log_sds)
        return (;
            signal = raw_signal,
            uncertainty,
            snr = raw_signal / uncertainty,
            feature_weight_norm = norm(view(weight_mean, 1:N_BASIS)),
            intercept_shift = weight_mean[end] - fit.upper_center,
            direct_eta_signal =
                fit.spec.skip_weight / fit.upper_scale * std(mean.(fit.score[2, :])),
        )
    end
end

# ╔═╡ 05d95139-d537-4790-bf95-d2376fa7facc
begin
    fits = map(model_specs) do spec
        elapsed = @elapsed fit = fit_model(spec)
        @printf("%-22s fitted in %5.1f s\n", spec.label, elapsed)
        fit
    end

    validation_predictions =
        map(fit -> predict_model(fit, data.x_validation), fits)
    test_predictions = map(fit -> predict_model(fit, data.x_test), fits)
    grid_predictions = map(fit -> predict_model(fit, grid), fits)

    validation_scores = [
        score_prediction(prediction, data.x_validation, data.y_validation)
        for prediction in validation_predictions
    ]
    test_scores = [
        score_prediction(prediction, data.x_test, data.y_test)
        for prediction in test_predictions
    ]
    diagnostics = carrier_diagnostic.(fits)

    residual_indices = 2:length(model_specs)
    selected_index = residual_indices[
        argmax([validation_scores[index].logpdf for index in residual_indices])
    ]
    selected_spec = model_specs[selected_index]
    nothing
end

# ╔═╡ aa1fb765-971b-4156-88d9-bb12840f20dd
md"""
## Does the skip deliver information?

The validation column chooses the coefficient. Test metrics are shown for every
coefficient for transparency, but only the validation-selected residual is used in the
headline comparison.

`upper SNR` is measured on ``q(\lambda_2)`` exactly as in the serial hierarchy:
between-input variation in posterior mean log-precision divided by average pointwise
posterior standard deviation. `direct η signal` is the standard deviation of the
upper level's actual residual contribution to the output log-precision.
"""

# ╔═╡ 32666093-f901-4c12-9266-27cdfef37860
begin
    @printf(
        "%-22s %10s %10s %8s %8s %10s %10s %10s %11s\n",
        "model", "val logpdf", "test logpdf", "RMSE", "cov95",
        "noise corr", "upper SNR", "|w₂ feat|", "skip signal",
    )
    for index in eachindex(model_specs)
        @printf(
            "%-22s %10.4f %10.4f %8.4f %8.3f %10.3f %10.4f %10.4f %11.4f\n",
            model_specs[index].label,
            validation_scores[index].logpdf,
            test_scores[index].logpdf,
            test_scores[index].rmse,
            test_scores[index].coverage,
            test_scores[index].noise_corr,
            diagnostics[index].snr,
            diagnostics[index].feature_weight_norm,
            diagnostics[index].direct_eta_signal,
        )
    end
    @printf("\nvalidation-selected residual: %s\n", selected_spec.label)
end

# ╔═╡ c7398f6a-5c7b-497a-8d47-1c7716eec904
begin
    serial_test = test_scores[1]
    selected_test = test_scores[selected_index]
    serial_diagnostic = diagnostics[1]
    selected_diagnostic = diagnostics[selected_index]

    test_logpdf_gain = selected_test.logpdf - serial_test.logpdf
    snr_multiplier = selected_diagnostic.snr / serial_diagnostic.snr

    selected_label_display = selected_spec.label
    serial_test_logpdf_display = round(serial_test.logpdf; digits = 4)
    serial_test_rmse_display = round(serial_test.rmse; digits = 4)
    serial_test_coverage_display = round(serial_test.coverage; digits = 3)
    serial_noise_corr_display = round(serial_test.noise_corr; digits = 3)
    serial_diagnostic_snr_display = round(serial_diagnostic.snr; digits = 4)
    selected_test_logpdf_display = round(selected_test.logpdf; digits = 4)
    selected_test_rmse_display = round(selected_test.rmse; digits = 4)
    selected_test_coverage_display = round(selected_test.coverage; digits = 3)
    selected_noise_corr_display = round(selected_test.noise_corr; digits = 3)
    selected_diagnostic_snr_display =
        round(selected_diagnostic.snr; digits = 4)
    comparison_logpdf_gain_display = round(test_logpdf_gain; digits = 4)
    comparison_snr_multiplier_display = round(snr_multiplier; digits = 2)

    md"""
    ## Locked test comparison

    Validation selected **$(selected_label_display)**.

    | model | test logpdf | RMSE | coverage | noise corr | upper SNR |
    |:--|--:|--:|--:|--:|--:|
    | serial L=3 | $(serial_test_logpdf_display) | $(serial_test_rmse_display) | $(serial_test_coverage_display) | $(serial_noise_corr_display) | $(serial_diagnostic_snr_display) |
    | residual L=3 | $(selected_test_logpdf_display) | $(selected_test_rmse_display) | $(selected_test_coverage_display) | $(selected_noise_corr_display) | $(selected_diagnostic_snr_display) |

    The residual changes held-out logpdf by
    **$(comparison_logpdf_gain_display) nats per point** and multiplies the upper
    carrier SNR by **$(comparison_snr_multiplier_display)**.
    """
end

# ╔═╡ 93ade203-827b-4524-a546-100c30f7ed73
begin
    residual_weights = [spec.skip_weight for spec in residual_specs]
    residual_validation_logpdf = [
        validation_scores[index].logpdf for index in residual_indices
    ]
    residual_snr = [diagnostics[index].snr for index in residual_indices]

    coefficient_panel = plot(
        residual_weights,
        residual_validation_logpdf;
        marker = :circle,
        linewidth = 2,
        color = :steelblue,
        xlabel = "skip strength a₂",
        ylabel = "validation logpdf",
        label = "residual",
        title = "Architecture selection",
    )
    hline!(
        coefficient_panel,
        [validation_scores[1].logpdf];
        color = :black,
        linestyle = :dash,
        label = "serial",
    )

    snr_panel = plot(
        residual_weights,
        residual_snr;
        marker = :circle,
        linewidth = 2,
        color = :darkorange,
        xlabel = "skip strength a₂",
        ylabel = "upper carrier SNR",
        label = "residual",
        title = "Does direct evidence reach level 2?",
    )
    hline!(
        snr_panel,
        [diagnostics[1].snr];
        color = :black,
        linestyle = :dash,
        label = "serial",
    )

    selection_figure =
        plot(coefficient_panel, snr_panel; layout = (1, 2), size = (1_000, 380))
end

# ╔═╡ d2795094-ad1d-42c1-8197-830d303c1176
begin
    comparison_indices = [1, selected_index]
    panels = []
    for index in comparison_indices
        prediction = grid_predictions[index]
        band = 1.96 .* sqrt.(max.(prediction.variance, 0.0))
        mean_panel = plot(
            grid,
            prediction.mean;
            ribbon = band,
            fillalpha = 0.2,
            color = :steelblue,
            linewidth = 2,
            xlabel = "x",
            ylabel = "y",
            label = "q(y*) ± 1.96 SD",
            title = model_specs[index].label,
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
            color = :gray55,
            markersize = 2,
            markeralpha = 0.3,
            markerstrokewidth = 0,
            label = "train",
        )

        variance_panel = plot(
            grid,
            max.(prediction.noise_variance, 1e-4);
            color = :steelblue,
            linewidth = 2,
            yscale = :log10,
            xlabel = "x",
            ylabel = "noise variance",
            label = "posterior mean",
            title = "$(model_specs[index].label): noise",
        )
        plot!(
            variance_panel,
            grid,
            max.(true_noise_variance.(grid), 1e-4);
            color = :black,
            linestyle = :dashdot,
            linewidth = 2,
            label = "truth",
        )
        push!(panels, mean_panel, variance_panel)
    end

    prediction_figure =
        plot(panels...; layout = (2, 2), size = (1_050, 780))
    haskey(ENV, "SNR_FIGURE") &&
        savefig(prediction_figure, ENV["SNR_FIGURE"])
    prediction_figure
end

# ╔═╡ 3ad95346-1f18-4a76-a3c5-8918c0a9e841
begin
    rmse_change = selected_test.rmse - serial_test.rmse
    weight_norm_multiplier =
        selected_diagnostic.feature_weight_norm /
        serial_diagnostic.feature_weight_norm

    serial_snr_display = round(serial_diagnostic.snr; digits = 4)
    serial_weight_norm_display =
        round(serial_diagnostic.feature_weight_norm; digits = 4)
    selected_snr_display = round(selected_diagnostic.snr; digits = 4)
    selected_weight_norm_display =
        round(selected_diagnostic.feature_weight_norm; digits = 4)
    snr_multiplier_display = round(snr_multiplier; digits = 1)
    weight_norm_multiplier_display =
        round(weight_norm_multiplier; digits = 1)
    direct_eta_signal_display =
        round(selected_diagnostic.direct_eta_signal; digits = 4)
    test_logpdf_gain_display = round(test_logpdf_gain; digits = 4)
    serial_coverage_display = round(serial_test.coverage; digits = 3)
    rmse_change_display = round(rmse_change; digits = 4)

    md"""
    ## Finding: the skip restores upper-level SNR

    The serial upper level is almost inactive: its SNR is
    ``$(serial_snr_display)`` and its feature-weight norm is
    ``$(serial_weight_norm_display)``. With the
    validation-selected residual readout, those become
    ``$(selected_snr_display)`` and
    ``$(selected_weight_norm_display)`` — SNR rises by
    **$(snr_multiplier_display)×** and the learned feature norm by
    **$(weight_norm_multiplier_display)×**. The upper score contributes a
    nonzero ``$(direct_eta_signal_display)`` standard
    deviation to the output log-precision.

    That recovered information is predictively useful, but the effect sizes differ.
    Locked-test logpdf improves by **$(test_logpdf_gain_display) nats per
    point**, while coverage remains
    ``$(serial_coverage_display)`` and RMSE changes by only
    ``$(rmse_change_display)``. The skip therefore improves how
    predictive variance is distributed, not the fitted mean.

    The conclusion is narrower than “more depth always helps.” The serial information
    bottleneck was real: useful upper-level structure becomes learnable when it receives
    a direct likelihood path. But all skipped scores still predict the same scalar
    log-precision, so the likelihood identifies their weighted sum more strongly than
    the individual decomposition. The instability at ``a_2=0.8`` also shows that the
    direct path needs controlled residual scaling. Centring, shrinkage, and distinct
    lengthscales become increasingly important before extending the construction to
    several skipped levels.
    """
end

# ╔═╡ Cell order:
# ╠═2e8ec1ca-b936-4cb9-a0c7-43b49d3fa651
# ╠═461debad-a740-4a9b-8052-5f02ac9b377d
# ╟─320ee55d-aa4b-493a-8151-dad80c24814a
# ╟─763eb91a-8bca-4963-bd94-49bd41db4180
# ╟─0ce41e5e-cd87-47d9-a7a1-f6652fe0ec98
# ╠═0e3ec438-9042-4775-b4ba-f5c27c9fe362
# ╠═b8fce073-6351-44a9-9ea1-b46f6c3ac21e
# ╠═664917d2-90b6-432f-abac-4b016cb205e8
# ╟─b25c7791-53ef-4868-a85f-267a10a7b2f9
# ╠═d16d24a8-dece-492f-9404-22cf1c92e8b7
# ╠═f4cd8911-bff7-4cfb-9bca-375cbb21d46b
# ╠═1715cb35-b364-48cb-a02f-b34e02959506
# ╠═c8290a25-06d5-4ac7-9965-c729d687a391
# ╠═8a21ea56-b916-420d-ab52-c8f21738e345
# ╠═64af094b-8158-47d5-98c7-8beeb743bd69
# ╠═c1bc27bd-b28f-4a86-8e0b-cd05814d3e14
# ╠═a99195d7-c2ed-44e4-b88c-53d82d1929b4
# ╠═05d95139-d537-4790-bf95-d2376fa7facc
# ╟─aa1fb765-971b-4156-88d9-bb12840f20dd
# ╠═32666093-f901-4c12-9266-27cdfef37860
# ╠═c7398f6a-5c7b-497a-8d47-1c7716eec904
# ╠═93ade203-827b-4524-a546-100c30f7ed73
# ╠═d2795094-ad1d-42c1-8197-830d303c1176
# ╠═3ad95346-1f18-4a76-a3c5-8918c0a9e841
