### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 04c6ad7a-58a4-4bab-966d-9ee3b8a124ea
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 79c682ba-eb24-47d0-a9fd-e65061727534
begin
    # Load shared definitions in an isolated module. Pluto reruns cells in its
    # workspace, where directly including the UCI files would redefine their
    # UCIPaperProtocol module and fail. No UCI names leak from this module.
    ENV["UCI_DIRECT_VECTOR_TRANSPORT_ALPHA"] = "0.6"
    ENV["UCI_DIRECT_VECTOR_TRANSPORT_NESTEROV_ALPHA"] = "0.6"
    backend_module = Module(
        gensym(:XORDeepKernelBackendRuntime),
        true,
        true,
    )
    Core.eval(
        backend_module,
        :(include(path) = Base.include($backend_module, path)),
    )
    Base.include(backend_module, joinpath(
        @__DIR__, "..", "scripts",
        "uci_deep_kernel_direct_paper_benchmark.jl",
    ))

    using Distributions: Chisq
    using LinearAlgebra
    using Plots
    using Printf
    using Random
    using RxInfer
    using StableRNGs
    using Statistics
    using SurrogateModelling
end

# ╔═╡ ab5e1eb1-4389-4ce2-80cf-0588470db53c
md"""
# XOR and checkerboards with a deep kernel

This notebook fits one depth-three heteroscedastic random-feature model. Choose either
the direct frozen-Gaussian backend or RxInfer as the inference engine. XOR/checkerboard
labels are represented as noisy regression targets near zero and one; classification
uses a threshold of ``0.5``.

The default configuration is the best shared UCI configuration found in the sweep:

```text
Matérn-3/2 multiscale preprocessing, 400 random features, depth 3,
fixed lengthscale 1.5, precision-prior gain 1.5,
vector transport beta 0.8, alpha 0.6.
```
"""

# ╔═╡ 73642bb5-4d87-4edf-8ee6-66d3404f737e
md"""
## Controls

Edit `BACKEND` and `PREPROCESSING`, then run the notebook from this cell downward.

Valid backends:

```julia
:direct
:rxinfer
```

Valid preprocessing choices:

```julia
:rbf
:multiscale_rbf_linear
:multiscale_matern32_linear
```
"""

# ╔═╡ c65b6b97-e4c9-4f31-9ea2-5372f7222a47
begin
    # ---- primary controls -------------------------------------------------
    BACKEND = :direct
    PREPROCESSING = :multiscale_matern32_linear

    # ---- selected UCI defaults -------------------------------------------
    FEATURE_DIMENSION = 400
    MODEL_DEPTH = 5
    FIXED_LENGTHSCALE = 1.5
    PRECISION_PRIOR_GAIN = 1.5
    OPTIMIZER = :vector_transport
    OPTIMIZER_BETA = 0.8
    OPTIMIZER_ALPHA = 0.6

    # ---- XOR experiment ---------------------------------------------------
    # (2, 2) is XOR. Rectangular examples: (3, 1), (3, 2), (4, 2).
    CHECKERBOARD_SIZE = (2, 2)
    N_SAMPLES_XOR = 400
    TRAIN_FRACTION = 0.70
    TARGET_NOISE_SD = 0.08
    DATA_SEED_XOR = 20_260_802
    FEATURE_SEED_XOR = 20_260_803
    GRID_RESOLUTION = 61
    RXINFER_PREDICTION_BATCH = 256

    BACKEND in (:direct, :rxinfer) ||
        throw(ArgumentError("BACKEND must be :direct or :rxinfer"))
    PREPROCESSING in (
        :rbf,
        :multiscale_rbf_linear,
        :multiscale_matern32_linear,
    ) || throw(ArgumentError("unknown preprocessing: $PREPROCESSING"))
    MODEL_DEPTH >= 1 || throw(ArgumentError("MODEL_DEPTH must be positive"))
    FEATURE_DIMENSION > 0 ||
        throw(ArgumentError("FEATURE_DIMENSION must be positive"))
    FIXED_LENGTHSCALE > 0 ||
        throw(ArgumentError("FIXED_LENGTHSCALE must be positive"))
    PRECISION_PRIOR_GAIN > 0 ||
        throw(ArgumentError("PRECISION_PRIOR_GAIN must be positive"))
    all(dimension -> dimension > 0, CHECKERBOARD_SIZE) ||
        throw(ArgumentError("checkerboard dimensions must be positive"))
end

# ╔═╡ 2f21cf06-604e-4c6d-8199-a26bb9d83bbd
md"""
## Data

Inputs are uniform on ``[-2,2]^2``. `CHECKERBOARD_SIZE = (2, 2)` is classic XOR.
Independent horizontal and vertical counts allow rectangular patterns such as
``3×1``, ``3×2``, or ``4×2``. The clean label is perturbed with Gaussian noise
and clipped to ``[0,1]``. Inputs and targets are standardized using training data
only; predictions are transformed back before scoring and plotting.
"""

# ╔═╡ d83fc607-c50d-4b75-bde8-98c177f024c4
begin
    function checkerboard_label(x1, x2, checkerboard_size)
        x_cells, y_cells = checkerboard_size
        cell_x = clamp(
            floor(Int, x_cells * (x1 + 2) / 4),
            0,
            x_cells - 1,
        )
        cell_y = clamp(
            floor(Int, y_cells * (x2 + 2) / 4),
            0,
            y_cells - 1,
        )
        return Float64(isodd(cell_x + cell_y))
    end

    function make_xor_data(
        n,
        noise_sd,
        seed,
        train_fraction;
        checkerboard_size,
    )
        rng = StableRNG(seed)
        X = 4 .* rand(rng, n, 2) .- 2
        clean = checkerboard_label.(
            X[:, 1],
            X[:, 2],
            Ref(checkerboard_size),
        )
        y = clamp.(clean .+ noise_sd .* randn(rng, n), 0.0, 1.0)
        order = randperm(rng, n)
        n_train = round(Int, train_fraction * n)
        train = order[1:n_train]
        test = order[(n_train + 1):end]
        return (;
            X_train = X[train, :],
            y_train = y[train],
            clean_train = clean[train],
            X_test = X[test, :],
            y_test = y[test],
            clean_test = clean[test],
        )
    end

    xor_data = make_xor_data(
        N_SAMPLES_XOR,
        TARGET_NOISE_SD,
        DATA_SEED_XOR,
        TRAIN_FRACTION,
        ;
        checkerboard_size = CHECKERBOARD_SIZE,
    )

    x_center = vec(mean(xor_data.X_train; dims = 1))
    x_scale = max.(vec(std(xor_data.X_train; dims = 1)), sqrt(eps()))
    standardize_x(X) = (Matrix{Float64}(X) .- x_center') ./ x_scale'

    y_center = mean(xor_data.y_train)
    y_scale = max(std(xor_data.y_train), sqrt(eps()))
    standardize_y(y) = (Float64.(y) .- y_center) ./ y_scale

    X_train_std = standardize_x(xor_data.X_train)
    X_test_std = standardize_x(xor_data.X_test)
    y_train_std = standardize_y(xor_data.y_train)

    axis_values = collect(range(-2.0, 2.0; length = GRID_RESOLUTION))
    X_grid = reduce(vcat, (
        reshape([x1, x2], 1, 2)
        for x2 in axis_values for x1 in axis_values
    ))
    X_grid_std = standardize_x(X_grid)
end

# ╔═╡ a4aef29f-a2ce-4bb4-84c0-c22749d0eb6e
md"""
## Three preprocessing variants

`rbf` uses one RBF random-feature bank. The multiscale variants divide the bank over
lengthscale bands ``0.5ℓ``, ``ℓ``, and ``2ℓ`` and append standardized linear
coordinates. The Matérn-3/2 version uses its Student-t spectral density.
"""

# ╔═╡ 8039dc6b-c363-4f36-a62c-28b10f82313b
begin
    function xor_feature_design(
        matrices...;
        preprocessing,
        feature_dimension,
        lengthscale,
        seed,
    )
        rng = StableRNG(seed)
        d = size(first(matrices), 2)

        if preprocessing == :rbf
            frequencies = randn(rng, feature_dimension, d) ./ lengthscale
        else
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
                        base .*= reshape(
                            sqrt.(3 ./ rand(rng, Chisq(3), count)),
                            :,
                            1,
                        )
                    end
                    base ./ (scale * lengthscale)
                end,
            )
        end

        phases = 2pi .* rand(rng, feature_dimension)
        transform(X) = begin
            random = sqrt(2 / feature_dimension) .*
                cos.(X * frequencies' .+ phases')
            preprocessing == :rbf ?
                hcat(random, ones(size(X, 1))) :
                hcat(random, X, ones(size(X, 1)))
        end
        return map(transform, matrices)
    end

    Φ_train, Φ_test, Φ_grid = xor_feature_design(
        X_train_std,
        X_test_std,
        X_grid_std;
        preprocessing = PREPROCESSING,
        feature_dimension = FEATURE_DIMENSION,
        lengthscale = FIXED_LENGTHSCALE,
        seed = FEATURE_SEED_XOR,
    )
end

# ╔═╡ e88773cc-63f0-4abc-a642-83dd16c07aa7
md"""
## Matched priors

Both backends start from the direct benchmark prior. The gain multiplies only the
standard deviation of non-intercept weights in precision levels. Mean weights and
precision anchors are unchanged.

For every precision-level feature weight, the baseline prior is

```math
w_{k,j} \sim \mathcal N(0, 0.4^2).
```

With prior gain ``g`` it becomes

```math
w_{k,j} \sim \mathcal N\!\left(0,(0.4g)^2\right).
```

At the default ``g=1.5``, the weight standard deviation is ``0.6`` and its variance is
``0.36`` instead of ``0.16``. This gives the learned log-precision functions more room
to vary across XOR/checkerboard inputs. It does **not** change the mean-function prior,
the intercept anchors, the optimizer step size, or the random features. Larger gain can
model stronger input-dependent variance, but it can also make uncertainty and inference
less stable, which is why it is a controlled hyperparameter.
"""

# ╔═╡ b04ee6bc-795b-405a-93c1-749c55e8ecff
begin
    function xor_direct_prior(depth, Φ, targets)
        prior = backend_module.direct_prior_parameters(depth, Φ, targets)
        gain_squared = abs2(PRECISION_PRIOR_GAIN)
        level_prior_precisions = map(prior.level_prior_precisions) do precision
            gained = copy(precision)
            gained[1:(end - 1), 1:(end - 1)] ./= gain_squared
            gained
        end
        return merge(prior, (; level_prior_precisions))
    end

    function xor_rxinfer_prior(depth, Φ, targets)
        direct = xor_direct_prior(depth, Φ, targets)
        return (;
            v = MvNormalMeanCovariance(
                direct.mean_prior_mean,
                inv(direct.mean_prior_precision),
            ),
            w = [
                MvNormalMeanCovariance(mean, inv(precision))
                for (mean, precision) in zip(
                    direct.level_prior_means,
                    direct.level_prior_precisions,
                )
            ],
            noise = GammaShapeRate(
                2.0,
                2.0 / backend_module.TOP_CARRIER,
            ),
        )
    end
end

# ╔═╡ 0ce0de9d-7a90-4e75-b798-9a76653f0dd5
begin
    function fit_xor_rxinfer(depth, Φ, targets)
        rows = collect(eachrow(Φ))
        prior = xor_rxinfer_prior(depth, Φ, targets)
        states = depth == 1 ? nothing :
            backend_module.initial_states(depth, prior, rows)
        init = depth == 1 ?
            @initialization(begin
                q(v) = deepcopy(prior.v)
                q(γ) = deepcopy(prior.noise)
            end) :
            @initialization(begin
                q(v) = deepcopy(prior.v)
                q(w) = deepcopy(prior.w)
                q(score) = states.score
                q(precision) = states.precision
            end)
        result = infer(
            model = backend_module.model(
                depth,
                prior,
                OPTIMIZER,
                OPTIMIZER_BETA,
                OPTIMIZER_ALPHA,
            ),
            data = (y = targets, features = rows),
            constraints = depth == 1 ?
                backend_module.gp_constraints() :
                backend_module.hierarchy_constraints(),
            initialization = init,
            returnvars = depth == 1 ?
                (v = KeepLast(), γ = KeepLast()) :
                (v = KeepLast(), w = KeepLast()),
            iterations = backend_module.ITERATIONS,
            free_energy = false,
            showprogress = false,
            options = (limit_stack_depth = 100,),
            disable_inference_error_hint = true,
        )
        return (;
            depth,
            v = result.posteriors[:v],
            w = depth == 1 ? [] : collect(vec(result.posteriors[:w])),
            noise = depth == 1 ? result.posteriors[:γ] : prior.noise,
        )
    end

    function predict_xor_rxinfer_batch(fit, Φ)
        rows = collect(eachrow(Φ))
        prior = (; v = fit.v, w = fit.w, noise = fit.noise)
        states = fit.depth == 1 ? nothing :
            backend_module.initial_states(fit.depth, prior, rows)
        init = fit.depth == 1 ?
            @initialization(begin
                q(v) = deepcopy(prior.v)
                q(γ) = deepcopy(prior.noise)
            end) :
            @initialization(begin
                q(v) = deepcopy(prior.v)
                q(w) = deepcopy(prior.w)
                q(score) = states.score
                q(precision) = states.precision
            end)
        result = infer(
            model = backend_module.model(
                fit.depth,
                prior,
                OPTIMIZER,
                OPTIMIZER_BETA,
                OPTIMIZER_ALPHA,
            ),
            data = (features = rows,),
            constraints = fit.depth == 1 ?
                backend_module.gp_constraints() :
                backend_module.hierarchy_constraints(),
            initialization = init,
            predictvars = (y = KeepLast(),),
            returnvars = fit.depth == 1 ?
                (γ = KeepLast(),) : (precision = KeepLast(),),
            iterations = backend_module.PREDICT_ITERATIONS,
            free_energy = false,
            showprogress = false,
            options = (limit_stack_depth = 100,),
            disable_inference_error_hint = true,
        )
        marginals = collect(vec(result.predictions[:y]))
        return (; mean = mean.(marginals), variance = var.(marginals))
    end

    function predict_xor_rxinfer(fit, Φ; batch_size)
        means = Float64[]
        variances = Float64[]
        for first_index in 1:batch_size:size(Φ, 1)
            indices = first_index:min(first_index + batch_size - 1, size(Φ, 1))
            prediction = predict_xor_rxinfer_batch(fit, Φ[indices, :])
            append!(means, prediction.mean)
            append!(variances, prediction.variance)
        end
        return (; mean = means, variance = variances)
    end
end

# ╔═╡ c8563798-a4a3-448f-bd8f-28dc39be7054
md"""
## Fit selected backend

The direct backend uses the closed-form frozen-Gaussian iterations. RxInfer constructs
the hierarchy graph and performs message passing. With 400 features, RxInfer is
expected to take substantially longer.
"""

# ╔═╡ b5467d15-0481-4fef-ad0a-a1ff90bfeb17
begin
    fit_seconds = @elapsed begin
        fitted_model = BACKEND == :direct ?
            backend_module.fit_direct_model(
                MODEL_DEPTH,
                Φ_train,
                y_train_std,
                OPTIMIZER,
                OPTIMIZER_BETA;
                prior_builder = xor_direct_prior,
            ) :
            fit_xor_rxinfer(MODEL_DEPTH, Φ_train, y_train_std)
    end
    @printf("%s backend fitted in %.2f seconds\n", BACKEND, fit_seconds)
end

# ╔═╡ c9bf37d4-9fb6-48fb-9a52-d932656d6909
begin
    function backend_prediction(fit, Φ)
        BACKEND == :direct ?
            backend_module.predict_direct_model(fit, Φ) :
            predict_xor_rxinfer(
                fit,
                Φ;
                batch_size = RXINFER_PREDICTION_BATCH,
            )
    end

    prediction_seconds = @elapsed begin
        test_prediction_std = backend_prediction(fitted_model, Φ_test)
        grid_prediction_std = backend_prediction(fitted_model, Φ_grid)
    end

    original_units(prediction) = (;
        mean = y_center .+ y_scale .* prediction.mean,
        variance = abs2(y_scale) .* prediction.variance,
    )
    test_prediction = original_units(test_prediction_std)
    grid_prediction = original_units(grid_prediction_std)
end

# ╔═╡ d9486644-da2b-47a6-8859-b210eb76978b
begin
    function xor_metrics(prediction, noisy_targets, clean_targets)
        variance = max.(prediction.variance, 1e-10)
        residual = noisy_targets .- prediction.mean
        probability = clamp.(prediction.mean, 0.0, 1.0)
        return (;
            mean_logpdf = mean(-0.5 .* (
                log.(2pi .* variance) .+ abs2.(residual) ./ variance
            )),
            rmse = sqrt(mean(abs2, residual)),
            accuracy = mean((probability .>= 0.5) .== (clean_targets .>= 0.5)),
            brier = mean(abs2.(probability .- clean_targets)),
        )
    end

    metrics = xor_metrics(
        test_prediction,
        xor_data.y_test,
        xor_data.clean_test,
    )

    md"""
    ## Held-out results

    | backend | preprocessing | log-PDF | RMSE | accuracy | Brier | fit seconds | prediction seconds |
    |:--|:--|--:|--:|--:|--:|--:|--:|
    | $(BACKEND) | $(PREPROCESSING) | $(round(metrics.mean_logpdf; digits = 4)) | $(round(metrics.rmse; digits = 4)) | $(round(metrics.accuracy; digits = 4)) | $(round(metrics.brier; digits = 4)) | $(round(fit_seconds; digits = 2)) | $(round(prediction_seconds; digits = 2)) |
    """
end

# ╔═╡ d505540e-c251-4c34-b256-8214de69929a
begin
    mean_surface = reshape(grid_prediction.mean, GRID_RESOLUTION, GRID_RESOLUTION)
    variance_surface = reshape(
        max.(grid_prediction.variance, 0.0),
        GRID_RESOLUTION,
        GRID_RESOLUTION,
    )

    mean_panel = heatmap(
        axis_values,
        axis_values,
        mean_surface;
        clim = (0, 1),
        color = :viridis,
        xlabel = "x₁",
        ylabel = "x₂",
        title = "predictive mean",
        aspect_ratio = :equal,
    )
    contour!(
        mean_panel,
        axis_values,
        axis_values,
        mean_surface;
        levels = [0.5],
        color = :white,
        linewidth = 2,
        label = false,
    )
    scatter!(
        mean_panel,
        xor_data.X_train[:, 1],
        xor_data.X_train[:, 2];
        marker_z = xor_data.clean_train,
        color = :coolwarm,
        markersize = 2,
        markeralpha = 0.45,
        markerstrokewidth = 0,
        colorbar = false,
        label = false,
    )

    uncertainty_panel = heatmap(
        axis_values,
        axis_values,
        variance_surface;
        color = :magma,
        xlabel = "x₁",
        ylabel = "x₂",
        title = "predictive variance",
        aspect_ratio = :equal,
    )

    plot(
        mean_panel,
        uncertainty_panel;
        layout = (1, 2),
        size = (1_050, 440),
        plot_title = "$(CHECKERBOARD_SIZE[1])×$(CHECKERBOARD_SIZE[2]) deep kernel — $(BACKEND), $(PREPROCESSING)",
    )
end

# ╔═╡ b5af08cb-40a6-46dd-943b-330e577decc2
md"""
## Suggested runs

Choose the desired checkerboard, preprocessing, and backend in the controls cell. XOR
is the ``2×2`` special case:

```julia
CHECKERBOARD_SIZE = (2, 2)  # XOR
CHECKERBOARD_SIZE = (3, 1)
CHECKERBOARD_SIZE = (3, 2)
CHECKERBOARD_SIZE = (6, 4)
BACKEND = :direct
```

Reduce `FEATURE_DIMENSION` temporarily if RxInfer is slow during interactive testing.

Both backends receive the same standardized data, random-feature realization, prior
gain, hierarchy depth, optimizer parameters, and prediction grid. The right-hand plot
shows predictive variance across the complete XOR/checkerboard surface.
"""

# ╔═╡ 4df404ce-bd26-42cc-8725-18c2d80d7945
md"""
## Run all suggested checkerboards

The final experiment uses the active backend, preprocessing, and model configuration
for ``2×2`` (XOR), ``3×1``, ``3×2``, and ``6×4``. The current checkerboard fit is
reused when possible. With `BACKEND = :rxinfer`, this section can take substantially
longer than the default direct backend.
"""

# ╔═╡ 97fe1bdb-340b-490f-acb9-91ed63eaf6c6
begin
    suggested_checkerboard_sizes = [
        (2, 2),
        (3, 1),
        (3, 2),
        
    ]

    function run_checkerboard_size(checkerboard_size)
        if checkerboard_size == CHECKERBOARD_SIZE
            return (;
                checkerboard_size,
                metrics,
                fit_seconds,
                prediction_seconds,
                grid_prediction,
            )
        end

        local_data = make_xor_data(
            N_SAMPLES_XOR,
            TARGET_NOISE_SD,
            DATA_SEED_XOR,
            TRAIN_FRACTION;
            checkerboard_size,
        )
        local_x_center = vec(mean(local_data.X_train; dims = 1))
        local_x_scale = max.(
            vec(std(local_data.X_train; dims = 1)),
            sqrt(eps()),
        )
        local_standardize_x(X) =
            (Matrix{Float64}(X) .- local_x_center') ./ local_x_scale'
        local_y_center = mean(local_data.y_train)
        local_y_scale = max(std(local_data.y_train), sqrt(eps()))
        local_y_train_std =
            (Float64.(local_data.y_train) .- local_y_center) ./ local_y_scale

        local_Φ_train, local_Φ_test, local_Φ_grid = xor_feature_design(
            local_standardize_x(local_data.X_train),
            local_standardize_x(local_data.X_test),
            local_standardize_x(X_grid);
            preprocessing = PREPROCESSING,
            feature_dimension = FEATURE_DIMENSION,
            lengthscale = FIXED_LENGTHSCALE,
            seed = FEATURE_SEED_XOR,
        )

        local_fit_seconds = @elapsed begin
            local_fit = BACKEND == :direct ?
                backend_module.fit_direct_model(
                    MODEL_DEPTH,
                    local_Φ_train,
                    local_y_train_std,
                    OPTIMIZER,
                    OPTIMIZER_BETA;
                    prior_builder = xor_direct_prior,
                ) :
                fit_xor_rxinfer(
                    MODEL_DEPTH,
                    local_Φ_train,
                    local_y_train_std,
                )
        end

        local_prediction_seconds = @elapsed begin
            local_test_std = backend_prediction(local_fit, local_Φ_test)
            local_grid_std = backend_prediction(local_fit, local_Φ_grid)
        end
        local_test_prediction = (;
            mean = local_y_center .+ local_y_scale .* local_test_std.mean,
            variance = abs2(local_y_scale) .* local_test_std.variance,
        )
        local_grid_prediction = (;
            mean = local_y_center .+ local_y_scale .* local_grid_std.mean,
            variance = abs2(local_y_scale) .* local_grid_std.variance,
        )
        local_metrics = xor_metrics(
            local_test_prediction,
            local_data.y_test,
            local_data.clean_test,
        )

        return (;
            checkerboard_size,
            metrics = local_metrics,
            fit_seconds = local_fit_seconds,
            prediction_seconds = local_prediction_seconds,
            grid_prediction = local_grid_prediction,
        )
    end

    suggested_runs = run_checkerboard_size.(suggested_checkerboard_sizes)
end

# ╔═╡ 99dc74ba-d771-47f3-bc86-13e19a6cf7da
begin
    suggested_summary = [
        (;
            checkerboard = "$(run.checkerboard_size[1])×$(run.checkerboard_size[2])",
            mean_logpdf = run.metrics.mean_logpdf,
            rmse = run.metrics.rmse,
            accuracy = run.metrics.accuracy,
            brier = run.metrics.brier,
            fit_seconds = run.fit_seconds,
            prediction_seconds = run.prediction_seconds,
        )
        for run in suggested_runs
    ]

    table_lines = [
        "| checkerboard | log-PDF | RMSE | accuracy | Brier | fit s | predict s |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    append!(table_lines, [
        "| $(row.checkerboard) | $(round(row.mean_logpdf; digits = 4)) | $(round(row.rmse; digits = 4)) | $(round(row.accuracy; digits = 4)) | $(round(row.brier; digits = 4)) | $(round(row.fit_seconds; digits = 2)) | $(round(row.prediction_seconds; digits = 2)) |"
        for row in suggested_summary
    ])
    Markdown.parse(join(table_lines, '\n'))
end

# ╔═╡ acf43fb8-e0aa-4991-9737-d0ade37a9ed4
begin
    suggested_panels = Any[]
    for run in suggested_runs
        board = "$(run.checkerboard_size[1])×$(run.checkerboard_size[2])"
        run_mean = reshape(
            run.grid_prediction.mean,
            GRID_RESOLUTION,
            GRID_RESOLUTION,
        )
        run_variance = reshape(
            max.(run.grid_prediction.variance, 0.0),
            GRID_RESOLUTION,
            GRID_RESOLUTION,
        )
        mean_plot = heatmap(
            axis_values,
            axis_values,
            run_mean;
            clim = (0, 1),
            color = :viridis,
            xlabel = "x₁",
            ylabel = "x₂",
            title = "$board mean",
            aspect_ratio = :equal,
        )
        contour!(
            mean_plot,
            axis_values,
            axis_values,
            run_mean;
            levels = [0.5],
            color = :white,
            linewidth = 1.5,
            label = false,
        )
        variance_plot = heatmap(
            axis_values,
            axis_values,
            run_variance;
            color = :magma,
            xlabel = "x₁",
            ylabel = "x₂",
            title = "$board variance",
            aspect_ratio = :equal,
        )
        push!(suggested_panels, mean_plot, variance_plot)
    end

    plot(
        suggested_panels...;
        layout = (length(suggested_runs), 2),
        size = (1_000, 330 * length(suggested_runs)),
        plot_title = "Suggested checkerboards — $(BACKEND), $(PREPROCESSING)",
    )
end

# ╔═╡ Cell order:
# ╠═04c6ad7a-58a4-4bab-966d-9ee3b8a124ea
# ╠═79c682ba-eb24-47d0-a9fd-e65061727534
# ╟─ab5e1eb1-4389-4ce2-80cf-0588470db53c
# ╟─73642bb5-4d87-4edf-8ee6-66d3404f737e
# ╠═c65b6b97-e4c9-4f31-9ea2-5372f7222a47
# ╟─2f21cf06-604e-4c6d-8199-a26bb9d83bbd
# ╠═d83fc607-c50d-4b75-bde8-98c177f024c4
# ╟─a4aef29f-a2ce-4bb4-84c0-c22749d0eb6e
# ╠═8039dc6b-c363-4f36-a62c-28b10f82313b
# ╟─e88773cc-63f0-4abc-a642-83dd16c07aa7
# ╠═b04ee6bc-795b-405a-93c1-749c55e8ecff
# ╠═0ce0de9d-7a90-4e75-b798-9a76653f0dd5
# ╟─c8563798-a4a3-448f-bd8f-28dc39be7054
# ╠═b5467d15-0481-4fef-ad0a-a1ff90bfeb17
# ╠═c9bf37d4-9fb6-48fb-9a52-d932656d6909
# ╠═d9486644-da2b-47a6-8859-b210eb76978b
# ╠═d505540e-c251-4c34-b256-8214de69929a
# ╟─b5af08cb-40a6-46dd-943b-330e577decc2
# ╟─4df404ce-bd26-42cc-8725-18c2d80d7945
# ╠═97fe1bdb-340b-490f-acb9-91ed63eaf6c6
# ╠═99dc74ba-d771-47f3-bc86-13e19a6cf7da
# ╠═acf43fb8-e0aa-4991-9737-d0ade37a9ed4
