### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 09d13d0c-8287-4f22-a9a9-29fdb77e4cf9
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ aa6b9c07-e1d1-45f4-af10-547b28735542
begin
    ENV["GKSwstype"] = "100"

    using DataFrames
    using FastGaussQuadrature
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

# ╔═╡ c47a37f2-48c6-48aa-aaf6-6426ddbbadf6
md"""
# Deep tensor kernels on two-dimensional XOR

The XOR notebook supplies only the **problem**: noisy two-dimensional
checkerboard regression and a prediction grid. The model here is independent
of `ManyPlus`, residual-sine neurons, and learned neural transitions.

Every function in the hierarchy is its own CP-decomposed tensor kernel. For
the mean (level zero) and every log-precision level ``\ell``,

```math
f_\ell(x)
=b_\ell+\sum_{r=1}^R\prod_{d=1}^D
z_d(x_d)^\top A_{\ell d}[:,r],
\qquad
A_{\ell d}\in\mathbb R^{\hat M\times R}.
```

The recursive probabilistic model is

```math
\begin{aligned}
y_n&\sim\mathcal N(f_0(x_n),\lambda_{1n}^{-1}),\\
s_{\ell n}&\sim\mathcal N(f_\ell(x_n),\lambda_{\ell+1,n}^{-1}),
&\lambda_{\ell n}&=\exp(s_{\ell n}),\\
s_{L-1,n}&\sim\mathcal N(f_{L-1}(x_n),\tau_{\rm top}^{-1}).
\end{aligned}
```

`L = 1` is a homoscedastic tensor-kernel regressor. `L = 2` makes the
observation precision a tensor-kernel function. Each further layer makes the
carrier precision of the preceding score another tensor-kernel function.

Training alternates over input dimensions. When all factors except
``A_{\ell d}`` are fixed, every level is linear in
``vec(A_{\ell d})``; the corresponding conditional problem is handled by the
existing Gaussian `softdot` rules. The exponential links use the package's
natural-gradient messages.
"""

# ╔═╡ c018da7a-2060-4d96-81dc-52a43232346b
begin
    env_int(name, default) = parse(Int, get(ENV, name, string(default)))
    env_float(name, default) = parse(Float64, get(ENV, name, string(default)))

    function parse_depths(value)
        depths = sort!(unique(parse.(Int, strip.(split(value, ',')))))
        isempty(depths) && throw(ArgumentError("at least one depth is required"))
        all(>=(1), depths) ||
            throw(ArgumentError("all depths must be positive"))
        return depths
    end

    config = (
        n_samples = env_int("TENSOR_KERNEL_SAMPLES", 400),
        train_fraction = 0.60,
        noise_std = 0.10,
        data_seed = 2_026,
        split_seed = 2_027,

        depths = parse_depths(get(ENV, "TENSOR_KERNEL_DEPTHS", "1,2,3")),
        basis_order = env_int("TENSOR_KERNEL_BASIS", 10),
        lengthscale = env_float("TENSOR_KERNEL_LENGTHSCALE", 0.45),
        cp_rank = env_int("TENSOR_KERNEL_RANK", 4),
        cp_regularization = env_float(
            "TENSOR_KERNEL_REGULARIZATION",
            2e-3,
        ),
        cp_jitter = 1e-7,
        sweeps = env_int("TENSOR_KERNEL_SWEEPS", 3),
        block_iterations = env_int("TENSOR_KERNEL_BLOCK_ITERATIONS", 15),
        fixed_observation_precision = 100.0,
        top_carrier = 25.0,
        factor_seed = 7_031,

        ngmp_alpha = 0.15,
        ngmp_max_step = 0.5,
        mean_intercept_variance = 4.0,
        level_intercept_variance = 1.0,
        initial_score_variance = 0.10,

        grid_points = env_int("TENSOR_KERNEL_GRID_POINTS", 48),
        grid_limits = (-5.0, 5.0),
    )

    config.n_samples >= 20 ||
        throw(ArgumentError("TENSOR_KERNEL_SAMPLES must be at least 20"))
    config.basis_order >= 2 ||
        throw(ArgumentError("TENSOR_KERNEL_BASIS must be at least 2"))
    config.cp_rank >= 1 ||
        throw(ArgumentError("TENSOR_KERNEL_RANK must be positive"))
    config.sweeps >= 1 ||
        throw(ArgumentError("TENSOR_KERNEL_SWEEPS must be positive"))
    config.block_iterations >= 1 ||
        throw(ArgumentError("TENSOR_KERNEL_BLOCK_ITERATIONS must be positive"))
end

# ╔═╡ 18009c59-cb11-43cf-a5b7-a6bb325257f7
md"""
## XOR data

Inputs are uniform on ``[-2,2]^2``. The clean ``2\times2`` checkerboard is
perturbed by Gaussian noise and clipped to ``[0,1]``. The Fourier map is
standardized using training inputs only.
"""

# ╔═╡ f72c5c98-7248-4ec9-8326-9c0483e59640
begin
    function checkerboard_label(x1, x2)
        cell_x = clamp(floor(Int, 2 * (x1 + 2) / 4), 0, 1)
        cell_y = clamp(floor(Int, 2 * (x2 + 2) / 4), 0, 1)
        return Float64(isodd(cell_x + cell_y))
    end

    function make_xor_dataset(; n, noise_std, seed)
        rng = StableRNG(seed)
        x1 = 4 .* rand(rng, n) .- 2
        x2 = 4 .* rand(rng, n) .- 2
        clean = checkerboard_label.(x1, x2)
        target = clamp.(clean .+ noise_std .* randn(rng, n), 0.0, 1.0)
        return DataFrame(x1 = x1, x2 = x2, clean = clean, target = target)
    end

    function split_dataset(data; train_fraction, seed)
        order = randperm(StableRNG(seed), nrow(data))
        n_train = round(Int, train_fraction * nrow(data))
        return (
            train = data[order[1:n_train], :],
            test = data[order[(n_train + 1):end], :],
        )
    end

    input_matrix(data) = Matrix{Float64}(data[:, [:x1, :x2]])
end

# ╔═╡ 9b799428-a70a-40b9-84e5-ab9756219902
begin
    dataset = make_xor_dataset(
        n = config.n_samples,
        noise_std = config.noise_std,
        seed = config.data_seed,
    )
    dataset_split = split_dataset(
        dataset;
        train_fraction = config.train_fraction,
        seed = config.split_seed,
    )
    train_data, test_data = dataset_split.train, dataset_split.test
    X_train, X_test = input_matrix(train_data), input_matrix(test_data)
    y_train, y_test =
        Float64.(train_data.target), Float64.(test_data.target)

    x_center = vec(mean(X_train; dims = 1))
    x_scale = map(
        value -> isfinite(value) && value > sqrt(eps()) ? value : 1.0,
        vec(std(X_train; dims = 1, corrected = true)),
    )
    standardize_inputs(X) = (Matrix{Float64}(X) .- x_center') ./ x_scale'
    X_train_standardized = standardize_inputs(X_train)
    X_test_standardized = standardize_inputs(X_test)
    y_anchor = mean(y_train)
    # The checkerboard's marginal target variance is mostly signal, not noise.
    # Anchor the first precision tensor at the known benchmark noise scale.
    noise_precision_anchor = config.fixed_observation_precision
end

# ╔═╡ d7dbdb76-6db4-408e-a02e-da46e00d3388
data_summary = DataFrame(
    observations = nrow(dataset),
    training_observations = nrow(train_data),
    test_observations = nrow(test_data),
    input_dimensions = size(X_train, 2),
    target_mean = y_anchor,
    constant_test_mse = mean(abs2, y_test .- y_anchor),
)

# ╔═╡ 23049167-df57-4df8-8179-bc4e18f1adcc
md"""
## Deterministic coordinate features

For the RBF kernel, an order-``\hat M`` Gauss-Hermite rule deterministically
approximates its Gaussian spectral integral. Symmetric nodes are compressed
into cosine/sine pairs, leaving exactly ``\hat M`` real features per input
coordinate.

Only ``D`` matrices ``Z_d\in\mathbb R^{N\times\hat M}`` are stored. The
Cartesian product with ``\hat M^D`` entries is implicit.
"""

# ╔═╡ dcb28919-5913-4be8-9d42-fe8cfcaa7646
begin
    struct GaussHermiteFourierBasis
        frequencies::Vector{Float64}
        pair_scales::Vector{Float64}
        zero_scale::Float64
    end

    function GaussHermiteFourierBasis(order::Int, lengthscale::Real)
        nodes, weights = FastGaussQuadrature.gausshermite(order)
        normalized_weights = weights ./ sqrt(pi)
        tolerance = 100eps(Float64)
        positive = findall(node -> node > tolerance, nodes)
        zero_index = findfirst(node -> abs(node) <= tolerance, nodes)
        frequencies =
            sqrt(2) .* Float64.(nodes[positive]) ./ Float64(lengthscale)
        pair_scales = sqrt.(2 .* Float64.(normalized_weights[positive]))
        zero_scale = isnothing(zero_index) ?
            0.0 : sqrt(Float64(normalized_weights[zero_index]))
        return GaussHermiteFourierBasis(
            frequencies,
            pair_scales,
            zero_scale,
        )
    end

    Base.length(basis::GaussHermiteFourierBasis) =
        2length(basis.frequencies) + Int(!iszero(basis.zero_scale))

    function (basis::GaussHermiteFourierBasis)(x::Real)
        result = Vector{Float64}(undef, length(basis))
        offset = 0
        if !iszero(basis.zero_scale)
            result[1] = basis.zero_scale
            offset = 1
        end
        for index in eachindex(basis.frequencies)
            phase = basis.frequencies[index] * x
            scale = basis.pair_scales[index]
            result[offset + 2index - 1] = scale * cos(phase)
            result[offset + 2index] = scale * sin(phase)
        end
        return result
    end

    function coordinate_features(inputs, bases)
        size(inputs, 2) == length(bases) ||
            throw(DimensionMismatch("one basis is required per coordinate"))
        return map(eachindex(bases)) do dimension
            result = Matrix{Float64}(
                undef,
                size(inputs, 1),
                length(bases[dimension]),
            )
            for observation in axes(inputs, 1)
                @views result[observation, :] .=
                    bases[dimension](inputs[observation, dimension])
            end
            result
        end
    end

    coordinate_bases = [
        GaussHermiteFourierBasis(
            config.basis_order,
            config.lengthscale,
        )
        for _ in axes(X_train_standardized, 2)
    ]
    Z_train = coordinate_features(
        X_train_standardized,
        coordinate_bases,
    )
    Z_test = coordinate_features(
        X_test_standardized,
        coordinate_bases,
    )
end

# ╔═╡ f5e3cf69-66d3-4c50-b86f-63b4b37cc9c2
begin
    pair_count = min(100, size(X_test_standardized, 1) ÷ 2)
    kernel_errors = map(1:pair_count) do pair
        left, right = pair, size(X_test_standardized, 1) - pair + 1
        approximate = prod(
            dot(Z_test[d][left, :], Z_test[d][right, :])
            for d in eachindex(Z_test)
        )
        squared_distance = sum(
            abs2,
            X_test_standardized[left, :] .-
            X_test_standardized[right, :],
        )
        exact = exp(-squared_distance / (2config.lengthscale^2))
        abs(approximate - exact)
    end
    maximum_kernel_error = maximum(kernel_errors)
end

# ╔═╡ 0ff859fc-9986-4c16-a14f-a5be49a5b896
md"""
## Tensor contractions

For one hierarchy function, the ``N\times R`` component matrix is

```math
\Psi=(Z_1A_1)\odot\cdots\odot(Z_DA_D).
```

When coordinate ``d`` is active, all other coordinate factors contract into
``H_{-d}``, and the conditional design has only ``\hat M R`` columns:

```math
G_d[n,(r-1)\hat M+i]=H_{-d}[n,r]Z_d[n,i].
```

The prior precision uses

```math
\left(\mathop{\odot}_{j\ne d}A_j^\top A_j\right)
\otimes I_{\hat M},
```

which is the conditional form of the tensor Frobenius regularizer from the
paper.
"""

# ╔═╡ c91874da-4d31-47a5-a223-ce5752bbaabf
begin
    function tensor_components(Z, factors)
        components = ones(
            Float64,
            size(first(Z), 1),
            size(first(factors), 2),
        )
        for dimension in eachindex(Z)
            components .*= Z[dimension] * factors[dimension]
        end
        return components
    end

    tensor_value(Z, factors, intercept) =
        vec(sum(tensor_components(Z, factors); dims = 2)) .+ intercept

    function conditional_design(Z, factors, active_dimension)
        n = size(first(Z), 1)
        basis_count, rank = size(factors[active_dimension])
        contracted = ones(Float64, n, rank)
        for dimension in eachindex(Z)
            dimension == active_dimension && continue
            contracted .*= Z[dimension] * factors[dimension]
        end

        design = Matrix{Float64}(undef, n, basis_count * rank + 1)
        for component in 1:rank
            columns =
                ((component - 1) * basis_count + 1):(component * basis_count)
            @views design[:, columns] .=
                Z[active_dimension] .* reshape(
                    contracted[:, component],
                    :,
                    1,
                )
        end
        design[:, end] .= 1.0
        return design
    end

    function other_factor_gram(factors, active_dimension)
        rank = size(first(factors), 2)
        result = ones(Float64, rank, rank)
        for dimension in eachindex(factors)
            dimension == active_dimension && continue
            result .*= transpose(factors[dimension]) * factors[dimension]
        end
        return result
    end

    function normalize_tensor_columns!(factors)
        dimensions = length(factors)
        rank = size(first(factors), 2)
        for component in 1:rank
            norms = [
                norm(@view factor[:, component])
                for factor in factors
            ]
            all(>(sqrt(eps())), norms) || continue
            shared_norm = exp(mean(log.(norms)))
            for dimension in 1:dimensions
                @views factors[dimension][:, component] .*= (
                    shared_norm / norms[dimension]
                )
            end
        end
        return factors
    end

    function active_factor_prior(
        factors,
        active_dimension,
        intercept_anchor,
        intercept_variance,
        precision_scale,
        config,
    )
        basis_count, rank = size(factors[active_dimension])
        factor_count = basis_count * rank
        tensor_precision = kron(
            other_factor_gram(factors, active_dimension),
            Matrix{Float64}(I, basis_count, basis_count),
        )
        precision = zeros(Float64, factor_count + 1, factor_count + 1)
        precision[1:factor_count, 1:factor_count] .=
            precision_scale * config.cp_regularization .* tensor_precision
        for index in 1:factor_count
            precision[index, index] += config.cp_jitter
        end
        precision[end, end] = inv(intercept_variance)
        weighted_mean = zeros(Float64, factor_count + 1)
        weighted_mean[end] = precision[end, end] * intercept_anchor
        return MvNormalWeightedMeanPrecision(
            weighted_mean,
            Matrix(Symmetric(precision)),
        )
    end

    function update_active_factor!(factors, active_dimension, coefficients)
        basis_count, rank = size(factors[active_dimension])
        factor_count = basis_count * rank
        factors[active_dimension] .= reshape(
            @view(coefficients[1:factor_count]),
            basis_count,
            rank,
        )
        return coefficients[end]
    end
end

# ╔═╡ da0dff78-3460-4c4f-9409-5291b3297f57
md"""
## Conditional tensor models

The depth-one update is exactly a tensor-kernel ridge block with fixed
observation precision. For depth two and above, a single RxInfer graph updates
the active CP factor of the mean and the active CP factor of every
log-precision level together. All remaining coordinate factors appear only
inside the conditional design matrices.
"""

# ╔═╡ 6c19d702-8a8a-4eed-95bc-6933fc43f0ef
@model function tensor_depth_one_block(
    y,
    mean_features,
    mean_prior,
    observation_precision,
)
    v ~ mean_prior
    for observation in eachindex(mean_features)
        y[observation] ~ softdot(
            mean_features[observation],
            v,
            observation_precision,
        )
    end
end

# ╔═╡ 07d06ce6-e628-4572-8085-508b3a8ac5e8
@model function tensor_deep_block(
    y,
    mean_features,
    level_features,
    n_levels,
    mean_prior,
    level_priors,
    top_carrier,
    exp_dependencies,
    exp_damping,
)
    local w, score, precision
    v ~ mean_prior
    for level in 1:n_levels
        w[level] ~ level_priors[level]
    end
    for observation in eachindex(mean_features)
        score[n_levels, observation] ~ softdot(
            level_features[n_levels, observation],
            w[n_levels],
            top_carrier,
        )
        precision[n_levels, observation] ~
            Exp(score[n_levels, observation]) where {
                dependencies = exp_dependencies,
                meta = exp_damping,
            }
        for level in (n_levels - 1):-1:1
            score[level, observation] ~ softdot(
                level_features[level, observation],
                w[level],
                precision[level + 1, observation],
            )
            precision[level, observation] ~
                Exp(score[level, observation]) where {
                    dependencies = exp_dependencies,
                    meta = exp_damping,
                }
        end
        y[observation] ~ softdot(
            mean_features[observation],
            v,
            precision[1, observation],
        )
    end
end

# ╔═╡ b13b2775-2de4-46bf-9a57-821087d2735b
@constraints function tensor_deep_constraints()
    q(v, w, score, precision, y) =
        q(v, y)q(w, score)q(precision)
    q(v)::MomentForm()
    q(w)::MomentForm()
end

# ╔═╡ 381f3e46-c782-493d-873b-8708c9218ec0
begin
    @initialization function tensor_deep_initialization(priors, states)
        q(v) = deepcopy(priors.mean)
        q(w) = deepcopy(priors.levels)
        q(score) = states.score
        q(precision) = states.precision
    end

    tensor_exp_dependencies() = NGMPDependencies(
        out = nothing,
        in = nothing;
        projection = TangentProjection(type = ClosedForm),
    )

    tensor_exp_damping(config) = DampingMeta(
        alpha = config.ngmp_alpha,
        beta = 0.0,
        max_step = config.ngmp_max_step,
        method = :damped,
    )
end

# ╔═╡ b35d19cc-f838-4d65-ae62-e43d24aa7728
md"""
## Alternating deep-tensor training

There is one independent factor set per hierarchy function:

```text
functions[1]       mean tensor
functions[2]       observation log-precision tensor
functions[3:end]   deeper carrier log-precision tensors
```

Depths are fitted in increasing order. A deeper model warm-starts all tensors
present at the previous depth and adds one small new carrier tensor. This makes
the depth comparison about the extra hierarchy rather than unrelated random
initializations.
"""

# ╔═╡ beae3216-0a66-4bfa-9581-942872317e99
begin
    function new_tensor_factors(rng, dimensions, basis_count, rank; scale)
        factors = [
            scale .* randn(rng, basis_count, rank) ./ sqrt(basis_count)
            for _ in 1:dimensions
        ]
        return normalize_tensor_columns!(factors)
    end

    function initialize_tensor_state(depth, Z, targets, config; warm = nothing)
        rng = StableRNG(config.factor_seed + depth)
        dimensions = length(Z)
        basis_count = size(first(Z), 2)
        functions = Vector{Vector{Matrix{Float64}}}()
        intercepts = Float64[]

        if !isnothing(warm)
            for function_index in eachindex(warm.functions)
                push!(
                    functions,
                    [copy(factor) for factor in warm.functions[function_index]],
                )
                push!(intercepts, warm.intercepts[function_index])
            end
        end

        if isempty(functions)
            push!(
                functions,
                new_tensor_factors(
                    rng,
                    dimensions,
                    basis_count,
                    config.cp_rank;
                    scale = 0.8,
                ),
            )
            push!(intercepts, mean(targets))
        end

        while length(functions) < depth
            level = length(functions)
            push!(
                functions,
                new_tensor_factors(
                    rng,
                    dimensions,
                    basis_count,
                    config.cp_rank;
                    scale = 0.05,
                ),
            )
            anchor = level == 1 ?
                log(noise_precision_anchor) : log(config.top_carrier)
            push!(intercepts, anchor)
        end
        return (; functions, intercepts)
    end

    function current_function_values(state, Z)
        return [
            tensor_value(
                Z,
                state.functions[index],
                state.intercepts[index],
            )
            for index in eachindex(state.functions)
        ]
    end

    function deep_initial_states(state, Z, config)
        n_levels = length(state.functions) - 1
        n = size(first(Z), 1)
        score = Matrix{NormalMeanVariance{Float64}}(
            undef,
            n_levels,
            n,
        )
        precision = Matrix{GammaShapeRate{Float64}}(
            undef,
            n_levels,
            n,
        )
        for level in 1:n_levels
            means = tensor_value(
                Z,
                state.functions[level + 1],
                state.intercepts[level + 1],
            )
            variances = fill(config.initial_score_variance, n)
            shapes = 1 .+ inv.(variances)
            score[level, :] = NormalMeanVariance.(means, variances)
            precision[level, :] =
                GammaShapeRate.(shapes, shapes .* exp.(-means))
        end
        return (; score, precision)
    end

    function active_priors(state, active_dimension, config)
        mean_prior = active_factor_prior(
            state.functions[1],
            active_dimension,
            y_anchor,
            config.mean_intercept_variance,
            config.fixed_observation_precision,
            config,
        )
        levels = [
            active_factor_prior(
                state.functions[level + 1],
                active_dimension,
                level == 1 ?
                    log(noise_precision_anchor) :
                    log(config.top_carrier),
                config.level_intercept_variance,
                config.top_carrier,
                config,
            )
            for level in 1:(length(state.functions) - 1)
        ]
        return (; mean = mean_prior, levels)
    end

    function update_depth_one!(
        state,
        active_dimension,
        Z,
        targets,
        config,
    )
        design = conditional_design(
            Z,
            state.functions[1],
            active_dimension,
        )
        prior = active_priors(state, active_dimension, config).mean
        result = infer(
            model = tensor_depth_one_block(
                mean_prior = prior,
                observation_precision =
                    config.fixed_observation_precision,
            ),
            data = (
                y = targets,
                mean_features = [collect(row) for row in eachrow(design)],
            ),
            returnvars = (v = KeepLast(),),
            iterations = 1,
            free_energy = false,
            showprogress = false,
            options = (limit_stack_depth = 100,),
            disable_inference_error_hint = true,
        )
        coefficients = mean(result.posteriors[:v])
        state.intercepts[1] = update_active_factor!(
            state.functions[1],
            active_dimension,
            coefficients,
        )
        return NaN
    end

    function update_deep!(
        state,
        active_dimension,
        Z,
        targets,
        config,
    )
        n_levels = length(state.functions) - 1
        designs = [
            conditional_design(
                Z,
                state.functions[index],
                active_dimension,
            )
            for index in eachindex(state.functions)
        ]
        mean_rows = [collect(row) for row in eachrow(designs[1])]
        level_rows = Matrix{Vector{Float64}}(
            undef,
            n_levels,
            length(mean_rows),
        )
        for level in 1:n_levels, observation in eachindex(mean_rows)
            level_rows[level, observation] =
                collect(@view designs[level + 1][observation, :])
        end

        priors = active_priors(state, active_dimension, config)
        states = deep_initial_states(state, Z, config)
        result = infer(
            model = tensor_deep_block(
                n_levels = n_levels,
                mean_prior = priors.mean,
                level_priors = priors.levels,
                top_carrier = config.top_carrier,
                exp_dependencies = tensor_exp_dependencies(),
                exp_damping = tensor_exp_damping(config),
            ),
            data = (
                y = targets,
                mean_features = mean_rows,
                level_features = level_rows,
            ),
            constraints = tensor_deep_constraints(),
            initialization = tensor_deep_initialization(priors, states),
            returnvars = (v = KeepLast(), w = KeepLast()),
            iterations = config.block_iterations,
            free_energy = true,
            showprogress = false,
            options = (limit_stack_depth = 100,),
            disable_inference_error_hint = true,
        )
        all(isfinite, result.free_energy) ||
            error("depth $(length(state.functions)) block produced non-finite FE")

        state.intercepts[1] = update_active_factor!(
            state.functions[1],
            active_dimension,
            mean(result.posteriors[:v]),
        )
        level_posteriors = collect(vec(result.posteriors[:w]))
        for level in 1:n_levels
            state.intercepts[level + 1] = update_active_factor!(
                state.functions[level + 1],
                active_dimension,
                mean(level_posteriors[level]),
            )
        end
        return Float64(last(result.free_energy))
    end

    function fit_tensor_depth(depth, Z, targets, config; warm = nothing)
        state = initialize_tensor_state(
            depth,
            Z,
            targets,
            config;
            warm = warm,
        )
        reports = NamedTuple[]
        for sweep in 1:config.sweeps
            block_energies = Float64[]
            for active_dimension in eachindex(Z)
                energy = depth == 1 ?
                    update_depth_one!(
                        state,
                        active_dimension,
                        Z,
                        targets,
                        config,
                    ) :
                    update_deep!(
                        state,
                        active_dimension,
                        Z,
                        targets,
                        config,
                    )
                push!(block_energies, energy)
            end
            for factors in state.functions
                normalize_tensor_columns!(factors)
            end
            values = current_function_values(state, Z)
            report = (
                sweep = sweep,
                train_mse = mean(abs2, values[1] .- targets),
                bottom_precision_minimum =
                    depth == 1 ?
                    config.fixed_observation_precision :
                    minimum(exp.(values[2])),
                bottom_precision_maximum =
                    depth == 1 ?
                    config.fixed_observation_precision :
                    maximum(exp.(values[2])),
                final_block_free_energy = last(block_energies),
            )
            push!(reports, report)
            @printf(
                "L=%d sweep %d/%d: train MSE %.6f, precision [%.3g, %.3g]\n",
                depth,
                sweep,
                config.sweeps,
                report.train_mse,
                report.bottom_precision_minimum,
                report.bottom_precision_maximum,
            )
        end
        return (; depth, state, reports)
    end
end

# ╔═╡ c6ddd160-89f0-4e86-bc3b-35a6964163e8
begin
    total_training_seconds, tensor_fits = let
        fits = Dict{Int, Any}()
        previous = nothing
        elapsed = @elapsed begin
            for depth in config.depths
                # Warm starts are available only when the immediately
                # shallower model has already been fitted.
                warm = !isnothing(previous) &&
                       previous.depth == depth - 1 ?
                       previous.state : nothing
                fit = fit_tensor_depth(
                    depth,
                    Z_train,
                    y_train,
                    config;
                    warm = warm,
                )
                fits[depth] = fit
                previous = fit
            end
        end
        elapsed, fits
    end
    selected_depth = maximum(config.depths)
    selected_fit = tensor_fits[selected_depth]
    nothing
end

# ╔═╡ 6650ff36-09dd-49ef-bab7-a49f468ccf62
md"""
## Tensor-only prediction

Prediction contracts the learned factors directly. There is no dense feature
head after the tensors.

The plotted variance is the conditional plug-in variance from the bottom
tensor precision,

```math
\operatorname{Var}(y_\star\mid A,b)
=\exp[-f_1(x_\star)].
```

It is aleatoric uncertainty conditional on the learned tensor factors. The
current alternating approximation does not integrate uncertainty over every
CP factor simultaneously; doing so requires a dedicated multilinear factor
rather than the Gaussian `ContinuousTransition`.
"""

# ╔═╡ ea19a1f8-6d4d-4199-a955-e4d3668731ef
begin
    function tensor_predict(fit, Z, config)
        values = current_function_values(fit.state, Z)
        predictive_variance = fit.depth == 1 ?
            fill(inv(config.fixed_observation_precision), length(values[1])) :
            exp.(-values[2])
        return (
            mean = values[1],
            variance = predictive_variance,
            level_scores = values[2:end],
        )
    end

    depth_rows = map(config.depths) do depth
        fit = tensor_fits[depth]
        train_prediction = tensor_predict(fit, Z_train, config)
        test_prediction = tensor_predict(fit, Z_test, config)
        (
            depth = depth,
            train_mse =
                mean(abs2, train_prediction.mean .- y_train),
            test_mse =
                mean(abs2, test_prediction.mean .- y_test),
            minimum_variance = minimum(test_prediction.variance),
            mean_variance = mean(test_prediction.variance),
            maximum_variance = maximum(test_prediction.variance),
        )
    end
    depth_summary = DataFrame(depth_rows)
    constant_test_mse = mean(abs2, y_test .- y_anchor)
end

# ╔═╡ e180ee25-8176-45c1-b009-5401a3a31bb1
model_summary = DataFrame(
    input_dimensions = length(Z_train),
    basis_functions_per_dimension = config.basis_order,
    implicit_tensor_features =
        config.basis_order^length(Z_train),
    rank_per_function = config.cp_rank,
    tensor_parameters_per_function =
        length(Z_train) * config.basis_order * config.cp_rank,
    deepest_functions = selected_depth,
    deepest_total_tensor_parameters =
        selected_depth * length(Z_train) *
        config.basis_order * config.cp_rank,
    maximum_kernel_error = maximum_kernel_error,
    training_seconds = total_training_seconds,
)

# ╔═╡ 7048c930-293e-4e91-a954-a97e48d6241a
begin
    grid_axis = range(
        config.grid_limits...;
        length = config.grid_points,
    )
    grid_matrix = reduce(
        vcat,
        (
            [x1 x2]
            for x2 in grid_axis
            for x1 in grid_axis
        ),
    )
    grid_standardized = standardize_inputs(grid_matrix)
    Z_grid = coordinate_features(grid_standardized, coordinate_bases)
    grid_statistics = tensor_predict(selected_fit, Z_grid, config)
    grid_prediction = (
        mean = Matrix(permutedims(reshape(
            grid_statistics.mean,
            config.grid_points,
            config.grid_points,
        ))),
        variance = Matrix(permutedims(reshape(
            grid_statistics.variance,
            config.grid_points,
            config.grid_points,
        ))),
    )
end

# ╔═╡ 9243b388-75d1-4605-8b76-2114845d2451
final_plot = let
    mse_panel = plot(
        xlabel = "tensor sweep",
        ylabel = "training MSE",
        title = "Alternating tensor blocks",
        yscale = :log10,
    )
    for depth in config.depths
        reports = tensor_fits[depth].reports
        plot!(
            mse_panel,
            getproperty.(reports, :sweep),
            getproperty.(reports, :train_mse);
            marker = :circle,
            markersize = 3,
            label = "L = $depth",
        )
    end
    mean_panel = heatmap(
        grid_axis,
        grid_axis,
        grid_prediction.mean;
        color = :RdBu,
        clims = (0, 1),
        xlabel = "x₁",
        ylabel = "x₂",
        title = "L=$selected_depth tensor mean",
        aspect_ratio = :equal,
    )
    variance_panel = heatmap(
        grid_axis,
        grid_axis,
        grid_prediction.variance;
        color = :viridis,
        xlabel = "x₁",
        ylabel = "x₂",
        title = "L=$selected_depth tensor variance",
        aspect_ratio = :equal,
    )
    observed_panel = scatter(
        dataset.x1,
        dataset.x2;
        marker_z = dataset.target,
        color = :RdBu,
        clims = (0, 1),
        markersize = 3.6,
        markeralpha = 0.68,
        markerstrokecolor = "#36454F",
        markerstrokewidth = 0.3,
        label = "",
        xlabel = "x₁",
        ylabel = "x₂",
        title = "Noisy 2×2 XOR targets",
        aspect_ratio = :equal,
        xlims = config.grid_limits,
        ylims = config.grid_limits,
    )
    figure = plot(
        mse_panel,
        mean_panel,
        variance_panel,
        observed_panel;
        layout = (1, 4),
        size = (1_700, 420),
    )
    if haskey(ENV, "TENSOR_KERNEL_NOTEBOOK_FIGURE")
        savefig(figure, ENV["TENSOR_KERNEL_NOTEBOOK_FIGURE"])
    end
    figure
end

# ╔═╡ af4422bd-019f-4932-850b-0d91dc12fc8c
md"""
## Result

The selected ``L=$selected_depth`` model reaches test MSE
**$(round(depth_summary[depth_summary.depth .== selected_depth, :test_mse][1];
digits = 4))**, compared with **$(round(constant_test_mse; digits = 4))** for
the training-mean predictor.

Each tensor function implicitly addresses
``$(config.basis_order)^2 = $(config.basis_order^2)`` product features, while
storing only
``2 × $(config.basis_order) × $(config.cp_rank) =
$(2config.basis_order * config.cp_rank)`` CP parameters. The deepest model has
``$selected_depth`` such tensor functions: one for the mean and
``$(selected_depth - 1)`` for the recursive precision hierarchy.

The contraction and update code is dimension-generic. Replacing the two-column
XOR input with a higher-dimensional matrix only adds coordinate feature
matrices and CP factors; no ``\hat M^D`` object is allocated.
"""

# ╔═╡ Cell order:
# ╠═09d13d0c-8287-4f22-a9a9-29fdb77e4cf9
# ╠═aa6b9c07-e1d1-45f4-af10-547b28735542
# ╠═c47a37f2-48c6-48aa-aaf6-6426ddbbadf6
# ╠═c018da7a-2060-4d96-81dc-52a43232346b
# ╠═18009c59-cb11-43cf-a5b7-a6bb325257f7
# ╠═f72c5c98-7248-4ec9-8326-9c0483e59640
# ╠═9b799428-a70a-40b9-84e5-ab9756219902
# ╠═d7dbdb76-6db4-408e-a02e-da46e00d3388
# ╠═23049167-df57-4df8-8179-bc4e18f1adcc
# ╠═dcb28919-5913-4be8-9d42-fe8cfcaa7646
# ╠═f5e3cf69-66d3-4c50-b86f-63b4b37cc9c2
# ╠═0ff859fc-9986-4c16-a14f-a5be49a5b896
# ╠═c91874da-4d31-47a5-a223-ce5752bbaabf
# ╠═da0dff78-3460-4c4f-9409-5291b3297f57
# ╠═6c19d702-8a8a-4eed-95bc-6933fc43f0ef
# ╠═07d06ce6-e628-4572-8085-508b3a8ac5e8
# ╠═b13b2775-2de4-46bf-9a57-821087d2735b
# ╠═381f3e46-c782-493d-873b-8708c9218ec0
# ╠═b35d19cc-f838-4d65-ae62-e43d24aa7728
# ╠═beae3216-0a66-4bfa-9581-942872317e99
# ╠═c6ddd160-89f0-4e86-bc3b-35a6964163e8
# ╠═6650ff36-09dd-49ef-bab7-a49f468ccf62
# ╠═ea19a1f8-6d4d-4199-a955-e4d3668731ef
# ╠═e180ee25-8176-45c1-b009-5401a3a31bb1
# ╠═7048c930-293e-4e91-a954-a97e48d6241a
# ╠═9243b388-75d1-4605-8b76-2114845d2451
# ╠═af4422bd-019f-4932-850b-0d91dc12fc8c
