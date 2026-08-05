### A Pluto.jl notebook ###
# v1.0.3
#
# Fast command-line validation:
#   DEEP_HIERARCHY_SMOKE=true OPENBLAS_NUM_THREADS=1 \
#     julia --project=. notebooks/why_hierarchy_deep_learned_features.jl

using Markdown
using InteractiveUtils

# ╔═╡ 8dc98342-8d11-4f27-b77c-2153af7f0a11
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ 92c569e8-11c2-4ca5-ae84-577736ea3993
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

    import BayesBase: mean_cov
    import ExponentialFamily: WishartFast
    import ProbabilisticEnsembling: Exp
end

# ╔═╡ 2b76f226-13a6-4aab-92eb-69b6668e05c4
md"""
# Deep learned hierarchy: split/activate/stack transitions

Each `ContinuousTransition` emits a multivariate Gaussian. The same vector is
attached to the output side of an exact `MvStack`, whose backward rules expose
one scalar marginal per coordinate. Those scalars pass through independent
`ResidualSine` factors and a second `MvStack` collects them for the next
transition:

```math
\begin{aligned}
h_0 &= [1,x],\\
z_\ell\mid h_{\ell-1},A_\ell,\Lambda_\ell
  &\sim \mathcal N(A_\ell h_{\ell-1},\Lambda_\ell^{-1}),\\
\tilde z_{\ell,j} &= [z_\ell]_j,\\
\tilde h_{\ell,j} &=
  \tilde z_{\ell,j}+
  \frac{\rho_\ell}{\omega_\ell}\sin(\omega_\ell\tilde z_{\ell,j}),\\
h_\ell &= \operatorname{stack}
  (\tilde h_{\ell,1},\ldots,\tilde h_{\ell,H}),
  \qquad \ell=1,\ldots,L.
\end{aligned}
```

The forward stack is diagonal because its inbound edges are scalar messages,
but the split stack's exact backward rule combines the dense CT cavity with all
other coordinate sites. This deliberately trades a joint nonlinear projection
for the scalar learning path that works in
`why_hierarchy_learned_features.jl`.

The final shared representation forks directly into independent Gaussian mean
and noise readout heads:

```math
\begin{aligned}
\mu(x) &= v^\mathsf T h_L,\\
\lambda(x) &= \exp\{g^\mathsf T h_L+a_0\},\\
y\mid x &\sim \mathcal N(\mu(x),\lambda(x)^{-1}).
\end{aligned}
```

The default `lowrank` transition layout uses a dense `LinearReshapeMeta` for the
small raw-input map and identity-offset `LinearLowRankMeta` transitions in the
repeated trunk. Thus each deep block learns a controlled correction to an
explicit skip path, while width can increase without dense ``H^2`` Gaussian
transition posteriors. The two final Gaussian weight vectors are deliberately
direct readouts: an additional CT at either head proved too strongly
regularized to move.
"""

# ╔═╡ 86812ffc-f260-425f-97d1-e035f53b75b4
begin
    env_bool(name, default) =
        lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")
    env_int(name, default) = parse(Int, get(ENV, name, string(default)))
    env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
    env_symbol(name, default) =
        Symbol(lowercase(strip(get(ENV, name, string(default)))))

    const DEEP_HIERARCHY_SMOKE = env_bool("DEEP_HIERARCHY_SMOKE", false)
    const DEEP_CONFIG = (
        n_samples = env_int(
            "N_SAMPLES", DEEP_HIERARCHY_SMOKE ? 48 : 600,
        ),
        holdout_fraction = env_float("HOLDOUT_FRACTION", 1 / 3),
        data_seed = env_int("DATA_SEED", 7),
        width = env_int("DEEP_WIDTH", DEEP_HIERARCHY_SMOKE ? 3 : 16),
        depth = env_int("DEEP_DEPTH", 2),
        iterations = env_int(
            "N_ITERATIONS", DEEP_HIERARCHY_SMOKE ? 2 : 40,
        ),
        transition_layout = env_symbol("TRANSITION_LAYOUT", :lowrank),
        activation_forward = env_symbol("ACTIVATION_FORWARD", :moments),
        low_rank_parameters = env_int(
            "LOW_RANK_PARAMETERS",
            DEEP_HIERARCHY_SMOKE ? 3 : 32,
        ),
        transition_precision_mean = env_float(
            "TRANSITION_PRECISION_MEAN", 250.0,
        ),
        input_bias_sd = env_float("INPUT_BIAS_SD", 0.35),
        input_frequency_floor_sd =
            env_float("INPUT_FREQUENCY_FLOOR_SD", 0.15),
        input_relative_frequency_sd =
            env_float("INPUT_RELATIVE_FREQUENCY_SD", 0.12),
        hidden_weight_variance = env_float("HIDDEN_WEIGHT_VARIANCE", 0.005),
        hidden_skip = env_float("HIDDEN_SKIP", 1.0),
        score_carrier = env_float("SCORE_CARRIER", 25.0),
        feature_jitter = env_float("FEATURE_JITTER", 1e-8),
        ngmp_alpha = env_float("NGMP_ALPHA", 0.01),
        ngmp_max_step = env_float("NGMP_MAX_STEP", 0.25),
        ct_alpha = env_float("CT_ALPHA", 0.01),
        ct_max_step = env_float("CT_MAX_STEP", 0.25),
        damping_method = env_symbol("DAMPING_METHOD", :damped),
        prediction_draws = env_int(
            "PREDICTION_DRAWS", DEEP_HIERARCHY_SMOKE ? 64 : 500,
        ),
        prediction_mode = env_symbol("PREDICTION_MODE", :rxinfer),
        prediction_iterations = env_int(
            "PREDICTION_ITERATIONS", DEEP_HIERARCHY_SMOKE ? 2 : 10,
        ),
        prediction_batch_size = env_int(
            "PREDICTION_BATCH_SIZE", DEEP_HIERARCHY_SMOKE ? 32 : 64,
        ),
        prediction_seed = env_int("PREDICTION_SEED", 9001),
        sample_transition_noise = env_bool("SAMPLE_TRANSITION_NOISE", true),
        grid_points = env_int(
            "GRID_POINTS", DEEP_HIERARCHY_SMOKE ? 17 : 81,
        ),
        free_energy = env_bool("FREE_ENERGY", DEEP_HIERARCHY_SMOKE),
        show_progress = env_bool("SHOW_PROGRESS", false),
        save_outputs = env_bool("SAVE_OUTPUTS", false),
        output_prefix = get(
            ENV,
            "OUTPUT_PREFIX",
            joinpath(@__DIR__, "..", "viz", "deep_learned_hierarchy"),
        ),
    )

    DEEP_CONFIG.depth >= 1 ||
        throw(ArgumentError("DEEP_DEPTH must be at least one"))
    DEEP_CONFIG.width >= 2 ||
        throw(ArgumentError("DEEP_WIDTH must be at least two"))
    DEEP_CONFIG.transition_layout in (:mixed, :reshape, :lowrank) ||
        throw(ArgumentError(
            "TRANSITION_LAYOUT must be mixed, reshape, or lowrank",
        ))
    DEEP_CONFIG.activation_forward in (:moments, :delta) ||
        throw(ArgumentError(
            "ACTIVATION_FORWARD must be moments or delta",
        ))
    DEEP_CONFIG.prediction_mode in (:rxinfer, :montecarlo, :both) ||
        throw(ArgumentError(
            "PREDICTION_MODE must be rxinfer, montecarlo, or both",
        ))
end

# ╔═╡ 93b993c4-a100-4acf-a6ca-21fa06a2760a
begin
    true_mean(x) = -(x + 0.5) * sin(3pi * x)
    true_noise_variance(x) = abs2(0.45 * (x + 0.5))

    deep_data = let
        rng = StableRNG(DEEP_CONFIG.data_seed)
        x = randn(rng, DEEP_CONFIG.n_samples)
        y = true_mean.(x) .+
            sqrt.(true_noise_variance.(x)) .* randn(rng, DEEP_CONFIG.n_samples)
        order = randperm(rng, DEEP_CONFIG.n_samples)
        n_test = round(
            Int,
            DEEP_CONFIG.holdout_fraction * DEEP_CONFIG.n_samples,
        )
        test = order[1:n_test]
        train = order[(n_test + 1):end]
        (
            x_train = x[train],
            y_train = y[train],
            x_test = x[test],
            y_test = y[test],
        )
    end

    raw_features(xs) = [[1.0, Float64(x)] for x in xs]
    deep_grid = collect(range(-3.0, 3.0; length = DEEP_CONFIG.grid_points))

    @printf(
        "deep hierarchy: %d train, %d test, width %d, trunk depth %d, layout %s\n",
        length(deep_data.y_train),
        length(deep_data.y_test),
        DEEP_CONFIG.width,
        DEEP_CONFIG.depth,
        string(DEEP_CONFIG.transition_layout),
    )
end

# ╔═╡ 07081021-37cb-494c-b9ab-e20e57ea2a30
begin
    gaussian(mean_vector, covariance) = MvNormalMeanCovariance(
        collect(mean_vector),
        Matrix(covariance),
    )

    noise_anchor(xs, ys) =
        -log(max(mean(abs2.(diff(ys[sortperm(xs)]))) / 2, 1e-8))

    function paired_input_matrix(
        width,
        low_frequency,
        high_frequency;
        seed,
    )
        rng = StableRNG(seed)
        n_pairs = cld(width, 2)
        frequencies = n_pairs == 1 ?
            [sqrt(low_frequency * high_frequency)] :
            exp.(range(
                log(low_frequency),
                log(high_frequency);
                length = n_pairs,
            ))
        matrix = Matrix{Float64}(undef, width, 2)
        for neuron in 1:width
            pair = cld(neuron, 2)
            sign = isodd(neuron) ? 1.0 : -1.0
            matrix[neuron, 1] =
                sign * (pi / 4) * (1 + 0.03 * randn(rng))
            matrix[neuron, 2] =
                frequencies[pair] * (1 + 0.02 * randn(rng))
        end
        return matrix
    end

    function ring_low_rank_meta(width, parameter_count, skip)
        count = clamp(parameter_count, 1, width^2)
        U = zeros(Float64, width, count)
        V = zeros(Float64, width, count)

        # Deterministic rank-one dictionary: diagonal atoms first, followed by
        # cyclic offsets. The fixed A0 supplies the main residual/skip path.
        pairs = Tuple{Int, Int}[]
        for offset in 0:(width - 1)
            for row in 1:width
                push!(pairs, (row, mod1(row + offset, width)))
            end
        end
        for (column, (row, input)) in
            enumerate(Iterators.take(pairs, count))
            U[row, column] = 1.0
            V[input, column] = 1.0
        end
        return LinearLowRankMeta(fill(Float64(skip), width), U, V)
    end

    function dense_transition_spec(reference, variance)
        output_dim, input_dim = size(reference)
        variances = variance isa Real ?
            fill(Float64(variance), length(reference)) :
            collect(Float64, variance)
        length(variances) == length(reference) ||
            throw(DimensionMismatch(
                "dense transition needs $(length(reference)) variances; " *
                "got $(length(variances))",
            ))
        return (
            meta = LinearReshapeMeta(output_dim, input_dim),
            prior = gaussian(
                vec(reference),
                Diagonal(variances),
            ),
            kind = :reshape,
        )
    end

    function hidden_transition_spec(config; seed)
        rng = StableRNG(seed)
        width = config.width
        skip = config.hidden_skip
        variance = config.hidden_weight_variance
        kind = config.transition_layout === :reshape ? :reshape : :lowrank

        if kind === :reshape
            reference =
                skip .* Matrix{Float64}(I, width, width) .+
                0.02 .* randn(rng, width, width)
            return dense_transition_spec(reference, variance)
        end

        meta = ring_low_rank_meta(
            width,
            config.low_rank_parameters,
            skip,
        )
        return (
            meta = meta,
            prior = gaussian(
                0.02 .* randn(rng, size(meta.U, 2)),
                Diagonal(fill(variance, size(meta.U, 2))),
            ),
            kind = :lowrank,
        )
    end

    function precision_prior(width, precision_mean)
        degrees = width + 2.0
        inverse_scale = Matrix(Diagonal(
            fill(degrees / precision_mean, width),
        ))
        return WishartFast(degrees, inverse_scale)
    end

    function make_deep_priors(xs, ys, config)
        width = config.width
        input_reference = paired_input_matrix(
            width,
            0.75,
            12.0;
            seed = 42,
        )
        input_spec = dense_transition_spec(
            input_reference,
            vcat(
                fill(abs2(config.input_bias_sd), width),
                abs2.(
                    config.input_frequency_floor_sd .+
                    config.input_relative_frequency_sd .*
                    abs.(input_reference[:, 2]),
                ),
            ),
        )
        trunk_specs = Any[input_spec]
        for layer in 2:config.depth
            push!(
                trunk_specs,
                hidden_transition_spec(
                    config;
                    seed = 100 + layer,
                ),
            )
        end
        precision = precision_prior(
            width,
            config.transition_precision_mean,
        )

        return (
            width = width,
            a_trunk = [spec.prior for spec in trunk_specs],
            P_trunk = [deepcopy(precision) for _ in 1:config.depth],
            v = gaussian(
                [isodd(k) ? 0.15 : -0.15 for k in 1:width],
                Diagonal(fill(1 / width, width)),
            ),
            g = gaussian(
                [isodd(k) ? 0.03 : -0.03 for k in 1:width],
                Diagonal(fill(abs2(0.4) / width, width)),
            ),
            anchor = noise_anchor(xs, ys),
            trunk_metas = [spec.meta for spec in trunk_specs],
            trunk_kinds = [spec.kind for spec in trunk_specs],
        )
    end

    function make_activation_metas(config)
        trunk = [
            ResidualSineMeta(
                rho = max(0.35, 0.75 / sqrt(layer)),
                omega = 1.45^(layer - 1),
            )
            for layer in 1:config.depth
        ]
        return (; trunk)
    end
end

# ╔═╡ a3d7f9c8-95a6-42d0-a171-f04f0faf0eb9
begin
    function activation_dependencies(config)
        projection = config.activation_forward === :moments ?
            TangentProjection(type = ClosedForm) :
            TangentProjection(type = DeltaApproximation)
        return NGMPDependencies(
            out = nothing,
            in = nothing;
            projection = projection,
            damping = DampingMeta(
                alpha = config.ngmp_alpha,
                beta = 0.0,
                max_step = config.ngmp_max_step,
                method = config.damping_method,
            ),
        )
    end

    function transition_dependencies(config)
        return NGMPDependencies(
            a = nothing,
            damping = DampingMeta(
                alpha = config.ct_alpha,
                beta = 0.0,
                max_step = config.ct_max_step,
                method = config.damping_method,
            ),
        )
    end

    exp_dependencies() = NGMPDependencies(
        out = nothing,
        in = nothing;
        projection = TangentProjection(type = ClosedForm),
    )

    function exp_damping(config)
        return DampingMeta(
            alpha = config.ngmp_alpha,
            beta = 0.0,
            max_step = config.ngmp_max_step,
            method = config.damping_method,
        )
    end
end

# ╔═╡ f825bb72-b5fb-4148-881f-fe73492a1582
@model function deep_learned_hierarchy(
    y,
    features,
    depth,
    width,
    priors,
    feature_covariance,
    activation_metas,
    activation_deps,
    ct_deps,
    link_deps,
    link_meta,
    score_carrier,
)
    local a_trunk, P_trunk, x_f
    local z_trunk, z_trunk_unit, h_trunk_unit, h_trunk
    local score, shifted_score, precision

    for layer in 1:depth
        a_trunk[layer] ~ priors.a_trunk[layer]
        P_trunk[layer] ~ priors.P_trunk[layer]
    end
    v ~ priors.v
    g ~ priors.g

    for observation in eachindex(features)
        x_f[observation] ~ MvNormalMeanCovariance(
            features[observation],
            feature_covariance,
        )

        for layer in 1:depth
            layer_input = layer == 1 ?
                x_f[observation] :
                h_trunk[layer - 1, observation]
            z_trunk[layer, observation] ~ ContinuousTransition(
                layer_input,
                a_trunk[layer],
                P_trunk[layer],
            ) where {
                dependencies = ct_deps,
                meta = priors.trunk_metas[layer],
            }
            for unit in 1:width
                z_trunk_unit[layer, unit, observation] ~
                    Uninformative()
            end
            z_trunk[layer, observation] ~ MvStack(inputs = [
                z_trunk_unit[layer, unit, observation]
                for unit in 1:width
            ])
            for unit in 1:width
                h_trunk_unit[layer, unit, observation] ~ ResidualSine(
                    z_trunk_unit[layer, unit, observation],
                ) where {
                    dependencies = activation_deps,
                    meta = activation_metas.trunk[layer],
                }
            end
            h_trunk[layer, observation] ~ MvStack(inputs = [
                h_trunk_unit[layer, unit, observation]
                for unit in 1:width
            ])
        end

        score[observation] ~ softdot(
            h_trunk[depth, observation],
            g,
            score_carrier,
        )
        shifted_score[observation] :=
            score[observation] + priors.anchor
        precision[observation] ~ Exp(
            shifted_score[observation],
        ) where {
            dependencies = link_deps,
            meta = link_meta,
        }
        y[observation] ~ softdot(
            h_trunk[depth, observation],
            v,
            precision[observation],
        )
    end
end

# ╔═╡ e62e2606-bda7-4711-941d-aee7af141568
@constraints function deep_learned_constraints()
    q(
        x_f,
        z_trunk,
        z_trunk_unit,
        h_trunk_unit,
        h_trunk,
        a_trunk,
        P_trunk,
        v,
        g,
        score,
        shifted_score,
        precision,
        y,
    ) =
        q(
            x_f,
            z_trunk,
            z_trunk_unit,
            h_trunk_unit,
            h_trunk,
        ) *
        q(a_trunk) *
        q(P_trunk) *
        q(v, y) *
        q(g, score, shifted_score, precision)

    q(a_trunk)::MomentForm()
    q(v)::MomentForm()
    q(g)::MomentForm()
end

# ╔═╡ ae30f54c-3e91-4b2b-b04f-f443f4fb1390
begin
    scalar_base() = NormalMeanVariance(0.0, 1.0)

    function scalar_residual_initial(meta)
        transformed_mean, transformed_variance =
            SurrogateModelling._residual_sine_mean_var_1d(
                0.0,
                1.0,
                meta,
            )
        return NormalMeanVariance(
            transformed_mean,
            transformed_variance,
        )
    end

    function residual_initial(width, meta)
        transformed_mean, transformed_covariance =
            SurrogateModelling._mv_residual_sine_mean_cov(
                zeros(width),
                Matrix(Diagonal(ones(width))),
                meta,
            )
        return MvNormalMeanCovariance(
            transformed_mean,
            transformed_covariance,
        )
    end

    function make_initial_states(
        n_observations,
        width,
        depth,
        activation_metas,
    )
        base = MvNormalMeanCovariance(
            zeros(width),
            Matrix(Diagonal(ones(width))),
        )
        z_trunk = Matrix{Any}(undef, depth, n_observations)
        h_trunk = Matrix{Any}(undef, depth, n_observations)
        z_trunk_unit =
            Array{Any}(undef, depth, width, n_observations)
        h_trunk_unit =
            Array{Any}(undef, depth, width, n_observations)
        for observation in 1:n_observations
            for layer in 1:depth
                z_trunk[layer, observation] = deepcopy(base)
                h_trunk[layer, observation] = residual_initial(
                    width,
                    activation_metas.trunk[layer],
                )
                for unit in 1:width
                    z_trunk_unit[layer, unit, observation] =
                        scalar_base()
                    h_trunk_unit[layer, unit, observation] =
                        scalar_residual_initial(
                            activation_metas.trunk[layer],
                        )
                end
            end
        end
        return (
            z_trunk = z_trunk,
            z_trunk_unit = z_trunk_unit,
            h_trunk_unit = h_trunk_unit,
            h_trunk = h_trunk,
        )
    end

    @initialization function deep_learned_initialization(priors, states)
        q(a_trunk) = deepcopy(priors.a_trunk)
        q(P_trunk) = deepcopy(priors.P_trunk)
        q(v) = deepcopy(priors.v)
        q(g) = deepcopy(priors.g)
        q(z_trunk) = states.z_trunk
        q(z_trunk_unit) = states.z_trunk_unit
        q(h_trunk_unit) = states.h_trunk_unit
        q(h_trunk) = states.h_trunk
        q(score) = NormalMeanVariance(0.0, 1.0)
        q(shifted_score) = NormalMeanVariance(priors.anchor, 1.0)
        q(precision) = GammaShapeRate(
            2.0,
            2.0 * exp(-priors.anchor),
        )
        μ(z_trunk) = states.z_trunk
        μ(z_trunk_unit) = states.z_trunk_unit
        μ(h_trunk_unit) = states.h_trunk_unit
        μ(h_trunk) = states.h_trunk
    end
end

# ╔═╡ 5350ca3a-0d69-4414-98e6-46359313b868
begin
    function inference_options()
        if Threads.nthreads() == 1
            return (limit_stack_depth = 250,)
        end
        return (
            limit_stack_depth = 250,
            fold_strategy = ReactiveMP.ParallelMessageProduct(
                nchunks = Threads.nthreads(),
            ),
        )
    end

    function fit_deep_hierarchy(data, config)
        priors = make_deep_priors(
            data.x_train,
            data.y_train,
            config,
        )
        activation_metas = make_activation_metas(config)
        states = make_initial_states(
            length(data.y_train),
            config.width,
            config.depth,
            activation_metas,
        )
        ct_deps = transition_dependencies(config)
        activation_deps = activation_dependencies(config)

        timed = @timed infer(
            model = deep_learned_hierarchy(
                depth = config.depth,
                width = config.width,
                priors = priors,
                feature_covariance = Matrix(Diagonal(
                    fill(config.feature_jitter, 2),
                )),
                activation_metas = activation_metas,
                activation_deps = activation_deps,
                ct_deps = ct_deps,
                link_deps = exp_dependencies(),
                link_meta = exp_damping(config),
                score_carrier = config.score_carrier,
            ),
            data = (
                y = data.y_train,
                features = raw_features(data.x_train),
            ),
            constraints = deep_learned_constraints(),
            initialization = deep_learned_initialization(priors, states),
            iterations = config.iterations,
            free_energy = config.free_energy,
            showprogress = config.show_progress,
            returnvars = (
                a_trunk = KeepLast(),
                P_trunk = KeepLast(),
                v = KeepLast(),
                g = KeepLast(),
            ),
            options = inference_options(),
            disable_inference_error_hint = true,
        )
        result = timed.value
        posterior = (
            a_trunk = collect(vec(result.posteriors[:a_trunk])),
            P_trunk = collect(vec(result.posteriors[:P_trunk])),
            v = result.posteriors[:v],
            g = result.posteriors[:g],
            anchor = priors.anchor,
            trunk_metas = priors.trunk_metas,
            activation_metas = activation_metas,
        )
        return (
            posterior = posterior,
            priors = priors,
            result = result,
            seconds = timed.time,
            allocated_gib = timed.bytes / 2.0^30,
            ct_deps = ct_deps,
            activation_deps = activation_deps,
        )
    end

    deep_fit = fit_deep_hierarchy(deep_data, DEEP_CONFIG)
    @printf(
        "fit completed in %.1f s (%.2f GiB allocated)\n",
        deep_fit.seconds,
        deep_fit.allocated_gib,
    )
end

# ╔═╡ 179f9e3e-59bf-4275-b3e8-873675073f44
md"""
## Native prediction

Prediction reruns `deep_learned_hierarchy` itself with the learned marginals as
priors and declares `y` through RxInfer's `predictvars`. The plotted mean and
variance therefore come directly from `result.predictions[:y]`; there is no
second hand-written prediction model and no auxiliary output prior.
"""

# ╔═╡ 08fce057-e0c7-44cf-8285-8397712b62d8
deep_prediction_constraints() = deep_learned_constraints()

# ╔═╡ 11e65ffd-0b59-4c4b-80ea-865fb8bb599d
@initialization function deep_prediction_initialization(
    priors,
    states,
)
    q(a_trunk) = deepcopy(priors.a_trunk)
    q(P_trunk) = deepcopy(priors.P_trunk)
    q(v) = deepcopy(priors.v)
    q(g) = deepcopy(priors.g)
    q(z_trunk) = states.z_trunk
    q(z_trunk_unit) = states.z_trunk_unit
    q(h_trunk_unit) = states.h_trunk_unit
    q(h_trunk) = states.h_trunk
    q(score) = NormalMeanVariance(0.0, 1.0)
    q(shifted_score) = NormalMeanVariance(priors.anchor, 1.0)
    q(precision) = GammaShapeRate(
        2.0,
        2.0 * exp(-priors.anchor),
    )
    μ(z_trunk) = states.z_trunk
    μ(z_trunk_unit) = states.z_trunk_unit
    μ(h_trunk_unit) = states.h_trunk_unit
    μ(h_trunk) = states.h_trunk
end

# ╔═╡ 90822741-f862-482a-bca4-1be2262a3876
begin
    function inverse_precision_mean(distribution)
        if distribution isa GammaShapeRate
            return distribution.a > 1 ?
                distribution.b / (distribution.a - 1) :
                inv(mean(distribution))
        end
        return inv(mean(distribution))
    end

    function run_rxinfer_prediction_batch(
        posterior,
        xs,
        config,
    )
        isempty(xs) && return (y = Any[], precision = Any[])
        activation_deps = activation_dependencies(config)
        ct_deps = transition_dependencies(config)
        states = make_initial_states(
            length(xs),
            config.width,
            config.depth,
            posterior.activation_metas,
        )
        result = infer(
            model = deep_learned_hierarchy(
                depth = config.depth,
                width = config.width,
                priors = posterior,
                feature_covariance = Matrix(Diagonal(
                    fill(config.feature_jitter, 2),
                )),
                activation_metas = posterior.activation_metas,
                activation_deps = activation_deps,
                ct_deps = ct_deps,
                link_deps = exp_dependencies(),
                link_meta = exp_damping(config),
                score_carrier = config.score_carrier,
            ),
            data = (features = raw_features(xs),),
            constraints = deep_prediction_constraints(),
            initialization = deep_prediction_initialization(
                posterior,
                states,
            ),
            iterations = config.prediction_iterations,
            free_energy = false,
            showprogress = false,
            predictvars = (y = KeepLast(),),
            returnvars = (precision = KeepLast(),),
            options = inference_options(),
            disable_inference_error_hint = true,
        )
        return (
            y = collect(vec(result.predictions[:y])),
            precision = collect(vec(result.posteriors[:precision])),
        )
    end

    function rxinfer_posterior_predictive(posterior, xs, config)
        y_marginals = Vector{Any}(undef, length(xs))
        precision_marginals = Vector{Any}(undef, length(xs))
        batch_size = config.prediction_batch_size
        for first_index in 1:batch_size:length(xs)
            indices =
                first_index:min(first_index + batch_size - 1, length(xs))
            batch = run_rxinfer_prediction_batch(
                posterior,
                xs[indices],
                config,
            )
            y_marginals[indices] = batch.y
            precision_marginals[indices] = batch.precision
        end

        predictive_mean = mean.(y_marginals)
        total_variance = max.(var.(y_marginals), 0.0)
        aleatoric = inverse_precision_mean.(precision_marginals)
        return (
            mean = predictive_mean,
            # The projected q(y) and q(precision) marginals are not a joint
            # distribution, so subtracting E[1 / precision] from Var(q(y))
            # is not a valid law-of-total-variance decomposition.
            epistemic = fill(NaN, length(xs)),
            aleatoric = aleatoric,
            variance = total_variance,
            y_marginals = y_marginals,
            precision_marginals = precision_marginals,
        )
    end
end

# ╔═╡ 897f60ef-75b0-42fd-825f-342b17fbed1a
begin
    function transition_matrix(meta::LinearReshapeMeta, parameter)
        return reshape(
            parameter,
            meta.output_dim,
            meta.input_dim,
        )
    end

    function transition_matrix(meta::LinearLowRankMeta, parameter)
        matrix = meta.U * Diagonal(parameter) * transpose(meta.V)
        for index in eachindex(meta.a0_diagonal)
            matrix[index, index] += meta.a0_diagonal[index]
        end
        return matrix
    end

    residual_sine(value, meta) =
        value + (meta.rho / meta.omega) * sin(meta.omega * value)

    function gaussian_sampler(distribution)
        mean_vector, covariance = mean_cov(distribution)
        decomposition = eigen(Symmetric(Matrix(covariance)))
        root = decomposition.vectors * Diagonal(
            sqrt.(max.(decomposition.values, 0.0)),
        )
        return (mean = collect(mean_vector), root = root)
    end

    draw_gaussian(rng, sampler) =
        sampler.mean + sampler.root * randn(rng, length(sampler.mean))

    function process_root(precision_distribution)
        covariance = inv(Symmetric(Matrix(mean(precision_distribution))))
        decomposition = eigen(Symmetric(Matrix(covariance)))
        return decomposition.vectors * Diagonal(
            sqrt.(max.(decomposition.values, 0.0)),
        )
    end

    function prepare_prediction(posterior)
        return (
            a_trunk = gaussian_sampler.(posterior.a_trunk),
            P_trunk = process_root.(posterior.P_trunk),
            v = gaussian_sampler(posterior.v),
            g = gaussian_sampler(posterior.g),
        )
    end

    function posterior_predictive(
        posterior,
        xs,
        config;
        seed = config.prediction_seed,
    )
        prepared = prepare_prediction(posterior)
        rng = StableRNG(seed)
        count = length(xs)
        sum_mean = zeros(count)
        sum_mean_square = zeros(count)
        sum_aleatoric = zeros(count)

        for _ in 1:config.prediction_draws
            a_trunk = draw_gaussian.(Ref(rng), prepared.a_trunk)
            trunk_matrices = [
                transition_matrix(
                    posterior.trunk_metas[layer],
                    a_trunk[layer],
                )
                for layer in 1:config.depth
            ]
            v = draw_gaussian(rng, prepared.v)
            g = draw_gaussian(rng, prepared.g)

            for (index, x) in enumerate(xs)
                hidden = [1.0, Float64(x)]
                for layer in 1:config.depth
                    preactivation = trunk_matrices[layer] * hidden
                    if config.sample_transition_noise
                        preactivation .+=
                            prepared.P_trunk[layer] *
                            randn(rng, config.width)
                    end
                    hidden = residual_sine.(
                        preactivation,
                        Ref(posterior.activation_metas.trunk[layer]),
                    )
                end

                latent_mean = dot(v, hidden)
                log_precision =
                    dot(g, hidden) +
                    posterior.anchor +
                    randn(rng) / sqrt(config.score_carrier)
                observation_variance =
                    exp(clamp(-log_precision, -40.0, 40.0))

                sum_mean[index] += latent_mean
                sum_mean_square[index] += abs2(latent_mean)
                sum_aleatoric[index] += observation_variance
            end
        end

        predictive_mean = sum_mean ./ config.prediction_draws
        epistemic = max.(
            sum_mean_square ./ config.prediction_draws .-
            abs2.(predictive_mean),
            0.0,
        )
        aleatoric = sum_aleatoric ./ config.prediction_draws
        return (
            mean = predictive_mean,
            epistemic = epistemic,
            aleatoric = aleatoric,
            variance = epistemic + aleatoric,
        )
    end

    montecarlo_grid_prediction =
        DEEP_CONFIG.prediction_mode in (:montecarlo, :both) ?
        posterior_predictive(
            deep_fit.posterior,
            deep_grid,
            DEEP_CONFIG,
        ) : nothing
    montecarlo_test_prediction =
        DEEP_CONFIG.prediction_mode in (:montecarlo, :both) ?
        posterior_predictive(
            deep_fit.posterior,
            deep_data.x_test,
            DEEP_CONFIG;
            seed = DEEP_CONFIG.prediction_seed + 1,
        ) : nothing

    rxinfer_grid_prediction =
        DEEP_CONFIG.prediction_mode in (:rxinfer, :both) ?
        rxinfer_posterior_predictive(
            deep_fit.posterior,
            deep_grid,
            DEEP_CONFIG,
        ) : nothing
    rxinfer_test_prediction =
        DEEP_CONFIG.prediction_mode in (:rxinfer, :both) ?
        rxinfer_posterior_predictive(
            deep_fit.posterior,
            deep_data.x_test,
            DEEP_CONFIG,
        ) : nothing

    deep_grid_prediction = DEEP_CONFIG.prediction_mode === :montecarlo ?
        montecarlo_grid_prediction :
        rxinfer_grid_prediction
    deep_test_prediction = DEEP_CONFIG.prediction_mode === :montecarlo ?
        montecarlo_test_prediction :
        rxinfer_test_prediction
end

# ╔═╡ fd1933f7-02cd-47a2-834f-77889c65aa83
begin
    function predictive_scores(prediction, xs, ys)
        variance = max.(prediction.variance, 1e-10)
        residual = ys .- prediction.mean
        truth = true_noise_variance.(xs)
        return (
            logpdf = mean(
                -0.5 .* (
                    log.(2pi .* variance) .+
                    abs2.(residual) ./ variance
                ),
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

    deep_test_scores = predictive_scores(
        deep_test_prediction,
        deep_data.x_test,
        deep_data.y_test,
    )
    if DEEP_CONFIG.prediction_mode === :both
        montecarlo_test_scores = predictive_scores(
            montecarlo_test_prediction,
            deep_data.x_test,
            deep_data.y_test,
        )
        @printf(
            "prediction comparison: RxInfer RMSE %.4f/logpdf %.4f | Monte Carlo RMSE %.4f/logpdf %.4f\n",
            deep_test_scores.rmse,
            deep_test_scores.logpdf,
            montecarlo_test_scores.rmse,
            montecarlo_test_scores.logpdf,
        )
    end

    function prior_scale_movement(posterior, prior)
        scale = sqrt(max(tr(Matrix(cov(prior))), eps(Float64)))
        return norm(mean(posterior) - mean(prior)) / scale
    end

    initial_a = deep_fit.priors.a_trunk
    learned_a = deep_fit.posterior.a_trunk
    trunk_movements = [
        prior_scale_movement(learned_a[layer], initial_a[layer])
        for layer in 1:DEEP_CONFIG.depth
    ]
    mean_head_movement = prior_scale_movement(
        deep_fit.posterior.v,
        deep_fit.priors.v,
    )
    noise_head_movement = prior_scale_movement(
        deep_fit.posterior.g,
        deep_fit.priors.g,
    )
    ct_firings = getproperty.(deep_fit.ct_deps.states, :nfired)
    activation_firings =
        getproperty.(deep_fit.activation_deps.states, :nfired)

    epistemic_text = isfinite(deep_test_scores.mean_epistemic) ?
        @sprintf("%.4f", deep_test_scores.mean_epistemic) :
        "n/a for projected marginals"
    @printf(
        "test logpdf %.4f | RMSE %.4f | latent RMSE %.4f | noise corr %.4f | E[1/lambda] %.4f (truth %.4f) | epistemic %s\n",
        deep_test_scores.logpdf,
        deep_test_scores.rmse,
        deep_test_scores.latent_mean_rmse,
        deep_test_scores.noise_correlation,
        deep_test_scores.mean_aleatoric,
        deep_test_scores.true_mean_aleatoric,
        epistemic_text,
    )
    @printf(
        "prior-scale weight movement: trunk [%s], mean readout %.3f, noise readout %.3f\n",
        join(round.(trunk_movements; digits = 3), ", "),
        mean_head_movement,
        noise_head_movement,
    )
    @printf(
        "NGMP states: %d CT (%d..%d firings), %d activations (%d..%d firings)\n",
        length(ct_firings),
        isempty(ct_firings) ? 0 : minimum(ct_firings),
        isempty(ct_firings) ? 0 : maximum(ct_firings),
        length(activation_firings),
        isempty(activation_firings) ? 0 : minimum(activation_firings),
        isempty(activation_firings) ? 0 : maximum(activation_firings),
    )
    all(firings -> firings >= DEEP_CONFIG.iterations, ct_firings) ||
        error("not every CT natural-gradient state fired per iteration")
    all(
        firings -> firings >= DEEP_CONFIG.iterations,
        activation_firings,
    ) || error("not every activation state fired per iteration")
    if DEEP_CONFIG.free_energy
        all(isfinite, deep_fit.result.free_energy) ||
            error("inference produced non-finite free energy")
        @printf(
            "free energy first/last: %.3f / %.3f\n",
            first(deep_fit.result.free_energy),
            last(deep_fit.result.free_energy),
        )
    end
end

# ╔═╡ b4bec151-80a7-4bbc-b911-327521474157
begin
    prediction_band =
        1.96 .* sqrt.(max.(deep_grid_prediction.variance, 0.0))
    mean_panel = plot(
        deep_grid,
        deep_grid_prediction.mean;
        ribbon = prediction_band,
        fillalpha = 0.20,
        color = :steelblue,
        linewidth = 2,
        label = "q(y*) ± 1.96 SD",
        xlabel = "x",
        ylabel = "y",
        title = "Native RxInfer prediction",
        titlefontsize = 9,
        legend = :topleft,
    )
    plot!(
        mean_panel,
        deep_grid,
        true_mean.(deep_grid);
        color = :black,
        linewidth = 2,
        label = "true mean",
    )
    scatter!(
        mean_panel,
        deep_data.x_train,
        deep_data.y_train;
        color = :black,
        markersize = 2,
        markerstrokewidth = 0,
        alpha = 0.35,
        label = "train",
    )

    variance_panel = plot(
        deep_grid,
        max.(deep_grid_prediction.variance, 1e-6);
        color = :steelblue,
        linewidth = 2,
        label = "Var(q(y*))",
        xlabel = "x",
        ylabel = "variance",
        yscale = :log10,
        title = "Projected predictive marginals",
        titlefontsize = 9,
        legend = :topleft,
    )
    if all(isfinite, deep_grid_prediction.epistemic)
        plot!(
            variance_panel,
            deep_grid,
            max.(deep_grid_prediction.epistemic, 1e-6);
            color = :darkorange,
            linewidth = 2,
            label = "epistemic",
        )
    end
    plot!(
        variance_panel,
        deep_grid,
        max.(deep_grid_prediction.aleatoric, 1e-6);
        color = :purple,
        linewidth = 2,
        label = "E[1/λ(x)]",
    )
    plot!(
        variance_panel,
        deep_grid,
        max.(true_noise_variance.(deep_grid), 1e-6);
        color = :black,
        linestyle = :dashdot,
        label = "true noise",
    )

    deep_hierarchy_plot = plot(
        mean_panel,
        variance_panel;
        layout = (1, 2),
        size = (1120, 420),
        margin = 4Plots.mm,
    )

    if DEEP_CONFIG.save_outputs
        filename = DEEP_CONFIG.output_prefix * ".png"
        mkpath(dirname(filename))
        savefig(deep_hierarchy_plot, filename)
        println("saved ", filename)
    end
end

# ╔═╡ 02549c33-1fae-4303-be8d-f08af5623fe5
md"""
## Reading the experiment

`TRANSITION_LAYOUT=lowrank` is the intended parameter-efficient architecture:

- raw-input CT: dense `LinearReshapeMeta` with the same frequency-aware weight
  uncertainty as the working shallow notebook;
- repeated trunk CTs: identity-offset `LinearLowRankMeta`;
- nonlinearities: scalar `ResidualSine` factors between an exact backward
  `MvStack` split and a forward `MvStack` collection;
- heads: direct dense-Gaussian readout vectors `v` and `g`.

For an ablation, set `TRANSITION_LAYOUT=reshape` to make every repeated
transition dense. `mixed` remains an alias for the default low-rank trunk.
`DEEP_DEPTH` counts the shared `CT → split → activate → stack` blocks.

`ACTIVATION_FORWARD=moments` uses the exact first two pushforward moments.
`ACTIVATION_FORWARD=delta` instead uses the package's implicit-function delta
tangent site, while retaining the same exact analytic backward Fisher
projection. On this deep benchmark the delta-forward arm is an experimental
ablation: it was numerically unstable, so `moments` is deliberately the default.

`PREDICTION_MODE=rxinfer` is the default. It reruns the exact training model,
uses `predictvars = (y = KeepLast(),)`, and reads the plotted marginals from
`result.predictions[:y]`. `PREDICTION_MODE=both` additionally evaluates the
older posterior Monte Carlo rollout as a diagnostic, but never substitutes it
for the plotted RxInfer result.

The full default run is stable, and the scalar learning path is genuinely
active: all 800 CT states and 25,600 scalar activation-edge states fired, the
two trunk transitions moved 0.942 and 3.858 prior scales, and the mean/noise
readouts moved 0.585 and 0.294. The frequency-aware input prior and true
identity-offset second CT improve on the discarded joint-vector graph.

It is nevertheless a negative predictive result. The native RxInfer marginals
still attenuate most of the target oscillation: held-out RMSE is 0.821,
latent-mean RMSE 0.712, and log predictive density -2.298. The learned noise
shape is more credible (correlation 0.702), but internal parameter movement is
not enough to claim that the hierarchy learned a useful mean function. A
depth-one isolation run was worse, so merely removing the second block is not
the remedy.

The projected `q(y)` and `q(precision)` marginals also do not define a joint
posterior, so this notebook does not manufacture an epistemic curve by clipping
`Var(q(y)) - E_q[1 / precision]`. The plot shows those two marginal diagnostics
separately. The result suggests the next experiment should change the learning
schedule (for example, shallow pretraining followed by an identity-initialized
second block) rather than adding another layer.
"""

# ╔═╡ Cell order:
# ╠═8dc98342-8d11-4f27-b77c-2153af7f0a11
# ╠═92c569e8-11c2-4ca5-ae84-577736ea3993
# ╟─2b76f226-13a6-4aab-92eb-69b6668e05c4
# ╠═86812ffc-f260-425f-97d1-e035f53b75b4
# ╠═93b993c4-a100-4acf-a6ca-21fa06a2760a
# ╠═07081021-37cb-494c-b9ab-e20e57ea2a30
# ╠═a3d7f9c8-95a6-42d0-a171-f04f0faf0eb9
# ╠═f825bb72-b5fb-4148-881f-fe73492a1582
# ╠═e62e2606-bda7-4711-941d-aee7af141568
# ╠═ae30f54c-3e91-4b2b-b04f-f443f4fb1390
# ╠═5350ca3a-0d69-4414-98e6-46359313b868
# ╟─179f9e3e-59bf-4275-b3e8-873675073f44
# ╠═08fce057-e0c7-44cf-8285-8397712b62d8
# ╠═11e65ffd-0b59-4c4b-80ea-865fb8bb599d
# ╠═90822741-f862-482a-bca4-1be2262a3876
# ╠═897f60ef-75b0-42fd-825f-342b17fbed1a
# ╠═fd1933f7-02cd-47a2-834f-77889c65aa83
# ╠═b4bec151-80a7-4bbc-b911-327521474157
# ╟─02549c33-1fae-4303-be8d-f08af5623fe5
