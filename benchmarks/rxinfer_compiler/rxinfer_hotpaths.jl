using Pkg

const PERF_PROJECT =
    get(ENV, "PERF_PROJECT", joinpath(@__DIR__, "..", ".."))
Pkg.activate(PERF_PROJECT)

using Distributions
using LinearAlgebra
using Random
using ReactiveMP
using RxInfer
using Statistics
using SurrogateModelling

LinearAlgebra.BLAS.set_num_threads(1)

const PERF_CONFIG = (
    n_neurons = 8,
    prior_seed = 42,
    phi_rho = 0.9,
    phi_omega = 1.0,
    w_prior_scale = 2.2,
    w_prior_variance = 0.005,
    bias_prior_variance = 0.5,
    v_prior_scale = 0.5,
    v_prior_variance = 0.01,
    tau_prior = (1e3, 1.0),
    tau_c_prior = (1e4, 1.0),
    obs_noise_prior = (100.0, 1.0),
    ngmp_alpha = 0.05,
    ngmp_beta = 0.0,
    ngmp_max_step = 1.0,
    prediction_iterations = 8,
    prediction_batch_size = 1_024,
    prediction_prior_variance = 1e12,
)

function make_perf_priors(config)
    rng = MersenneTwister(config.prior_seed)
    n_pairs = config.n_neurons ÷ 2
    prior_precision = Diagonal([
        1 / config.bias_prior_variance,
        1 / config.w_prior_variance,
        1 / config.w_prior_variance,
    ])
    w = Vector{Any}(undef, config.n_neurons)
    v = Vector{Any}(undef, config.n_neurons)

    for pair in 1:n_pairs
        angle = π * (pair - 1) / n_pairs + 0.1 * randn(rng)
        radius = config.w_prior_scale * (1 + 0.05 * randn(rng))
        direction = [radius * cos(angle), radius * sin(angle)]
        bias = (isodd(pair) ? 1.0 : -1.0) *
               (π / 4) * (1 + 0.1 * randn(rng))

        for (slot, sign) in ((2pair - 1, 1.0), (2pair, -1.0))
            prior_mean = [sign * bias, direction[1], direction[2]]
            w[slot] = MvNormalWeightedMeanPrecision(
                prior_precision * prior_mean,
                prior_precision,
            )
            v[slot] = NormalMeanVariance(
                sign * config.v_prior_scale,
                config.v_prior_variance,
            )
        end
    end

    return Dict{Symbol, Any}(
        :intercept => NormalMeanVariance(0, 100),
        :w => w,
        :v => v,
        :τ => GammaShapeRate(config.tau_prior...),
        :τ_c => GammaShapeRate(config.tau_c_prior...),
        :obs_noise => GammaShapeRate(config.obs_noise_prior...),
    )
end

function make_perf_dependencies(config)
    return NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = TangentProjection(type = ClosedForm),
        damping = DampingMeta(
            alpha = config.ngmp_alpha,
            beta = config.ngmp_beta,
            max_step = config.ngmp_max_step,
        ),
    )
end

@model function perf_prediction_model(
    n_neurons,
    features,
    priors,
    activation,
    activation_deps,
    y_prior_variance,
)
    local w, v, za, h, c, out, intercept, mean_output, y

    τ ~ priors[:τ]
    τ_c ~ priors[:τ_c]
    obs_noise ~ priors[:obs_noise]
    intercept ~ priors[:intercept]

    for neuron in 1:n_neurons
        w[neuron] ~ priors[:w][neuron]
        v[neuron] ~ priors[:v][neuron]
    end

    for observation in eachindex(features)
        for neuron in 1:n_neurons
            za[neuron, observation] ~
                softdot(features[observation], w[neuron], τ)
            h[neuron, observation] ~ ResidualSine(za[neuron, observation]) where {
                dependencies = activation_deps,
                meta = activation,
            }
            c[neuron, observation] ~
                softdot(v[neuron], h[neuron, observation], τ_c)
        end
        out[observation] ~ ManyPlus(
            inputs = [c[neuron, observation] for neuron in 1:n_neurons],
        )
        mean_output[observation] := out[observation] + intercept
        y[observation] ~ NormalMeanPrecision(
            mean_output[observation],
            obs_noise,
        )
        y[observation] ~ NormalMeanVariance(0.0, y_prior_variance)
    end
end

@constraints function perf_prediction_constraints(priors)
    q(
        w, v, za, h, c, out, intercept, mean_output,
        τ, τ_c, obs_noise, y,
    ) = q(w)q(v)q(τ)q(τ_c)q(obs_noise)q(intercept) *
        q(za, h, c, out, mean_output, y)

    q(τ)::RxInfer.FixedMarginalFormConstraint(priors[:τ])
    q(τ_c)::RxInfer.FixedMarginalFormConstraint(priors[:τ_c])
    q(obs_noise)::RxInfer.FixedMarginalFormConstraint(priors[:obs_noise])
    q(intercept)::RxInfer.FixedMarginalFormConstraint(priors[:intercept])

    for (neuron, prior) in enumerate(priors[:w])
        q(w[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    for (neuron, prior) in enumerate(priors[:v])
        q(v[neuron])::RxInfer.FixedMarginalFormConstraint(prior)
    end
end

@initialization function perf_prediction_initialization(
    priors,
    inits,
    output_mean,
    y_prior_variance,
)
    q(w) = deepcopy(priors[:w])
    q(v) = deepcopy(priors[:v])
    q(za) = inits.za
    q(h) = inits.h
    q(c) = inits.c
    q(out) = inits.out
    q(mean_output) = NormalMeanVariance(output_mean, 10.0)
    q(τ) = priors[:τ]
    q(τ_c) = priors[:τ_c]
    q(obs_noise) = priors[:obs_noise]
    q(intercept) = priors[:intercept]
    q(y) = NormalMeanVariance(output_mean, y_prior_variance)

    μ(out) = NormalMeanVariance(
        output_mean - mean(priors[:intercept]),
        10.0,
    )
    μ(y) = NormalMeanVariance(output_mean, y_prior_variance)
end

function perf_pushforward_inits(priors, features, config)
    activation = ResidualSineMeta(
        rho = config.phi_rho,
        omega = config.phi_omega,
    )
    phi(x) = SurrogateModelling._residual_sine(x, activation)
    w_means = mean.(priors[:w])
    v_means = mean.(priors[:v])
    n = length(features)

    za = [
        NormalMeanVariance(dot(w_means[k], features[i]), 0.5)
        for k in 1:config.n_neurons, i in 1:n
    ]
    h = [
        NormalMeanVariance(phi(mean(za[k, i])), 1.0)
        for k in 1:config.n_neurons, i in 1:n
    ]
    c = [
        NormalMeanVariance(v_means[k] * mean(h[k, i]), 1.0)
        for k in 1:config.n_neurons, i in 1:n
    ]
    out = [
        NormalMeanVariance(
            sum(mean(c[k, i]) for k in 1:config.n_neurons),
            1.0,
        )
        for i in 1:n
    ]
    return (za = za, h = h, c = c, out = out)
end

const PERF_SAMPLES = parse(Int, get(ENV, "PERF_SAMPLES", "8"))
const PERF_LABEL = get(ENV, "PERF_LABEL", "working-tree")
const PERF_OUTPUT = get(ENV, "PERF_OUTPUT", "")
const PERF_CALLBACK_MODE = Symbol(get(ENV, "PERF_CALLBACK_MODE", "none"))
const PERF_STACK_LIMIT = get(ENV, "PERF_STACK_LIMIT", "100")
const PERF_BATCH_SIZE = parse(
    Int, get(ENV, "PERF_BATCH_SIZE", string(PERF_CONFIG.prediction_batch_size))
)
const PERF_POSTERIOR = get(
    ENV,
    "PERF_POSTERIOR_OUTPUT",
    get(ENV, "PERF_POSTERIOR", ""),
)

PERF_SAMPLES > 0 || throw(ArgumentError("PERF_SAMPLES must be positive"))
1 <= PERF_BATCH_SIZE <= PERF_CONFIG.prediction_batch_size || throw(
    ArgumentError(
        "PERF_BATCH_SIZE must be between 1 and $(PERF_CONFIG.prediction_batch_size)",
    ),
)
PERF_CALLBACK_MODE in (:none, :benchmark) ||
    throw(ArgumentError("PERF_CALLBACK_MODE must be `none` or `benchmark`"))
const ITERATION_CALLBACKS =
    PERF_CALLBACK_MODE === :benchmark ?
    RxInferBenchmarkCallbacks(capacity = PERF_SAMPLES + 4) : nothing
const ITERATION_STACK_OPTIONS =
    PERF_STACK_LIMIT == "none" ? (;) :
    (limit_stack_depth = parse(Int, PERF_STACK_LIMIT),)
const ITERATION_OPTIONS = ITERATION_STACK_OPTIONS

const ITERATION_FEATURE_GRID = [
    [1.0, x1, x2]
    for x2 in range(-2.0, 2.0; length = 32)
    for x1 in range(-2.0, 2.0; length = 32)
]
const ITERATION_FEATURES = ITERATION_FEATURE_GRID[1:PERF_BATCH_SIZE]

const ITERATION_CONFIG = PERF_CONFIG
const ITERATION_PRIORS = make_perf_priors(ITERATION_CONFIG)
const ITERATION_OUTPUT_MEAN = 0.0
const ITERATION_DEPENDENCIES = make_perf_dependencies(ITERATION_CONFIG)
const ITERATION_ACTIVATION = ResidualSineMeta(
    rho = ITERATION_CONFIG.phi_rho,
    omega = ITERATION_CONFIG.phi_omega,
)
const INITIAL_INITIALIZATION = perf_prediction_initialization(
    ITERATION_PRIORS,
    perf_pushforward_inits(
        ITERATION_PRIORS,
        ITERATION_FEATURES,
        ITERATION_CONFIG,
    ),
    ITERATION_OUTPUT_MEAN,
    ITERATION_CONFIG.prediction_prior_variance,
)

const ITERATION_MODEL = prepare_inference(
    perf_prediction_model(
        n_neurons = ITERATION_CONFIG.n_neurons,
        priors = ITERATION_PRIORS,
        activation = ITERATION_ACTIVATION,
        activation_deps = ITERATION_DEPENDENCIES,
        y_prior_variance = ITERATION_CONFIG.prediction_prior_variance,
    );
    data = (features = ITERATION_FEATURES,),
    constraints = perf_prediction_constraints(ITERATION_PRIORS),
    initialization = INITIAL_INITIALIZATION,
    options = ITERATION_OPTIONS,
    callbacks = ITERATION_CALLBACKS,
)

infer!(
    ITERATION_MODEL;
    data = (features = ITERATION_FEATURES,),
    iterations = ITERATION_CONFIG.prediction_iterations,
    free_energy = false,
    showprogress = false,
    returnvars = (y = KeepLast(),),
    session = nothing,
    disable_inference_error_hint = true,
)

const ITERATION_STATE = RxInfer.__prepared_batch_state(ITERATION_MODEL)
const ITERATION_DATA, ITERATION_PREDICTVARS = RxInfer.__prepare_batch_data(
    (features = ITERATION_FEATURES,),
    nothing,
    ITERATION_CONFIG.prediction_iterations,
)

function iteration_initialization()
    return perf_prediction_initialization(
        ITERATION_PRIORS,
        perf_pushforward_inits(
            ITERATION_PRIORS,
            ITERATION_FEATURES,
            ITERATION_CONFIG,
        ),
        ITERATION_OUTPUT_MEAN,
        ITERATION_CONFIG.prediction_prior_variance,
    )
end

function rematerialize_for_iteration!()
    empty!(ITERATION_DEPENDENCIES.states)
    initialization = iteration_initialization()
    rematerialize = RxInfer.__rematerialize_batch_runtime!
    arguments = (
        ITERATION_MODEL,
        ITERATION_STATE.options,
        initialization,
        ITERATION_STATE.constraints,
        nothing,
    )
    if applicable(rematerialize, arguments..., false)
        rematerialize(arguments..., false)
    else
        rematerialize(arguments...)
    end
    return initialization
end

function run_iteration_wave!(initialization)
    return RxInfer.__run_batch_inference!(
        ITERATION_MODEL;
        rematerialize = false,
        initialization = initialization,
        constraints = ITERATION_STATE.constraints,
        data = ITERATION_DATA,
        returnvars = (y = KeepLast(),),
        predictvars = ITERATION_PREDICTVARS,
        iterations = ITERATION_CONFIG.prediction_iterations,
        free_energy = false,
        free_energy_diagnostics =
            RxInfer.DefaultObjectiveDiagnosticChecks,
        showprogress = false,
        callbacks = ITERATION_CALLBACKS,
        postprocess = RxInfer.__default_batch_postprocess(
            ITERATION_STATE.options,
        ),
        warn = false,
        catch_exception = false,
        disable_inference_error_hint = true,
    )
end

function posterior_moment_vectors(result)
    posteriors = collect(vec(result.posteriors[:y]))
    return Float64.(mean.(posteriors)), Float64.(var.(posteriors))
end

function assert_same_posterior(reference, candidate)
    all(
        isapprox.(reference[1], candidate[1]; atol = 1e-8, rtol = 1e-7),
    ) || error("posterior means changed between benchmark samples")
    all(
        isapprox.(reference[2], candidate[2]; atol = 1e-8, rtol = 1e-7),
    ) || error("posterior variances changed between benchmark samples")
    return nothing
end

function validate_firings!()
    states = ITERATION_DEPENDENCIES.states
    expected_states =
        2 * ITERATION_CONFIG.n_neurons * length(ITERATION_FEATURES)
    length(states) == expected_states ||
        error("expected $expected_states NGMP states, got $(length(states))")
    nfired = getfield.(states, :nfired)
    all(==(ITERATION_CONFIG.prediction_iterations), nfired) ||
        error("each NGMP state must fire exactly once per iteration")
    return (states = length(states), firings = sum(nfired))
end

function latest_lifecycle_timing(callbacks::RxInferBenchmarkCallbacks)
    inference_seconds = Float64(
        last(callbacks.after_inference_ts) -
        last(callbacks.before_inference_ts),
    ) / 1e9
    iteration_nanoseconds =
        last(callbacks.after_iteration_ts) .-
        last(callbacks.before_iteration_ts)
    iteration_callback_seconds =
        sum(Float64, iteration_nanoseconds) / 1e9
    return (
        inference_span_seconds = inference_seconds,
        iteration_callback_seconds = iteration_callback_seconds,
    )
end

latest_lifecycle_timing(::Nothing) = (
    inference_span_seconds = NaN,
    iteration_callback_seconds = NaN,
)

function run_hotpath_benchmark()
# Compile every operation before collecting samples.
run_iteration_wave!(rematerialize_for_iteration!())

rows = NamedTuple[]
reference_posterior = Ref{Any}(nothing)
final_posterior = Ref{Any}(nothing)

for sample in 1:PERF_SAMPLES
    GC.gc()
    rematerialization = @timed rematerialize_for_iteration!()

    GC.gc()
    iteration_wave = @timed run_iteration_wave!(rematerialization.value)
    lifecycle = latest_lifecycle_timing(ITERATION_CALLBACKS)
    firings = validate_firings!()
    posterior = posterior_moment_vectors(iteration_wave.value)

    if isnothing(reference_posterior[])
        reference_posterior[] = posterior
    else
        assert_same_posterior(reference_posterior[], posterior)
    end
    final_posterior[] = posterior

    push!(
        rows,
        (
            label = PERF_LABEL,
            sample = sample,
            rematerialization_seconds = rematerialization.time,
            rematerialization_bytes = rematerialization.bytes,
            rematerialization_gc_seconds = rematerialization.gctime,
            iteration_seconds = iteration_wave.time,
            seconds_per_iteration =
                iteration_wave.time /
                ITERATION_CONFIG.prediction_iterations,
            iteration_bytes = iteration_wave.bytes,
            bytes_per_iteration =
                iteration_wave.bytes ÷
                ITERATION_CONFIG.prediction_iterations,
            iteration_gc_seconds = iteration_wave.gctime,
            inference_span_seconds = lifecycle.inference_span_seconds,
            iteration_callback_seconds =
                lifecycle.iteration_callback_seconds,
            outside_iteration_seconds =
                isfinite(lifecycle.iteration_callback_seconds) ?
                iteration_wave.time -
                lifecycle.iteration_callback_seconds : NaN,
            states = firings.states,
            firings = firings.firings,
        ),
    )
end

function write_rows(io, rows)
    println(
        io,
        join(
            (
                "label",
                "sample",
                "rematerialization_seconds",
                "rematerialization_bytes",
                "rematerialization_gc_seconds",
                "iteration_seconds",
                "seconds_per_iteration",
                "iteration_bytes",
                "bytes_per_iteration",
                "iteration_gc_seconds",
                "inference_span_seconds",
                "iteration_callback_seconds",
                "outside_iteration_seconds",
                "states",
                "firings",
            ),
            ',',
        ),
    )
    for row in rows
        println(io, join(values(row), ','))
    end
end

if isempty(PERF_OUTPUT)
    write_rows(stdout, rows)
else
    open(PERF_OUTPUT, "w") do io
        write_rows(io, rows)
    end
end

if !isempty(PERF_POSTERIOR)
    open(PERF_POSTERIOR, "w") do io
        println(io, "index,mean,variance")
        for index in eachindex(final_posterior[][1], final_posterior[][2])
            println(
                io,
                index,
                ',',
                final_posterior[][1][index],
                ',',
                final_posterior[][2][index],
            )
        end
    end
end

println(
    "PERF_SUMMARY ",
    (
        label = PERF_LABEL,
        samples = PERF_SAMPLES,
        median_rematerialization_seconds =
            median(getindex.(rows, :rematerialization_seconds)),
        median_iteration_seconds =
            median(getindex.(rows, :iteration_seconds)),
        median_seconds_per_iteration =
            median(getindex.(rows, :seconds_per_iteration)),
        median_iteration_bytes =
            median(getindex.(rows, :iteration_bytes)),
        median_bytes_per_iteration =
            median(getindex.(rows, :bytes_per_iteration)),
        median_iteration_callback_seconds =
            median(getindex.(rows, :iteration_callback_seconds)),
        median_outside_iteration_seconds =
            median(getindex.(rows, :outside_iteration_seconds)),
        states = first(rows).states,
        firings = first(rows).firings,
    ),
)
return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_hotpath_benchmark()
end
