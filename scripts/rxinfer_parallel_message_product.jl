using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BenchmarkTools
using LinearAlgebra
using Printf
using RxInfer
using StableRNGs
using Statistics
using SurrogateModelling

import ProbabilisticEnsembling: Exp

# L = 2, 800 observations, and the same 33-dimensional Fourier features as the
# hierarchy notebook. Julia owns the parallelism; nested BLAS threads stay off.
BLAS.set_num_threads(1)
const N = 800
const D = 33
const ITERATIONS = 5
const PARALLEL_FOLD = ReactiveMP.ParallelMessageProduct(
    nchunks = min(Threads.nthreads(), 6),
)

rng = StableRNG(20260728)
x = randn(rng, N)
y = -(x .+ 0.5) .* sin.(3pi .* x) .+
    abs.(0.45 .* (x .+ 0.5)) .* randn(rng, N)

frequencies = randn(rng, D - 1) ./ 0.25
phases = 2pi .* rand(rng, D - 1)
features = [
    vcat(sqrt(2 / (D - 1)) .* cos.(frequencies .* xi .+ phases), 1.0)
    for xi in x
]

@model function two_layer_model(
    y,
    features,
    v_prior,
    w_prior,
    dependencies,
    damping,
)
    v ~ v_prior
    w ~ w_prior
    for i in eachindex(y)
        score[i] ~ softdot(features[i], w, 25.0)
        precision[i] ~ Exp(score[i]) where {
            dependencies = dependencies,
            meta = damping,
        }
        y[i] ~ softdot(features[i], v, precision[i])
    end
end

@constraints function constraints()
    q(v, w, score, precision, y) = q(v, y)q(w, score)q(precision)
    q(v)::MomentForm()
    q(w)::MomentForm()
end

# Only the two 33-dimensional, degree-801 hubs are opted in here.
dense_meta() = @meta begin
    v -> (
        meta = nothing,
        messages_fold_strategy = PARALLEL_FOLD,
        marginal_fold_strategy = PARALLEL_FOLD,
    )
    w -> (
        meta = nothing,
        messages_fold_strategy = PARALLEL_FOLD,
        marginal_fold_strategy = PARALLEL_FOLD,
    )
end

prior = MvNormalMeanCovariance(zeros(D), Matrix{Float64}(I, D, D))
score0 = fill(NormalMeanVariance(0.0, 1.0), N)
precision0 = fill(GammaShapeRate(2.0, 2.0), N)
dependencies = NGMPDependencies(
    out = nothing,
    in = nothing;
    projection = TangentProjection(type = ClosedForm),
)
damping = DampingMeta(
    alpha = 0.6,
    beta = 0.0,
    max_step = 0.5,
    method = :damped,
)

function fit(mode)
    initialization = @initialization begin
        q(v) = deepcopy(prior)
        q(w) = deepcopy(prior)
        q(score) = deepcopy(score0)
        q(precision) = deepcopy(precision0)
    end

    # :dense assigns the strategy only to v and w. :all also offers it to all
    # scalar variables; ParallelMessageProduct should reject those as too small.
    meta = mode === :dense ? dense_meta() : nothing
    options = mode === :all ?
        (limit_stack_depth = 100, fold_strategy = PARALLEL_FOLD) :
        (limit_stack_depth = 100,)

    result = infer(
        model = two_layer_model(
            v_prior = prior,
            w_prior = prior,
            dependencies = dependencies,
            damping = damping,
        ),
        data = (y = y, features = features),
        constraints = constraints(),
        initialization = initialization,
        meta = meta,
        options = options,
        returnvars = (v = KeepLast(), w = KeepLast()),
        iterations = ITERATIONS,
        free_energy = false,
        showprogress = false,
    )
    return result.posteriors
end

Threads.nthreads() > 1 ||
    error("Restart Julia with --threads=6 (or another value greater than one).")

# Warm every compiled path, then let BenchmarkTools collect independent infer calls.
reference = fit(:sequential)
dense_result = fit(:dense)
all_result = fit(:all)

sequential = @benchmark fit(:sequential) samples = 5 evals = 1
dense_only = @benchmark fit(:dense) samples = 5 evals = 1
including_scalars = @benchmark fit(:all) samples = 5 evals = 1

seconds(trial) = median(trial).time / 1e9
equivalent(result) =
    all(
        isapprox(
            mean(result[name]),
            mean(reference[name]);
            rtol = 1e-8,
            atol = 1e-10,
        ) &&
        isapprox(
            cov(result[name]),
            cov(reference[name]);
            rtol = 1e-8,
            atol = 1e-10,
        )
        for name in (:v, :w)
    )

t_sequential = seconds(sequential)
t_dense = seconds(dense_only)
t_all = seconds(including_scalars)

@printf(
    "L=2, %d observations, %d features, %d iterations, %d Julia threads\n\n",
    N,
    D,
    ITERATIONS,
    Threads.nthreads(),
)
@printf("%-27s %9s %9s %12s\n", "strategy", "median", "speedup", "same result")
@printf("%-27s %8.3f s %8.2fx %12s\n", "sequential", t_sequential, 1.0, "—")
@printf(
    "%-27s %8.3f s %8.2fx %12s\n",
    "parallel: v and w",
    t_dense,
    t_sequential / t_dense,
    equivalent(dense_result),
)
@printf(
    "%-27s %8.3f s %8.2fx %12s\n",
    "parallel: all variables",
    t_all,
    t_sequential / t_all,
    equivalent(all_result),
)
