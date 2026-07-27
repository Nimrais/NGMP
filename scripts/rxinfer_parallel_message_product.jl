using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BenchmarkTools
using LinearAlgebra
using Printf
using RxInfer
using StableRNGs
using Statistics

# A focused version of the L=3 workload: v, w[1], and w[2] are the three
# high-degree Gaussian hubs. Conjugate observations isolate their message products
# from the nonlinear optimizer used in the full hierarchy notebook.
BLAS.set_num_threads(1)
const N = 800
const D = 128
const COVARIANCE = Matrix{Float64}(I, D, D)
const FOLD = ReactiveMP.ParallelMessageProduct(
    nchunks = min(Threads.nthreads(), 6),
)

@model function l3_message_hubs(observation, residual)
    local w
    v ~ MvNormalMeanCovariance(zeros(D), COVARIANCE)
    for level in 1:2
        w[level] ~ MvNormalMeanCovariance(zeros(D), COVARIANCE)
    end

    # A high-degree scalar hub checks that offering the strategy to scalar
    # variables is harmless: ParallelMessageProduct should reject it as too small.
    γ ~ GammaShapeRate(2.0, 2.0)

    for i in 1:N
        observation[1, i] ~ MvNormalMeanCovariance(v, COVARIANCE)
        for level in 1:2
            observation[level + 1, i] ~ MvNormalMeanCovariance(
                w[level],
                COVARIANCE,
            )
        end
        residual[i] ~ NormalMeanPrecision(0.0, γ)
    end
end

# Parallelize only the three variables that combine 801 multivariate messages.
dense_hubs = @meta begin
    v -> (meta = nothing, marginal_fold_strategy = FOLD)
    w -> (meta = nothing, marginal_fold_strategy = FOLD)
end

rng = StableRNG(42)
data = (
    observation = reshape(
        [randn(rng, D) for _ in 1:(3 * N)],
        3,
        N,
    ),
    residual = randn(rng, N),
)

function fit(mode)
    result = infer(
        model = l3_message_hubs(),
        data = data,
        meta = mode === :dense ? dense_hubs : nothing,
        options = mode === :all ? (fold_strategy = FOLD,) : nothing,
        returnvars = (v = KeepLast(), w = KeepLast(), γ = KeepLast()),
        free_energy = false,
        showprogress = false,
    )
    return result.posteriors
end

Threads.nthreads() > 1 ||
    error("Run with `julia --threads=6 --project=. $(PROGRAM_FILE)`.")

# Warm all code paths before BenchmarkTools starts sampling.
reference = fit(:sequential)
dense_result = fit(:dense)
all_result = fit(:all)

sequential = @benchmark fit(:sequential) samples = 5 evals = 1
dense_only = @benchmark fit(:dense) samples = 5 evals = 1
including_scalar = @benchmark fit(:all) samples = 5 evals = 1

seconds(trial) = median(trial).time / 1e9
function equivalent(result)
    dense_reference = (reference[:v], vec(reference[:w])...)
    dense_candidate = (result[:v], vec(result[:w])...)
    dense_same = all(
        isapprox(mean(a), mean(b); rtol = 1e-10, atol = 1e-12) &&
        isapprox(cov(a), cov(b); rtol = 1e-10, atol = 1e-12)
        for (a, b) in zip(dense_reference, dense_candidate)
    )
    scalar_same =
        isapprox(
            mean(reference[:γ]),
            mean(result[:γ]);
            rtol = 1e-10,
            atol = 1e-12,
        ) &&
        isapprox(
            var(reference[:γ]),
            var(result[:γ]);
            rtol = 1e-10,
            atol = 1e-12,
        )
    return dense_same && scalar_same
end

t_sequential = seconds(sequential)
t_dense = seconds(dense_only)
t_all = seconds(including_scalar)

@printf(
    "L=3 message hubs, %d observations, %d dimensions, %d Julia threads\n\n",
    N,
    D,
    Threads.nthreads(),
)
@printf("%-28s %9s %9s %12s\n", "strategy", "median", "speedup", "same result")
@printf("%-28s %8.3f s %8.2fx %12s\n", "sequential", t_sequential, 1.0, "—")
@printf(
    "%-28s %8.3f s %8.2fx %12s\n",
    "parallel: v, w[1], w[2]",
    t_dense,
    t_sequential / t_dense,
    equivalent(dense_result),
)
@printf(
    "%-28s %8.3f s %8.2fx %12s\n",
    "parallel: plus scalar γ",
    t_all,
    t_sequential / t_all,
    equivalent(all_result),
)
