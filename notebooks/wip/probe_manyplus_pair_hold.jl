# Diagnostic probe: can the za -> ResidualSine -> softdot(v, h) -> ManyPlus
# chain HOLD a solution it is initialized at? One difference pair, pure ridge
# cosine target that the pair represents exactly, priors centered at the true
# solution, observation precision FIXED (no noise race).
#
# Findings (2026-07-22), each verified by toggling one ingredient:
#   1. q(za) = N(0,1) init  -> w snaps to ~0 in ONE sweep: the ResidualSine
#      backward sites are projected at marginals inconsistent with the
#      prior-mean ridge, and the softdot least-squares crushes w before the
#      marginals can catch up. Coupled prior-pushforward init (below) fixes it.
#   2. Diffuse q(v) (var 0.25) shrinks every backward target by E[v]/E[v^2]
#      in the first sweep. v prior var 0.01 fixes it.
#   3. Even then, the ridge FREQUENCY shrinks (1.1 -> 0.46; the fit lands in
#      the exact cos-evenness symmetry family but low-pass filtered). The NGMP
#      fixed point under-rewards high frequencies because sine moments carry
#      exp(-omega^2 v / 2) attenuation. A hard ridge anchor (precision 200,
#      as configured here) recovers the true solution to the noise floor
#      (train MSE ~0.003) with monotonically decreasing free energy;
#      precision 10 does not.
#
# Run: julia --project=. experiments/probe_manyplus_pair_hold.jl

using SurrogateModelling
using RxInfer
using StableRNGs
using LinearAlgebra

const RHO = 0.9
const OMEGA = 1.0
const A = 1.1          # ridge frequency along s = x1 + x2 (za = a*s ± b)
const B = π / 4
const VTRUE = 0.35

phi(x) = x + (RHO / OMEGA) * sin(OMEGA * x)

rng = StableRNG(3)
n = 200
x1 = 4 .* rand(rng, n) .- 2
x2 = 4 .* rand(rng, n) .- 2
s = x1 .+ x2
# Exactly what the pair computes: v [phi(a s + b) - phi(a s - b)]
target = VTRUE .* (phi.(A .* s .+ B) .- phi.(A .* s .- B)) .+ 0.05 .* randn(rng, n)
features = [[1.0, x1[i], x2[i]] for i in 1:n]
println("target range: ", extrema(target))

@model function probe_model(features, y, priors, activation, activation_deps)
    local w, v, za, h, c, out
    τ ~ priors[:τ]
    τ_c ~ priors[:τ_c]
    for k in 1:2
        w[k] ~ priors[:w][k]
        v[k] ~ priors[:v][k]
    end
    for i in eachindex(y)
        for k in 1:2
            za[k, i] ~ softdot(features[i], w[k], τ)
            h[k, i] ~ ResidualSine(za[k, i]) where {
                dependencies = activation_deps,
                meta = activation,
            }
            c[k, i] ~ softdot(v[k], h[k, i], τ_c)
        end
        out[i] ~ ManyPlus(inputs = [c[k, i] for k in 1:2])
        y[i] ~ NormalMeanPrecision(out[i], 100.0)
    end
end

@constraints function probe_constraints()
    q(w, v, za, h, c, out, τ, τ_c) = q(w, za, h, c, out)q(v)q(τ)q(τ_c)
    q(w)::MomentForm()
end

# Coupled initialization: the NGMP projection points q(za), q(h) and the BP
# seeds q(c), q(out) start at the PRIOR PUSHFORWARD, not at a generic N(0, 1) —
# otherwise the first backward sites are projected at marginals inconsistent
# with the prior-mean ridge and the softdot least-squares snaps w to zero.
@initialization function probe_initialization(priors, za_init, h_init, c_init, out_init)
    q(v) = deepcopy(priors[:v])
    q(za) = za_init
    q(h) = h_init
    q(c) = c_init
    q(out) = out_init
    q(τ) = priors[:τ]
    q(τ_c) = priors[:τ_c]
    μ(w) = deepcopy(priors[:w])
end

function pushforward_inits(priors, features)
    n = length(features)
    w_means = [mean(p) for p in priors[:w]]
    v_means = [mean(p) for p in priors[:v]]
    za_init = [NormalMeanVariance(dot(w_means[k], features[i]), 0.5) for k in 1:2, i in 1:n]
    h_init = [NormalMeanVariance(phi(mean(za_init[k, i])), 1.0) for k in 1:2, i in 1:n]
    c_init = [NormalMeanVariance(v_means[k] * mean(h_init[k, i]), 1.0) for k in 1:2, i in 1:n]
    out_init = [NormalMeanVariance(sum(mean(c_init[k, i]) for k in 1:2), 1.0) for i in 1:n]
    return za_init, h_init, c_init, out_init
end

precision_w = Diagonal([2.0, 200.0, 200.0])
w_true_1 = [B, A, A]
w_true_2 = [-B, A, A]
priors = Dict{Symbol, Any}(
    :w => [
        MvNormalWeightedMeanPrecision(precision_w * w_true_1, precision_w),
        MvNormalWeightedMeanPrecision(precision_w * w_true_2, precision_w),
    ],
    :v => [NormalMeanVariance(VTRUE, 0.01), NormalMeanVariance(-VTRUE, 0.01)],
    :τ => GammaShapeRate(1e3, 1.0),
    :τ_c => GammaShapeRate(1e4, 1.0),
)

activation = ResidualSineMeta(rho = RHO, omega = OMEGA)
activation_deps = NGMPDependencies(
    out = nothing,
    in = nothing,
    projection = TangentProjection(type = ClosedForm),
    damping = DampingMeta(alpha = 0.05, beta = 0.0, max_step = 1.0),
)

result = infer(
    model = probe_model(
        priors = priors,
        activation = activation,
        activation_deps = activation_deps,
    ),
    data = (y = target, features = features),
    constraints = probe_constraints(),
    initialization = probe_initialization(priors, pushforward_inits(priors, features)...),
    iterations = 200,
    free_energy = true,
    showprogress = false,
    options = (limit_stack_depth = 100,),
    disable_inference_error_hint = true,
)

for it in (1, 5, 10, 25, 50, 100, 200)
    w1 = mean(result.posteriors[:w][it][1])
    w2 = mean(result.posteriors[:w][it][2])
    v1 = mean(result.posteriors[:v][it][1])
    v2 = mean(result.posteriors[:v][it][2])
    println("iter $it: w1 = ", round.(w1; digits = 3),
        "  w2 = ", round.(w2; digits = 3),
        "  v = (", round(v1; digits = 3), ", ", round(v2; digits = 3), ")",
        "  FE = ", round(result.free_energy[it]; digits = 2))
end
println("true: w1 = ", round.(w_true_1; digits = 3),
    " w2 = ", round.(w_true_2; digits = 3), " v = ±", VTRUE)

# Post-fit predictions on training inputs from posterior means.
w1 = mean(result.posteriors[:w][end][1])
w2 = mean(result.posteriors[:w][end][2])
v1 = mean(result.posteriors[:v][end][1])
v2 = mean(result.posteriors[:v][end][2])
predicted = [
    v1 * phi(dot(w1, f)) + v2 * phi(dot(w2, f)) for f in features
]
println("train MSE from posterior means: ",
    round(sum(abs2, predicted .- target) / n; digits = 4),
    " | constant MSE: ",
    round(sum(abs2, mean(target) .- target) / n; digits = 4))
