# =============================================================================
# Direct frozen-Gaussian backend for the dynamic ensemble with a multi-level
# VAE-dependent precision hierarchy — ETTh2, horizons 96/192/336/720.
# =============================================================================
# Ports the deep-kernel direct backend (scripts/uci_deep_kernel_direct_paper_
# benchmark.jl) to the dynamic β-ensemble: expert predictions stay FIXED (no
# mean solve), and the scalar τ is replaced by L levels of input-dependent
# log-precision, each linear in a shared feature basis:
#
#   score^k_ij ~ N(ψ_j' u^k_i, 1/carrier),  carrier = TOP_CARRIER at k = L,
#                                            exp(m^{k+1}+v^{k+1}/2) below
#   level-1 exp-site on z = score¹ (∝ exp((a−1)z − b eᶻ)):
#     no-β arm: a = 1.5, b = r²/2            (r = y − pred, fixed data)
#     β arm:    a = 2.5, b = E[β] + r²/2     (Gamma(1,β) carrier collapsed
#                                             through the log-Jacobian)
#   level k≥2 site: a = 1.5, b = residual cascade from level k−1 (with the
#     −2·carrier_fraction·conditional_variance cross-term — the τ-trap fix).
#
# Recipe under test (colleague's UCI/XOR configuration): Matérn-3/2 multiscale
# RFF (400) at fixed lengthscale 1.5, precision-prior gain 1.5, anchored
# intercepts, damped NG exp-sites with vector transport (α = 0.6, β = 0.8).
#
# Arms: feature setup ∈ {rff_all, linear_l1_rff_up, linear_all} ×
#       L ∈ {2, 3} × carrier ∈ {beta, nobeta}. Train on the validation split,
#       score on the full test split (repo convention).
#
# Usage (one process per horizon — long shell loops get truncated here):
#   julia --project=. experiments/etth2_direct_precision_hierarchy.jl smoke 96
#   julia --project=. experiments/etth2_direct_precision_hierarchy.jl full 96
#
# Env: EDH_ITERATIONS (60), EDH_TOLERANCE (1e-5), EDH_N_OBS (0 = full split),
#      EDH_RFF_SEED (12345), EDH_RESULTS_DIR, EDH_FORCE=1 (refit existing arms)
# =============================================================================

using JLD2, Distributions, LinearAlgebra, Printf, Random, Statistics
using SurrogateModelling
import SurrogateModelling.NaturalGradientMP: DampingMeta, NGMPEdgeState

# Load the direct backend in an isolated module (pattern from
# notebooks/xor_deep_kernel.jl): its include chain defines UCIPaperProtocol,
# which must not collide with anything else loaded in this process.
const BACKEND = Module(gensym(:ETTh2DirectBackend), true, true)
Core.eval(BACKEND, :(include(path) = Base.include($BACKEND, path)))
Base.include(BACKEND, joinpath(
    @__DIR__, "..", "scripts", "uci_deep_kernel_direct_paper_benchmark.jl",
))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = get(
    ENV, "EDH_RESULTS_DIR",
    joinpath(ROOT, "results", "etth2_direct_hierarchy"),
)
const ITERATIONS = parse(Int, get(ENV, "EDH_ITERATIONS", "60"))
const TOLERANCE = parse(Float64, get(ENV, "EDH_TOLERANCE", "1e-5"))
const ENV_N_OBS = parse(Int, get(ENV, "EDH_N_OBS", "0"))
const RFF_SEED = parse(Int, get(ENV, "EDH_RFF_SEED", "12345"))
const FORCE = get(ENV, "EDH_FORCE", "0") == "1"

const PRIOR_GAIN = 1.5
const BASE_WEIGHT_SD = 0.4
const N_RFF = 400
const FIXED_LENGTHSCALE = 1.5
const KAPPA = 1.0
const BETA_SHAPE0 = 1.0        # sessions/dynamic/vae/dynamic_ETTh2_*.yaml
const BETA_RATE0 = 1e3
const TOP_CARRIER = BACKEND.TOP_CARRIER
const JITTER = BACKEND.DIRECT_JITTER
const SETUPS = (:rff_all, :linear_l1_rff_up, :linear_all)

# ETTh2 Dynamic reference NLL per horizon (results/etth_ct_table/table.txt).
const REFERENCE_NLL = Dict(
    96 => 0.93418, 192 => 0.92366, 336 => 0.96119, 720 => 0.86993,
)

damping_meta() = DampingMeta(
    alpha = 0.6, beta = 0.8, max_step = 0.5, method = :vector_transport,
)

# --- helpers copied from dynamic_cavi_compare.jl (validated there) -----------

# moment-matched LogNormal(m,v) → Gamma(a,b): E[γ]=exp(m+v/2), Var/E² = eᵛ−1
function zforward_gamma(m, v)
    em = exp(clamp(v, 1e-6, 12.0)) - 1
    a = 1 / em
    b = exp(-m - v / 2) / em
    return a, b
end

# metrics with 95% CIs: ±1.96·SE over test points (delta method for rmse,
# binomial for cov95); ll is a positive-sense mean log-density, nll = −ll
# matches the etth_ct_table sign convention.
function score_metrics(μ, σ, y; quantiles = (0.1, 0.9))
    n = length(y)
    ll_terms = [logpdf(Normal(μ[j], σ[j]), y[j]) for j in 1:n]
    zc = 1.959963984540054
    ae = abs.(μ .- y); sq = (μ .- y) .^ 2
    covi = (y .>= μ .- zc .* σ) .& (y .<= μ .+ zc .* σ)
    pint = zeros(n)
    for q in quantiles
        zq = quantile(Normal(), q); qhat = μ .+ zq .* σ
        pint .+= max.(q .* (y .- qhat), (q - 1) .* (y .- qhat)) ./
            length(quantiles)
    end
    mae = mean(ae); rmse = sqrt(mean(sq)); ll = mean(ll_terms); p = mean(covi)
    return (; mae, mae_ci = zc * std(ae) / sqrt(n),
            rmse, rmse_ci = zc * std(sq) / sqrt(n) / (2 * rmse),
            ll, ll_ci = zc * std(ll_terms) / sqrt(n),
            nll = -ll, nll_ci = zc * std(ll_terms) / sqrt(n),
            cov95 = p, cov_ci = zc * sqrt(p * (1 - p) / n),
            pinball = mean(pint), pin_ci = zc * std(pint) / sqrt(n))
end

# precision-weighted Gaussian fusion across forecasters (dyn_predict in
# dynamic_cavi_compare.jl): V is nf×n per-forecaster predictive variance.
function fuse_predictions(V, predictions)
    P = clamp.(1 ./ V, 1e-6, 1e6)
    nf, no = size(P)
    μ = Vector{Float64}(undef, no); σ = Vector{Float64}(undef, no)
    for j in 1:no
        τc = sum(@view P[:, j])
        μ[j] = sum(P[i, j] * predictions[i, j] for i in 1:nf) / τc
        σ[j] = sqrt(1 / τc)
    end
    return μ, σ
end

# --- feature designs ---------------------------------------------------------

# The cache stores features as vcat(1.0, 64-d VAE latent): intercept FIRST for
# the raw/linear design; the RFF design appends its intercept LAST.
struct LevelDesign
    train::Matrix{Float64}
    test::Matrix{Float64}
    intercept::Int
end

function check_intercept(design::LevelDesign)
    for Φ in (design.train, design.test)
        all(x -> isapprox(x, 1.0; atol = 1e-9), @view Φ[:, design.intercept]) ||
            error("intercept column $(design.intercept) is not all ones")
    end
    return design
end

function base_designs(features_val, features_test, n_obs, seed)
    F_val = permutedims(reduce(hcat, features_val))
    F_test = permutedims(reduce(hcat, features_test))
    linear = check_intercept(LevelDesign(F_val[1:n_obs, :], F_test, 1))

    X_train = F_val[1:n_obs, 2:end]
    X_test = F_test[:, 2:end]
    centre = mean(X_train, dims = 1)
    spread = max.(std(X_train, dims = 1), 1e-9)
    Φr_train, Φr_test = BACKEND.direct_rff_design(
        (X_train .- centre) ./ spread,
        (X_test .- centre) ./ spread,
        seed,
        :multiscale_matern32_linear,
        N_RFF,
        FIXED_LENGTHSCALE,
    )
    rff = check_intercept(LevelDesign(Φr_train, Φr_test, size(Φr_train, 2)))
    return (; linear, rff)
end

function level_designs(setup, L, bases)
    setup == :linear_all && return fill(bases.linear, L)
    setup == :rff_all && return fill(bases.rff, L)
    setup == :linear_l1_rff_up &&
        return [k == 1 ? bases.linear : bases.rff for k in 1:L]
    error("unknown feature setup: $setup")
end

# --- priors ------------------------------------------------------------------

# direct_prior_parameters generalized to arbitrary intercept position, with
# the prior gain applied as in notebooks/xor_deep_kernel.jl (non-intercept
# weight sd 0.4 → 0.4·gain; intercept precision stays 1.0).
function level_prior(p, intercept_index, anchor; gain = PRIOR_GAIN)
    prior_mean = zeros(p)
    prior_mean[intercept_index] = anchor
    diagonal = fill(inv(BASE_WEIGHT_SD^2) / gain^2, p)
    diagonal[intercept_index] = 1.0
    return (mean = prior_mean, precision = Matrix(Diagonal(diagonal)))
end

function hierarchy_priors(designs, residuals)
    anchor1 = -log(max(mean(abs2, residuals), 1e-8))
    return [
        level_prior(
            size(designs[k].train, 2),
            designs[k].intercept,
            k == 1 ? anchor1 : log(TOP_CARRIER),
        )
        for k in eachindex(designs)
    ]
end

# --- core fit (fit_direct_deep minus the mean block) -------------------------

function fit_hierarchy_no_mean(
    Φlevels,
    r2half,
    level1_shape,
    priors;
    beta_prior = nothing,
    iterations = ITERATIONS,
    tolerance = TOLERANCE,
)
    L = length(Φlevels)
    n = length(r2half)
    level_weights = [copy(prior.mean) for prior in priors]
    level_covariances = [inv(prior.precision) for prior in priors]
    score_means = [Φlevels[k] * level_weights[k] for k in 1:L]
    score_variances = [
        BACKEND.row_quadratic_forms(Φlevels[k], level_covariances[k]) .+
            inv(TOP_CARRIER)
        for k in 1:L
    ]
    expected_precisions = [
        exp.(score_means[k] .+ score_variances[k] ./ 2) for k in 1:L
    ]
    states = [NGMPEdgeState(damping_meta()) for _ in 1:L, _ in 1:n]
    expected_beta = isnothing(beta_prior) ? 0.0 :
        beta_prior.shape / beta_prior.rate
    expected_gamma = zeros(n)

    previous_vector = reduce(vcat, level_weights)
    iterations_run = 0
    final_delta = NaN
    for iteration in 1:iterations
        iterations_run = iteration
        gamma_rate = isnothing(beta_prior) ? copy(r2half) :
            r2half .+ expected_beta

        for level in 1:L
            carrier_precision = level == L ? fill(TOP_CARRIER, n) :
                max.(expected_precisions[level + 1], JITTER)
            site_xi, site_precision = BACKEND.damp_exp_sites!(
                states,
                level,
                level == 1 ? level1_shape : 1.5,
                max.(gamma_rate, JITTER),
                score_means[level],
                score_variances[level],
            )
            positive_site_precision = max.(site_precision, JITTER)
            posterior_score_precision =
                carrier_precision + positive_site_precision
            effective_precision =
                carrier_precision .* positive_site_precision ./
                posterior_score_precision
            site_target = site_xi ./ positive_site_precision
            level_weights[level], level_covariances[level] =
                BACKEND.direct_gaussian_update(
                    Φlevels[level],
                    site_target,
                    effective_precision,
                    priors[level].mean,
                    priors[level].precision,
                )

            updated_conditional_mean = Φlevels[level] * level_weights[level]
            updated_conditional_variance = BACKEND.row_quadratic_forms(
                Φlevels[level], level_covariances[level],
            )
            carrier_fraction = carrier_precision ./ posterior_score_precision
            posterior_score_mean =
                (
                    carrier_precision .* updated_conditional_mean + site_xi
                ) ./ posterior_score_precision
            posterior_score_variance =
                inv.(posterior_score_precision) +
                carrier_fraction .^ 2 .* updated_conditional_variance

            score_means[level] = posterior_score_mean
            score_variances[level] = posterior_score_variance
            expected_precisions[level] = exp.(
                posterior_score_mean + posterior_score_variance / 2,
            )

            # residual cascade with the cross-term: dropping the
            # −2·carrier_fraction·conditional_variance piece makes the level
            # above inert (τ-trap).
            residual_variance =
                posterior_score_variance +
                updated_conditional_variance -
                2 .* carrier_fraction .* updated_conditional_variance
            gamma_rate =
                (
                    abs2.(posterior_score_mean - updated_conditional_mean) +
                    max.(residual_variance, JITTER)
                ) ./ 2
        end

        if !isnothing(beta_prior)
            for j in 1:n
                a_g, b_g = zforward_gamma(
                    clamp(score_means[1][j], -30.0, 30.0),
                    clamp(score_variances[1][j], 1e-6, 8.0),
                )
                expected_gamma[j] =
                    (a_g + 0.5) / (b_g + expected_beta + r2half[j])
            end
            expected_beta = (beta_prior.shape + n) /
                (beta_prior.rate + sum(expected_gamma))
        end

        all(weights -> all(isfinite, weights), level_weights) &&
            all(matrix -> all(isfinite, matrix), level_covariances) ||
            error("non-finite posterior at iteration $iteration")
        all(
            level -> all(
                value -> isfinite(value) && value > 0,
                expected_precisions[level],
            ),
            1:L,
        ) || error("non-finite precision at iteration $iteration")

        current_vector = reduce(vcat, level_weights)
        final_delta = norm(current_vector - previous_vector) /
            max(1.0, norm(previous_vector))
        final_delta < tolerance && break
        previous_vector = current_vector
    end

    return (;
        level_weights,
        level_covariances,
        expected_beta,
        iterations_run,
        final_delta,
        level2_score_std = std(score_means[min(2, L)]),
    )
end

# --- prediction --------------------------------------------------------------

# Hierarchy-aware predictive variance: recurse the carrier from the top level
# down — v^L includes 1/TOP_CARRIER, and level k−1 receives E[1/λ_k] =
# exp(−m_k + v_k/2) as its carrier variance (the input-dependent analogue of
# the CT arm's `+ 1/E[τ]` term). After the level-1 pass the accumulator IS
# E[1/γ]; carrier arms then add the κ·E[β] noise floor.
function forecaster_variance(fit, designs; κ = KAPPA)
    L = length(designs)
    carrier_variance = fill(1 / TOP_CARRIER, size(designs[1].test, 1))
    for level in L:-1:1
        m = designs[level].test * fit.level_weights[level]
        v = BACKEND.row_quadratic_forms(
            designs[level].test, fit.level_covariances[level],
        ) .+ carrier_variance
        carrier_variance = exp.(-m .+ v ./ 2)
    end
    return carrier_variance .+ κ * fit.expected_beta
end

# --- arm runner --------------------------------------------------------------

arm_name(setup, L, carrier) = "$(setup)_L$(L)_$(carrier)"
arm_path(tag, setup, L, carrier) = joinpath(
    RESULTS_DIR, tag, arm_name(setup, L, carrier) * ".jld2",
)

function run_arm(cache, horizon, tag, setup, L, carrier, n_obs, bases)
    path = arm_path(tag, setup, L, carrier)
    if isfile(path) && !FORCE
        @printf("skip (exists): %s\n", arm_name(setup, L, carrier))
        return
    end
    mkpath(dirname(path))

    y = cache["y_val"][1:n_obs]
    predictions = cache["predictions_val"][:, 1:n_obs]
    y_test = cache["y_test"]
    predictions_test = cache["predictions_test"]
    n_forecasters = size(predictions, 1)
    designs = level_designs(setup, L, bases)
    beta_prior = carrier == :beta ?
        (shape = BETA_SHAPE0, rate = BETA_RATE0) : nothing
    level1_shape = carrier == :beta ? 2.5 : 1.5

    fits = Vector{Any}(undef, n_forecasters)
    status = "ok"
    failure = ""
    elapsed = @elapsed try
        for i in 1:n_forecasters
            residuals = y .- predictions[i, :]
            r2half = max.(abs2.(residuals) ./ 2, JITTER)
            fits[i] = fit_hierarchy_no_mean(
                [design.train for design in designs],
                r2half,
                level1_shape,
                hierarchy_priors(designs, residuals);
                beta_prior,
            )
        end
    catch exception
        status = "unstable"
        failure = sprint(showerror, exception)
    end

    result = Dict{String, Any}(
        "horizon" => horizon,
        "setup" => String(setup),
        "L" => L,
        "carrier" => String(carrier),
        "n_obs" => n_obs,
        "rff_seed" => RFF_SEED,
        "status" => status,
        "error" => failure,
        "elapsed" => elapsed,
    )
    if status == "ok"
        V = reduce(
            vcat,
            (forecaster_variance(fits[i], designs)' for i in 1:n_forecasters),
        )
        μ, σ = fuse_predictions(V, predictions_test)
        m = score_metrics(μ, σ, y_test)
        result["metrics"] = Dict(
            String(key) => Float64(value) for (key, value) in pairs(m)
        )
        result["mu"] = μ
        result["sigma"] = σ
        result["expected_beta"] = [fit.expected_beta for fit in fits]
        result["iterations_run"] = [fit.iterations_run for fit in fits]
        result["final_delta"] = [fit.final_delta for fit in fits]
        result["level2_score_std"] = [fit.level2_score_std for fit in fits]
        result["level_weight_means"] = [fit.level_weights for fit in fits]
        result["level_cov_diags"] = [
            [diag(cov) for cov in fit.level_covariances] for fit in fits
        ]
        @printf(
            "%-28s  nll %.4f±%.4f  rmse %.4f  cov95 %.3f  Eβ %.3g  iters %s  Δ %.2g  l2σ %.3g  (%.0fs)\n",
            arm_name(setup, L, carrier),
            m.nll, m.nll_ci, m.rmse, m.cov95,
            mean(result["expected_beta"]),
            maximum(result["iterations_run"]),
            maximum(result["final_delta"]),
            minimum(result["level2_score_std"]),
            elapsed,
        )
    else
        @printf(
            "%-28s  UNSTABLE: %s\n",
            arm_name(setup, L, carrier),
            first(failure, 120),
        )
    end
    jldsave(path; result)
    write_summary(horizon, tag)
    return
end

# --- summary -----------------------------------------------------------------

function write_summary(horizon, tag)
    directory = joinpath(RESULTS_DIR, tag)
    isdir(directory) || return
    rows = []
    for file in sort(filter(endswith(".jld2"), readdir(directory)))
        push!(rows, load(joinpath(directory, file))["result"])
    end
    isempty(rows) && return
    path = joinpath(RESULTS_DIR, "summary_$(tag).md")
    open(path, "w") do io
        println(io, "# ETTh2 h$(horizon) — direct precision hierarchy\n")
        println(io, "Reference (paper Dynamic, etth_ct_table): NLL ",
            get(REFERENCE_NLL, horizon, NaN), " (lower is better)\n")
        println(io, "| setup | L | carrier | status | nll | ll | rmse | mae | cov95 | pinball | mean Eβ | iters | Δ | l2σ | secs |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in rows
            if row["status"] == "ok"
                m = row["metrics"]
                @printf(io,
                    "| %s | %d | %s | ok | %.4f±%.4f | %.4f±%.4f | %.4f±%.4f | %.4f | %.4f±%.4f | %.4f | %.3g | %d | %.2g | %.3g | %.0f |\n",
                    row["setup"], row["L"], row["carrier"],
                    m["nll"], m["nll_ci"], m["ll"], m["ll_ci"],
                    m["rmse"], m["rmse_ci"], m["mae"],
                    m["cov95"], m["cov_ci"], m["pinball"],
                    mean(row["expected_beta"]),
                    maximum(row["iterations_run"]),
                    maximum(row["final_delta"]),
                    minimum(row["level2_score_std"]),
                    row["elapsed"],
                )
            else
                @printf(io, "| %s | %d | %s | unstable | %s ||||||||||||\n",
                    row["setup"], row["L"], row["carrier"],
                    first(row["error"], 80))
            end
        end
    end
    return
end

# --- driver ------------------------------------------------------------------

function main(args)
    length(args) == 2 || error(
        "usage: etth2_direct_precision_hierarchy.jl <smoke|full> <horizon>",
    )
    phase = Symbol(args[1])
    phase in (:smoke, :full) || error("unknown phase: $phase")
    horizon = parse(Int, args[2])
    haskey(REFERENCE_NLL, horizon) || error("unknown horizon: $horizon")

    cache_path = joinpath(ROOT, "cache", "dynamic_etth2_h$(horizon)_cache.jld2")
    isfile(cache_path) || error("missing cache: $cache_path")
    cache = load(cache_path)
    # Full validation split in BOTH phases: the first ~500 val points are a
    # much easier regime than test, and the conjugate β update needs n large
    # enough to wash out the Gamma(1, 1e3) prior rate. EDH_N_OBS still allows
    # quick mechanics-only runs.
    n_val = length(cache["y_val"])
    n_obs = ENV_N_OBS > 0 ? min(ENV_N_OBS, n_val) : n_val
    arms = phase == :smoke ?
        [(setup, 2, carrier) for setup in SETUPS
            for carrier in (:nobeta, :beta)] :
        [(setup, L, carrier) for setup in SETUPS
            for L in (2, 3) for carrier in (:nobeta, :beta)]

    @printf(
        "ETTh2 h%d %s: n_obs=%d, %d arms, iterations=%d, seed=%d\n",
        horizon, phase, n_obs, length(arms), ITERATIONS, RFF_SEED,
    )
    tag = phase == :smoke ? "h$(horizon)_smoke" : "h$(horizon)"
    bases = base_designs(
        cache["features_val"], cache["features_test"], n_obs, RFF_SEED,
    )
    for (setup, L, carrier) in arms
        run_arm(cache, horizon, tag, setup, L, carrier, n_obs, bases)
    end
    @printf(
        "done: summary at %s\n",
        joinpath(RESULTS_DIR, "summary_$(tag).md"),
    )
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
