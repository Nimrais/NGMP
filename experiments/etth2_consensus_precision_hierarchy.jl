# =============================================================================
# Consensus-x precision hierarchy for the dynamic ensemble — ETTh2, direct
# backend, horizons 96/192/336/720.
# =============================================================================
# Follow-up to experiments/etth2_direct_precision_hierarchy.jl (findings in
# paper_materials/etth2_direct_precision_hierarchy_findings.md), which showed
# the per-(i,j) precision updates share nothing across forecasters while the
# expert errors are strongly correlated — the independence fusion is then
# structurally overconfident, and the no-β likelihood is ill-posed.
#
# Fix: a latent consensus x_j between predictions and observation:
#
#   pred_ij ~ N(x_j, 1/γ_ij)    i = 1..7   (experts = noisy measurements of x)
#   y_j     ~ N(x_j, 1/τ_y)                (shared observation link)
#   γ_ij    = exp(z¹ᵢⱼ)         softdot hierarchy per forecaster, unchanged
#
# Consequences: all γ_ij couple through q(x_j); the common error component
# lives in 1/τ_y and passes through the fusion UNDIVIDED; the level-1 site
# rate becomes E[(x−pred)²]/2 ≥ v_x/2 > 0, curing the no-β ill-posedness
# (this restores the role the learned mean's quadform played in the UCI
# deep-kernel model). τ_y is either a conjugate Gamma scalar or an
# input-dependent function exp(ψᵀw_s) fit with the same exp-site machinery
# (the shared-difficulty channel the β floor cannot eat).
#
# Arms (lean grid): {linear_all, rff_all} × L=2 × {beta, nobeta} ×
# τ_y ∈ {scalar, func} = 8, plus one L=3 probe (rff_all, beta, func).
# Train on the full validation split, score on the full test split.
#
# Usage (one process per horizon, run horizons SEQUENTIALLY — concurrent
# processes previously caused ~100× per-arm slowdowns):
#   julia --project=. experiments/etth2_consensus_precision_hierarchy.jl smoke 96
#   julia --project=. experiments/etth2_consensus_precision_hierarchy.jl full 96
#
# Env: EDH_ITERATIONS (60), EDH_TOLERANCE (1e-5), EDH_N_OBS (0 = full split),
#      EDH_RFF_SEED (12345), EDH_RESULTS_DIR, EDH_FORCE=1 (refit existing arms)
# =============================================================================

using JLD2, Distributions, LinearAlgebra, Printf, Random, Statistics
using SurrogateModelling
import SurrogateModelling.NaturalGradientMP: DampingMeta, NGMPEdgeState,
    apply_damping!
import SurrogateModelling: TangentProjection, project, StudentTMessage
import SurrogateModelling.UnscentedTransforms: UnscentedTransform
import ClosedFormExpectations: Logpdf
import ExponentialFamily: getnaturalparameters, NormalMeanVariance
import BayesBase

const BACKEND = Module(gensym(:ETTh2ConsensusBackend), true, true)
Core.eval(BACKEND, :(include(path) = Base.include($BACKEND, path)))
Base.include(BACKEND, joinpath(
    @__DIR__, "..", "scripts", "uci_deep_kernel_direct_paper_benchmark.jl",
))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = get(
    ENV, "EDH_RESULTS_DIR",
    joinpath(ROOT, "results", "etth2_consensus_hierarchy"),
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
# Shape 2 so E[1/τ_y] = rate/(shape−1) exists (his depth-1 noise prior).
const TAU_Y_SHAPE0 = 2.0
const TAU_Y_RATE0 = 2.0 / TOP_CARRIER
const PRECISION_CLAMP = 1e8    # plug-in weights entering the x update

# Paper Dynamic reference NLL (results/etth_ct_table/table.txt).
# EDH_XMSG=student replaces the plug-in E[γ] weighting in the x update with
# the tangent-projected (Unscented) Student-t BP message toward x: the factor
# N(pred; x, 1/γ) marginalized over the moment-matched Gamma q(γ) is
# ∝ (2b̃ + (x−pred)²)^−(ã+½), projected at the current q(x_j) and damped in
# η-space. Heavier tails ⇒ self-limiting weights (no x-pinning). VMP scheme
# otherwise unchanged (no cavity/EP).
const XMSG = Symbol(get(ENV, "EDH_XMSG", "plugin"))
const T_PROJECTION = TangentProjection{UnscentedTransform}()

# EDH_TOP=learned replaces the fixed TOP_CARRIER=25 cap on the top hierarchy
# level with a learned per-forecaster Gamma carrier τ2 ~ Γ(1, 1e-3) (the CT
# arm's slack precision), updated conjugately from the top level's own
# residual cascade. Prediction then uses E[1/τ2] = rate/(shape−1) as the top
# carrier variance instead of 1/25.
const TOPC = Symbol(get(ENV, "EDH_TOP", "fixed"))
const TAU2_SHAPE0 = 1.0
const TAU2_RATE0 = 1e-3

# EDH_CALIB=1 inserts a calibration link y ~ N(a·x + c, 1/τ_y): systematic
# consensus bias lands in the transferable (a, c) instead of being laundered
# into τ_y's variance (the misspecification that made early stopping look
# good). Joint 2-d conjugate Gaussian for θ = (a, c), prior N([1,0], 0.5²·I).
const CALIB = get(ENV, "EDH_CALIB", "0") == "1"
const CALIB_PRIOR_MEAN = [1.0, 0.0]
const CALIB_PRIOR_PRECISION = Matrix(Diagonal([4.0, 4.0]))   # sd 0.5 each

const REFERENCE_NLL = Dict(
    ("ETTh2", 96) => 0.93418, ("ETTh2", 192) => 0.92366,
    ("ETTh2", 336) => 0.96119, ("ETTh2", 720) => 0.86993,
    ("ETTh1", 96) => 0.41203, ("ETTh1", 192) => 0.37006,
    ("ETTh1", 336) => 0.31413, ("ETTh1", 720) => 0.37633,
)
# Uncoupled port (etth2_direct_hierarchy), best β arm per horizon (ETTh2 only).
const UNCOUPLED_NLL = Dict(
    ("ETTh2", 96) => 1.589, ("ETTh2", 192) => 1.169,
    ("ETTh2", 336) => 1.098, ("ETTh2", 720) => 1.365,
)

function cache_candidates(dataset, horizon)
    stem = "dynamic_$(lowercase(dataset))_h$(horizon)_cache.jld2"
    return [
        joinpath(ROOT, "cache", stem),
        joinpath(ROOT, "notebooks", "vmp_vs_ngmp", stem),
    ]
end

# Sweepable hyperparameters (defaults = the XOR deep-kernel recipe + paper
# priors). The `sweep` phase mutates this Ref per combination.
const HYPER = Ref((
    alpha = 0.6, momentum = 0.8, method = :vector_transport,
    gain = PRIOR_GAIN, beta_rate0 = BETA_RATE0, anchor_var = 1.0,
))

hyper_suffix() = begin
    h = HYPER[]
    "_a$(h.alpha)m$(h.momentum)$(h.method == :vector_transport ? "vt" : "d")g$(h.gain)b$(h.beta_rate0)av$(h.anchor_var)"
end

damping_meta() = DampingMeta(
    alpha = HYPER[].alpha, beta = HYPER[].momentum, max_step = 0.5,
    method = HYPER[].method,
)

# --- helpers copied from dynamic_cavi_compare.jl (validated there) -----------

function zforward_gamma(m, v)
    em = exp(clamp(v, 1e-6, 12.0)) - 1
    a = 1 / em
    b = exp(-m - v / 2) / em
    return a, b
end

function score_metrics(μ, σ, y; quantiles = (0.1, 0.9))
    n = length(y)
    length(μ) == n == length(σ) || error("prediction/target length mismatch")
    all(isfinite, μ) || error("non-finite predictive mean")
    all(value -> isfinite(value) && value > 0, σ) ||
        error("improper predictive standard deviation")
    ll_terms = [logpdf(Normal(μ[j], σ[j]), y[j]) for j in 1:n]
    zc = 1.959963984540054
    ae = abs.(μ .- y); sq = (μ .- y) .^ 2
    covi = (y .>= μ .- zc .* σ) .& (y .<= μ .+ zc .* σ)
    widths = 2zc .* σ
    pint = zeros(n)
    for q in quantiles
        zq = quantile(Normal(), q); qhat = μ .+ zq .* σ
        pint .+= max.(q .* (y .- qhat), (q - 1) .* (y .- qhat)) ./
            length(quantiles)
    end
    mae = mean(ae); mse = mean(sq); rmse = sqrt(mse)
    ll = mean(ll_terms); p = mean(covi)
    return (; mae, mae_ci = zc * std(ae) / sqrt(n),
            mse, mse_ci = zc * std(sq) / sqrt(n),
            rmse, rmse_ci = zc * std(sq) / sqrt(n) / (2 * rmse),
            ll, ll_ci = zc * std(ll_terms) / sqrt(n),
            nll = -ll, nll_ci = zc * std(ll_terms) / sqrt(n),
            cov95 = p, cov_ci = zc * sqrt(p * (1 - p) / n),
            interval_width = mean(widths),
            interval_width_ci = zc * std(widths) / sqrt(n),
            pinball = mean(pint), pin_ci = zc * std(pint) / sqrt(n))
end

# --- feature designs (as in etth2_direct_precision_hierarchy.jl) -------------

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

function base_designs(features_val, features_test, n_obs, seed; need_rff)
    F_val = permutedims(reduce(hcat, features_val))
    F_test = permutedims(reduce(hcat, features_test))
    linear = check_intercept(LevelDesign(F_val[1:n_obs, :], F_test, 1))
    need_rff || return (; linear, rff = linear)

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
    error("unknown feature setup: $setup")
end

# --- priors ------------------------------------------------------------------

function level_prior(p, intercept_index, anchor; gain = HYPER[].gain)
    prior_mean = zeros(p)
    prior_mean[intercept_index] = anchor
    diagonal = fill(inv(BASE_WEIGHT_SD^2) / gain^2, p)
    diagonal[intercept_index] = inv(HYPER[].anchor_var)
    return (mean = prior_mean, precision = Matrix(Diagonal(diagonal)))
end

function hierarchy_priors(designs, anchor1)
    return [
        level_prior(
            size(designs[k].train, 2),
            designs[k].intercept,
            k == 1 ? anchor1 : log(TOP_CARRIER),
        )
        for k in eachindex(designs)
    ]
end

# --- per-forecaster hierarchy state (one sweep at a time) --------------------
# The τ_y function arm reuses this with L = 1 and no β: the shared-difficulty
# level s(x) is just one more "forecaster" whose pseudo-data is (y − x).

mutable struct HierarchyState
    Φlevels::Vector{Matrix{Float64}}
    priors::Vector{NamedTuple{(:mean, :precision), Tuple{Vector{Float64}, Matrix{Float64}}}}
    level_weights::Vector{Vector{Float64}}
    level_covariances::Vector{Matrix{Float64}}
    score_means::Vector{Vector{Float64}}
    score_variances::Vector{Vector{Float64}}
    expected_precisions::Vector{Vector{Float64}}
    states::Matrix{NGMPEdgeState}
    level1_shape::Float64
    beta_prior::Union{Nothing, NamedTuple{(:shape, :rate), Tuple{Float64, Float64}}}
    expected_beta::Float64
    expected_gamma::Vector{Float64}
    tau2_shape::Float64
    tau2_rate::Float64
end

function HierarchyState(Φlevels, priors, level1_shape, beta_prior, n)
    L = length(Φlevels)
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
    return HierarchyState(
        Φlevels, priors, level_weights, level_covariances,
        score_means, score_variances, expected_precisions, states,
        level1_shape, beta_prior, expected_beta, zeros(n),
        TAU2_SHAPE0, TAU2_RATE0,
    )
end

top_carrier_precision(state::HierarchyState) = TOPC == :learned ?
    max(state.tau2_shape / state.tau2_rate, JITTER) : TOP_CARRIER
top_carrier_variance(state::HierarchyState) = TOPC == :learned ?
    state.tau2_rate / max(state.tau2_shape - 1, JITTER) : inv(TOP_CARRIER)

# One outer-iteration sweep given this iteration's level-1 rate r2half
# (consensus-based, recomputed by the caller every iteration). Body is the
# fit_direct_deep level loop minus the mean block, cross-term cascade intact.
function sweep!(state::HierarchyState, r2half)
    L = length(state.Φlevels)
    n = length(r2half)
    gamma_rate = isnothing(state.beta_prior) ? copy(r2half) :
        r2half .+ state.expected_beta

    for level in 1:L
        carrier_precision = level == L ?
            fill(top_carrier_precision(state), n) :
            max.(state.expected_precisions[level + 1], JITTER)
        site_xi, site_precision = BACKEND.damp_exp_sites!(
            state.states,
            level,
            level == 1 ? state.level1_shape : 1.5,
            max.(gamma_rate, JITTER),
            state.score_means[level],
            state.score_variances[level],
        )
        positive_site_precision = max.(site_precision, JITTER)
        posterior_score_precision = carrier_precision + positive_site_precision
        effective_precision =
            carrier_precision .* positive_site_precision ./
            posterior_score_precision
        site_target = site_xi ./ positive_site_precision
        state.level_weights[level], state.level_covariances[level] =
            BACKEND.direct_gaussian_update(
                state.Φlevels[level],
                site_target,
                effective_precision,
                state.priors[level].mean,
                state.priors[level].precision,
            )

        updated_conditional_mean =
            state.Φlevels[level] * state.level_weights[level]
        updated_conditional_variance = BACKEND.row_quadratic_forms(
            state.Φlevels[level], state.level_covariances[level],
        )
        carrier_fraction = carrier_precision ./ posterior_score_precision
        posterior_score_mean =
            (
                carrier_precision .* updated_conditional_mean + site_xi
            ) ./ posterior_score_precision
        posterior_score_variance =
            inv.(posterior_score_precision) +
            carrier_fraction .^ 2 .* updated_conditional_variance

        state.score_means[level] = posterior_score_mean
        state.score_variances[level] = posterior_score_variance
        state.expected_precisions[level] = exp.(
            posterior_score_mean + posterior_score_variance / 2,
        )

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

    # Learned top carrier: conjugate Gamma update from the top level's own
    # residual cascade (gamma_rate holds it after the level-L pass) — the CT
    # arm's per-forecaster slack precision, fit_direct_depth_one pattern.
    if TOPC == :learned
        state.tau2_shape = TAU2_SHAPE0 + n / 2
        state.tau2_rate = TAU2_RATE0 + sum(gamma_rate)
    end

    if !isnothing(state.beta_prior)
        for j in 1:n
            a_g, b_g = zforward_gamma(
                clamp(state.score_means[1][j], -30.0, 30.0),
                clamp(state.score_variances[1][j], 1e-6, 8.0),
            )
            state.expected_gamma[j] =
                (a_g + 0.5) / (b_g + state.expected_beta + r2half[j])
        end
        state.expected_beta = (state.beta_prior.shape + n) /
            (state.beta_prior.rate + sum(state.expected_gamma))
    end

    all(weights -> all(isfinite, weights), state.level_weights) &&
        all(matrix -> all(isfinite, matrix), state.level_covariances) ||
        error("non-finite posterior in hierarchy sweep")
    all(
        level -> all(
            value -> isfinite(value) && value > 0,
            state.expected_precisions[level],
        ),
        1:L,
    ) || error("non-finite precision in hierarchy sweep")
    return state
end

flat_weights(state::HierarchyState) = reduce(vcat, state.level_weights)

# Plug-in E[γ] on the TRAINING rows (Jensen-lower variance; his convention).
training_precision(state::HierarchyState) = clamp.(
    state.expected_precisions[1], JITTER, PRECISION_CLAMP,
)

# Hierarchy-recursed honest predictive variance on TEST rows:
# E[1/γ] with carrier recursion (1/TOP_CARRIER at the top) + κ·E[β].
function forecaster_variance(state::HierarchyState, designs; κ = KAPPA)
    L = length(designs)
    carrier_variance = fill(
        top_carrier_variance(state), size(designs[1].test, 1),
    )
    for level in L:-1:1
        m = designs[level].test * state.level_weights[level]
        v = BACKEND.row_quadratic_forms(
            designs[level].test, state.level_covariances[level],
        ) .+ carrier_variance
        carrier_variance = exp.(-m .+ v ./ 2)
    end
    return carrier_variance .+ κ * state.expected_beta
end

# --- arm runner --------------------------------------------------------------

arm_name(setup, L, carrier, tauy) = "$(setup)_L$(L)_$(carrier)_$(tauy)"

function run_arm(cache, cell, tag, setup, L, carrier, tauy, n_obs, bases;
                 suffix = "")
    dataset, horizon = cell
    name = arm_name(setup, L, carrier, tauy) * suffix *
        (XMSG == :student ? "_tx" : "") *
        (TOPC == :learned ? "_lt" : "") *
        (CALIB ? "_cal" : "")
    path = joinpath(RESULTS_DIR, tag, name * ".jld2")
    if isfile(path) && !FORCE
        @printf("skip (exists): %s\n", name)
        return
    end
    mkpath(dirname(path))

    y = cache["y_val"][1:n_obs]
    predictions = cache["predictions_val"][:, 1:n_obs]
    y_test = cache["y_test"]
    predictions_test = cache["predictions_test"]
    n_forecasters = size(predictions, 1)
    n = length(y)
    designs = level_designs(setup, L, bases)
    s_design = setup == :linear_all ? bases.linear : bases.rff

    result = Dict{String, Any}(
        "dataset" => dataset, "horizon" => horizon,
        "setup" => String(setup), "L" => L,
        "carrier" => String(carrier), "tauy" => String(tauy),
        "n_obs" => n_obs, "rff_seed" => RFF_SEED,
        "x_message" => String(XMSG),
        "alpha" => HYPER[].alpha, "momentum" => HYPER[].momentum,
        "damping_method" => String(HYPER[].method),
        "gain" => HYPER[].gain, "beta_rate0" => HYPER[].beta_rate0,
        "anchor_var" => HYPER[].anchor_var,
        "n_rff" => N_RFF, "rff_lengthscale" => FIXED_LENGTHSCALE,
        "status" => "ok", "error" => "",
    )

    elapsed = @elapsed try
        # Anchors from residuals to the INITIAL consensus (equal-weight mean),
        # not to y — anchoring on y would bias toward the old uncoupled fit.
        m_x = vec(mean(predictions, dims = 1))
        v_x = fill(1 / TOP_CARRIER, n)

        beta_prior = carrier == :beta ?
            (shape = BETA_SHAPE0, rate = HYPER[].beta_rate0) : nothing
        forecasters = map(1:n_forecasters) do i
            anchor = -log(max(
                mean(abs2, m_x .- predictions[i, :]), 1e-8,
            ))
            HierarchyState(
                [design.train for design in designs],
                hierarchy_priors(designs, anchor),
                carrier == :beta ? 2.5 : 1.5,
                beta_prior,
                n,
            )
        end

        # τ_y: conjugate Gamma scalar, or a shared single-level exp function.
        tau_shape, tau_rate = TAU_Y_SHAPE0, TAU_Y_RATE0
        s_state = nothing
        if tauy == :func
            anchor_s = -log(max(mean(abs2, y .- m_x), 1e-8))
            s_state = HierarchyState(
                [s_design.train],
                [level_prior(
                    size(s_design.train, 2), s_design.intercept, anchor_s,
                )],
                1.5,
                nothing,
                n,
            )
        end
        tau_bar = tauy == :func ? training_precision(s_state) :
            fill(tau_shape / tau_rate, n)
        x_states = XMSG == :student ?
            [NGMPEdgeState(damping_meta())
                for _ in 1:n_forecasters, _ in 1:n] : nothing
        # calibration link state: θ = (a, c), q(θ) = N(θ̄, Cθ)
        θ̄ = copy(CALIB_PRIOR_MEAN)
        Cθ = inv(CALIB_PRIOR_PRECISION)

        iterations_run = 0
        final_delta = NaN
        min_v_x = Inf
        previous_vector = vcat(
            reduce(vcat, flat_weights.(forecasters)),
            isnothing(s_state) ? Float64[] : flat_weights(s_state),
        )
        for iteration in 1:ITERATIONS
            iterations_run = iteration

            # 1. consensus update (flat prior on x). With the calibration
            # link the y-factor message toward x has precision E[a²]·τ̄ and
            # weighted mean τ̄·(ā(y−c̄) − Cov(a,c)).
            Ea2 = abs2(θ̄[1]) + Cθ[1, 1]
            ylink_precision(j) = CALIB ? Ea2 * tau_bar[j] : tau_bar[j]
            ylink_xi(j) = CALIB ?
                tau_bar[j] * (θ̄[1] * (y[j] - θ̄[2]) - Cθ[1, 2]) :
                tau_bar[j] * y[j]
            if XMSG == :student
                for j in 1:n
                    Λ = ylink_precision(j)
                    ξ = ylink_xi(j)
                    for i in 1:n_forecasters
                        fs = forecasters[i]
                        a_g, b_g = zforward_gamma(
                            clamp(fs.score_means[1][j], -30.0, 30.0),
                            clamp(fs.score_variances[1][j], 1e-6, 8.0),
                        )
                        site = project(
                            T_PROJECTION,
                            NormalMeanVariance(m_x[j], v_x[j]),
                            Logpdf(StudentTMessage(
                                predictions[i, j], a_g, b_g,
                            )),
                        )
                        η = getnaturalparameters(site)
                        ξt, Λt = η[1], -2 * η[2]
                        if !(isfinite(ξt) && isfinite(Λt) && Λt > 0)
                            ξt, Λt = 0.0, 0.0     # improper tail → flat site
                        end
                        message = apply_damping!(x_states[i, j], ξt, Λt)
                        Λc = BayesBase.precision(message)
                        if isfinite(Λc) && Λc > 0
                            Λ += Λc
                            ξ += BayesBase.weightedmean(message)
                        end
                    end
                    m_x[j] = ξ / Λ
                    v_x[j] = 1 / Λ
                end
            else
                weights = [training_precision(fs) for fs in forecasters]
                for j in 1:n
                    Λ = ylink_precision(j)
                    ξ = ylink_xi(j)
                    for i in 1:n_forecasters
                        γ̄ = weights[i][j]
                        Λ += γ̄
                        ξ += γ̄ * predictions[i, j]
                    end
                    m_x[j] = ξ / Λ
                    v_x[j] = 1 / Λ
                end
            end
            min_v_x = min(min_v_x, minimum(v_x))

            # 2. per-forecaster hierarchy sweeps against the consensus.
            for i in 1:n_forecasters
                r2half = max.(
                    (abs2.(m_x .- predictions[i, :]) .+ v_x) ./ 2, JITTER,
                )
                sweep!(forecasters[i], r2half)
            end

            # 2b. calibration update: conjugate 2-d Gaussian regression of y
            # on z = [x, 1] with E[zzᵀ] using the x moments.
            if CALIB
                Λθ = copy(CALIB_PRIOR_PRECISION)
                ξθ = CALIB_PRIOR_PRECISION * CALIB_PRIOR_MEAN
                for j in 1:n
                    τj = tau_bar[j]
                    Λθ[1, 1] += τj * (abs2(m_x[j]) + v_x[j])
                    Λθ[1, 2] += τj * m_x[j]
                    Λθ[2, 1] += τj * m_x[j]
                    Λθ[2, 2] += τj
                    ξθ[1] += τj * y[j] * m_x[j]
                    ξθ[2] += τj * y[j]
                end
                Cθ = inv(Symmetric(Λθ))
                θ̄ = Cθ * ξθ
            end

            # 3. τ_y update from the observation link (calibrated scatter:
            # E[(y − aᵀx − c)²] with θ- and x-uncertainty).
            y_scatter_half = if CALIB
                map(1:n) do j
                    residual = y[j] - θ̄[1] * m_x[j] - θ̄[2]
                    z = [m_x[j], 1.0]
                    (abs2(residual) + dot(z, Cθ, z) +
                        (abs2(θ̄[1]) + Cθ[1, 1]) * v_x[j]) / 2
                end
            else
                (abs2.(y .- m_x) .+ v_x) ./ 2
            end
            if tauy == :func
                sweep!(s_state, max.(y_scatter_half, JITTER))
                tau_bar = training_precision(s_state)
            else
                tau_shape = TAU_Y_SHAPE0 + n / 2
                tau_rate = TAU_Y_RATE0 + sum(y_scatter_half)
                tau_bar = fill(tau_shape / tau_rate, n)
            end

            current_vector = vcat(
                reduce(vcat, flat_weights.(forecasters)),
                isnothing(s_state) ? Float64[] : flat_weights(s_state),
            )
            final_delta = norm(current_vector - previous_vector) /
                max(1.0, norm(previous_vector))
            final_delta < TOLERANCE && break
            previous_vector = current_vector
        end

        # --- prediction: q(x*) from expert factors only, then + E[1/τ_y] ----
        V = reduce(
            vcat,
            (forecaster_variance(forecasters[i], designs)'
                for i in 1:n_forecasters),
        )
        P = clamp.(1 ./ V, 1e-6, 1e6)
        n_test = length(y_test)
        μ = Vector{Float64}(undef, n_test)
        σ = Vector{Float64}(undef, n_test)
        inv_tau_test = if tauy == :func
            forecaster_variance(s_state, [s_design]; κ = 0.0)
        else
            fill(tau_rate / (tau_shape - 1), n_test)
        end
        for j in 1:n_test
            total = sum(@view P[:, j])
            mx = sum(
                P[i, j] * predictions_test[i, j] for i in 1:n_forecasters
            ) / total
            vx = 1 / total
            if CALIB
                # y* = a·x* + c + ε: Var = ā²v* + Caa(m*²+v*) + Ccc +
                # 2m*Cac + E[1/τ_y]
                μ[j] = θ̄[1] * mx + θ̄[2]
                z = [mx, 1.0]
                σ[j] = sqrt(
                    abs2(θ̄[1]) * vx + dot(z, Cθ, z) + Cθ[1, 1] * vx +
                    inv_tau_test[j],
                )
            else
                μ[j] = mx
                σ[j] = sqrt(vx + inv_tau_test[j])
            end
        end
        m = score_metrics(μ, σ, y_test)

        result["metrics"] = Dict(
            String(key) => Float64(value) for (key, value) in pairs(m)
        )
        result["mu"] = μ
        result["sigma"] = σ
        result["expected_beta"] =
            [fs.expected_beta for fs in forecasters]
        result["iterations_run"] = iterations_run
        result["final_delta"] = final_delta
        result["min_v_x"] = min_v_x
        result["median_v_x"] = median(v_x)
        result["level2_score_std"] = [
            std(fs.score_means[min(2, L)]) for fs in forecasters
        ]
        result["top_carrier"] = [
            top_carrier_precision(fs) for fs in forecasters
        ]
        result["calibration"] = CALIB ?
            Dict("a" => θ̄[1], "c" => θ̄[2],
                 "a_sd" => sqrt(Cθ[1, 1]), "c_sd" => sqrt(Cθ[2, 2])) :
            Dict{String, Float64}()
        result["inv_tau_mean"] = mean(inv_tau_test)
        result["inv_tau_std"] = std(inv_tau_test)
        result["level_weight_means"] =
            [fs.level_weights for fs in forecasters]
        s_summary = isnothing(s_state) ? Float64[] :
            s_state.level_weights[1]
        result["s_weights"] = s_summary

        @printf(
            "%-34s  nll %.4f±%.4f  rmse %.4f  cov95 %.3f  E[1/τy] %.3f±%.3f  Eβ %.3g  vx %.2g  iters %d  Δ %.2g  l2σ %.3g\n",
            name, m.nll, m.nll_ci, m.rmse, m.cov95,
            mean(inv_tau_test), std(inv_tau_test),
            mean(result["expected_beta"]),
            min_v_x, iterations_run, final_delta,
            minimum(result["level2_score_std"]),
        )
    catch exception
        result["status"] = "unstable"
        result["error"] = sprint(showerror, exception)
        @printf("%-34s  UNSTABLE: %s\n", name, first(result["error"], 120))
    end
    result["elapsed"] = elapsed
    jldsave(path; result)
    write_summary(cell, tag)
    return
end

# --- summary -----------------------------------------------------------------

function write_summary(cell, tag)
    dataset, horizon = cell
    directory = joinpath(RESULTS_DIR, tag)
    isdir(directory) || return
    rows = []
    for file in sort(filter(endswith(".jld2"), readdir(directory)))
        push!(rows, load(joinpath(directory, file))["result"])
    end
    isempty(rows) && return
    path = joinpath(RESULTS_DIR, "summary_$(tag).md")
    open(path, "w") do io
        println(io, "# $dataset h$(horizon) — consensus-x precision hierarchy\n")
        println(io, "Reference (paper Dynamic): NLL ",
            get(REFERENCE_NLL, cell, NaN),
            "; uncoupled direct port best: NLL ",
            get(UNCOUPLED_NLL, cell, NaN), " (lower is better)\n")
        println(io, "| setup | L | carrier | τy | status | nll | ll | rmse | mae | cov95 | pinball | E[1/τy] | Eβ | min vx | iters | Δ | l2σ | secs |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in rows
            if row["status"] == "ok"
                m = row["metrics"]
                @printf(io,
                    "| %s | %d | %s | %s | ok | %.4f±%.4f | %.4f±%.4f | %.4f±%.4f | %.4f | %.4f±%.4f | %.4f | %.3f±%.3f | %.3g | %.2g | %d | %.2g | %.3g | %.0f |\n",
                    row["setup"], row["L"], row["carrier"], row["tauy"],
                    m["nll"], m["nll_ci"], m["ll"], m["ll_ci"],
                    m["rmse"], m["rmse_ci"], m["mae"],
                    m["cov95"], m["cov_ci"], m["pinball"],
                    row["inv_tau_mean"], row["inv_tau_std"],
                    mean(row["expected_beta"]),
                    row["min_v_x"],
                    row["iterations_run"], row["final_delta"],
                    minimum(row["level2_score_std"]),
                    row["elapsed"],
                )
            else
                @printf(io, "| %s | %d | %s | %s | unstable | %s |||||||||||||\n",
                    row["setup"], row["L"], row["carrier"], row["tauy"],
                    first(row["error"], 80))
            end
        end
    end
    return
end

# --- driver ------------------------------------------------------------------

function main(args)
    length(args) in (2, 3) || error(
        "usage: etth2_consensus_precision_hierarchy.jl <smoke|full> <horizon> [dataset=ETTh2]",
    )
    phase = Symbol(args[1])
    phase in (:smoke, :full, :sweep) || error("unknown phase: $phase")
    horizon = parse(Int, args[2])
    dataset = length(args) == 3 ? args[3] : "ETTh2"
    haskey(REFERENCE_NLL, (dataset, horizon)) ||
        error("unknown cell: $dataset h$horizon")

    candidates = cache_candidates(dataset, horizon)
    cache_path_index = findfirst(isfile, candidates)
    isnothing(cache_path_index) &&
        error("missing cache, tried: $(join(candidates, ", "))")
    cache = load(candidates[cache_path_index])
    n_val = length(cache["y_val"])
    n_obs = ENV_N_OBS > 0 ? min(ENV_N_OBS, n_val) : n_val

    if phase == :sweep
        # Hyperparameter research on the winning arm structures: optimizer ×
        # weight-prior gain × intercept-anchor variance × β-prior rate.
        prefix = dataset == "ETTh2" ? "" : lowercase(dataset) * "_"
        tag = prefix * "h$(horizon)_sweep"
        need_rff = true
        bases = base_designs(
            cache["features_val"], cache["features_test"], n_obs, RFF_SEED;
            need_rff,
        )
        # EDH_WINNER=1 collapses the grid to one combination (defaults = the
        # rff sweep winner; override via EDH_W_* envs) for validation runs.
        # EDH_SWEEP_SETUP picks the feature basis (rff_all | linear_all).
        winner_only = get(ENV, "EDH_WINNER", "0") == "1"
        sweep_setup = Symbol(get(ENV, "EDH_SWEEP_SETUP", "rff_all"))
        w_gain = parse(Float64, get(ENV, "EDH_W_GAIN", "3.0"))
        w_brate = parse(Float64, get(ENV, "EDH_W_BRATE", "1.0"))
        w_alpha = parse(Float64, get(ENV, "EDH_W_ALPHA", "0.2"))
        w_mom = parse(Float64, get(ENV, "EDH_W_MOM", "0.0"))
        w_method = Symbol(get(ENV, "EDH_W_METHOD", "damped"))
        optimizers = winner_only ? ((w_alpha, w_mom, w_method),) :
            ((0.6, 0.8, :vector_transport), (0.2, 0.0, :damped))
        gains = winner_only ? (w_gain,) : (1.5, 3.0, 5.0)
        anchor_vars = winner_only ? (1.0,) : (1.0, 2.0)
        beta_rates = winner_only ? (w_brate,) : (1e3, 10.0, 1.0)
        default = HYPER[]
        count = 0
        for (alpha, momentum, method) in optimizers,
            gain in gains, anchor_var in anchor_vars

            for carrier in (:nobeta, :beta),
                beta_rate0 in (carrier == :beta ? beta_rates : (BETA_RATE0,))

                HYPER[] = (; alpha, momentum, method, gain,
                    beta_rate0, anchor_var)
                run_arm(cache, (dataset, horizon), tag, sweep_setup, 2,
                    carrier, :scalar, n_obs, bases; suffix = hyper_suffix())
                count += 1
            end
        end
        HYPER[] = default
        @printf("sweep done (%d combos): summary at %s\n", count,
            joinpath(RESULTS_DIR, "summary_$(tag).md"))
        return
    end

    arms = if phase == :smoke
        [(:linear_all, 2, carrier, tauy)
            for carrier in (:nobeta, :beta) for tauy in (:scalar, :func)]
    else
        vcat(
            [(setup, 2, carrier, tauy)
                for setup in (:linear_all, :rff_all)
                for carrier in (:nobeta, :beta)
                for tauy in (:scalar, :func)],
            # L=3 probes under the exact XOR deep-kernel recipe (rff basis):
            # func arm plus the scalar-τy winners.
            [(:rff_all, 3, :beta, :func),
             (:rff_all, 3, :beta, :scalar),
             (:rff_all, 3, :nobeta, :scalar)],
        )
    end

    @printf(
        "%s h%d %s (consensus-x): n_obs=%d, %d arms, iterations=%d, seed=%d\n",
        dataset, horizon, phase, n_obs, length(arms), ITERATIONS, RFF_SEED,
    )
    need_rff = any(arm -> arm[1] == :rff_all, arms)
    bases = base_designs(
        cache["features_val"], cache["features_test"], n_obs, RFF_SEED;
        need_rff,
    )
    # ETTh2 keeps its original tags ("h96") so existing results stay valid.
    prefix = dataset == "ETTh2" ? "" : lowercase(dataset) * "_"
    tag = prefix * (phase == :smoke ? "h$(horizon)_smoke" : "h$(horizon)")
    for (setup, L, carrier, tauy) in arms
        run_arm(cache, (dataset, horizon), tag, setup, L, carrier, tauy,
            n_obs, bases)
    end
    @printf(
        "done: summary at %s\n",
        joinpath(RESULTS_DIR, "summary_$(tag).md"),
    )
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
