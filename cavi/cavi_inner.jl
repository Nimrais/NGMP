# =============================================================================
# Hand-rolled CAVI for the inner conjugate model of dynamic_exp_ngmp_surrogate.jl
# =============================================================================
# Replaces `infer` on `inner_exp_ensemble`: per forecaster i and obs j,
#
#     z[i,j] ~ N(w[i]'f_j, 1/τ[i])          (softdot, mean-field q(w)q(z)q(τ))
#     obsz[i,j] ~ N(z[i,j], Rz[i,j])         (Gaussian pseudo-observation)
#
# with w[i] ~ MvNormal prior, τ[i] ~ Gamma prior. All VMP updates are closed
# form and match the ReactiveMP rules exactly:
#
#   q(z_j)  : precision λ_j = E[τ] + 1/R_j,  mean = (E[τ]·f_j'μ_w + o_j/R_j)/λ_j
#             (softdot(:y) mean-field rule × NormalMeanVariance obs message)
#   q(w)    : Λ = Λ₀ + E[τ]·FF',  ξ = ξ₀ + E[τ]·F·E[z]
#             (product of prior with the rank-1 softdot(:x) messages)
#   q(τ)    : shape = a₀ + n/2,
#             rate  = b₀ + ½ Σ_j [ v_j + (m_j − f_j'μ_w)² + f_j'Σ_w f_j ]
#             (product of prior with the softdot(:γ) Gamma(3/2, β_j) messages)
#
# FF' (d×d) is computed once per call — O(d²n) — and everything else is
# GEMV/GEMM-level BLAS. Cost per sweep ≈ O(nf·(d²n + d³)), i.e. milliseconds
# for d=65, n=3398, nf=7.
#
# `schedule` controls the within-sweep update order. Validated against RxInfer
# (compare_cavi_vs_rxinfer.jl, nf=7, no=800): every schedule converges to the
# SAME fixed point as `infer` (max |Δ| ~1e-15 at 50 iterations); at finite
# iteration counts (:τ, :z, :w) tracks the reactive engine most closely
# (max |Δ| ≤ 8e-5 at iterations = 5), so it is the default. Measured speedup
# vs `infer`: 594× at no=800; 16 ms per call at full 7×3398 scale.
# =============================================================================

using LinearAlgebra
import BayesBase: mean, cov, precision, weightedmean, shape, rate
import ExponentialFamily: GammaShapeRate, NormalMeanVariance, MvNormalMeanCovariance

"""
    cavi_inner_infer(features, obsz, Rz, w_priors, τ_priors;
                     iterations = 5, z_init = (0.0, 1.0), schedule = (:τ, :z, :w))

CAVI equivalent of `infer` on `inner_exp_ensemble`. `features` is a vector of
`n` feature vectors (length `d`, shared across forecasters), `obsz`/`Rz` are
`nf × n` matrices. Returns `(w = Vector{MvNormalMeanCovariance},
τ = Vector{GammaShapeRate}, z = Matrix{NormalMeanVariance})`, matching the
`[end]` posteriors of the RxInfer call.
"""
function cavi_inner_infer(features::AbstractVector, obsz::AbstractMatrix, Rz::AbstractMatrix,
                          w_priors::AbstractVector, τ_priors::AbstractVector;
                          iterations::Int = 5, z_init::Tuple{Real,Real} = (0.0, 1.0),
                          schedule::NTuple{3,Symbol} = (:τ, :z, :w))
    nf, n = size(obsz)
    d = length(features[1])
    F = reduce(hcat, features)::Matrix{Float64}          # d × n
    FFt = F * F'                                          # d × d, once per call

    # prior natural parameters (any Gaussian/Gamma parametrization accepted)
    Λ0 = [Matrix{Float64}(precision(w_priors[i])) for i in 1:nf]
    ξ0 = [Λ0[i] * mean(w_priors[i]) for i in 1:nf]
    a0 = [shape(τ_priors[i]) for i in 1:nf]
    b0 = [rate(τ_priors[i]) for i in 1:nf]

    # state, initialized like `inner_init`
    μw = [Vector{Float64}(mean(w_priors[i])) for i in 1:nf]
    Σw = [Matrix{Float64}(cov(w_priors[i])) for i in 1:nf]
    a = copy(a0); b = copy(b0)
    m = fill(float(z_init[1]), nf, n)
    v = fill(float(z_init[2]), nf, n)

    s = Matrix{Float64}(undef, nf, n)                     # s[i,:] = F'μw[i]
    for i in 1:nf
        mul!(view(s, i, :), F', μw[i])
    end
    G = Matrix{Float64}(undef, d, n)                      # scratch: Σw[i] * F

    for _ in 1:iterations, step in schedule
        if step === :z
            @inbounds for i in 1:nf
                Eτ = a[i] / b[i]
                for j in 1:n
                    λ = Eτ + 1.0 / Rz[i, j]
                    m[i, j] = (Eτ * s[i, j] + obsz[i, j] / Rz[i, j]) / λ
                    v[i, j] = 1.0 / λ
                end
            end
        elseif step === :w
            for i in 1:nf
                Eτ = a[i] / b[i]
                Λ = Symmetric(Λ0[i] + Eτ * FFt)
                ξ = ξ0[i] + Eτ * (F * view(m, i, :))
                C = cholesky(Λ)
                μw[i] = C \ ξ
                Σw[i] = inv(C)
                mul!(view(s, i, :), F', μw[i])
            end
        elseif step === :τ
            for i in 1:nf
                mul!(G, Σw[i], F)
                quad = 0.0
                @inbounds for j in 1:n
                    quad += dot(view(F, :, j), view(G, :, j))
                end
                misfit = 0.0
                @inbounds for j in 1:n
                    misfit += v[i, j] + (m[i, j] - s[i, j])^2
                end
                a[i] = a0[i] + n / 2
                b[i] = b0[i] + (misfit + quad) / 2
            end
        else
            error("unknown schedule step $step")
        end
    end

    return (w = [MvNormalMeanCovariance(μw[i], Σw[i]) for i in 1:nf],
            τ = [GammaShapeRate(a[i], b[i]) for i in 1:nf],
            z = [NormalMeanVariance(m[i, j], v[i, j]) for i in 1:nf, j in 1:n])
end

# =============================================================================
# Structured variant: q(w, z) q(τ)  (the factorization now used by the script)
# =============================================================================
# Given τ̄ = E[τ], the (w, z) cluster is a Gaussian TREE (w at the center, each
# z_j a leaf through its softdot + pseudo-obs), so one sweep of BP is exact:
#
#   λ_j  = 1/(Rz_j + 1/τ̄)            rank-1 obs message precision on f_j'w
#   Λ_w  = Λ₀ + F·diag(λ)·F',  ξ_w = ξ₀ + F·(λ∘o)      →  μ_w, Σ_w
#   z_j | w ~ N((τ̄·f_j'w + o_j/R_j)/p_j, 1/p_j),  p_j = τ̄ + 1/R_j
#   q(z_j): m_j = (τ̄·s_j + o_j/R_j)/p_j,  v_j = 1/p_j + (τ̄/p_j)²·f_j'Σ_w f_j
#   Cov(z_j, f_j'w) = (τ̄/p_j)·f_j'Σ_w f_j                  (c_j = τ̄/p_j below)
#
# τ update (softdot(:γ) with the joint q(z_j, w), shape 3/2 per node):
#   a = a₀ + n/2
#   b = b₀ + ½ Σ_j E[(z_j − f_j'w)²]
#     = b₀ + ½ Σ_j [ v_j + (m_j − s_j)² + (1 − 2c_j)·f_j'Σ_w f_j ]
#     ( = b₀ + ½ Σ_j [ 1/p_j + ((o_j−s_j)² + f_j'Σ_w f_j)/(R_j τ̄ + 1)² ] )
#
# Note Λ_w depends on τ̄ through λ, so F·diag(λ)·F' is one d×n×d GEMM per
# forecaster per sweep (unlike mean-field, FF' is not hoistable).
# `τ_scatter = :joint` is the exact structured VMP update (τ sees the joint
# scatter incl. the z–w cross-covariance — cancels the misfit, τ sticks at its
# prior mean when the prior is tight/large). `:marginal` evaluates the scatter
# under the product of the marginals (drops the −2c·qf cross term) — a hybrid
# that keeps the structured q(w,z) while letting τ learn from the misfit.
function cavi_inner_infer_structured(features::AbstractVector, obsz::AbstractMatrix, Rz::AbstractMatrix,
                                     w_priors::AbstractVector, τ_priors::AbstractVector;
                                     iterations::Int = 5, z_init::Tuple{Real,Real} = (0.0, 1.0),
                                     schedule::NTuple{2,Symbol} = (:wz, :τ), τ_scatter::Symbol = :joint)
    nf, n = size(obsz)
    d = length(features[1])
    F = reduce(hcat, features)::Matrix{Float64}           # d × n

    Λ0 = [Matrix{Float64}(precision(w_priors[i])) for i in 1:nf]
    ξ0 = [Λ0[i] * mean(w_priors[i]) for i in 1:nf]
    a0 = [shape(τ_priors[i]) for i in 1:nf]
    b0 = [rate(τ_priors[i]) for i in 1:nf]

    μw = [Vector{Float64}(mean(w_priors[i])) for i in 1:nf]
    Σw = [Matrix{Float64}(cov(w_priors[i])) for i in 1:nf]
    a = copy(a0); b = copy(b0)
    m = fill(float(z_init[1]), nf, n)
    v = fill(float(z_init[2]), nf, n)
    c = zeros(nf, n)                                      # Cov(z_j, f'w)/f'Σ_w f; 0 at init (independent marginals)

    s = Matrix{Float64}(undef, nf, n)                     # s[i,:]  = F'μw[i]
    qf = Matrix{Float64}(undef, nf, n)                    # qf[i,:] = diag(F'Σw[i]F)
    G = Matrix{Float64}(undef, d, n)
    for i in 1:nf
        mul!(view(s, i, :), F', μw[i])
        mul!(G, Σw[i], F)
        @inbounds for j in 1:n
            qf[i, j] = dot(view(F, :, j), view(G, :, j))
        end
    end

    for _ in 1:iterations, step in schedule
        if step === :wz
            for i in 1:nf
                τ̄ = a[i] / b[i]
                λ = [1.0 / (Rz[i, j] + 1.0 / τ̄) for j in 1:n]
                Λ = Symmetric(Λ0[i] + (F .* λ') * F')
                ξ = ξ0[i] + F * (λ .* view(obsz, i, :))
                C = cholesky(Λ)
                μw[i] = C \ ξ
                Σw[i] = inv(C)
                mul!(view(s, i, :), F', μw[i])
                mul!(G, Σw[i], F)
                @inbounds for j in 1:n
                    qf[i, j] = dot(view(F, :, j), view(G, :, j))
                    p = τ̄ + 1.0 / Rz[i, j]
                    m[i, j] = (τ̄ * s[i, j] + obsz[i, j] / Rz[i, j]) / p
                    v[i, j] = 1.0 / p + (τ̄ / p)^2 * qf[i, j]
                    c[i, j] = τ̄ / p
                end
            end
        elseif step === :τ
            crossfac = τ_scatter === :joint ? 2.0 : 0.0
            for i in 1:nf
                scatter = 0.0
                @inbounds for j in 1:n
                    scatter += v[i, j] + (m[i, j] - s[i, j])^2 + (1.0 - crossfac * c[i, j]) * qf[i, j]
                end
                a[i] = a0[i] + n / 2
                b[i] = b0[i] + scatter / 2
            end
        else
            error("unknown schedule step $step")
        end
    end

    return (w = [MvNormalMeanCovariance(μw[i], Σw[i]) for i in 1:nf],
            τ = [GammaShapeRate(a[i], b[i]) for i in 1:nf],
            z = [NormalMeanVariance(m[i, j], v[i, j]) for i in 1:nf, j in 1:n])
end
