# β-NLL-style faithful-heteroscedastic twins of the MvNormalExpPrecision mean-
# pathway rules (see PrecisionTempering in NaturalGradientMP/damping.jl): every
# message that carries the likelihood precision into the mean pathway uses the
# TEMPERED precision ρ^(1−β) instead of ρ = E[e^s], so hard points cannot mute
# their own mean-learning signal by inflating their variance. The s-site rules
# are deliberately NOT specialized — the log-precision channel trains on the
# residuals of the tempered-mean posterior (the stop-gradient scheme of Stirn
# et al. 2023). Predictive moments in the scripts are assembled post-hoc from
# marginals with the UNtempered ρ, so calibration at prediction time is intact.

_mnep_tempered_rho(ms, vs, meta::PrecisionTempering) =
    _mnep_rho(ms, vs) .^ (1 - meta.beta)
_mnep_tempered_rho(ms::Real, vs::Real, meta::PrecisionTempering) =
    exp(_mnep_clamp_exponent(ms + vs / 2))^(1 - meta.beta)

# --- mean-field multivariate ------------------------------------------------

@rule MvNormalExpPrecision(:out, Marginalisation) (q_μ::Any, q_s::MultivariateNormalDistributionsFamily, meta::PrecisionTempering) = begin
    ms, Vs = mean_cov(q_s)
    return MvNormalMeanPrecision(mean(q_μ), Matrix(Diagonal(_mnep_tempered_rho(ms, diag(Vs), meta))))
end

@rule MvNormalExpPrecision(:μ, Marginalisation) (q_out::Any, q_s::MultivariateNormalDistributionsFamily, meta::PrecisionTempering) = begin
    ms, Vs = mean_cov(q_s)
    return MvNormalMeanPrecision(mean(q_out), Matrix(Diagonal(_mnep_tempered_rho(ms, diag(Vs), meta))))
end

# --- mean-field univariate --------------------------------------------------

@rule MvNormalExpPrecision(:out, Marginalisation) (q_μ::Any, q_s::UnivariateNormalDistributionsFamily, meta::PrecisionTempering) = begin
    m, v = mean_var(q_s)
    return NormalMeanPrecision(mean(q_μ), _mnep_tempered_rho(m, v, meta))
end

@rule MvNormalExpPrecision(:μ, Marginalisation) (q_out::Any, q_s::UnivariateNormalDistributionsFamily, meta::PrecisionTempering) = begin
    m, v = mean_var(q_s)
    return NormalMeanPrecision(mean(q_out), _mnep_tempered_rho(m, v, meta))
end

# --- structured (out, μ) cluster --------------------------------------------

@rule MvNormalExpPrecision(:out, Marginalisation) (m_μ::MultivariateNormalDistributionsFamily, q_s::MultivariateNormalDistributionsFamily, meta::PrecisionTempering) = begin
    ms, Vs = mean_cov(q_s)
    ρt = _mnep_tempered_rho(ms, diag(Vs), meta)
    m̃, Ṽ = mean_cov(m_μ)
    return MvNormalMeanCovariance(m̃, Ṽ + Diagonal(inv.(ρt)))
end

@rule MvNormalExpPrecision(:μ, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, q_s::MultivariateNormalDistributionsFamily, meta::PrecisionTempering) = begin
    ms, Vs = mean_cov(q_s)
    ρt = _mnep_tempered_rho(ms, diag(Vs), meta)
    m̃, Ṽ = mean_cov(m_out)
    return MvNormalMeanCovariance(m̃, Ṽ + Diagonal(inv.(ρt)))
end

@marginalrule MvNormalExpPrecision(:out_μ) (m_out::MultivariateNormalDistributionsFamily, m_μ::MultivariateNormalDistributionsFamily, q_s::MultivariateNormalDistributionsFamily, meta::PrecisionTempering) = begin
    ms, Vs = mean_cov(q_s)
    P = Diagonal(_mnep_tempered_rho(ms, diag(Vs), meta))
    ξν, Λν = weightedmean_precision(m_out)
    ξμ, Λμ = weightedmean_precision(m_μ)
    return MvNormalWeightedMeanPrecision(
        [ξν; ξμ],
        [(Λν + P) (-Matrix(P)); (-Matrix(P)) (Λμ + P)],
    )
end
