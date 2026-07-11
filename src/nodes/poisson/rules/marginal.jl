@rule PoissonExp(:in, Marginalisation) (q_out::PointMass,) = begin
    y = mean(q_out)
    return PoissonExpression(y)
end

# Predictive rule for unobserved counts: plug-in rate E_q[e^z].
@rule PoissonExp(:out, Marginalisation) (q_in::UnivariateNormalDistributionsFamily, meta::Any) = begin
    m, v = mean_var(q_in)
    return Poisson(exp(m + v / 2))
end