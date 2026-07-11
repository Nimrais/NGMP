# Reference implementation of the NGMP surrogate outer loop from
# `notebooks/poisson_surrogate.jl`, with the inner conjugate smoother hand-rolled
# as forward filtering plus RTS smoothing on the random-walk chain.

function poisson_surrogate_reference(y::Real, m::Real, v::Real)
    rho = exp(m + v / 2)
    return (y + (m - 1) * rho, rho)
end

function kalman_smoother_reference(xi, Lambda; sigma, m0, v0)
    N = length(xi)
    mf = zeros(N)
    vf = zeros(N)
    mp = m0
    vp = v0
    for k in 1:N
        p = 1 / vp + Lambda[k]
        w = mp / vp + xi[k]
        vf[k] = 1 / p
        mf[k] = w / p
        mp = mf[k]
        vp = vf[k] + sigma
    end
    ms = copy(mf)
    vs = copy(vf)
    for k in (N - 1):-1:1
        vpred = vf[k] + sigma
        J = vf[k] / vpred
        ms[k] = mf[k] + J * (ms[k + 1] - mf[k])
        vs[k] = vf[k] + J^2 * (vs[k + 1] - vpred)
    end
    return ms, vs
end

function ngmp_smoother_reference(y; sigma = 0.1, m0 = 0.0, v0 = 10.0, iters = 10, alpha = 0.5, beta = 0.2)
    N = length(y)
    m = log.(y .+ 1.0)
    v = fill(1.0, N)
    xi = zeros(N)
    Lambda = zeros(N)
    vxi = zeros(N)
    vLambda = zeros(N)
    for _ in 1:iters
        s = poisson_surrogate_reference.(y, m, v)
        etaxi = first.(s)
        etaLambda = last.(s)
        @. vxi = beta * vxi + alpha * (etaxi - xi)
        @. xi += vxi
        @. vLambda = beta * vLambda + alpha * (etaLambda - Lambda)
        @. Lambda += vLambda
        m, v = kalman_smoother_reference(xi, Lambda; sigma = sigma, m0 = m0, v0 = v0)
    end
    return m, v
end
