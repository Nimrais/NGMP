@model function ngmp_poisson_ssm(y, sigma, m0, v0, deps, damping)
    z[1] ~ Normal(mean = m0, variance = v0)
    y[1] ~ PoissonExp(z[1]) where { dependencies = deps, meta = damping }
    for k in 2:length(y)
        z[k] ~ Normal(mean = z[k - 1], variance = sigma)
        y[k] ~ PoissonExp(z[k]) where { dependencies = deps, meta = damping }
    end
end

@initialization function ngmp_init(y)
    q(z) = NormalMeanVariance.(log.(coalesce.(y, 1) .+ 1.0), 1.0)
end

function simulate_counts(rng, N)
    ztrue = cumsum(0.3 .* randn(rng, N)) .+ 1.0
    return rand.(Ref(rng), Poisson.(exp.(ztrue)))
end

@testset "integration: native NGMP inference" begin
    rng = Random.MersenneTwister(20260706)
    N, iters = 30, 12
    sigma, m0, v0 = 0.1, 0.0, 10.0
    alpha, beta = 0.5, 0.2
    y = simulate_counts(rng, N)

    @testset "matches the surrogate outer-loop reference" begin
        deps = NGMPDependencies(in = nothing)
        result = infer(
            model = ngmp_poisson_ssm(
                sigma = sigma,
                m0 = m0,
                v0 = v0,
                deps = deps,
                damping = DampingMeta(alpha = alpha, beta = beta),
            ),
            data = (y = y,),
            initialization = ngmp_init(y),
            iterations = iters,
            free_energy = true,
            options = (limit_stack_depth = 100,),
        )
        post = last(result.posteriors[:z])
        mref, vref = ngmp_smoother_reference(y; sigma = sigma, m0 = m0, v0 = v0, iters = iters, alpha = alpha, beta = beta)
        @test maximum(abs.(mean.(post) .- mref)) < 1e-3
        @test maximum(abs.(var.(post) .- vref)) < 1e-3
        @test maximum(abs.(mean.(post) .- mref)) < 1e-10

        @test length(deps.states) == N
        @test all(state -> state.nfired == iters, deps.states)

        @test length(result.free_energy) == iters
        @test all(isfinite, result.free_energy)
        @test abs(result.free_energy[end] - result.free_energy[end - 1]) < 1e-2
    end

    @testset "missing observations: leaf stays flat, prediction is returned" begin
        ymiss = Vector{Union{Int, Missing}}(y)
        ymiss[10] = missing
        deps = NGMPDependencies(in = nothing)
        result = infer(
            model = ngmp_poisson_ssm(
                sigma = sigma,
                m0 = m0,
                v0 = v0,
                deps = deps,
                damping = DampingMeta(alpha = alpha, beta = beta),
            ),
            data = (y = ymiss,),
            initialization = ngmp_init(ymiss),
            iterations = iters,
            options = (limit_stack_depth = 100,),
        )
        @test sort(unique(state.nfired for state in deps.states)) == [0, iters]
        @test count(state -> state.nfired == 0, deps.states) == 1

        pred = last(result.predictions[:y])[10]
        @test pred isa Poisson
        post = last(result.posteriors[:z])[10]
        @test rate(pred) ≈ exp(mean(post) + var(post) / 2) rtol = 1e-6
    end
end
