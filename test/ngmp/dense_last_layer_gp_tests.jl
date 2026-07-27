# A dense last layer makes message passing exactly a GP; a mean-field one cannot.
#
# These tests pin the structural claim behind the parametric-GP reference arm
# (experiments/parametric_gp_reference.jl):
#
#   1. With deterministic features on the theta edge and a DENSE Gaussian weight
#      vector on the x edge, `softdot` inference reproduces the analytic Bayesian
#      linear regression posterior, and its predictive matches the function-space
#      GP with kernel k(x, x') = phi(x)' phi(x') -- to machine precision.
#
#   2. Diagonalising that same posterior destroys the behaviour: the predictive
#      variance becomes a non-negative combination of fixed non-negative basis
#      functions, so it can no longer dip where the data are. This is what makes
#      the production ManyPlus arm structurally unable to be GP-like, since
#      src/ManyPlusNode/rules.jl sums variances with no covariance term.
#
# The second test is the important one: it is a regression guard against anyone
# "simplifying" the weight vector back to per-coordinate marginals.

using LinearAlgebra

# `@model` expands to a top-level function definition, so it cannot live inside
# the `@testset` block.
@model function dense_last_layer_gp_test_model(y, features, n_features, noise_precision)
    v ~ MvNormalMeanCovariance(zeros(n_features), diageye(n_features))
    for observation in eachindex(y)
        y[observation] ~ softdot(features[observation], v, noise_precision)
    end
end

@testset "dense last layer == exact GP" begin
    # A small deterministic feature map with a genuine gap in the inputs, so the
    # contraction structure is visible.
    n_features = 12
    lengthscale = 0.4
    rng = Random.MersenneTwister(20260725)
    frequencies = randn(rng, n_features) ./ lengthscale
    phases = 2pi .* rand(rng, n_features)
    scale = sqrt(2 / n_features)
    feature_map(x) = scale .* cos.(frequencies .* x .+ phases)

    x_train = vcat(range(-2.0, -1.0; length = 9), range(1.0, 2.0; length = 9))
    y_train = sin.(x_train) .+ 0.05 .* randn(rng, length(x_train))
    noise_precision = 25.0

    design = reduce(hcat, (feature_map(x) for x in x_train))'
    x_star = collect(range(-2.5, 2.5; length = 41))
    design_star = reduce(hcat, (feature_map(x) for x in x_star))'

    result = infer(
        model = dense_last_layer_gp_test_model(
            features = collect(eachrow(design)),
            n_features = n_features,
            noise_precision = noise_precision,
        ),
        data = (y = y_train,),
        returnvars = (v = KeepLast(),),
        iterations = 5,
        free_energy = false,
        showprogress = false,
    )
    q_v = result.posteriors[:v]

    # Analytic weight-space posterior with an N(0, I) prior.
    exact_precision =
        Matrix{Float64}(I, n_features, n_features) + noise_precision .* (design' * design)
    exact_covariance = inv(Symmetric(exact_precision))
    exact_mean = noise_precision .* (exact_covariance * (design' * y_train))

    @test mean(q_v) ≈ exact_mean atol = 1e-8
    @test cov(q_v) ≈ exact_covariance atol = 1e-8

    # The posterior covariance must be genuinely dense: the off-diagonal mass is
    # what produces contraction at the data.
    off_diagonal = exact_covariance - Diagonal(diag(exact_covariance))
    @test maximum(abs, off_diagonal) > 1e-3

    # Weight space vs function space, same kernel, independent computation.
    mp_mean = design_star * mean(q_v)
    mp_variance = vec(sum((design_star * cov(q_v)) .* design_star; dims = 2))

    gram = design * design'
    cross = design_star * design'
    prior_variance = vec(sum(abs2, design_star; dims = 2))
    factorization = cholesky(Symmetric(gram + inv(noise_precision) * I))
    gp_mean = cross * (factorization \ y_train)
    gp_variance =
        prior_variance .- vec(sum(cross .* (factorization \ cross')'; dims = 2))

    @test mp_mean ≈ gp_mean atol = 1e-8
    @test mp_variance ≈ gp_variance atol = 1e-8

    # GP behaviour: variance contracts on the observed blocks and grows in the
    # interior gap and beyond the data.
    observed = @. (-2.0 <= x_star <= -1.0) | (1.0 <= x_star <= 2.0)
    gap = @. abs(x_star) < 0.75
    outer = @. abs(x_star) > 2.25

    @test mean(gp_variance[gap]) / mean(gp_variance[observed]) > 5
    @test mean(gp_variance[outer]) / mean(gp_variance[observed]) > 5

    @testset "diagonalising the posterior destroys contraction" begin
        # Keep the exact marginal variances, drop only the correlations -- which
        # is precisely what a mean-field weight posterior and a variance-summing
        # aggregation node retain and discard respectively.
        diagonal_variance = vec(
            sum((design_star * Diagonal(diag(exact_covariance))) .* design_star; dims = 2),
        )

        # A non-negative combination of non-negative basis functions cannot carve
        # a dip at the data, so the contraction ratio collapses toward one.
        dense_ratio = mean(gp_variance[gap]) / mean(gp_variance[observed])
        diagonal_ratio =
            mean(diagonal_variance[gap]) / mean(diagonal_variance[observed])
        @test diagonal_ratio < dense_ratio / 5

        # And it cannot contract below the prior scale at the observed inputs.
        @test mean(diagonal_variance[observed]) > 10 * mean(gp_variance[observed])
    end
end
