# Minimal reproduction of the ETTh2 q10/q90 CT-Probit validation result.
#
# Run from the repository root:
#
#   julia --project=. experiments/etth2_q10_q90_ct_probit_minimal.jl
#
# This script is intentionally fixed to the experiment that produced mean RMSE
# 0.554974. It reads validation history only; the held-out test set is not read.

ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"

include(joinpath(@__DIR__, "dynamic_ngmp_shared_projector.jl"))
import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

const REPRO_CACHE = normpath(joinpath(
    @__DIR__,
    "..",
    "cache",
    "dynamic_etth2_h96_cache.jld2",
))
const REPRO_SEEDS = (2_026, 2_027, 2_028)
const REPRO_HIDDEN_WIDTH = 4
const REPRO_CT_RANK = 8
const REPRO_TRAINING_ITERATIONS = 50
const REPRO_PREDICTION_ITERATIONS = 20
const REPRO_PREDICTION_PRIOR_VARIANCE = 1.0e12

@model function repro_q10_q90_training_model(
    n_observations,
    y,
    features,
    predictions,
    priors,
    map_meta,
    hidden_meta,
    map_dependencies,
    hidden_dependencies,
    activation,
    activation_dependencies,
)
    local h1, s, h2, z, switch

    a_map ~ priors[:a_map]
    a_hidden ~ priors[:a_hidden]
    P_map ~ priors[:P_map]
    P_hidden ~ priors[:P_hidden]
    theta ~ priors[:theta]
    gamma_score ~ priors[:gamma_score]
    gamma_q10 ~ priors[:gamma_q10]
    gamma_q90 ~ priors[:gamma_q90]

    for observation in 1:n_observations
        h1[observation] ~ ContinuousTransition(
            features[observation],
            a_map,
            P_map,
        ) where {
            dependencies = map_dependencies,
            meta = map_meta,
        }
        s[observation] ~ MvResidualSine(h1[observation]) where {
            dependencies = activation_dependencies,
            meta = activation,
        }
        h2[observation] ~ ContinuousTransition(
            s[observation],
            a_hidden,
            P_hidden,
        ) where {
            dependencies = hidden_dependencies,
            meta = hidden_meta,
        }
        z[observation] ~ softdot(theta, h2[observation], gamma_score)
        switch[observation] ~ Probit(z[observation])
        y[observation] ~ NormalMixture(
            switch = switch[observation],
            m = (predictions[1, observation], predictions[2, observation]),
            p = (gamma_q10, gamma_q90),
        )
    end
end

@constraints function repro_q10_q90_training_constraints()
    q(
        h1,
        s,
        h2,
        z,
        switch,
        a_map,
        a_hidden,
        P_map,
        P_hidden,
        theta,
        gamma_score,
        gamma_q10,
        gamma_q90,
    ) = q(h1)q(s, h2)q(z)q(switch)q(a_map)q(a_hidden)q(P_map)q(P_hidden)q(theta)q(gamma_score)q(gamma_q10)q(gamma_q90)
end

# Prediction is a second RxInfer graph. The learned global marginals are fixed,
# and a diffuse Gaussian closes each otherwise dangling mixture output y.
@model function repro_q10_q90_prediction_model(
    features,
    predictions,
    priors,
    map_meta,
    hidden_meta,
    map_dependencies,
    hidden_dependencies,
    activation,
    activation_dependencies,
)
    local h1, s, h2, z, switch, y

    a_map ~ priors[:a_map]
    a_hidden ~ priors[:a_hidden]
    P_map ~ priors[:P_map]
    P_hidden ~ priors[:P_hidden]
    theta ~ priors[:theta]
    gamma_score ~ priors[:gamma_score]
    gamma_q10 ~ priors[:gamma_q10]
    gamma_q90 ~ priors[:gamma_q90]

    for observation in eachindex(features)
        h1[observation] ~ ContinuousTransition(
            features[observation],
            a_map,
            P_map,
        ) where {
            dependencies = map_dependencies,
            meta = map_meta,
        }
        s[observation] ~ MvResidualSine(h1[observation]) where {
            dependencies = activation_dependencies,
            meta = activation,
        }
        h2[observation] ~ ContinuousTransition(
            s[observation],
            a_hidden,
            P_hidden,
        ) where {
            dependencies = hidden_dependencies,
            meta = hidden_meta,
        }
        z[observation] ~ softdot(theta, h2[observation], gamma_score)
        switch[observation] ~ Probit(z[observation])
        y[observation] ~ NormalMixture(
            switch = switch[observation],
            m = (predictions[1, observation], predictions[2, observation]),
            p = (gamma_q10, gamma_q90),
        )
        y[observation] ~ NormalMeanVariance(
            0.0,
            REPRO_PREDICTION_PRIOR_VARIANCE,
        )
    end
end

@constraints function repro_q10_q90_prediction_constraints(priors)
    q(
        h1,
        s,
        h2,
        z,
        switch,
        y,
        a_map,
        a_hidden,
        P_map,
        P_hidden,
        theta,
        gamma_score,
        gamma_q10,
        gamma_q90,
    ) = q(h1)q(s, h2)q(z)q(switch)q(y)q(a_map)q(a_hidden)q(P_map)q(P_hidden)q(theta)q(gamma_score)q(gamma_q10)q(gamma_q90)

    q(a_map)::RxInfer.FixedMarginalFormConstraint(priors[:a_map])
    q(a_hidden)::RxInfer.FixedMarginalFormConstraint(priors[:a_hidden])
    q(P_map)::RxInfer.FixedMarginalFormConstraint(priors[:P_map])
    q(P_hidden)::RxInfer.FixedMarginalFormConstraint(priors[:P_hidden])
    q(theta)::RxInfer.FixedMarginalFormConstraint(priors[:theta])
    q(gamma_score)::RxInfer.FixedMarginalFormConstraint(priors[:gamma_score])
    q(gamma_q10)::RxInfer.FixedMarginalFormConstraint(priors[:gamma_q10])
    q(gamma_q90)::RxInfer.FixedMarginalFormConstraint(priors[:gamma_q90])
end

function repro_dependencies()
    map_dependencies = NGMPDependencies(
        a = nothing,
        damping = DampingMeta(alpha = 0.5, beta = 0.2, max_step = Inf),
    )
    hidden_dependencies = NGMPDependencies(
        a = nothing,
        damping = DampingMeta(alpha = 0.5, beta = 0.2, max_step = Inf),
    )
    activation_dependencies = NGMPDependencies(
        out = nothing,
        in = nothing,
        projection = TangentProjection(type = ClosedForm),
        damping = DampingMeta(alpha = 0.4, beta = 0.2, max_step = 1.0),
    )
    return (; map_dependencies, hidden_dependencies, activation_dependencies)
end

function repro_setup(features, seed)
    feature_setup = projector_reference_map(features, REPRO_HIDDEN_WIDTH)
    map = projector_svd_low_rank_meta(
        feature_setup.reference,
        REPRO_CT_RANK;
        offset_scale = 0.0,
        prior_variance = 0.25,
        initial_jitter = 0.05,
        rng = MersenneTwister(seed),
    )
    hidden = projector_svd_low_rank_meta(
        Matrix{Float64}(I, REPRO_HIDDEN_WIDTH, REPRO_HIDDEN_WIDTH),
        REPRO_CT_RANK;
        offset_scale = 1.0,
        prior_variance = 0.25,
        initial_jitter = 0.05,
        rng = MersenneTwister(seed + 100_000),
    )

    degrees_of_freedom = 6.0
    precision_prior = ExponentialFamily.WishartFast(
        degrees_of_freedom,
        Matrix(Diagonal(fill(
            degrees_of_freedom / 10.0,
            REPRO_HIDDEN_WIDTH,
        ))),
    )
    priors = Dict{Symbol, Any}(
        :a_map => map.prior,
        :a_hidden => hidden.prior,
        :P_map => precision_prior,
        :P_hidden => deepcopy(precision_prior),
        :theta => MvNormalMeanCovariance(
            zeros(REPRO_HIDDEN_WIDTH),
            Matrix(Diagonal(fill(0.3, REPRO_HIDDEN_WIDTH))),
        ),
        :gamma_score => GammaShapeRate(10.0, 1.0),
        :gamma_q10 => GammaShapeRate(10.0, 1.0),
        :gamma_q90 => GammaShapeRate(10.0, 1.0),
    )
    return (;
        priors,
        map_meta = map.meta,
        hidden_meta = hidden.meta,
        activation = ResidualSineMeta(rho = 0.9, omega = 1.0),
    )
end

function repro_training_initialization(setup)
    sine_mean, sine_covariance = SurrogateModelling._mv_residual_sine_mean_cov(
        zeros(REPRO_HIDDEN_WIDTH),
        Matrix(Diagonal(ones(REPRO_HIDDEN_WIDTH))),
        setup.activation,
    )
    return @initialization begin
        q(a_map) = setup.priors[:a_map]
        q(a_hidden) = setup.priors[:a_hidden]
        q(P_map) = setup.priors[:P_map]
        q(P_hidden) = setup.priors[:P_hidden]
        q(theta) = setup.priors[:theta]
        q(gamma_score) = setup.priors[:gamma_score]
        q(gamma_q10) = setup.priors[:gamma_q10]
        q(gamma_q90) = setup.priors[:gamma_q90]
        q(h1) = MvNormalMeanCovariance(
            zeros(REPRO_HIDDEN_WIDTH),
            Matrix(Diagonal(ones(REPRO_HIDDEN_WIDTH))),
        )
        q(s) = MvNormalMeanCovariance(sine_mean, sine_covariance)
        q(h2) = MvNormalMeanCovariance(
            zeros(REPRO_HIDDEN_WIDTH),
            Matrix(Diagonal(ones(REPRO_HIDDEN_WIDTH))),
        )
        q(z) = NormalMeanVariance(0.0, 1.0)
        q(switch) = Bernoulli(0.5)
    end
end

function repro_prediction_initialization(priors, setup, output_mean)
    sine_mean, sine_covariance = SurrogateModelling._mv_residual_sine_mean_cov(
        zeros(REPRO_HIDDEN_WIDTH),
        Matrix(Diagonal(ones(REPRO_HIDDEN_WIDTH))),
        setup.activation,
    )
    return @initialization begin
        q(a_map) = priors[:a_map]
        q(a_hidden) = priors[:a_hidden]
        q(P_map) = priors[:P_map]
        q(P_hidden) = priors[:P_hidden]
        q(theta) = priors[:theta]
        q(gamma_score) = priors[:gamma_score]
        q(gamma_q10) = priors[:gamma_q10]
        q(gamma_q90) = priors[:gamma_q90]
        q(h1) = MvNormalMeanCovariance(
            zeros(REPRO_HIDDEN_WIDTH),
            Matrix(Diagonal(ones(REPRO_HIDDEN_WIDTH))),
        )
        q(s) = MvNormalMeanCovariance(sine_mean, sine_covariance)
        q(h2) = MvNormalMeanCovariance(
            zeros(REPRO_HIDDEN_WIDTH),
            Matrix(Diagonal(ones(REPRO_HIDDEN_WIDTH))),
        )
        q(z) = NormalMeanVariance(0.0, 1.0)
        q(switch) = Bernoulli(0.5)
        q(y) = NormalMeanVariance(
            output_mean,
            REPRO_PREDICTION_PRIOR_VARIANCE,
        )
        μ(y) = NormalMeanVariance(
            output_mean,
            REPRO_PREDICTION_PRIOR_VARIANCE,
        )
    end
end

function repro_load_data()
    isfile(REPRO_CACHE) || error("missing ETTh2 cache: $REPRO_CACHE")
    history = jldopen(REPRO_CACHE, "r") do file
        y = collect(Float64, file["y_val"])
        all_predictions = Matrix{Float64}(file["predictions_val"])
        features = [collect(Float64, value) for value in file["features_val"]]
        predictions = all_predictions[[6, 7], :]
        (; y, predictions, features)
    end
    length(history.y) == 3_446 || error("unexpected validation-history length")
    training_indices = 1:2_048
    validation_indices = 2_049:3_072
    training = (;
        y = history.y[training_indices],
        predictions = history.predictions[:, training_indices],
        features = history.features[training_indices],
    )
    validation = (;
        y = history.y[validation_indices],
        predictions = history.predictions[:, validation_indices],
        features = history.features[validation_indices],
    )
    return (; training, validation)
end

function repro_fit(training, seed)
    setup = repro_setup(training.features, seed)
    dependencies = repro_dependencies()
    result = infer(
        model = repro_q10_q90_training_model(
            n_observations = length(training.y),
            predictions = training.predictions,
            priors = setup.priors,
            map_meta = setup.map_meta,
            hidden_meta = setup.hidden_meta,
            map_dependencies = dependencies.map_dependencies,
            hidden_dependencies = dependencies.hidden_dependencies,
            activation = setup.activation,
            activation_dependencies = dependencies.activation_dependencies,
        ),
        data = (y = training.y, features = training.features),
        constraints = repro_q10_q90_training_constraints(),
        initialization = repro_training_initialization(setup),
        iterations = REPRO_TRAINING_ITERATIONS,
        free_energy = false,
        showprogress = false,
        options = (limit_stack_depth = 500,),
        disable_inference_error_hint = true,
        returnvars = (
            a_map = KeepLast(),
            a_hidden = KeepLast(),
            P_map = KeepLast(),
            P_hidden = KeepLast(),
            theta = KeepLast(),
            gamma_score = KeepLast(),
            gamma_q10 = KeepLast(),
            gamma_q90 = KeepLast(),
            h1 = KeepLast(),
            s = KeepLast(),
            h2 = KeepLast(),
            z = KeepLast(),
            switch = KeepLast(),
        ),
    )
    posterior = Dict{Symbol, Any}(
        key => result.posteriors[key] for key in (
            :a_map,
            :a_hidden,
            :P_map,
            :P_hidden,
            :theta,
            :gamma_score,
            :gamma_q10,
            :gamma_q90,
        )
    )
    return (; posterior, setup)
end

function repro_predict(fit, validation)
    priors = Dict(key => deepcopy(value) for (key, value) in fit.posterior)
    dependencies = repro_dependencies()
    output_mean = mean(validation.predictions)
    result = infer(
        model = repro_q10_q90_prediction_model(
            predictions = validation.predictions,
            priors = priors,
            map_meta = fit.setup.map_meta,
            hidden_meta = fit.setup.hidden_meta,
            map_dependencies = dependencies.map_dependencies,
            hidden_dependencies = dependencies.hidden_dependencies,
            activation = fit.setup.activation,
            activation_dependencies = dependencies.activation_dependencies,
        ),
        data = (features = validation.features,),
        constraints = repro_q10_q90_prediction_constraints(priors),
        initialization = repro_prediction_initialization(
            priors,
            fit.setup,
            output_mean,
        ),
        iterations = REPRO_PREDICTION_ITERATIONS,
        free_energy = false,
        showprogress = false,
        returnvars = (y = KeepLast(), switch = KeepLast()),
        options = (limit_stack_depth = 500,),
        disable_inference_error_hint = true,
    )
    prediction = Float64.(mean.(vec(result.posteriors[:y])))
    p_q90 = [probvec(value)[2] for value in vec(result.posteriors[:switch])]
    all(isfinite, prediction) || error("non-finite prediction")
    all(isfinite, p_q90) || error("non-finite gate probability")
    return (; prediction, p_q90)
end

function reproduce_etth2_q10_q90_ct_probit()
    data = repro_load_data()
    q10_rmse = sqrt(mean(abs2, data.validation.predictions[1, :] .- data.validation.y))
    println("q10 validation RMSE: $(round(q10_rmse; digits = 6))")

    rows = map(REPRO_SEEDS) do seed
        GC.gc()
        fit = repro_fit(data.training, seed)
        prediction = repro_predict(fit, data.validation)
        rmse = sqrt(mean(abs2, prediction.prediction .- data.validation.y))
        println(
            "seed $seed: RMSE=$(round(rmse; digits = 6)), " *
            "mean p(q90)=$(round(mean(prediction.p_q90); digits = 6))",
        )
        (; seed, rmse, mean_p_q90 = mean(prediction.p_q90))
    end

    mean_rmse = mean(row.rmse for row in rows)
    println("three-seed mean RMSE: $(round(mean_rmse; digits = 6))")
    isapprox(mean_rmse, 0.5549736634650514; atol = 5.0e-6) || error(
        "expected approximately 0.554974, got $mean_rmse",
    )
    return (; q10_rmse, rows, mean_rmse)
end

if abspath(PROGRAM_FILE) == @__FILE__
    etth2_q10_q90_reproduction = reproduce_etth2_q10_q90_ct_probit()
end
