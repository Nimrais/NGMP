# Depth sweep for the ContinuousTransition / MvSoftplus XOR model.
#
# A depth-L arm is
#
#   x -> CT(a[1]) -> h[1] -> softplus -> CT(a[2]) -> ...
#     -> softplus -> CT(a[L]) -> h[L] -> softdot(theta) -> y
#
# Thus depth 2 is the architecture in xor_ctransition_mvsoftplus.jl, and every
# extra depth adds one nonlinear hidden block. All hidden vectors have the same
# deliberately small width. Training retains a structured q(output, input) at
# every ContinuousTransition. MSE is evaluated with the posterior-mean network;
# no prediction graph or damping state is allocated.
#
# Quick smoke check (the default):
#   julia --project=. experiments/xor_ctransition_mvsoftplus_depth_comparison.jl
#
# More useful comparison:
#   XOR_CT_DEPTH_SMOKE=false N_SAMPLES=200 D_HIDDEN=3 N_ITERATIONS=60 \
#   CT_DEPTHS=2,3,4 SAVE_OUTPUTS=true OPENBLAS_NUM_THREADS=1 \
#   julia --project=. experiments/xor_ctransition_mvsoftplus_depth_comparison.jl

const XOR_CT_DEPTH_SMOKE =
    lowercase(get(ENV, "XOR_CT_DEPTH_SMOKE", "true")) in ("1", "true", "yes", "on")
get!(ENV, "XOR_CT_SMOKE", string(XOR_CT_DEPTH_SMOKE))
get!(
    ENV,
    "OUTPUT_PREFIX",
    joinpath(@__DIR__, "..", "viz", "xor_ctransition_mvsoftplus_depth"),
)

include(joinpath(@__DIR__, "xor_ctransition_mvsoftplus.jl"))

using CSV
using LinearAlgebra: norm

function env_depths(name, default)
    values = parse.(Int, strip.(split(get(ENV, name, default), ',')))
    isempty(values) && throw(ArgumentError("$name must contain at least one depth"))
    all(>=(2), values) || throw(ArgumentError("all $name values must be at least 2"))
    length(values) == length(unique(values)) ||
        throw(ArgumentError("$name must not contain duplicate depths"))
    return values
end

const CT_DEPTHS = env_depths("CT_DEPTHS", "2,3,4")
const CT_DEPTH_SHOW_PROGRESS = env_bool("SHOW_PROGRESS", false)

@model function xor_ct_mvsoftplus_deep(
    y,
    features,
    priors,
    feature_cov,
    metas,
    depth,
    ct_a_deps,
    sp_deps,
    sp_damping,
)
    local a, transition_precision, x_f, h, s, theta, gamma_obs

    for layer in 1:depth
        a[layer] ~ priors[:a][layer]
        transition_precision[layer] ~ priors[:transition_precision][layer]
    end
    theta ~ priors[:theta]
    gamma_obs ~ priors[:gamma_obs]

    for observation in eachindex(y)
        x_f[observation] ~ MvNormalMeanCovariance(
            features[observation],
            feature_cov,
        )
        h[1, observation] ~ ContinuousTransition(
            x_f[observation],
            a[1],
            transition_precision[1],
        ) where {
            dependencies = ct_a_deps,
            meta = metas[1],
        }

        for layer in 1:(depth - 1)
            s[layer, observation] ~ MvSoftplus(h[layer, observation]) where {
                dependencies = sp_deps,
                meta = sp_damping,
            }
            h[layer + 1, observation] ~ ContinuousTransition(
                s[layer, observation],
                a[layer + 1],
                transition_precision[layer + 1],
            ) where {
                dependencies = ct_a_deps,
                meta = metas[layer + 1],
            }
        end

        y[observation] ~ softdot(theta, h[depth, observation], gamma_obs)
    end
end

# The local cluster makes each transition's input/output pair structured. Only
# the variables incident to a given factor enter its local marginal, so this
# works for a dynamically sized stack. MvSoftplus remains deterministically
# factorized by ReactiveMP.
@constraints function xor_ct_mvsoftplus_deep_constraints()
    q(x_f, h, s, a, transition_precision, theta, gamma_obs) =
        q(x_f, h, s)q(a)q(transition_precision)q(theta)q(gamma_obs)
end

function make_deep_priors(; depth, d_h, d_f, seed, ct_precision_mean)
    rng = StableRNG(seed)
    ν = d_h + 2.0
    inv_scale = Matrix(Diagonal(fill(ν / ct_precision_mean, d_h)))
    weight_priors = Vector{Any}(undef, depth)
    weight_priors[1] = MvNormalMeanCovariance(
        0.5 .* randn(rng, d_h * d_f),
        Diagonal(ones(d_h * d_f)),
    )
    for layer in 2:depth
        weight_priors[layer] = MvNormalMeanCovariance(
            0.5 .* randn(rng, d_h * d_h),
            Diagonal(ones(d_h * d_h)),
        )
    end
    return Dict{Symbol, Any}(
        :a => weight_priors,
        :transition_precision => [WishartFast(ν, inv_scale) for _ in 1:depth],
        :theta => MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h))),
        :gamma_obs => GammaShapeRate(1.0, 1.0),
    )
end

function make_deep_initialization(
    priors,
    d_h;
    softplus_output_initial_mean,
    softplus_output_initial_variance,
)
    return @initialization begin
        q(a) = deepcopy(priors[:a])
        q(transition_precision) = deepcopy(priors[:transition_precision])
        q(theta) = priors[:theta]
        q(gamma_obs) = priors[:gamma_obs]
        q(h) = MvNormalMeanCovariance(zeros(d_h), Diagonal(ones(d_h)))
        q(s) = MvNormalMeanCovariance(
            fill(softplus_output_initial_mean, d_h),
            Diagonal(fill(softplus_output_initial_variance, d_h)),
        )
    end
end

function posterior_mean_predictions(result, features, depth, d_h, d_f)
    weight_posteriors = last(result.posteriors[:a])
    theta = mean(last(result.posteriors[:theta]))
    matrices = Vector{Matrix{Float64}}(undef, depth)
    matrices[1] = reshape(mean(weight_posteriors[1]), d_h, d_f)
    for layer in 2:depth
        matrices[layer] = reshape(mean(weight_posteriors[layer]), d_h, d_h)
    end

    return [
        begin
            hidden = matrices[1] * feature
            for layer in 2:depth
                hidden = matrices[layer] * _softplus.(hidden)
            end
            dot(theta, hidden)
        end for feature in features
    ]
end

function relative_weight_movements(result, priors, depth)
    weight_posteriors = last(result.posteriors[:a])
    return [
        norm(mean(weight_posteriors[layer]) .- mean(priors[:a][layer])) /
        norm(mean(priors[:a][layer])) for layer in 1:depth
    ]
end

function dependency_diagnostics(dependencies)
    firings = getproperty.(dependencies.states, :nfired)
    return (
        count = length(firings),
        minimum_firings = isempty(firings) ? 0 : minimum(firings),
        maximum_firings = isempty(firings) ? 0 : maximum(firings),
    )
end

function run_depth_arm(depth, config, train_data, test_data, train_features, test_features)
    d_h, d_f = config.d_hidden, 3
    priors = make_deep_priors(
        depth = depth,
        d_h = d_h,
        d_f = d_f,
        seed = config.prior_seed,
        ct_precision_mean = config.ct_precision_mean,
    )
    metas = [
        layer == 1 ? LinearReshapeMeta(d_h, d_f) : LinearReshapeMeta(d_h, d_h) for
        layer in 1:depth
    ]
    ct_a_deps = NGMPDependencies(
        a = nothing,
        damping = DampingMeta(
            alpha = config.ct_a_alpha,
            beta = config.ct_a_beta,
            max_step = config.ct_a_max_step,
        ),
    )
    sp_deps = make_mvsoftplus_dependencies(config)
    sp_damping = DampingMeta(
        alpha = config.ngmp_alpha,
        beta = config.ngmp_beta,
        max_step = config.ngmp_max_step,
    )

    timed = @timed infer(
        model = xor_ct_mvsoftplus_deep(
            priors = priors,
            feature_cov = Matrix(Diagonal(fill(config.feature_jitter, d_f))),
            metas = metas,
            depth = depth,
            ct_a_deps = ct_a_deps,
            sp_deps = sp_deps,
            sp_damping = sp_damping,
        ),
        data = (y = train_data.OT, features = train_features),
        constraints = xor_ct_mvsoftplus_deep_constraints(),
        initialization = make_deep_initialization(
            priors,
            d_h,
            softplus_output_initial_mean = config.softplus_output_initial_mean,
            softplus_output_initial_variance = config.softplus_output_initial_variance,
        ),
        iterations = config.iterations,
        free_energy = true,
        showprogress = CT_DEPTH_SHOW_PROGRESS,
        returnvars = (a = KeepEach(), theta = KeepEach()),
        options = (limit_stack_depth = 200,),
        disable_inference_error_hint = true,
    )

    result = timed.value
    train_prediction = posterior_mean_predictions(result, train_features, depth, d_h, d_f)
    test_prediction = posterior_mean_predictions(result, test_features, depth, d_h, d_f)
    movements = relative_weight_movements(result, priors, depth)
    ct_diagnostics = dependency_diagnostics(ct_a_deps)
    sp_diagnostics = dependency_diagnostics(sp_deps)
    expected_ct_states = depth * nrow(train_data)
    expected_sp_states = 2 * (depth - 1) * nrow(train_data)

    ct_diagnostics.count == expected_ct_states || error(
        "depth $depth created $(ct_diagnostics.count) CT states; expected $expected_ct_states",
    )
    sp_diagnostics.count == expected_sp_states || error(
        "depth $depth created $(sp_diagnostics.count) MvSoftplus states; expected $expected_sp_states",
    )
    ct_diagnostics.minimum_firings == config.iterations || error(
        "some depth-$depth CT weight states did not fire once per iteration",
    )
    ct_diagnostics.maximum_firings == config.iterations || error(
        "some depth-$depth CT weight states fired more than once per iteration",
    )
    all(isfinite, result.free_energy) || error("depth $depth produced non-finite free energy")
    all(isfinite, train_prediction) || error("depth $depth produced non-finite train predictions")
    all(isfinite, test_prediction) || error("depth $depth produced non-finite test predictions")

    return (
        result = result,
        seconds = timed.time,
        allocated_gib = timed.bytes / 2.0^30,
        train_mse = mean(abs2, train_prediction .- train_data.OT),
        test_mse = mean(abs2, test_prediction .- test_data.OT),
        free_energy_first = first(result.free_energy),
        free_energy_last = last(result.free_energy),
        decreasing_steps = count(<(0), diff(result.free_energy)),
        weight_movements = movements,
        ct_diagnostics = ct_diagnostics,
        sp_diagnostics = sp_diagnostics,
    )
end

function save_depth_comparison(results, config)
    csv_path = config.output_prefix * "_depth_results.csv"
    plot_path = config.output_prefix * "_depth_mse.png"
    mkpath(dirname(csv_path))
    CSV.write(csv_path, results)

    successful = filter(:status => ==("ok"), results)
    if !isempty(successful)
        baseline = first(successful.baseline_mse)
        figure = plot(
            successful.depth,
            successful.test_mse;
            marker = :circle,
            linewidth = 2,
            xlabel = "ContinuousTransition depth",
            ylabel = "MSE",
            label = "test",
            title = "Small-width CT / MvSoftplus depth comparison",
        )
        plot!(figure, successful.depth, successful.train_mse; marker = :square, label = "train")
        hline!(figure, [baseline]; linestyle = :dash, label = "constant baseline")
        savefig(figure, plot_path)
    end
    return (csv = csv_path, plot = plot_path)
end

function run_depth_comparison(config = CONFIG; depths = CT_DEPTHS)
    dataset = make_checkerboard_dataset(
        n = config.n_samples,
        noise_std = config.noise_std,
        seed = config.data_seed,
    )
    train_data, test_data = split_dataset(
        dataset;
        train_fraction = config.train_fraction,
        seed = config.split_seed,
    )
    train_features = build_features(train_data)
    test_features = build_features(test_data)
    baseline = mean(abs2, mean(train_data.OT) .- test_data.OT)
    rows = DataFrame(
        depth = Int[],
        width = Int[],
        prior_seed = Int[],
        status = String[],
        seconds = Float64[],
        allocated_gib = Float64[],
        train_mse = Float64[],
        test_mse = Float64[],
        baseline_mse = Float64[],
        free_energy_first = Float64[],
        free_energy_last = Float64[],
        decreasing_steps = Int[],
        ct_states = Int[],
        softplus_states = Int[],
        weight_movements = String[],
        error = Union{Missing, String}[],
    )

    println("CT/MvSoftplus depth comparison: depths=$(join(depths, ',')), " *
            "width=$(config.d_hidden), n_train=$(nrow(train_data)), " *
            "n_test=$(nrow(test_data)), iterations=$(config.iterations), " *
            "prior_seed=$(config.prior_seed)")
    for depth in depths
        println("\n=== depth $depth")
        try
            arm = run_depth_arm(
                depth,
                config,
                train_data,
                test_data,
                train_features,
                test_features,
            )
            push!(rows, (
                depth = depth,
                width = config.d_hidden,
                prior_seed = config.prior_seed,
                status = "ok",
                seconds = arm.seconds,
                allocated_gib = arm.allocated_gib,
                train_mse = arm.train_mse,
                test_mse = arm.test_mse,
                baseline_mse = baseline,
                free_energy_first = arm.free_energy_first,
                free_energy_last = arm.free_energy_last,
                decreasing_steps = arm.decreasing_steps,
                ct_states = arm.ct_diagnostics.count,
                softplus_states = arm.sp_diagnostics.count,
                weight_movements = join(round.(arm.weight_movements; digits = 3), ";"),
                error = missing,
            ))
            println("train/test MSE        : ", round(arm.train_mse, digits = 4),
                    " / ", round(arm.test_mse, digits = 4))
            println("free energy           : ", round(arm.free_energy_first, digits = 3),
                    " -> ", round(arm.free_energy_last, digits = 3),
                    " ($(arm.decreasing_steps)/$(config.iterations - 1) decreases)")
            println("CT / softplus states  : ", arm.ct_diagnostics.count,
                    " / ", arm.sp_diagnostics.count)
            println("relative weight moves : ", round.(arm.weight_movements; digits = 3))
            println("time / allocations    : ", round(arm.seconds, digits = 2),
                    "s / ", round(arm.allocated_gib, digits = 2), " GiB")
        catch exception
            message = sprint(showerror, exception, catch_backtrace())
            @error "depth arm failed" depth exception
            push!(rows, (
                depth = depth,
                width = config.d_hidden,
                prior_seed = config.prior_seed,
                status = "failed",
                seconds = NaN,
                allocated_gib = NaN,
                train_mse = NaN,
                test_mse = NaN,
                baseline_mse = baseline,
                free_energy_first = NaN,
                free_energy_last = NaN,
                decreasing_steps = 0,
                ct_states = 0,
                softplus_states = 0,
                weight_movements = "",
                error = message,
            ))
        end
    end

    println("\n=== depth summary")
    show(rows; allrows = true, allcols = true, truncate = 80)
    println()
    if config.save_outputs
        paths = save_depth_comparison(rows, config)
        println("results CSV : ", paths.csv)
        println("MSE plot    : ", paths.plot)
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    depth_results = run_depth_comparison()
end
