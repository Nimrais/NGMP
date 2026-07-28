#!/usr/bin/env julia

# UCI protocol from Wu et al. (ICLR 2019): 20 random 90/10 splits and mean
# test log predictive density in original target units.
include(joinpath(@__DIR__, "uci_paper_protocol.jl"))
using .UCIPaperProtocol
using Distributions: Chisq, InverseGamma
using LinearAlgebra, Printf, Random, RxInfer, Statistics, SurrogateModelling
import ProbabilisticEnsembling: Exp

const N_RFF = parse(Int, get(ENV, "UCI_RFFS", "50"))
const DEPTHS = parse.(Int, split(get(ENV, "UCI_DEPTHS", "1,2,3,4,5"), ','))
const N_SPLITS = parse(Int, get(ENV, "UCI_SPLITS", "20"))
const ITERATIONS = parse(Int, get(ENV, "UCI_ITERATIONS", "240"))
const PREDICT_ITERATIONS = parse(Int, get(ENV, "UCI_PREDICT_ITERATIONS", "60"))
const FEATURE_MAPS = Symbol.(split(
    get(ENV, "UCI_FEATURE_MAPS", "rbf"), ',',
))
const ALPHA, MAX_STEP, TOP_CARRIER = 0.6, 0.5, 25.0
const OPTIMIZERS = [
    (:damped, 0.0),
    [(:vector_transport, beta) for beta in (0.05, 0.10, 0.20, 0.50, 0.80)]...,
    [(:vector_transport_nesterov, beta)
     for beta in (0.05, 0.10, 0.20, 0.50, 0.80)]...,
]

@model function uci_gp(y, features, v_prior, noise_prior)
    v ~ v_prior; γ ~ noise_prior
    for o in eachindex(features); y[o] ~ softdot(features[o], v, γ); end
end

@model function uci_hierarchy(y, features, levels, v_prior, w_priors, deps, damping)
    local w, score, precision
    v ~ v_prior
    for k in 1:levels; w[k] ~ w_priors[k]; end
    for o in eachindex(features)
        score[levels, o] ~ softdot(features[o], w[levels], TOP_CARRIER)
        precision[levels, o] ~ Exp(score[levels, o]) where {
            dependencies = deps, meta = damping,
        }
        for k in (levels - 1):-1:1
            score[k, o] ~ softdot(features[o], w[k], precision[k + 1, o])
            precision[k, o] ~ Exp(score[k, o]) where {
                dependencies = deps, meta = damping,
            }
        end
        y[o] ~ softdot(features[o], v, precision[1, o])
    end
end

@constraints function gp_constraints()
    q(v, γ, y) = q(v, y)q(γ); q(v)::MomentForm()
end
@constraints function hierarchy_constraints()
    q(v, w, score, precision, y) = q(v, y)q(w, score)q(precision)
    q(v)::MomentForm(); q(w)::MomentForm()
end

gaussian(m, variances) =
    MvNormalMeanCovariance(collect(m), Matrix(Diagonal(collect(variances))))

function rff_design(x_train, x_test, seed, mode = :rbf)
    mode in (:rbf, :multiscale_rbf_linear, :multiscale_matern32_linear) ||
        throw(ArgumentError("unknown UCI feature map: $mode"))
    d = size(x_train, 2)
    rng = MersenneTwister(seed)
    # Median-distance heuristic on at most 512 training pairs.
    sample = x_train[rand(rng, 1:size(x_train, 1), min(512, size(x_train, 1))), :]
    distances = Float64[]
    for i in 2:size(sample, 1); push!(distances, norm(sample[i, :] - sample[i - 1, :])); end
    lengthscale = max(median(distances), 0.25)
    if mode == :rbf
        frequencies = randn(rng, N_RFF, d) ./ lengthscale
    else
        counts = [div(N_RFF, 3), div(N_RFF, 3),
                  N_RFF - 2 * div(N_RFF, 3)]
        frequencies = reduce(vcat, map(zip(counts, (0.5, 1.0, 2.0))) do (count, scale)
            base = randn(rng, count, d)
            if mode == :multiscale_matern32_linear
                # Matérn-3/2 spectral density is a multivariate Student-t
                # distribution with three degrees of freedom.
                base .*= reshape(
                    sqrt.(3 ./ rand(rng, Chisq(3), count)), :, 1,
                )
            end
            base ./ (scale * lengthscale)
        end)
    end
    phases = 2pi .* rand(rng, size(frequencies, 1))
    function transform(X)
        random = sqrt(2 / size(frequencies, 1)) .*
            cos.(X * frequencies' .+ phases')
        return mode == :rbf ?
            hcat(random, ones(size(X, 1))) :
            hcat(random, X, ones(size(X, 1)))
    end
    return transform(x_train), transform(x_test)
end

function priors(depth, p, y)
    anchor = -log(max(mean(abs2, diff(sort(y))) / 2, 1e-8))
    level(k) = gaussian(
        vcat(zeros(p - 1), k == 1 ? anchor : log(TOP_CARRIER)),
        vcat(fill(0.4^2, p - 1), 1.0),
    )
    return (; v = gaussian(zeros(p), ones(p)),
              w = [level(k) for k in 1:(depth - 1)],
              noise = GammaShapeRate(2.0, 2.0 / TOP_CARRIER))
end

function initial_states(depth, prior, rows)
    Φ = reduce(hcat, rows)'
    score = Matrix{NormalMeanVariance{Float64}}(undef, depth - 1, length(rows))
    precision = Matrix{GammaShapeRate{Float64}}(undef, depth - 1, length(rows))
    for k in 1:(depth - 1)
        m, V = mean_cov(prior.w[k]); means = Φ * m
        vars = vec(sum((Φ * V) .* Φ; dims = 2)) .+ inv(TOP_CARRIER)
        shapes = 1 .+ inv.(max.(vars, 1e-6))
        score[k, :] = NormalMeanVariance.(means, vars)
        precision[k, :] = GammaShapeRate.(shapes, shapes .* exp.(-means))
    end
    (; score, precision)
end

deps() = NGMPDependencies(out = nothing, in = nothing;
    projection = TangentProjection(type = ClosedForm))
damping(method, beta, alpha = ALPHA) =
    DampingMeta(alpha = alpha, beta = beta, max_step = MAX_STEP, method = method)

function model(depth, prior, method, beta, alpha)
    depth == 1 && return uci_gp(v_prior = prior.v, noise_prior = prior.noise)
    uci_hierarchy(levels = depth - 1, v_prior = prior.v, w_priors = prior.w,
                  deps = deps(), damping = damping(method, beta, alpha))
end

function fit_model(depth, Φ, y, method, beta)
    rows = collect(eachrow(Φ)); prior = priors(depth, size(Φ, 2), y)
    states = depth == 1 ? nothing : initial_states(depth, prior, rows)
    init = depth == 1 ? @initialization(begin
        q(v) = deepcopy(prior.v); q(γ) = deepcopy(prior.noise)
    end) : @initialization(begin
        q(v) = deepcopy(prior.v); q(w) = deepcopy(prior.w)
        q(score) = states.score; q(precision) = states.precision
    end)
    result = infer(model = model(depth, prior, method, beta, ALPHA),
        data = (y = y, features = rows),
        constraints = depth == 1 ? gp_constraints() : hierarchy_constraints(),
        initialization = init,
        returnvars = depth == 1 ? (v = KeepLast(), γ = KeepLast()) :
                                  (v = KeepLast(), w = KeepLast()),
        iterations = ITERATIONS, free_energy = false, showprogress = false,
        options = (limit_stack_depth = 100,), disable_inference_error_hint = true)
    (; depth, method, beta, v = result.posteriors[:v],
       w = depth == 1 ? [] : collect(vec(result.posteriors[:w])),
       noise = depth == 1 ? result.posteriors[:γ] : prior.noise)
end

function predict_model(fit, Φ)
    rows = collect(eachrow(Φ)); prior = (; v = fit.v, w = fit.w, noise = fit.noise)
    states = fit.depth == 1 ? nothing : initial_states(fit.depth, prior, rows)
    init = fit.depth == 1 ? @initialization(begin
        q(v) = deepcopy(prior.v); q(γ) = deepcopy(prior.noise)
    end) : @initialization(begin
        q(v) = deepcopy(prior.v); q(w) = deepcopy(prior.w)
        q(score) = states.score; q(precision) = states.precision
    end)
    result = infer(
        model = model(fit.depth, prior, fit.method, fit.beta, 0.5),
        data = (features = rows,),
        constraints = fit.depth == 1 ? gp_constraints() : hierarchy_constraints(),
        initialization = init, predictvars = (y = KeepLast(),),
        returnvars = fit.depth == 1 ? (γ = KeepLast(),) : (precision = KeepLast(),),
        iterations = PREDICT_ITERATIONS, free_energy = false, showprogress = false,
        options = (limit_stack_depth = 100,), disable_inference_error_hint = true)
    marginals = collect(vec(result.predictions[:y]))
    (; mean = mean.(marginals), variance = var.(marginals))
end

function summary_rows(rows, datasets)
    summaries = NamedTuple[]
    for dataset in datasets, feature_map in FEATURE_MAPS, depth in DEPTHS,
        (method, beta) in OPTIMIZERS
        selected = filter(
            r -> r.dataset == dataset && r.feature_map == feature_map &&
                 r.depth == depth &&
                 r.optimizer == method && r.beta == beta,
            rows,
        )
        summary = summarize_rows(selected)
        push!(summaries, (;
            dataset,
            feature_map,
            depth,
            optimizer = method,
            beta,
            successful_splits = summary.successful,
            requested_splits = N_SPLITS,
            mean_logpdf = summary.mean_logpdf,
            std_logpdf = summary.std_logpdf,
            mean_rmse = summary.mean_rmse,
            paper_dvi = getproperty(DATASETS, dataset).paper_dvi,
        ))
    end
    return summaries
end

function main()
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    datasets = Symbol.(split(get(ENV, "UCI_DATASETS", "yacht,energy,concrete"), ','))
    rows = NamedTuple[]
    path = joinpath(dirname(@__DIR__), "results",
        "uci_deep_kernel_paper.csv")
    summary_path = joinpath(dirname(@__DIR__), "results",
        "uci_deep_kernel_paper_summary.csv")
    total_runs = length(datasets) * length(FEATURE_MAPS) * N_SPLITS *
        length(DEPTHS) * length(OPTIMIZERS)
    run_index = 0
    cached_splits = Dict{Tuple{Symbol, Symbol}, Any}()
    for dataset in datasets, feature_map in FEATURE_MAPS
        cached_splits[(dataset, feature_map)] = map(
            paper_splits(dataset; count = N_SPLITS),
        ) do split
            prepared = prepare_split(split)
            Φtrain, Φtest = rff_design(
                prepared.x_train_std,
                prepared.x_test_std,
                10_000 * split.split_id,
                feature_map,
            )
            (; split, prepared, Φtrain, Φtest)
        end
    end
    for dataset in datasets, feature_map in FEATURE_MAPS, depth in DEPTHS,
        (method, beta) in OPTIMIZERS
        first_split_unstable = false
        for (split_position, cached) in
            enumerate(cached_splits[(dataset, feature_map)])
            split = cached.split
            prepared = cached.prepared
            run_index += 1
            @printf(
                "[%d/%d] dataset=%s features=%s split=%d/%d depth=%d optimizer=%s beta=%.2f\n",
                run_index, total_runs, dataset, feature_map, split.split_id,
                N_SPLITS, depth, method, beta,
            )
            flush(stdout)
            if split_position > 1 && first_split_unstable
                println("  skipped: split 1 was unstable for this configuration")
                push!(rows, (; dataset, feature_map, split = split.split_id,
                    depth, optimizer = method, beta, status = "skipped",
                    logpdf_standardized = NaN, logpdf = NaN, rmse = NaN,
                    paper_dvi = split.paper_dvi,
                    error = "skipped because split 1 was unstable"))
                write_results(path, rows)
                write_results(summary_path, summary_rows(rows, datasets))
                continue
            end
            try
                fit = fit_model(
                    depth, cached.Φtrain, prepared.y_train_std, method, beta,
                )
                prediction = predict_model(fit, cached.Φtest)
                metrics = gaussian_logpdf_metrics(
                    prepared.y_test_std, prediction.mean, prediction.variance,
                    prepared.y_scale)
                push!(rows, (; dataset, feature_map, split = split.split_id, depth,
                    optimizer = method, beta, status = "ok", metrics...,
                    paper_dvi = split.paper_dvi, error = ""))
            catch error
                first_split_unstable = split_position == 1
                push!(rows, (; dataset, feature_map, split = split.split_id, depth,
                    optimizer = method, beta, status = "unstable",
                    logpdf_standardized = NaN, logpdf = NaN, rmse = NaN,
                    paper_dvi = split.paper_dvi,
                    error = replace(sprint(showerror, error), '\n' => ' ')))
            end
            write_results(path, rows)
            write_results(summary_path, summary_rows(rows, datasets))
        end
    end
    println("dataset feature_map depth optimizer beta successful mean_logpdf std_logpdf paper_DVI")
    for dataset in datasets, feature_map in FEATURE_MAPS, depth in DEPTHS,
        (method, beta) in OPTIMIZERS
        selected = filter(
            r -> r.dataset == dataset && r.feature_map == feature_map &&
                 r.depth == depth &&
                 r.optimizer == method && r.beta == beta,
            rows,
        )
        summary = summarize_rows(selected)
        @printf("%-9s %-28s %5d %-27s %4.2f %3d/%-3d %10.3f %10.3f %9.2f\n",
            dataset, feature_map, depth, method, beta,
            summary.successful, N_SPLITS,
            summary.mean_logpdf, summary.std_logpdf,
            getproperty(DATASETS, dataset).paper_dvi)
    end
    println("CSV: $path")
    println("Summary CSV: $summary_path")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
