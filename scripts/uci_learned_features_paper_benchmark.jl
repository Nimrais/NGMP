#!/usr/bin/env julia

# Standalone UCI wrapper around the learned-feature model. Model definitions
# are loaded from their Pluto cells without executing the notebook experiment.
include(joinpath(@__DIR__, "uci_paper_protocol.jl"))
using .UCIPaperProtocol
using LinearAlgebra, Printf, Random, RxInfer, Statistics, SurrogateModelling
import ProbabilisticEnsembling: Exp

const ROOT = dirname(@__DIR__)
const NOTEBOOK = joinpath(ROOT, "notebooks", "why_hierarchy_learned_features.jl")
const N_SPLITS = parse(Int, get(ENV, "UCI_SPLITS", "20"))
const OPTIMIZERS = [
    (:damped, 0.0),
    [(:vector_transport, beta) for beta in (0.05, 0.10, 0.20, 0.50, 0.80)]...,
    [(:vector_transport_nesterov, beta)
     for beta in (0.05, 0.10, 0.20, 0.50, 0.80)]...,
]

function notebook_cell(id)
    source = read(NOTEBOOK, String)
    marker = "# ╔═╡ $id"
    start = findfirst(marker, source)
    isnothing(start) && error("notebook cell $id not found")
    body_start = nextind(source, last(start))
    next_marker = findnext("\n# ╔═╡ ", source, body_start)
    stop = isnothing(next_marker) ? lastindex(source) : first(next_marker) - 1
    source[body_start:stop]
end

# Configuration, dependencies, factor graph, constraints, moment propagation,
# and initialization are the canonical notebook implementation.
for id in (
    "f45fa3c6-5df6-4983-ae20-2d44185b3ce1",
    "cc31c16e-7532-44f8-aa58-39b78f2e1bcd",
    "d374accc-1151-45d6-b8c2-bb62320b894d",
    "961232fc-5814-4764-915e-fe4e793f703e",
    "3d2a1af1-950c-4634-9a7b-b0638b032f92",
    "87b3a45a-7483-466f-a2c4-e7cf7399f5cc",
)
    code = notebook_cell(id)
    if id == "f45fa3c6-5df6-4983-ae20-2d44185b3ce1"
        neurons = parse(Int, get(ENV, "UCI_NEURONS", "16"))
        iterations = parse(Int, get(ENV, "UCI_ITERATIONS", "60"))
        code = replace(
            code,
            "const N_NEURONS = 16" => "const N_NEURONS = $neurons",
            "const MAX_BATCH_ITERATIONS = 60" =>
                "const MAX_BATCH_ITERATIONS = $iterations",
        )
    end
    Base.include_string(Main, code, NOTEBOOK)
end

gaussian(m, V) = MvNormalMeanCovariance(collect(m), Matrix(V))

function uci_priors(X, y, seed)
    rng = MersenneTwister(seed); p = size(X, 2) + 1
    weights(scale, frequency) = map(1:N_NEURONS) do k
        direction = randn(rng, p - 1); direction ./= norm(direction)
        m = vcat(isodd(k) ? pi / 4 : -pi / 4, frequency .* direction)
        gaussian(m, Diagonal(fill(scale^2, p)))
    end
    (; w = weights(0.35, 1.5), noise_w = weights(0.45, 0.5),
       v = gaussian([isodd(k) ? .2 : -.2 for k in 1:N_NEURONS],
                    Diagonal(fill(1 / N_NEURONS, N_NEURONS))),
       g = gaussian([isodd(k) ? .04 : -.04 for k in 1:N_NEURONS],
                    Diagonal(fill(.25 / N_NEURONS, N_NEURONS))),
       anchor = -log(max(mean(abs2, diff(sort(y))) / 2, 1e-8)))
end

raw(X) = [vcat(1.0, collect(row)) for row in eachrow(X)]

uci_activation_dependencies(method, beta) = NGMPDependencies(
    out = nothing, in = nothing;
    projection = TangentProjection(type = ClosedForm),
    damping = DampingMeta(
        alpha = ALPHA, beta = beta, max_step = MAX_STEP, method = method,
    ),
)

uci_exp_damping(method, beta) = DampingMeta(
    alpha = ALPHA, beta = beta, max_step = MAX_STEP, method = method,
)

function fit_model(X, y, seed, method, beta)
    features = raw(X); prior = uci_priors(X, y, seed)
    states = initial_marginals(prior, features, activation_meta())
    result = infer(model = learned_feature_hierarchy(
            n_neurons = N_NEURONS, priors = prior, activation = activation_meta(),
            activation_deps = uci_activation_dependencies(method, beta),
            link_deps = exp_dependencies(),
            link_meta = uci_exp_damping(method, beta)),
        data = (y = y, features = features),
        constraints = learned_feature_constraints(),
        initialization = learned_feature_initialization(prior, states),
        returnvars = (w = KeepLast(), noise_w = KeepLast(),
                      v = KeepLast(), g = KeepLast()),
        iterations = MAX_BATCH_ITERATIONS, free_energy = false,
        showprogress = false, options = (limit_stack_depth = 100,),
        disable_inference_error_hint = true)
    merge(prior, (; w = collect(vec(result.posteriors[:w])),
        noise_w = collect(vec(result.posteriors[:noise_w])),
        v = result.posteriors[:v], g = result.posteriors[:g]))
end

function predict_model(fit, X)
    activation = activation_meta()
    means = Float64[]; variances = Float64[]
    for row in eachrow(X)
        feature = vcat(1.0, collect(row))
        hm = Float64[]; hv = Float64[]; nm = Float64[]; nv = Float64[]
        for k in 1:N_NEURONS
            a, b = hidden_moments(fit.w[k], feature, activation); push!(hm,a); push!(hv,b)
            a, b = hidden_moments(fit.noise_w[k], feature, activation); push!(nm,a); push!(nv,b)
        end
        μ, epistemic = bilinear_moments(fit.v, hm, Matrix(Diagonal(hv)))
        score, scorevar = bilinear_moments(fit.g, nm, Matrix(Diagonal(nv)))
        score += fit.anchor; scorevar += inv(SCORE_CARRIER)
        push!(means, μ)
        push!(variances, max(epistemic, 0.0) + exp(clamp(-score + scorevar / 2, -40, 40)))
    end
    (; mean = means, variance = variances)
end

function summary_rows(rows, datasets)
    summaries = NamedTuple[]
    for dataset in datasets, (method, beta) in OPTIMIZERS
        selected = filter(
            r -> r.dataset == dataset && r.optimizer == method && r.beta == beta,
            rows,
        )
        summary = summarize_rows(selected)
        push!(summaries, (;
            dataset,
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
    path = joinpath(ROOT, "results", "uci_learned_features_paper.csv")
    summary_path = joinpath(
        ROOT, "results", "uci_learned_features_paper_summary.csv",
    )
    total_runs = length(datasets) * N_SPLITS * length(OPTIMIZERS)
    run_index = 0
    cached_splits = Dict(
        dataset => prepare_split.(
            paper_splits(dataset; count = N_SPLITS),
        )
        for dataset in datasets
    )
    for dataset in datasets, (method, beta) in OPTIMIZERS
        first_split_unstable = false
        for (split_position, p) in enumerate(cached_splits[dataset])
            split_id = p.split_id
            run_index += 1
            @printf(
                "[%d/%d] dataset=%s split=%d/%d optimizer=%s beta=%.2f\n",
                run_index, total_runs, dataset, split_id, N_SPLITS,
                method, beta,
            )
            flush(stdout)
            if split_position > 1 && first_split_unstable
                println("  skipped: split 1 was unstable for this configuration")
                push!(rows, (; dataset, split = split_id,
                    optimizer = method, beta, status = "skipped",
                    logpdf_standardized = NaN, logpdf = NaN, rmse = NaN,
                    paper_dvi = p.paper_dvi,
                    error = "skipped because split 1 was unstable"))
                write_results(path, rows)
                write_results(summary_path, summary_rows(rows, datasets))
                continue
            end
            try
                fit = fit_model(
                    p.x_train_std, p.y_train_std,
                    10_000 + split_id, method, beta,
                )
                prediction = predict_model(fit, p.x_test_std)
                metrics = gaussian_logpdf_metrics(
                    p.y_test_std, prediction.mean, prediction.variance, p.y_scale)
                push!(rows, (; dataset, split = split_id,
                    optimizer = method, beta, status = "ok", metrics...,
                    paper_dvi = p.paper_dvi, error = ""))
            catch error
                first_split_unstable = split_position == 1
                push!(rows, (; dataset, split = split_id,
                    optimizer = method, beta, status = "unstable",
                    logpdf_standardized = NaN, logpdf = NaN, rmse = NaN,
                    paper_dvi = p.paper_dvi,
                    error = replace(sprint(showerror, error), '\n' => ' ')))
            end
            write_results(path, rows)
            write_results(summary_path, summary_rows(rows, datasets))
        end
    end
    println("dataset optimizer beta successful mean_logpdf std_logpdf paper_DVI")
    for dataset in datasets, (method, beta) in OPTIMIZERS
        selected = filter(
            r -> r.dataset == dataset && r.optimizer == method && r.beta == beta,
            rows,
        )
        s = summarize_rows(selected)
        @printf("%-9s %-27s %4.2f %3d/%-3d %10.3f %10.3f %9.2f\n",
            dataset, method, beta, s.successful, N_SPLITS,
            s.mean_logpdf, s.std_logpdf,
            getproperty(DATASETS, dataset).paper_dvi)
    end
    println("CSV: $path")
    println("Summary CSV: $summary_path")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
