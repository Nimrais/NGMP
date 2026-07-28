#!/usr/bin/env julia

using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const NOTEBOOK = joinpath(ROOT, "notebooks", "why_hierarchy_deep_kernel.jl")
const METHODS = [
    (:damped, 0.0),
    (:vector_transport, 0.05),
    (:vector_transport, 0.10),
    (:vector_transport, 0.20),
    (:vector_transport_nesterov, 0.05),
    (:vector_transport_nesterov, 0.10),
    (:vector_transport_nesterov, 0.20),
]
const DEPTHS = 1:5

function experiment_source(method, beta, depth)
    source = read(NOTEBOOK, String)
    source = replace(
        source,
        "Pkg.activate(joinpath(@__DIR__, \"..\"))" =>
            "Pkg.activate($(repr(ROOT)))",
        "using StableRNGs" => """
        const _stable_rngs = Base.require(Base.PkgId(
            Base.UUID("860ef19b-820b-49d6-a774-d7a799459cd3"), "StableRNGs",
        ))
        const StableRNG = _stable_rngs.StableRNG
        """,
        "const LAYER_COUNTS = [1, 2, 3, 4, 5]" =>
            "const LAYER_COUNTS = [$depth]",
        "const METHOD = :damped" => "const METHOD = :$method",
        "DampingMeta(alpha = alpha, beta = 0.0, max_step = max_step, method = METHOD)" =>
            "DampingMeta(alpha = alpha, beta = $beta, max_step = max_step, method = METHOD)",
    )
    return source
end

function run_arm(method, beta, depth)
    return mktempdir() do directory
        path = joinpath(directory, "deep_kernel_arm.jl")
        write(path, experiment_source(method, beta, depth))
        output = IOBuffer()
        process = run(
            pipeline(
                ignorestatus(`$(Base.julia_cmd()) --project=$(ROOT) $path`),
                stdout = output,
                stderr = output,
            );
            wait = true,
        )
        text = String(take!(output))
        matched = match(
            r"L = \d+(?:  \(Gaussian process\)| layers)?\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)",
            text,
        )
        if success(process) && !isnothing(matched)
            return (;
                method, beta, depth, status = "ok",
                logpdf = parse(Float64, matched[1]),
                rmse = parse(Float64, matched[2]),
                noise_corr = parse(Float64, matched[3]),
                error = "",
            )
        end
        error_line = something(
            findfirst(line -> occursin("ERROR:", line), split(text, '\n')),
            0,
        )
        lines = split(text, '\n')
        message = error_line == 0 ? "run failed" :
            replace(lines[error_line], ',' => ';')
        return (;
            method, beta, depth, status = "unstable",
            logpdf = NaN, rmse = NaN, noise_corr = NaN, error = message,
        )
    end
end

rows = map(Iterators.product(METHODS, DEPTHS)) do ((method, beta), depth)
    @printf(stderr, "running %-27s beta %.2f L=%d\n", method, beta, depth)
    run_arm(method, beta, depth)
end

results_dir = joinpath(ROOT, "results")
mkpath(results_dir)
csv_path = joinpath(results_dir, "deep_kernel_optimizer_depth.csv")
open(csv_path, "w") do io
    println(io, "method,beta,depth,status,logpdf,rmse,noise_corr,error")
    for row in rows
        println(
            io,
            join((
                row.method, row.beta, row.depth, row.status,
                row.logpdf, row.rmse, row.noise_corr, row.error,
            ), ','),
        )
    end
end

println("| optimizer | beta | depth | status | logpdf | RMSE | noise corr |")
println("|---|---:|---:|---|---:|---:|---:|")
for row in rows
    values = row.status == "ok" ?
        @sprintf("%.4f | %.4f | %.3f", row.logpdf, row.rmse, row.noise_corr) :
        "— | — | —"
    println("| $(row.method) | $(row.beta) | $(row.depth) | $(row.status) | $values |")
end
println("\nCSV: $csv_path")
