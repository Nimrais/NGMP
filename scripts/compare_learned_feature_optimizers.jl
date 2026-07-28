#!/usr/bin/env julia

using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const NOTEBOOK = joinpath(ROOT, "notebooks", "why_hierarchy_learned_features.jl")
const METHODS = [
    (:damped, 0.0),
    (:vector_transport, 0.05),
    (:vector_transport, 0.10),
    (:vector_transport, 0.20),
    (:vector_transport_nesterov, 0.05),
    (:vector_transport_nesterov, 0.10),
    (:vector_transport_nesterov, 0.20),
]

function experiment_source(method, beta)
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
        "const METHOD = :damped" => "const METHOD = :$method",
        "alpha = alpha, beta = 0.0, max_step = MAX_STEP, method = METHOD" =>
            "alpha = alpha, beta = $beta, max_step = MAX_STEP, method = METHOD",
    )
    return source
end

function run_arm(method, beta)
    return mktempdir() do directory
        path = joinpath(directory, "learned_feature_arm.jl")
        write(path, experiment_source(method, beta))
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
            r"test logpdf (-?\d+\.\d+) \| observed RMSE (\d+\.\d+) \| latent-mean RMSE (\d+\.\d+) \| noise corr (-?\d+\.\d+) \| aleatoric (\d+\.\d+) \(truth (\d+\.\d+)\) \| epistemic (\d+\.\d+)",
            text,
        )
        if success(process) && !isnothing(matched)
            numbers = parse.(Float64, matched.captures)
            return (;
                method, beta, status = "ok", logpdf = numbers[1],
                rmse = numbers[2], latent_rmse = numbers[3],
                noise_corr = numbers[4], aleatoric = numbers[5],
                truth_aleatoric = numbers[6], epistemic = numbers[7], error = "",
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
            method, beta, status = "unstable", logpdf = NaN, rmse = NaN,
            latent_rmse = NaN, noise_corr = NaN, aleatoric = NaN,
            truth_aleatoric = NaN, epistemic = NaN, error = message,
        )
    end
end

rows = map(METHODS) do (method, beta)
    @printf(stderr, "running %-27s beta %.2f\n", method, beta)
    run_arm(method, beta)
end

results_dir = joinpath(ROOT, "results")
mkpath(results_dir)
csv_path = joinpath(results_dir, "learned_feature_optimizers.csv")
open(csv_path, "w") do io
    println(
        io,
        "method,beta,status,logpdf,rmse,latent_rmse,noise_corr,aleatoric,truth_aleatoric,epistemic,error",
    )
    for row in rows
        println(
            io,
            join((
                row.method, row.beta, row.status, row.logpdf, row.rmse,
                row.latent_rmse, row.noise_corr, row.aleatoric,
                row.truth_aleatoric, row.epistemic, row.error,
            ), ','),
        )
    end
end

println("| optimizer | beta | status | logpdf | RMSE | latent RMSE | noise corr |")
println("|---|---:|---|---:|---:|---:|---:|")
for row in rows
    values = row.status == "ok" ?
        @sprintf(
            "%.4f | %.4f | %.4f | %.4f",
            row.logpdf, row.rmse, row.latent_rmse, row.noise_corr,
        ) : "— | — | — | —"
    println("| $(row.method) | $(row.beta) | $(row.status) | $values |")
end
println("\nCSV: $csv_path")
