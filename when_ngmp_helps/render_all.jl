#!/usr/bin/env julia

# Redraw every figure, table, and summary from the persisted results/*.csv
# without running any inference (~seconds per study). Method display names
# come from METHOD_LABELS in common.jl: edit them there and re-run this
# script to relabel every artifact.

const STUDY_ROOT = @__DIR__
const PROJECT_ROOT = dirname(STUDY_ROOT)
const STUDIES = (
    "normal_mean_precision.jl",
    "poisson_state_space.jl",
    "streaming_hetero.jl",
)

function main()
    for study in STUDIES
        path = joinpath(STUDY_ROOT, study)
        println("\n=== rendering $(splitext(study)[1]) ===")
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$(PROJECT_ROOT) $path`,
            "WHEN_NGMP_RENDER_ONLY" => "true",
        )
        run(command)
    end
    println("\nAll when-NGMP-helps figures re-rendered from results/.")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
