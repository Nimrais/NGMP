#!/usr/bin/env julia

const STUDY_ROOT = @__DIR__
const PROJECT_ROOT = dirname(STUDY_ROOT)
const STUDIES = (
    "normal_mean_precision.jl",
    "poisson_state_space.jl",
    "poisson_damping.jl",
    "streaming_hetero.jl",
)

function main()
    for study in STUDIES
        path = joinpath(STUDY_ROOT, study)
        println("\n=== running $(splitext(study)[1]) ===")
        command = `$(Base.julia_cmd()) --startup-file=no --project=$(PROJECT_ROOT) $path`
        run(command)
    end
    println("\nAll when-NGMP-helps studies completed.")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
