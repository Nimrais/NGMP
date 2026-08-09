using Test

const STUDY_ROOT = normpath(joinpath(@__DIR__, ".."))
const PROJECT_ROOT = dirname(STUDY_ROOT)

@testset "when_ngmp_helps smoke reproduction" begin
    mktempdir() do output
        environment = copy(ENV)
        environment["WHEN_NGMP_SMOKE"] = "true"
        environment["WHEN_NGMP_OUTPUT_DIR"] = output
        environment["GKSwstype"] = "100"
        command = setenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$(PROJECT_ROOT) $(joinpath(STUDY_ROOT, "run_all.jl"))`,
            environment,
        )
        @test success(command)

        for stem in (
            "poisson_state_space",
            "poisson_state_space_free_energy",
            "gaussian_state_space",
        )
            for extension in ("pdf", "png")
                path = joinpath(output, "figures", "$stem.$extension")
                @test isfile(path)
                @test filesize(path) > 0
            end
            if stem != "poisson_state_space_free_energy"
                csv_path = joinpath(output, "results", "$(stem)_runs.csv")
                @test isfile(csv_path)
                @test filesize(csv_path) > 0
            end
        end
        normal_runs = joinpath(
            output,
            "results",
            "normal_mean_precision_runs.csv",
        )
        @test isfile(normal_runs)
        @test filesize(normal_runs) > 0
        for filename in (
            "poisson_state_space_metrics.csv",
            "poisson_state_space_metrics_table.csv",
            "poisson_state_space_seed_metrics.csv",
            "poisson_state_space_free_energy.csv",
            "poisson_state_space_metrics_table.tex",
        )
            path = joinpath(output, "results", filename)
            @test isfile(path)
            @test filesize(path) > 0
        end
        for stem in (
            "normal_kl_state",
            "normal_kl_precision",
            "poisson_bethe_heldout_5",
            "poisson_bethe_heldout_10",
            "poisson_bethe_heldout_20",
            "poisson_bethe_heldout_50",
            "gaussian_precision_posterior",
            "gaussian_state_posterior",
            "gaussian_precision_calibration",
            "gaussian_local_uncertainty",
        )
            path = joinpath(output, "figures", "$stem.pdf")
            @test isfile(path)
            @test filesize(path) > 0
        end
    end
end
