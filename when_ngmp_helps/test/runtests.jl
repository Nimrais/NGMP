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
            "poisson_outlier_5pct",
            "poisson_gap50",
            "poisson_depth_profile",
            "hetero_hierarchy_aleatoric",
            "hetero_hierarchy_epistemic",
            "streaming_hetero_predictions",
            "streaming_hetero_collapse",
        )
            for extension in ("pdf", "png")
                path = joinpath(output, "figures", "$stem.$extension")
                @test isfile(path)
                @test filesize(path) > 0
            end
        end
        for filename in (
            "poisson_state_space_runs.csv",
            "hetero_hierarchy_runs.csv",
            "hetero_hierarchy_aggregate.csv",
            "hetero_hierarchy_free_energy.csv",
            "streaming_hetero_runs.csv",
            "streaming_hetero_track.csv",
        )
            csv_path = joinpath(output, "results", filename)
            @test isfile(csv_path)
            @test filesize(csv_path) > 0
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
            "poisson_outlier_vmp",
            "poisson_outlier_ngmp",
            "poisson_gap50_vmp",
            "poisson_gap50_ngmp",
            "poisson_depth_nll",
            "poisson_depth_variance",
            "hetero_aleatoric_vmp_fit",
            "hetero_aleatoric_ngmp_fit",
            "hetero_aleatoric_vmp_variance",
            "hetero_aleatoric_ngmp_variance",
            "hetero_aleatoric_free_energy",
            "hetero_epistemic_vmp_fit",
            "hetero_epistemic_ngmp_fit",
            "hetero_epistemic_sd",
            "hetero_epistemic_free_energy",
            "streaming_vmp_full",
            "streaming_vmp_sequential",
            "streaming_cavity_full",
            "streaming_cavity_sequential",
            "streaming_collapse",
        )
            path = joinpath(output, "figures", "$stem.pdf")
            @test isfile(path)
            @test filesize(path) > 0
        end
    end
end
