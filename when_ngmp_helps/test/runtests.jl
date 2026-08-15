using Test

const STUDY_ROOT = normpath(joinpath(@__DIR__, ".."))
const PROJECT_ROOT = dirname(STUDY_ROOT)

# save_figure stems: pdf + png artifacts
const PREVIEW_STEMS = (
    "normal_kl_state",
    "normal_kl_precision",
    "poisson_state_space_free_energy",
    "poisson_damping_free_energy",
    "poisson_damping_synthetic_free_energy",
    "poisson_damping_synthetic_seed42",
    "poisson_gap50",
    "poisson_depth_profile",
    "streaming_hetero_predictions",
    "streaming_hetero_variances",
    "streaming_vmp_variance_full",
    "streaming_vmp_variance_sequential",
    "streaming_cavity_variance_full",
    "streaming_cavity_variance_sequential",
    "streaming_hetero_collapse",
    "streaming_hetero_free_energy",
)

# save_pdf stems: paper panels, pdf only
const PDF_STEMS = (
    "poisson_bethe_heldout_5",
    "poisson_bethe_heldout_10",
    "poisson_bethe_heldout_20",
    "poisson_bethe_heldout_50",
    "poisson_damping_heldout_5",
    "poisson_damping_heldout_10",
    "poisson_damping_heldout_20",
    "poisson_damping_heldout_50",
    "poisson_damping_synthetic_n20",
    "poisson_damping_synthetic_n40",
    "poisson_gap50_vmp",
    "poisson_gap50_ngmp",
    "poisson_depth_nll",
    "poisson_depth_variance",
    "streaming_vmp_full",
    "streaming_vmp_sequential",
    "streaming_cavity_full",
    "streaming_cavity_sequential",
    "streaming_collapse",
    "streaming_bethe_full",
    "streaming_bethe_sequential",
)

const RESULT_FILES = (
    "normal_mean_precision_runs.csv",
    "normal_mean_precision_aggregate.csv",
    "poisson_state_space_runs.csv",
    "poisson_state_space_holdout.csv",
    "poisson_state_space_seed_metrics.csv",
    "poisson_state_space_metrics.csv",
    "poisson_state_space_metrics_table.csv",
    "poisson_state_space_metrics_table.tex",
    "poisson_state_space_metrics_table_full.tex",
    "poisson_state_space_free_energy.csv",
    "poisson_damping_free_energy.csv",
    "poisson_damping_synthetic.csv",
    "poisson_damping_synthetic_fits.csv",
    "poisson_damping_config.toml",
    "poisson_damping_summary.md",
    "streaming_hetero_runs.csv",
    "streaming_hetero_track.csv",
    "streaming_hetero_panels.csv",
    "streaming_hetero_train.csv",
    "streaming_hetero_free_energy.csv",
)

function assert_artifacts(output)
    for stem in PREVIEW_STEMS, extension in ("pdf", "png")
        path = joinpath(output, "figures", "$stem.$extension")
        @test isfile(path)
        @test filesize(path) > 0
    end
    for stem in PDF_STEMS
        path = joinpath(output, "figures", "$stem.pdf")
        @test isfile(path)
        @test filesize(path) > 0
    end
    for filename in RESULT_FILES
        path = joinpath(output, "results", filename)
        @test isfile(path)
        @test filesize(path) > 0
    end
end

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
        assert_artifacts(output)

        # render-only pass: delete every figure, redraw the full set from the
        # persisted results/*.csv alone (no inference)
        for name in readdir(joinpath(output, "figures"))
            rm(joinpath(output, "figures", name))
        end
        render_command = setenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$(PROJECT_ROOT) $(joinpath(STUDY_ROOT, "render_all.jl"))`,
            environment,
        )
        @test success(render_command)
        assert_artifacts(output)
    end
end
