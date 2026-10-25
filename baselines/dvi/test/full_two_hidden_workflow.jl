using Test, TOML
module CampaignTests
include("../full_two_hidden_experiment.jl")
end
const C = CampaignTests
@testset "DVI big campaign protocol and saved configuration" begin
    mktempdir() do directory
        config = C.config_at(directory; execution_backend = "zygote", execution_device = "cpu")
        @test config.propagation == "full" && config.hidden_layers == 2
        @test config.full_selection_patience_steps == 50000
        @test config.selection_max_optimizer_steps == config.refit_max_optimizer_steps == 0
        @test config.split_ids == collect(1:20)
        @test config.kl_warmup_steps == 14000 && config.kl_anneal_steps == 1000
        @test length(config.datasets) == 6
        C.DVIUCI.save_config(config)
        @test_throws ArgumentError C.DVIUCI.validate_resume_config(joinpath(directory, "config.toml"),
            C.config_at(directory; execution_backend = "reactant", execution_device = "gpu"))
        @test_throws ErrorException C.check_complete(config)
    end
    # CSV.jl may return InlineString dataset keys; checkpoint validation accepts them.
    pilot_directory = joinpath(C.ROOT, "pilots", "reactant_gpu", "yacht")
    if isfile(joinpath(pilot_directory, "runs.csv"))
        saved = TOML.parsefile(joinpath(pilot_directory, "config.toml"))
        pilot_config = C.DVIConfig(; (Symbol(k) => v for (k, v) in saved)...)
        @test C.check_complete(pilot_config).status == ["success"]
    end
    @test length(C.source_fingerprint()) == 64
    @test length(C.Table.table_sources()) == 9
    @test length(C.Table.table_sources(include_dvi_big = true)) == 10
    @test_throws ErrorException C.Table.main(["--unknown-option"])
end
