using Test, CSV, DataFrames, TOML, DVIUCI
module HomoCampaignTests
include("../homoscedastic_two_hidden_experiment.jl")
end
const HC = HomoCampaignTests
const HT = HC.HomoTable

@testset "homoscedastic big protocol and isolation" begin
    @test_throws ErrorException HC.method_propagation("unknown")
    for method in ("ddvi", "dvi")
        mktempdir() do directory
            config = HC.homo_config(method, directory)
            @test config.hidden_layers == 2 && config.hidden_units == 50
            @test config.likelihoods == ["homoscedastic"]
            @test config.homo_log_variance == 0
            @test config.selection_max_optimizer_steps == config.refit_max_optimizer_steps == 50000
            @test config.training_budget_protocol == "selection-refit-max-50000-v1"
            @test (config.kl_warmup_steps, config.kl_anneal_steps) == (14000, 1000)
            @test config.split_ids == collect(1:20)
            @test length(config.datasets) == 6
            for d in (4, 6, 8, 11, 13)
                params = initialize_model(d, "homoscedastic", config; seed = 701)
                @test size(params.output.weight_mu) == (1, 50)
                @test DVIUCI.model_coefficient_count(d, "homoscedastic", config) == 50d + 2651
            end
            DVIUCI.ensure_output_directories(config)
            DVIUCI.save_config(config)
            @test DVIUCI.validate_resume_config(joinpath(directory, "config.toml"), config) === nothing
            for overrides in ((likelihoods = ["heteroscedastic"],),
                (selection_max_optimizer_steps = 0, refit_max_optimizer_steps = 0, training_budget_protocol = "unlimited-v1"),
                (split_ids = [1, 3],))
                @test_throws ArgumentError DVIUCI.save_config(HC.homo_config(method, directory; overrides...))
            end
            other = method == "ddvi" ? "dvi" : "ddvi"
            @test_throws ArgumentError DVIUCI.save_config(HC.homo_config(other, directory))
            # A smoke-sized table can never be accepted as a complete campaign.
            CSV.write(joinpath(directory, "runs.csv"), DataFrame(dataset = ["yacht"], split = [1],
                likelihood = ["homoscedastic"], propagation = [config.propagation], status = ["success"]))
            @test_throws Exception HC.completed_homo_runs(config)
        end
    end
    @test HC.method_directory("ddvi") != HC.method_directory("dvi")
    @test length(HC.homo_source_fingerprint()) == 64
    @test_throws ErrorException HC.prepare_homo("ddvi", 3)
end

@testset "independent table variants and coefficient counts" begin
    for hetero in (false, true), diagonal in (false, true), full in (false, true)
        sources = HT.table_sources(include_dvi_big = hetero,
            include_ddvi_big_homo = diagonal, include_dvi_big_homo = full)
        @test length(sources) == 9 + hetero + diagonal + full
        @test count(s -> s.method == "dDVI-2HL-Homo", sources) == diagonal
        @test count(s -> s.method == "DVI-2HL-Homo", sources) == full
    end
    @test_throws ErrorException HT.main(["--unknown"])
    @test_throws ErrorException HT.main(["--include-dvi-big-homo", "--include-dvi-big-homo"])
    for method in ("ddvi", "dvi")
        mktempdir() do directory
            config = HC.homo_config(method, directory)
            DVIUCI.save_config(config)
            source = HT.ResultSource("test", "homoscedastic", config.propagation, directory; hidden_layers = 2)
            rows = DataFrame(dataset = ["yacht"], hidden_layers = [2], hidden_units = [50], model_coefficients = [2951])
            @test HT.validate_source_architecture(source, rows, rows) === nothing
            rows.model_coefficients .= 3002 # The heteroscedastic model has 51 extra coefficients.
            @test_throws ErrorException HT.validate_source_architecture(source, rows, rows)
            rows.model_coefficients .= 2951
            rows.hidden_layers .= 1
            @test_throws ErrorException HT.validate_source_architecture(source, rows, rows)
        end
    end
    # Existing results remain usable and requesting missing homoscedastic results fails.
    sources = HT.table_sources(include_dvi_big = true)
    HT.validate_split_manifests(sources)
    @test length(HT.load_results(sources)) == 60
    # Temporary table fixtures exercise all twelve variants together without training
    # or publishing pretend big-model measurements as experiment results.
    mktempdir() do directory
        dimensions = Dict("concrete" => 8, "energy" => 8, "housing" => 13,
            "power" => 4, "wine" => 11, "yacht" => 6)
        for method in ("ddvi", "dvi")
            destination = joinpath(directory, method)
            config = HC.homo_config(method, destination)
            DVIUCI.ensure_output_directories(config)
            DVIUCI.save_config(config)
            label = method == "ddvi" ? "dDVI-Homo" : "DVI-Homo"
            original = only(s.directory for s in HT.SOURCES if s.method == label)
            for filename in ("runs.csv", "summary.csv")
                fixture = CSV.read(joinpath(original, filename), DataFrame)
                fixture.hidden_layers = fill(2, nrow(fixture))
                fixture.hidden_units = fill(50, nrow(fixture))
                fixture.model_coefficients = [50 * dimensions[String(key)] + 2651 for key in fixture.dataset]
                CSV.write(joinpath(destination, filename), fixture)
            end
            cp(joinpath(original, "split_manifest.jld2"), joinpath(destination, "split_manifest.jld2"))
        end
        sources = HT.table_sources(include_dvi_big = true, include_ddvi_big_homo = true,
            include_dvi_big_homo = true, ddvi_big_homo_directory = joinpath(directory, "ddvi"),
            dvi_big_homo_directory = joinpath(directory, "dvi"))
        @test HT.validate_split_manifests(sources) === nothing
        results = HT.load_results(sources)
        @test length(results) == 72
        io = IOBuffer()
        HT.render_markdown(io, sources, results)
        rendered = String(take!(io))
        @test occursin("dDVI-2HL-Homo", rendered) && occursin("DVI-2HL-Homo", rendered)
        partial = joinpath(directory, "ddvi", "runs.csv")
        rows = CSV.read(partial, DataFrame)
        CSV.write(partial, rows[1:end-1, :])
        @test_throws ErrorException HT.load_results(sources)
    end
    mktempdir() do directory
        sources = HT.table_sources(include_ddvi_big_homo = true, ddvi_big_homo_directory = directory)
        @test_throws ErrorException HT.validate_split_manifests(sources)
        @test_throws Exception HT.load_results(sources)
    end
end
