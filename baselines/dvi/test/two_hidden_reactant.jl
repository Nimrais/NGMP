# Optional backend parity check: julia --project=baselines/dvi baselines/dvi/test/two_hidden_reactant.jl
using DVIUCI, StableRNGs, Test

@testset "two-hidden-layer Reactant CPU parity" begin
    rng = StableRNG(501)
    features = randn(rng, 30, 3)
    targets = randn(rng, 30)
    spec = uci_regression_split(30, 1; base_seed = 502, test_fraction = 0.2)
    split = prepare_uci_regression_partition(features, targets, spec.train_indices, spec.test_indices)
    settings = (hidden_layers = 2, hidden_units = 4, propagation = "diagonal",
        initialization_scale = 0.1, batch_size = 8, kl_warmup_steps = 0,
        kl_anneal_steps = 0, show_progress = false, make_plot = false)
    cpu = DVIConfig(; settings..., execution_backend = "zygote")
    compiled = DVIConfig(; settings..., execution_backend = "reactant", execution_device = "cpu")
    reference = refit_model(split, 3, "heteroscedastic", cpu; seed = 503, epochs = 2)
    observed = refit_model(split, 3, "heteroscedastic", compiled; seed = 503, epochs = 2)
    @test observed.losses ≈ reference.losses rtol = 1e-4 atol = 1e-5
    for name in keys(reference.params), field in (:weight_mu, :weight_log_std, :bias_mu, :bias_log_std)
        @test getproperty(getproperty(observed.params, name), field) ≈
            getproperty(getproperty(reference.params, name), field) rtol = 1e-4 atol = 1e-5
    end
end
