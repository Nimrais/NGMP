# DVI_DEVICE=cpu|gpu julia --project=baselines/dvi baselines/dvi/test/full_two_hidden_reactant.jl
using DVIUCI, StableRNGs, Test, Zygote, Reactant, Enzyme, Optimisers
const RB = DVIUCI.load_reactant_backend()
const DEVICE = get(ENV, "DVI_DEVICE", "cpu")
Reactant.set_default_backend(DEVICE)

@testset "full two-hidden Reactant $DEVICE parity" begin
    rng = StableRNG(701)
    features, targets = randn(rng, Float32, 8, 3), randn(rng, Float32, 8)
    config = DVIConfig(hidden_layers = 2, hidden_units = 4, propagation = "full",
        initialization_scale = 0.1, batch_size = 8, kl_warmup_steps = 0,
        kl_anneal_steps = 0, show_progress = false, make_plot = false)
    params = initialize_model(3, "heteroscedastic", config; seed = 702)
    params.output.weight_mu[2, :] .= [0.1, -0.2, 0.3, 0.15]
    device_params = Reactant.to_rarray(params)
    x, y = Reactant.to_rarray(features), Reactant.to_rarray(targets)
    forward(p, f) = propagate_dvi(p, f, config)
    forward_kernel = Reactant.@compile sync = true forward(device_params, x)
    observed = RB.materialize(forward_kernel(device_params, x))
    expected = forward(params, features)
    @test maximum(abs, expected.covariance[:, 1, 2]) > 1e-5
    @test observed.mean ≈ expected.mean rtol = 2e-4 atol = 2e-6
    @test observed.covariance ≈ expected.covariance rtol = 2e-4 atol = 2e-6
    for weight in (0f0, 1f0)
        objective(p) = DVIUCI.dvi_loss_with_kl_weight(p, features, targets,
            "heteroscedastic", config, 8, weight)
        reference_loss, reference_gradient = Zygote.withgradient(objective, params)
        compiled_objective(p, f, t) = Enzyme.gradient(Enzyme.ReverseWithPrimal,
            q -> RB.reactant_dvi_loss(q, f, t, "heteroscedastic", config, 8, weight), p)
        kernel = Reactant.@compile sync = true compiled_objective(device_params, x, y)
        result = RB.materialize(kernel(device_params, x, y))
        @test RB.host_scalar(result.val) ≈ reference_loss rtol = 2e-4 atol = 2e-5
        for name in keys(params), field in keys(params[name])
            @test result.derivs[1][name][field] ≈ reference_gradient[1][name][field] rtol = 2e-3 atol = 2e-5
        end
    end
    # Same seeded minibatches and Adam steps on CPU and the compiled backend.
    compiled_config = DVIConfig(hidden_layers = 2, hidden_units = 4, propagation = "full",
        initialization_scale = 0.1, batch_size = 8, kl_warmup_steps = 0,
        kl_anneal_steps = 0, show_progress = false, make_plot = false,
        execution_backend = "reactant", execution_device = DEVICE)
    reference = DVIUCI.initialize_training_backend(params, features, targets, "heteroscedastic", config)
    compiled = DVIUCI.initialize_training_backend(params, features, targets, "heteroscedastic", compiled_config)
    for step in 1:3
        reference, loss, _ = DVIUCI.training_backend_epoch(reference, StableRNG(703 + step), step, step - 1;
            phase = "test", tracker = NumericalTracker())
        compiled, other, _ = DVIUCI.training_backend_epoch(compiled, StableRNG(703 + step), step, step - 1;
            phase = "test", tracker = NumericalTracker())
        @test loss ≈ other rtol = 2e-4 atol = 2e-5
    end
    actual = DVIUCI.training_backend_parameters(compiled)
    for name in keys(params), field in keys(params[name])
        @test actual[name][field] ≈ reference.params[name][field] rtol = 2e-3 atol = 2e-5
    end
end
