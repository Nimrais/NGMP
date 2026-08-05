#!/usr/bin/env julia

using BPCUCI

isempty(ARGS) && error("usage: smoke.jl OUTPUT_DIRECTORY [cpu|cuda]")
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
backend = length(ARGS) >= 2 ? lowercase(ARGS[2]) : "cpu"

config = BPCConfig(
    datasets = ["yacht"],
    n_splits = 1,
    hidden_units = 8,
    batch_size = 64,
    max_epochs = 2,
    min_epochs = 1,
    validation_every = 1,
    patience = 1,
    latent_steps = 2,
    eval_samples = 5,
    backend = backend,
    output_dir = abspath(first(ARGS)),
    resume = false,
    save_checkpoints = true,
    show_progress = false,
)

result = run_benchmark(config)
row = only(eachrow(result.runs))
row.status == "success" || error("BPC smoke benchmark failed: $(row.error)")

checkpoint_path = joinpath(
    config.output_dir,
    "checkpoints",
    "yacht_split01_homoscedastic.jld2",
)
checkpoint = load_posterior_checkpoint(checkpoint_path)
replayed = predict_holdout(checkpoint)
isfinite(replayed.metrics.lpd_original) || error("replayed LPD is not finite")
isfinite(replayed.metrics.rmse_original) || error("replayed RMSE is not finite")
isapprox(replayed.metrics.lpd_original, row.lpd_original; rtol = 1e-12) ||
    error("checkpoint replay LPD differs from the recorded run")
isapprox(replayed.metrics.rmse_original, row.rmse_original; rtol = 1e-12) ||
    error("checkpoint replay RMSE differs from the recorded run")

println(
    "BPC $backend smoke and checkpoint replay passed: " *
    "LPD=$(replayed.metrics.lpd_original), " *
    "RMSE=$(replayed.metrics.rmse_original)",
)
