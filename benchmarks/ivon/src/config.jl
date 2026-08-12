const BENCHMARK_ROOT = normpath(joinpath(@__DIR__, ".."))
const REPOSITORY_ROOT = normpath(joinpath(BENCHMARK_ROOT, "..", ".."))
const RESULTS_ROOT = joinpath(BENCHMARK_ROOT, "results")
const CHECKPOINT_ROOT = joinpath(RESULTS_ROOT, "checkpoints")
const PILOT_ROOT = joinpath(RESULTS_ROOT, "pilot")
const SMOKE_ROOT = joinpath(RESULTS_ROOT, "smoke")

const PGE_COMMIT = "8cfdef63384d48ba302b3dd25ffb0ff356fecaea"
const IVON_COMMIT = "871dc5e0cc85a0ed00ea9fb4bad42fc1a52799ed"
const DEFAULT_SEED = 12345
const DATASETS = ("ETTh1", "ETTh2")
const HORIZONS = (96, 192, 336, 720)
const ARCHITECTURES = (:moe, :moe_big)
const EXPERT_NAMES = ("CNN", "NLinear", "LSTM", "DLinear", "NConv", "q10", "q90")
const LEARNING_RATES = (0.001, 0.01, 0.1)
const ESS_MULTIPLIERS = (1, 100)

const FIXED_IVON = (
    hess_init = 0.1,
    beta1 = 0.9,
    beta2 = 0.9999,
    weight_decay = 1.0e-4,
    mc_samples = 1,
    hess_approx = :price,
    rescale_lr = true,
    debias = true,
)

architecture_name(architecture::Symbol) =
    architecture === :moe ? "moe" : architecture === :moe_big ? "moe_big" :
    error("Unknown gate architecture: $architecture")

function deterministic_seed(parts...; base_seed::Int = DEFAULT_SEED)
    digest = sha256(join(string.(parts), "|"))
    offset = reinterpret(UInt32, digest[1:4])[1]
    return Int(mod(UInt64(base_seed) + UInt64(offset), UInt64(typemax(Int32)))) + 1
end

function effective_config(; phase, dataset, horizon, architecture, learning_rate,
    ess_multiplier, n_epochs = 100, posterior_samples = 1000,
    max_train_observations = nothing, max_eval_observations = nothing)
    seed = deterministic_seed(phase, dataset, horizon, architecture, learning_rate,
        ess_multiplier)
    return (
        benchmark_version = 1,
        phase = String(phase),
        dataset = String(dataset),
        horizon = Int(horizon),
        architecture = architecture_name(architecture),
        target = "OT",
        sequence_length = 96,
        split = (train = 0.6, validation = 0.2, test = 0.2),
        train_set = false,
        objective = "softmax_weighted_expert_sse",
        learning_rate = Float64(learning_rate),
        ess_multiplier = Int(ess_multiplier),
        hess_init = FIXED_IVON.hess_init,
        beta1 = FIXED_IVON.beta1,
        beta2 = FIXED_IVON.beta2,
        weight_decay = FIXED_IVON.weight_decay,
        mc_samples = FIXED_IVON.mc_samples,
        hess_approx = String(FIXED_IVON.hess_approx),
        rescale_lr = FIXED_IVON.rescale_lr,
        debias = FIXED_IVON.debias,
        n_epochs = Int(n_epochs),
        patience = 1,
        min_delta = 1.0e-3,
        posterior_samples = Int(posterior_samples),
        max_train_observations = max_train_observations,
        max_eval_observations = max_eval_observations,
        base_seed = DEFAULT_SEED,
        training_seed = seed,
        prediction_seed = deterministic_seed("prediction", seed),
        interval_seed = deterministic_seed("interval", seed),
        pge_commit = PGE_COMMIT,
        ivonrepro_commit = IVON_COMMIT,
    )
end

config_fingerprint(config) = bytes2hex(sha256(repr(config)))

function manifest_commit(package_name::String)
    manifest = TOML.parsefile(joinpath(BENCHMARK_ROOT, "Manifest.toml"))
    entries = manifest["deps"][package_name]
    entry = entries isa Vector ? only(entries) : entries
    return get(entry, "repo-rev", "")
end

function verify_dependency_pins()
    manifest_commit("ProbabilisticEnsembling") == PGE_COMMIT ||
        error("Manifest does not pin PrecisionGatedExperts commit $PGE_COMMIT")
    manifest = TOML.parsefile(joinpath(BENCHMARK_ROOT, "Manifest.toml"))
    ivon_entries = manifest["deps"]["IVONRepro"]
    ivon_entry = ivon_entries isa Vector ? only(ivon_entries) : ivon_entries
    get(ivon_entry, "path", "") == "vendor/IVONRepro" ||
        error("Manifest does not use the benchmark-local IVONRepro snapshot")
    provenance_path = joinpath(BENCHMARK_ROOT, "vendor", "IVONRepro", "UPSTREAM.toml")
    provenance = TOML.parsefile(provenance_path)
    provenance["commit"] == IVON_COMMIT ||
        error("Vendored IVONRepro does not pin commit $IVON_COMMIT")
    vendor_root = dirname(provenance_path)
    for (relative, expected) in provenance["sha256"]
        actual = file_sha256(joinpath(vendor_root, relative))
        actual == expected || error("Vendored IVONRepro source changed: $relative")
    end
    return true
end
