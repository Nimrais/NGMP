using Pkg

Pkg.activate(
    get(
        ENV,
        "RXINFER_DIFF_PROJECT",
        normpath(joinpath(@__DIR__, "..", "..")),
    ),
)

include(joinpath(@__DIR__, "RxInferBackendHarness.jl"))
using .RxInferBackendHarness

parse_bool(name, default) =
    lowercase(get(ENV, name, string(default))) in
    ("1", "true", "yes", "on")

config = HarnessConfig(
    samples = parse(Int, get(ENV, "RXINFER_DIFF_SAMPLES", "3")),
    warmup = parse_bool("RXINFER_DIFF_WARMUP", true),
    chain_states =
        parse(Int, get(ENV, "RXINFER_DIFF_CHAIN_STATES", "24")),
    high_degree =
        parse(Int, get(ENV, "RXINFER_DIFF_HIGH_DEGREE", "128")),
    atol = parse(Float64, get(ENV, "RXINFER_DIFF_ATOL", "1e-8")),
    rtol = parse(Float64, get(ENV, "RXINFER_DIFF_RTOL", "1e-7")),
    reference =
        get(ENV, "RXINFER_DIFF_REFERENCE", "reactive"),
    output_dir = abspath(
        get(
            ENV,
            "RXINFER_DIFF_OUTPUT_DIR",
            joinpath(tempdir(), "rxinfer_backend_differential"),
        ),
    ),
    fail_on_mismatch =
        parse_bool("RXINFER_DIFF_FAIL_ON_MISMATCH", true),
    run_hotpath =
        parse_bool("RXINFER_DIFF_RUN_HOTPATH", false),
    hotpath_samples =
        parse(Int, get(ENV, "RXINFER_DIFF_HOTPATH_SAMPLES", "1")),
)

register_backend!(make_rxinfer_adapter("reactive"))

# An extension file can register one or more future backends without changing
# this harness. Example:
#
# using .RxInferBackendHarness
# using MyBackendPackage
# register_backend!(
#     make_rxinfer_adapter(
#         "my-backend";
#         prepare_transform = MyBackendPackage.prepare_keywords,
#     ),
# )
adapter_file = get(ENV, "RXINFER_DIFF_ADAPTER", "")
if !isempty(adapter_file)
    include(abspath(adapter_file))
end

run_harness(config)
