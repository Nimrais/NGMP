# ETTh2 entry point for the nonlinear two-expert gate:
#
#   features -> CT -> MvResidualSine -> CT -> softdot -> Probit
#            -> NormalMixture(expert 4, expert 6)
#
# The implementation and validation-only acceptance protocol live in the
# shared ETTh1/ETTh2 runner. Environment values remain overridable.

ENV["TWO_EXPERT_DATASET"] = get(ENV, "TWO_EXPERT_DATASET", "ETTh2")
ENV["TWO_EXPERT_ARCHITECTURE"] = get(
    ENV,
    "TWO_EXPERT_ARCHITECTURE",
    "residual_sine",
)

include(joinpath(@__DIR__, "etth1_two_expert_ct_probit_mixture.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    etth2_two_expert_residual_sine_report = run_etth1_two_expert_study()
end
