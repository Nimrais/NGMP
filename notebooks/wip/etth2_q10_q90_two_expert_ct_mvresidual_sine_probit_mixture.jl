# ETTh2 q10/q90 nonlinear gate:
#
#   features -> CT -> MvResidualSine -> CT -> softdot -> Probit
#            -> NormalMixture(q10, q90)
#
# Cache row 6 is the constant validation-history q10 and row 7 is q90. Positive
# scores select component 2, hence q90. The shared runner keeps held-out test
# keys sealed until the validation screen passes.

ENV["TWO_EXPERT_DATASET"] = get(ENV, "TWO_EXPERT_DATASET", "ETTh2")
ENV["TWO_EXPERT_ARCHITECTURE"] = get(
    ENV,
    "TWO_EXPERT_ARCHITECTURE",
    "residual_sine",
)
ENV["TWO_EXPERT_COMPONENT1_INDEX"] = get(
    ENV,
    "TWO_EXPERT_COMPONENT1_INDEX",
    "6",
)
ENV["TWO_EXPERT_COMPONENT2_INDEX"] = get(
    ENV,
    "TWO_EXPERT_COMPONENT2_INDEX",
    "7",
)
ENV["TWO_EXPERT_COMPONENT1_LABEL"] = get(
    ENV,
    "TWO_EXPERT_COMPONENT1_LABEL",
    "q10",
)
ENV["TWO_EXPERT_COMPONENT2_LABEL"] = get(
    ENV,
    "TWO_EXPERT_COMPONENT2_LABEL",
    "q90",
)
ENV["TWO_EXPERT_COMPONENT1_METHOD"] = get(
    ENV,
    "TWO_EXPERT_COMPONENT1_METHOD",
    "q10",
)
ENV["TWO_EXPERT_COMPONENT2_METHOD"] = get(
    ENV,
    "TWO_EXPERT_COMPONENT2_METHOD",
    "q90",
)

include(joinpath(@__DIR__, "etth1_two_expert_ct_probit_mixture.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    etth2_q10_q90_residual_sine_report = run_etth1_two_expert_study()
end
