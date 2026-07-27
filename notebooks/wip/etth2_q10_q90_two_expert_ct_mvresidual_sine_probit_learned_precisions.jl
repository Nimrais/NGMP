# ETTh2 q10/q90 nonlinear gate with three learned Gamma precisions:
#
#   gamma_score       for CT-sine-CT -> softdot -> z
#   gamma_component1  for the q10 NormalMixture component
#   gamma_component2  for the q90 NormalMixture component
#
# All three are global parameters with independent mean-field posteriors.

ENV["TWO_EXPERT_LEARN_PRECISIONS"] = get(
    ENV,
    "TWO_EXPERT_LEARN_PRECISIONS",
    "true",
)

include(joinpath(
    @__DIR__,
    "etth2_q10_q90_two_expert_ct_mvresidual_sine_probit_mixture.jl",
))

if abspath(PROGRAM_FILE) == @__FILE__
    etth2_q10_q90_learned_precision_report = run_etth1_two_expert_study()
end
