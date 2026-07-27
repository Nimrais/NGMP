# Controlled Exp-link version of the cubic epistemic-gap benchmark.
#
# All data, priors, architecture, training, and prediction settings come from
# manyplus_residual_sine_joint_cubic_epistemic.jl. Only the positive precision
# link and the output paths differ from the preserved Squareplus run.

ENV["JOINT_CUBIC_LINK"] = "exp"
ENV["JOINT_CUBIC_OUTPUT"] = get(
    ENV,
    "JOINT_CUBIC_EXP_OUTPUT",
    "/tmp/manyplus_joint_cubic_epistemic_exp.png",
)
ENV["JOINT_CUBIC_POSTERIORS"] = get(
    ENV,
    "JOINT_CUBIC_EXP_POSTERIORS",
    "/tmp/manyplus_joint_cubic_epistemic_exp_posteriors.jls",
)

include(joinpath(
    @__DIR__,
    "manyplus_residual_sine_joint_cubic_epistemic.jl",
))
