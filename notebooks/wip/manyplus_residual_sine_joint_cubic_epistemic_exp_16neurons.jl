# Capacity test for the Exp-link cubic epistemic benchmark.
#
# This preserves the data, priors, seeds, and 100-iteration caps from the
# 8-neuron Exp run. Only the width of both independent ResidualSine networks
# changes from 8 to 16 neurons.

ENV["JOINT_HETERO_NEURONS"] = "16"
ENV["JOINT_CUBIC_EXP_OUTPUT"] = get(
    ENV,
    "JOINT_CUBIC_EXP_16_OUTPUT",
    "/tmp/manyplus_joint_cubic_epistemic_exp_16neurons.png",
)
ENV["JOINT_CUBIC_EXP_POSTERIORS"] = get(
    ENV,
    "JOINT_CUBIC_EXP_16_POSTERIORS",
    "/tmp/manyplus_joint_cubic_epistemic_exp_16neurons_posteriors.jls",
)

include(joinpath(
    @__DIR__,
    "manyplus_residual_sine_joint_cubic_epistemic_exp.jl",
))
