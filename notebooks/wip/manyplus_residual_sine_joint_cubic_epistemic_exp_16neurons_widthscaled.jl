# Width-normalized capacity test for the Exp-link cubic benchmark.
#
# The original 8-neuron model is the reference parameterization. At 16
# neurons, both heads' output-weight means and standard deviations are
# multiplied by sqrt(8 / 16), while their variances are multiplied by 8 / 16.
# All other data, priors, seeds, and inference settings remain unchanged.

ENV["JOINT_HETERO_NEURONS"] = "16"
ENV["JOINT_HETERO_OUTPUT_WEIGHT_REFERENCE_NEURONS"] = "8"
ENV["JOINT_CUBIC_EXP_OUTPUT"] = get(
    ENV,
    "JOINT_CUBIC_EXP_16_SCALED_OUTPUT",
    "/tmp/manyplus_joint_cubic_epistemic_exp_16neurons_widthscaled.png",
)
ENV["JOINT_CUBIC_EXP_POSTERIORS"] = get(
    ENV,
    "JOINT_CUBIC_EXP_16_SCALED_POSTERIORS",
    "/tmp/manyplus_joint_cubic_epistemic_exp_16neurons_widthscaled_posteriors.jls",
)

include(joinpath(
    @__DIR__,
    "manyplus_residual_sine_joint_cubic_epistemic_exp.jl",
))
