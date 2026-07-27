# Interaction test after the two one-factor ablations:
# remove the Gamma precision prior and use alpha = 0.05.
#
# The output-weight initialization remains at the original 1e-4 because its
# independent 10,000x ablation was already found to have negligible effect.

ENV["JOINT_HETERO_USE_PRECISION_PRIOR"] = "false"
ENV["JOINT_HETERO_ACTIVATION_ALPHA"] = "0.05"
ENV["JOINT_HETERO_LOG_ALPHA"] = "0.05"
ENV["JOINT_HETERO_G_INITIAL_VARIANCE"] = "0.0001"
ENV["JOINT_HETERO_SEPARATE_OUTPUT"] = get(
    ENV,
    "JOINT_HETERO_NO_GAMMA_FAST_OUTPUT",
    "/tmp/manyplus_joint_separate_variance_no_gamma_fast_damping_qy.png",
)
ENV["JOINT_HETERO_SEPARATE_POSTERIORS"] = get(
    ENV,
    "JOINT_HETERO_NO_GAMMA_FAST_POSTERIORS",
    "/tmp/manyplus_joint_separate_variance_no_gamma_fast_damping_qy_posteriors.jls",
)

include(joinpath(
    @__DIR__,
    "manyplus_residual_sine_joint_heteroscedastic_separate_variance_squareplus.jl",
))
