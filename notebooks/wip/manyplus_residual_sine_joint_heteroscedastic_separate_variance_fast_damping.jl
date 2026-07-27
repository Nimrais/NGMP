# Controlled ablation 2:
# increase only the nonlinear natural-message update rate.
#
# The per-observation Gamma precision prior is retained. Both ResidualSine
# edges and the Squareplus edge use alpha = 0.05 instead of 0.005.
# All other baseline settings are unchanged.

ENV["JOINT_HETERO_USE_PRECISION_PRIOR"] = "true"
ENV["JOINT_HETERO_ACTIVATION_ALPHA"] = "0.05"
ENV["JOINT_HETERO_LOG_ALPHA"] = "0.05"
ENV["JOINT_HETERO_G_INITIAL_VARIANCE"] = "0.0001"
ENV["JOINT_HETERO_SEPARATE_OUTPUT"] = get(
    ENV,
    "JOINT_HETERO_FAST_DAMPING_OUTPUT",
    "/tmp/manyplus_joint_separate_variance_fast_damping_qy.png",
)
ENV["JOINT_HETERO_SEPARATE_POSTERIORS"] = get(
    ENV,
    "JOINT_HETERO_FAST_DAMPING_POSTERIORS",
    "/tmp/manyplus_joint_separate_variance_fast_damping_qy_posteriors.jls",
)

include(joinpath(
    @__DIR__,
    "manyplus_residual_sine_joint_heteroscedastic_separate_variance_squareplus.jl",
))
