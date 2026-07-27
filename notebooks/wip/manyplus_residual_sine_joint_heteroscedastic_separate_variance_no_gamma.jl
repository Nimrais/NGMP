# Controlled ablation 1:
# remove only the per-observation Gamma prior on precision.
#
# Everything else remains at the separate-network baseline:
# Squareplus, alpha = 0.005, q(noise_v) initialization variance = 1e-4,
# 60 observations, 8 neurons per network, and at most 100 iterations.

ENV["JOINT_HETERO_USE_PRECISION_PRIOR"] = "false"
ENV["JOINT_HETERO_ACTIVATION_ALPHA"] = "0.005"
ENV["JOINT_HETERO_LOG_ALPHA"] = "0.005"
ENV["JOINT_HETERO_G_INITIAL_VARIANCE"] = "0.0001"
ENV["JOINT_HETERO_SEPARATE_OUTPUT"] = get(
    ENV,
    "JOINT_HETERO_NO_GAMMA_OUTPUT",
    "/tmp/manyplus_joint_separate_variance_no_gamma_qy.png",
)
ENV["JOINT_HETERO_SEPARATE_POSTERIORS"] = get(
    ENV,
    "JOINT_HETERO_NO_GAMMA_POSTERIORS",
    "/tmp/manyplus_joint_separate_variance_no_gamma_qy_posteriors.jls",
)

include(joinpath(
    @__DIR__,
    "manyplus_residual_sine_joint_heteroscedastic_separate_variance_squareplus.jl",
))
