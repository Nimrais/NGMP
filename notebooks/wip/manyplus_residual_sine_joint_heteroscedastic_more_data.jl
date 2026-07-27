# Observation-count experiment for the corrected joint mean/variance model.
#
# Defaults to 240 observations (four times the original 60). Override with
# JOINT_HETERO_MORE_DATA_N while keeping all architecture and iteration
# settings unchanged.

more_data_n = get(ENV, "JOINT_HETERO_MORE_DATA_N", "240")
ENV["JOINT_HETERO_N"] = more_data_n
ENV["JOINT_HETERO_NO_GAMMA_FAST_OUTPUT"] = get(
    ENV,
    "JOINT_HETERO_MORE_DATA_OUTPUT",
    "/tmp/manyplus_joint_separate_variance_n$(more_data_n)_qy.png",
)
ENV["JOINT_HETERO_NO_GAMMA_FAST_POSTERIORS"] = get(
    ENV,
    "JOINT_HETERO_MORE_DATA_POSTERIORS",
    "/tmp/manyplus_joint_separate_variance_n$(more_data_n)_qy_posteriors.jls",
)

include(joinpath(
    @__DIR__,
    "manyplus_residual_sine_joint_heteroscedastic_separate_variance_no_gamma_fast_damping.jl",
))
