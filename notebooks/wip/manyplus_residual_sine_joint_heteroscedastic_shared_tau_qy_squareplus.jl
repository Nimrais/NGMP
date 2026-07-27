### Squareplus comparison runner
#
# This executes the same full-data synthetic regression experiment as
# `manyplus_residual_sine_joint_heteroscedastic.jl`, changing only the positive
# precision link:
#
#     precision = Exp(score)        ->        precision = Squareplus(score)
#
# The following remain identical:
# - 60 training observations and 8 neurons in the full run;
# - one shared tau and one shared tau_c;
# - simultaneous mean/precision learning in one infer call;
# - a hard cap of 100 training iterations;
# - direct structured q(y*) prediction with w kept joint;
# - the data seed, parameter priors, and prediction grid.
#
# Outputs are deliberately distinct from the Exp run, so its picture and saved
# posteriors are not overwritten.

ENV["JOINT_HETERO_LINK"] = "squareplus"

include(joinpath(
    @__DIR__,
    "manyplus_residual_sine_joint_heteroscedastic.jl",
))
