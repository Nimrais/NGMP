"""
    IVONRepro

Julia reproduction of the IVON optimizer (Improved Variational Online Newton)
from *Variational Learning is Effective for Large Deep Networks* (Shen et al.,
ICML 2024, [arXiv:2402.17641](https://arxiv.org/abs/2402.17641)), built on
Lux.jl, Optimisers.jl and Reactant.jl.

[`IVON`](@ref) is an `Optimisers.AbstractRule`, so it drops into Lux training
loops exactly like `AdamW`. Unlike `AdamW` it learns a *posterior distribution*
over the weights — after training you hold an ensemble of networks, sampled
with `rand(rng, ivon, opt_state, ps)`, instead of a single point estimate.

The public API is the optimizer alone: [`IVON`](@ref),
[`ivon_train_step!`](@ref), `rand` and [`posterior_variance`](@ref). The test
suite compares the implementation behaviorally against reference artifacts
recorded from the official PyTorch implementation
(`test_python_artifacts/artifacts/`); the Colab model and training drivers used
for that comparison live in the test setup (`test/colab_testsetup.jl`), not in
the package.
"""
module IVONRepro

using Accessors: @set
using Functors: fmap
using Lux: Training
using Optimisers
using Random
using Random: AbstractRNG

export IVON, ivon_train_step!, posterior_variance

include("ivon.jl")

end
