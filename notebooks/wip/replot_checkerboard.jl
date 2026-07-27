# Re-render a saved checkerboard result in the notebook's style, WITHOUT refitting.
#
# `experiments/deep_kernel_checkerboard.jl` takes ~100 s per fit, so changing the
# figure should not mean paying for inference again. Everything the plot needs is in
# the serialized result.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 julia --project=. experiments/replot_checkerboard.jl \
#       results/uncertainty_diagnosis/dkcb_nohole.jls out.png [notebook|full]

using Printf
using Serialization
using Statistics

using RxInfer
using SurrogateModelling

include(joinpath(@__DIR__, "checkerboard_plotting.jl"))

length(ARGS) >= 2 || error(
    "usage: replot_checkerboard.jl <result.jls> <output.png> [notebook|full]",
)
input_path, output_path = ARGS[1], ARGS[2]
panels = Symbol(length(ARGS) >= 3 ? ARGS[3] : "notebook")
isfile(input_path) || error("missing result artifact $input_path")

result = deserialize(input_path)
config = result.config

@printf("%s: cells %s, hole radius %.2f, H = %d, %d levels\n",
    basename(input_path), string(config.cells), config.hole_radius,
    config.n_features, config.levels)
@printf("  epistemic: observed %.5f | hole %s | beyond data %.5f\n",
    result.epistemic_observed,
    isnan(result.epistemic_hole) ? "none carved" :
        @sprintf("%.5f (ratio %.2f)", result.epistemic_hole,
                 result.epistemic_hole / result.epistemic_observed),
    result.epistemic_outer)

savefig(checkerboard_figure(result; panels), output_path)
@info "saved" output_path
