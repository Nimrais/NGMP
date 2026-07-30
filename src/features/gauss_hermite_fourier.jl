export GaussHermiteFourierBasis, coordinate_features, median_pairwise_distance

import FastGaussQuadrature
import Statistics: median

"""
    GaussHermiteFourierBasis(order, lengthscale)

Deterministic per-coordinate Fourier features whose frequencies and scales are
the Gauss–Hermite quadrature rule of the given `order` applied to the spectral
density of a squared-exponential kernel with the given `lengthscale`: the
positive nodes become cos/sin frequency pairs (`√2 · node / lengthscale`),
weighted so that `φ(x)ᵀφ(x′)` approximates the SE kernel; an odd `order`
contributes an additional constant coordinate from the zero node.

Callable: `basis(x::Real)` returns the feature vector of `length(basis)`
(= `2 · n_positive_nodes + (order isodd)`).

This is the shared implementation previously copied between
`scripts/uci_yacht_tensor_kernel_benchmark.jl` and the tensor-kernel
notebooks; new code should use this export instead of a local copy.
"""
struct GaussHermiteFourierBasis
    frequencies::Vector{Float64}
    pair_scales::Vector{Float64}
    zero_scale::Float64
end

function GaussHermiteFourierBasis(order::Int, lengthscale::Real)
    lengthscale > 0 || throw(ArgumentError("lengthscale must be positive"))
    nodes, weights = FastGaussQuadrature.gausshermite(order)
    normalized_weights = weights ./ sqrt(pi)
    tolerance = 100eps(Float64)
    positive = findall(node -> node > tolerance, nodes)
    zero_index = findfirst(node -> abs(node) <= tolerance, nodes)
    frequencies =
        sqrt(2) .* Float64.(nodes[positive]) ./ Float64(lengthscale)
    pair_scales = sqrt.(2 .* Float64.(normalized_weights[positive]))
    zero_scale = isnothing(zero_index) ?
        0.0 : sqrt(Float64(normalized_weights[zero_index]))
    return GaussHermiteFourierBasis(
        frequencies,
        pair_scales,
        zero_scale,
    )
end

Base.length(basis::GaussHermiteFourierBasis) =
    2length(basis.frequencies) + Int(!iszero(basis.zero_scale))

function (basis::GaussHermiteFourierBasis)(x::Real)
    result = Vector{Float64}(undef, length(basis))
    offset = 0
    if !iszero(basis.zero_scale)
        result[1] = basis.zero_scale
        offset = 1
    end
    for index in eachindex(basis.frequencies)
        phase = basis.frequencies[index] * x
        scale = basis.pair_scales[index]
        result[offset + 2index - 1] = scale * cos(phase)
        result[offset + 2index] = scale * sin(phase)
    end
    return result
end

"""
    coordinate_features(inputs, bases)

Evaluate one basis per input coordinate: for an `n × d` input matrix and a
`d`-vector of bases, return a `d`-vector of `n × length(basis)` feature
matrices — the per-coordinate designs consumed by the CP tensor machinery.
"""
function coordinate_features(inputs, bases)
    size(inputs, 2) == length(bases) ||
        throw(DimensionMismatch("one basis is required per coordinate"))
    return map(eachindex(bases)) do dimension
        result = Matrix{Float64}(
            undef,
            size(inputs, 1),
            length(bases[dimension]),
        )
        for observation in axes(inputs, 1)
            @views result[observation, :] .=
                bases[dimension](inputs[observation, dimension])
        end
        result
    end
end

"""
    median_pairwise_distance(inputs)

Median Euclidean distance between the rows of `inputs` — the training-only
lengthscale heuristic. Errors if the result is not finite and positive.
"""
function median_pairwise_distance(inputs)
    distances = Float64[]
    sizehint!(distances, size(inputs, 1) * (size(inputs, 1) - 1) ÷ 2)
    for left in 1:(size(inputs, 1) - 1)
        for right in (left + 1):size(inputs, 1)
            push!(
                distances,
                sqrt(sum(abs2, @view(inputs[left, :]) .-
                    @view(inputs[right, :]))),
            )
        end
    end
    result = median(distances)
    isfinite(result) && result > 0 ||
        error("could not determine a positive training-only lengthscale")
    return result
end
