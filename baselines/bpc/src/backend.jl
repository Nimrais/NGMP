backend_available(name::String) = name == "cpu" ||
    (name == "cuda" && try CUDA.functional() catch; false end)

to_device(array::AbstractArray, ::Val{:cpu}) = Array{Float32}(array)
to_device(array::AbstractArray, ::Val{:cuda}) = CUDA.CuArray{Float32}(array)

backend_value(config::BPCConfig) =
    config.backend == "cuda" ? Val(:cuda) : Val(:cpu)

to_device(array::AbstractArray, config::BPCConfig) =
    to_device(array, backend_value(config))

to_host(array::Array) = array
to_host(array::AbstractArray) = Array(array)

function synchronize_backend(config::BPCConfig)
    config.backend == "cuda" && CUDA.synchronize()
    return nothing
end

function device_ones(reference::AbstractArray, rows::Int, columns::Int)
    result = similar(reference, eltype(reference), rows, columns)
    fill!(result, one(eltype(reference)))
    return result
end

function device_identity(reference::AbstractArray, dimension::Int)
    host = Matrix{Float32}(I, dimension, dimension)
    return reference isa Array ? host : CUDA.CuArray(host)
end
