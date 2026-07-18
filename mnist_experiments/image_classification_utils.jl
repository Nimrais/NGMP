using MLDatasets: MNIST, FashionMNIST, CIFAR10, CIFAR100
using Random
using Statistics

const FLATTENED_DATASET_SPECS = Dict(
    :mnist => (constructor=MNIST, default_size=(14, 14)),
    :fashion_mnist => (constructor=FashionMNIST, default_size=(14, 14)),
    :cifar10 => (constructor=CIFAR10, default_size=(16, 16)),
    :cifar100 => (constructor=CIFAR100, default_size=(16, 16)),
)

function canonical_dataset_name(dataset)
    name = Symbol(replace(lowercase(String(dataset)), "-" => "_"))
    aliases = Dict(
        :fashionmnist => :fashion_mnist,
        :fashion => :fashion_mnist,
        :cifar_10 => :cifar10,
        :cifar_100 => :cifar100,
    )
    name = get(aliases, name, name)
    haskey(FLATTENED_DATASET_SPECS, name) ||
        throw(ArgumentError("Unsupported dataset $dataset. Use :mnist, " *
            ":fashion_mnist, :cifar10, or :cifar100."))
    return name
end

function balanced_dataset_counts(total, class_count)
    total >= 0 || throw(ArgumentError("split sizes must be nonnegative"))
    base, remainder = divrem(total, class_count)
    return [base + (i <= remainder) for i in 1:class_count]
end

function validate_image_size(image_size)
    length(image_size) == 2 ||
        throw(ArgumentError("image_size must contain height and width"))
    size_tuple = (Int(image_size[1]), Int(image_size[2]))
    all(>(0), size_tuple) ||
        throw(ArgumentError("image_size dimensions must be positive"))
    return size_tuple
end

image_count(images) = size(images, ndims(images))
image_channels(images) = ndims(images) == 3 ? 1 : size(images, 3)

function dataset_labels(data, name)
    labels = name === :cifar100 ? data.targets.fine : data.targets
    return Int.(vec(labels))
end

function image_at(images, index)
    if ndims(images) == 3
        return @view images[:, :, index]
    elseif ndims(images) == 4
        return @view images[:, :, :, index]
    end
    throw(ArgumentError("expected H×W×N or H×W×C×N image storage"))
end

function normalized_pixel(image, i, j, channel)
    value = ndims(image) == 2 ? Float64(image[i, j]) : Float64(image[i, j, channel])
    return eltype(image) <: Integer ? value / Float64(typemax(eltype(image))) : value
end

function resize_image_area(image, output_size)
    source_height, source_width = size(image, 1), size(image, 2)
    output_height, output_width = output_size
    channels = ndims(image) == 2 ? 1 : size(image, 3)
    row_scale = div(source_height, output_height)
    column_scale = div(source_width, output_width)
    output = Array{Float64}(undef, output_height, output_width, channels)
    normalization = inv(row_scale * column_scale)
    for channel in 1:channels, j in 1:output_width, i in 1:output_height
        total = 0.0
        for source_j in ((j - 1) * column_scale + 1):(j * column_scale)
            for source_i in ((i - 1) * row_scale + 1):(i * row_scale)
                total += normalized_pixel(image, source_i, source_j, channel)
            end
        end
        output[i, j, channel] = total * normalization
    end
    return output
end

function resize_image_bilinear(image, output_size)
    source_height, source_width = size(image, 1), size(image, 2)
    output_height, output_width = output_size
    channels = ndims(image) == 2 ? 1 : size(image, 3)
    output = Array{Float64}(undef, output_height, output_width, channels)
    for channel in 1:channels, j in 1:output_width, i in 1:output_height
        source_i = clamp((i - 0.5) * source_height / output_height + 0.5,
            1.0, Float64(source_height))
        source_j = clamp((j - 0.5) * source_width / output_width + 0.5,
            1.0, Float64(source_width))
        i0, j0 = floor(Int, source_i), floor(Int, source_j)
        i1, j1 = min(i0 + 1, source_height), min(j0 + 1, source_width)
        wi, wj = source_i - i0, source_j - j0
        top = (1 - wj) * normalized_pixel(image, i0, j0, channel) +
            wj * normalized_pixel(image, i0, j1, channel)
        bottom = (1 - wj) * normalized_pixel(image, i1, j0, channel) +
            wj * normalized_pixel(image, i1, j1, channel)
        output[i, j, channel] = (1 - wi) * top + wi * bottom
    end
    return output
end

function resize_image(image, output_size)
    source_size = (size(image, 1), size(image, 2))
    source_size == output_size && return reshape(
        [normalized_pixel(image, i, j, channel)
         for i in 1:source_size[1], j in 1:source_size[2],
             channel in 1:(ndims(image) == 2 ? 1 : size(image, 3))],
        source_size[1], source_size[2],
        ndims(image) == 2 ? 1 : size(image, 3))
    if source_size[1] % output_size[1] == 0 &&
       source_size[2] % output_size[2] == 0
        return resize_image_area(image, output_size)
    end
    return resize_image_bilinear(image, output_size)
end

function materialize_flattened_images(images, labels, indices, output_size)
    channels = image_channels(images)
    inputs = Matrix{Float64}(undef,
        output_size[1] * output_size[2] * channels, length(indices))
    outputs = Vector{Int}(undef, length(indices))
    for (column, index) in enumerate(indices)
        inputs[:, column] .= vec(resize_image(image_at(images, index), output_size))
        outputs[column] = Int(labels[index])
    end
    return inputs, outputs
end

function select_balanced_indices(labels, classes, counts, rng; used=nothing)
    indices = Int[]
    sizehint!(indices, sum(counts))
    for (class_index, label) in enumerate(classes)
        candidates = findall(==(label), labels)
        if !isnothing(used)
            filter!(index -> !(index in used), candidates)
        end
        length(candidates) >= counts[class_index] ||
            throw(ArgumentError("class $label has $(length(candidates)) available " *
                "images, but $(counts[class_index]) were requested"))
        shuffle!(rng, candidates)
        append!(indices, @view candidates[1:counts[class_index]])
    end
    shuffle!(rng, indices)
    return indices
end

function load_flattened_image_dataset(
    dataset=:mnist;
    ntrain=1000,
    nval=100,
    ntest=100,
    seed=1,
    classes=nothing,
    image_size=nothing,
)
    name = canonical_dataset_name(dataset)
    spec = FLATTENED_DATASET_SPECS[name]
    output_size = isnothing(image_size) ?
        spec.default_size : validate_image_size(image_size)
    rng = MersenneTwister(seed)
    train_data = spec.constructor(split=:train)
    test_data = spec.constructor(split=:test)
    train_images = train_data.features
    test_images = test_data.features
    train_labels = dataset_labels(train_data, name)
    test_labels = dataset_labels(test_data, name)

    available_classes = sort(intersect(unique(train_labels), unique(test_labels)))
    selected_classes = isnothing(classes) ? available_classes : collect(Int, classes)
    isempty(selected_classes) && throw(ArgumentError("classes cannot be empty"))
    missing_classes = setdiff(selected_classes, available_classes)
    isempty(missing_classes) ||
        throw(ArgumentError("classes $(missing_classes) are unavailable in $name"))

    train_counts = balanced_dataset_counts(ntrain, length(selected_classes))
    val_counts = balanced_dataset_counts(nval, length(selected_classes))
    test_counts = balanced_dataset_counts(ntest, length(selected_classes))
    train_indices = select_balanced_indices(
        train_labels, selected_classes, train_counts, rng)
    train_used = Set(train_indices)
    val_indices = select_balanced_indices(
        train_labels, selected_classes, val_counts, rng; used=train_used)
    test_indices = select_balanced_indices(
        test_labels, selected_classes, test_counts, rng)

    train_x, train_y = materialize_flattened_images(
        train_images, train_labels, train_indices, output_size)
    val_x, val_y = materialize_flattened_images(
        train_images, train_labels, val_indices, output_size)
    test_x, test_y = materialize_flattened_images(
        test_images, test_labels, test_indices, output_size)
    return (
        train_x, train_y, val_x, val_y, test_x, test_y,
        name=name,
        classes=selected_classes,
        image_size=output_size,
        channels=image_channels(train_images),
    )
end
