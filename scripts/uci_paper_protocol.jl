module UCIPaperProtocol

using CSV
using DataFrames
using Random
using Statistics
using SurrogateModelling: Yacht, Concrete, EnergyEfficiency, BostonHousing,
                          PowerPlant, WineQualityRed,
                          UCI_DEFAULT_SPLIT_SEED,
                          UCI_DEFAULT_TEST_FRACTION,
                          uci_regression_splits

export DATASETS, paper_splits, prepare_split, gaussian_logpdf_metrics,
       summarize_rows, write_results

const DATASETS = (
    yacht = (constructor = Yacht, paper_dvi = -0.47),
    energy = (constructor = EnergyEfficiency, paper_dvi = -1.01),
    concrete = (constructor = Concrete, paper_dvi = -3.06),
    housing = (constructor = BostonHousing, paper_dvi = -2.41),
    # The local red-wine mirror contains 1,599 rows; DVI Table 2 reports
    # 1,588, so this reference is informative rather than exactly comparable.
    wine = (constructor = WineQualityRed, paper_dvi = -0.90),
    power = (constructor = PowerPlant, paper_dvi = -2.80),
)

function load_dataset(name::Symbol)
    hasproperty(DATASETS, name) || throw(ArgumentError("unknown dataset $name"))
    spec = getproperty(DATASETS, name)
    dataset = spec.constructor(as_df = false)
    features, targets = dataset[:]
    return Matrix{Float64}(features), vec(Float64.(targets)), spec.paper_dvi
end

function paper_splits(
    name::Symbol;
    count::Int = 20,
    seed::Int = UCI_DEFAULT_SPLIT_SEED,
)
    features, targets, paper_dvi = load_dataset(name)
    n = length(targets)
    specifications = uci_regression_splits(
        n,
        count;
        base_seed = seed,
        test_fraction = UCI_DEFAULT_TEST_FRACTION,
    )
    return map(specifications) do specification
        train_indices = specification.train_indices
        test_indices = specification.test_indices
        (; split_id = specification.split_id, paper_dvi,
           train_indices, test_indices,
           x_train = features[train_indices, :],
           x_test = features[test_indices, :],
           y_train = targets[train_indices],
           y_test = targets[test_indices])
    end
end

function prepare_split(split)
    x_center = vec(mean(split.x_train; dims = 1))
    x_scale = vec(std(split.x_train; dims = 1))
    x_scale = map(s -> isfinite(s) && s > sqrt(eps()) ? s : 1.0, x_scale)
    y_center = mean(split.y_train)
    y_scale = std(split.y_train)
    y_scale > sqrt(eps()) || error("target has zero training variance")
    return merge(split, (;
        x_train_std = (split.x_train .- x_center') ./ x_scale',
        x_test_std = (split.x_test .- x_center') ./ x_scale',
        y_train_std = (split.y_train .- y_center) ./ y_scale,
        y_test_std = (split.y_test .- y_center) ./ y_scale,
        x_center, x_scale, y_center, y_scale,
    ))
end

function gaussian_logpdf_metrics(y_std, mean_std, variance_std, y_scale)
    variance = max.(variance_std, 1e-10)
    residual = y_std .- mean_std
    pointwise_std =
        -0.5 .* (log.(2pi .* variance) .+ abs2.(residual) ./ variance)
    return (;
        logpdf_standardized = mean(pointwise_std),
        # Change of variables: p(y) = p(y_std) / y_scale.
        logpdf = mean(pointwise_std) - log(y_scale),
        rmse = y_scale * sqrt(mean(abs2.(residual))),
    )
end

function summarize_rows(rows)
    successful = filter(row -> row.status == "ok", rows)
    isempty(successful) && return (; successful = 0, mean_logpdf = NaN,
                                    std_logpdf = NaN, mean_rmse = NaN)
    values = getproperty.(successful, :logpdf)
    return (;
        successful = length(successful),
        mean_logpdf = mean(values),
        std_logpdf = length(values) == 1 ? 0.0 : std(values),
        mean_rmse = mean(getproperty.(successful, :rmse)),
    )
end

function write_results(path, rows)
    mkpath(dirname(path))
    CSV.write(path, DataFrame(rows))
    return path
end

end
