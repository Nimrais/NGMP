import ReactiveMP: @call_marginalrule
using LinearAlgebra: Diagonal

function manyplus_rule_output(inputs)
    return @call_rule ManyPlus(:out, Marginalisation) (
        m_inputs = ReactiveMP.ManyOf(Tuple(inputs)),
    )
end

function manyplus_rule_input(output, other_inputs, target_index)
    messages = (
        ReactiveMP.Message(output, false, false),
        ReactiveMP.ManyOf(
            map(
                input -> ReactiveMP.Message(input, false, false),
                Tuple(other_inputs),
            )
        ),
    )
    return ReactiveMP.rule(
        ManyPlus,
        (Val(:inputs), target_index),
        Marginalisation(),
        Val((:out, :inputs)),
        messages,
        nothing,
        nothing,
        nothing,
        ReactiveMP.AnnotationDict(),
        nothing,
    )
end

function manyplus_test_normal(parameterisation, mean_value, variance_value, ::Type{T}) where {T}
    mean_t = convert(T, mean_value)
    variance_t = convert(T, variance_value)

    if parameterisation === :mean_variance
        return NormalMeanVariance(mean_t, variance_t)
    elseif parameterisation === :mean_precision
        return NormalMeanPrecision(mean_t, inv(variance_t))
    elseif parameterisation === :weighted_mean_precision
        precision = inv(variance_t)
        return NormalWeightedMeanPrecision(mean_t * precision, precision)
    elseif parameterisation === :normal
        return Normal(mean_t, sqrt(variance_t))
    end

    error("Unknown Normal parameterisation $(parameterisation).")
end

@model function manyplus_two_input_model(y)
    x1 ~ Normal(mean = -0.3, variance = 1.2)
    x2 ~ Normal(mean = 0.8, variance = 0.7)
    total := ManyPlus(inputs = [x1, x2])
    y ~ Normal(mean = total, variance = 0.4)
end

@model function binary_two_input_model(y)
    x1 ~ Normal(mean = -0.3, variance = 1.2)
    x2 ~ Normal(mean = 0.8, variance = 0.7)
    total := x1 + x2
    y ~ Normal(mean = total, variance = 0.4)
end

@model function manyplus_multi_input_model(y, means, variances)
    x[1] ~ Normal(mean = means[1], variance = variances[1])
    for index in 2:length(means)
        x[index] ~ Normal(mean = means[index], variance = variances[index])
    end
    total := ManyPlus(inputs = x)
    y ~ Normal(mean = total, variance = 0.6)
end

@model function binary_multi_input_model(y, means, variances)
    x[1] ~ Normal(mean = means[1], variance = variances[1])
    for index in 2:length(means)
        x[index] ~ Normal(mean = means[index], variance = variances[index])
    end
    partial[1] := x[1] + x[2]
    for index in 2:(length(means) - 1)
        partial[index] := partial[index - 1] + x[index + 1]
    end
    y ~ Normal(mean = partial[length(means) - 1], variance = 0.6)
end

@model function manyplus_graph_model(y, ninputs)
    x[1] ~ Normal(mean = 0.0, variance = 1.0)
    for index in 2:ninputs
        x[index] ~ Normal(mean = 0.0, variance = 1.0)
    end
    total := ManyPlus(inputs = x)
    y ~ Normal(mean = total, variance = 1.0)
end

@model function binary_graph_model(y, ninputs)
    x[1] ~ Normal(mean = 0.0, variance = 1.0)
    for index in 2:ninputs
        x[index] ~ Normal(mean = 0.0, variance = 1.0)
    end
    partial[1] := x[1] + x[2]
    for index in 2:(ninputs - 1)
        partial[index] := partial[index - 1] + x[index + 1]
    end
    y ~ Normal(mean = partial[ninputs - 1], variance = 1.0)
end

@model function manyplus_linear_regression(x, y)
    a ~ Normal(mean = 0.0, variance = 1.0)
    b ~ Normal(mean = 0.0, variance = 100.0)

    for index in eachindex(y)
        scaled[index] := a * x[index]
        intercept[index] := b * 1.0
        predictor[index] := ManyPlus(
            inputs = [scaled[index], intercept[index]]
        )
        y[index] ~ Normal(mean = predictor[index], variance = 1.0)
    end
end

@model function binary_linear_regression(x, y)
    a ~ Normal(mean = 0.0, variance = 1.0)
    b ~ Normal(mean = 0.0, variance = 100.0)

    for index in eachindex(y)
        scaled[index] := a * x[index]
        intercept[index] := b * 1.0
        predictor[index] := scaled[index] + intercept[index]
        y[index] ~ Normal(mean = predictor[index], variance = 1.0)
    end
end

@testset "rules" begin
    parameterisations = (
        :mean_variance,
        :mean_precision,
        :weighted_mean_precision,
        :normal,
    )

    for arity in (2, 3, 7)
        means = [(-1.0)^index * (index + 0.25) for index in 1:arity]
        variances = [0.2 + index / 3 for index in 1:arity]
        inputs = [
            manyplus_test_normal(
                parameterisations[mod1(index, length(parameterisations))],
                means[index],
                variances[index],
                Float64,
            ) for index in 1:arity
        ]

        forward = manyplus_rule_output(inputs)
        @test forward isa NormalMeanVariance
        @test collect(mean_var(forward)) ≈ [sum(means), sum(variances)]

        output = NormalWeightedMeanPrecision(4.0, 2.0)
        output_mean, output_variance = mean_var(output)
        for target_index in eachindex(inputs)
            other_indices = filter(!=(target_index), eachindex(inputs))
            other_inputs = inputs[other_indices]
            backward = manyplus_rule_input(output, other_inputs, target_index)

            @test backward isa NormalMeanVariance
            @test collect(mean_var(backward)) ≈ [
                output_mean - sum(means[other_indices]),
                output_variance + sum(variances[other_indices]),
            ]
        end
    end

    float32_inputs = [
        NormalMeanVariance(Float32(0.5), Float32(0.25)),
        NormalMeanPrecision(Float32(-0.25), Float32(2.0)),
    ]
    float32_output = manyplus_rule_output(float32_inputs)
    @test float32_output isa NormalMeanVariance{Float32}

    promoted_output = manyplus_rule_output((
        float32_inputs[1], NormalMeanVariance(0.25, 0.75)
    ))
    @test promoted_output isa NormalMeanVariance{Float64}

    big_output = manyplus_rule_output((
        NormalMeanVariance(big"0.5", big"0.25"), float32_inputs[2]
    ))
    @test big_output isa NormalMeanVariance{BigFloat}

    input1 = NormalMeanPrecision(-0.4, 2.0)
    input2 = NormalWeightedMeanPrecision(0.3, 1.5)
    output = NormalMeanVariance(1.2, 0.7)

    manyplus_forward = manyplus_rule_output((input1, input2))
    binary_forward = @call_rule typeof(+)(:out, Marginalisation) (
        m_in1 = input1, m_in2 = input2
    )
    @test collect(mean_var(manyplus_forward)) ≈ collect(mean_var(binary_forward))

    manyplus_backward1 = manyplus_rule_input(output, (input2,), 1)
    binary_backward1 = @call_rule typeof(+)(:in1, Marginalisation) (
        m_out = output, m_in2 = input2
    )
    @test collect(mean_var(manyplus_backward1)) ≈
        collect(mean_var(binary_backward1))

    manyplus_backward2 = manyplus_rule_input(output, (input1,), 2)
    binary_backward2 = @call_rule typeof(+)(:in2, Marginalisation) (
        m_out = output, m_in1 = input1
    )
    @test collect(mean_var(manyplus_backward2)) ≈
        collect(mean_var(binary_backward2))
end

@testset "construction and dependencies" begin
    for ninputs in (0, 1)
        interfaces = [
            (:out, randomvar());
            [(:inputs, randomvar()) for _ in 1:ninputs]
        ]
        error = try
            ReactiveMP.factornode(ManyPlus, interfaces, nothing)
            nothing
        catch caught
            caught
        end

        @test error isa ArgumentError
        @test occursin("at least two inputs", sprint(showerror, error))
    end

    interfaces = [
        (:out, randomvar());
        [(:inputs, randomvar()) for _ in 1:4]
    ]
    node = ReactiveMP.factornode(ManyPlus, interfaces, nothing)
    dependencies = ReactiveMP.collect_functional_dependencies(node, nothing)

    message_dependencies, marginal_dependencies =
        ReactiveMP.functional_dependencies(dependencies, node, node.inputs[2], 3)
    @test message_dependencies[1] === node.out
    @test length(message_dependencies[2]) == 3
    @test all(input -> input !== node.inputs[2], message_dependencies[2])
    @test isempty(marginal_dependencies)

    large_interfaces = [
        (:out, randomvar());
        [(:inputs, randomvar()) for _ in 1:64]
    ]
    large_node = ReactiveMP.factornode(ManyPlus, large_interfaces, nothing)
    @test length(ReactiveMP.getinterfaces(large_node)) == 65
end

@testset "factor free energy" begin
    output = NormalMeanVariance(0.7, 0.9)
    input1 = NormalMeanPrecision(-0.4, 2.0)
    input2 = NormalWeightedMeanPrecision(0.3, 1.5)

    binary_joint = @call_marginalrule typeof(+)(:in1_in2) (
        m_out = output, m_in1 = input1, m_in2 = input2
    )
    manyplus_score =
        SurrogateModelling.ManyPlusNode._manyplus_negative_entropy(
            output, (input1, input2)
        )
    @test manyplus_score ≈ -entropy(binary_joint)

    for dimension in (2, 3, 8)
        variances = [0.25 + index / 5 for index in 1:dimension]
        output_variance = 0.7
        inputs = Tuple(
            NormalMeanVariance(index / 3, variances[index]) for
            index in 1:dimension
        )

        analytic = SurrogateModelling.ManyPlusNode._manyplus_negative_entropy(
            NormalMeanVariance(-0.2, output_variance), inputs
        )
        precision =
            Diagonal(inv.(variances)) +
            fill(inv(output_variance), dimension, dimension)
        dense_joint = MvNormalWeightedMeanPrecision(
            zeros(dimension), Matrix(precision)
        )
        @test analytic ≈ -entropy(dense_joint) rtol = 1e-12 atol = 1e-12
    end

    manyplus_result = infer(
        model = manyplus_two_input_model(),
        data = (y = 0.35,),
        returnvars = (x1 = KeepLast(), x2 = KeepLast(), total = KeepLast()),
        free_energy = true,
    )
    binary_result = infer(
        model = binary_two_input_model(),
        data = (y = 0.35,),
        returnvars = (x1 = KeepLast(), x2 = KeepLast(), total = KeepLast()),
        free_energy = true,
    )

    for variable in (:x1, :x2, :total)
        @test collect(mean_var(manyplus_result.posteriors[variable])) ≈
            collect(mean_var(binary_result.posteriors[variable])) rtol = 1e-12 atol = 1e-12
    end
    @test manyplus_result.free_energy ≈ binary_result.free_energy rtol = 1e-12 atol = 1e-12
end

@testset "graph structure" begin
    graphppl = RxInfer.GraphPPL
    ninputs = 32

    manyplus_model = RxInfer.create_model(
        RxInfer.condition_on(
            manyplus_graph_model(ninputs = ninputs), y = 0.0
        )
    ).model
    binary_model = RxInfer.create_model(
        RxInfer.condition_on(
            binary_graph_model(ninputs = ninputs), y = 0.0
        )
    ).model

    manyplus_factors = collect(filter(graphppl.as_node(ManyPlus), manyplus_model))
    addition_factors = collect(filter(graphppl.as_node(+), binary_model))
    partial_variables = collect(
        filter(graphppl.as_variable(:partial), binary_model)
    )

    @test length(manyplus_factors) == 1
    @test length(graphppl.neighbors(manyplus_model, only(manyplus_factors))) == 33
    @test length(addition_factors) == 31
    @test length(partial_variables) - 1 == 30
end

@testset "multi-input inference matches a binary chain" begin
    means = [-1.0, 0.5, 1.25, -0.75, 0.2, 0.9]
    variances = [0.4, 1.1, 0.8, 1.7, 0.6, 1.3]
    observation = 1.4

    manyplus_result = infer(
        model = manyplus_multi_input_model(
            means = means, variances = variances
        ),
        data = (y = observation,),
        returnvars = (x = KeepLast(), total = KeepLast()),
        free_energy = true,
    )
    binary_result = infer(
        model = binary_multi_input_model(
            means = means, variances = variances
        ),
        data = (y = observation,),
        returnvars = (x = KeepLast(), partial = KeepLast()),
        free_energy = true,
    )

    @test mean.(manyplus_result.posteriors[:x]) ≈
        mean.(binary_result.posteriors[:x]) rtol = 1e-10 atol = 1e-12
    @test var.(manyplus_result.posteriors[:x]) ≈
        var.(binary_result.posteriors[:x]) rtol = 1e-10 atol = 1e-12
    @test collect(mean_var(manyplus_result.posteriors[:total])) ≈
        collect(mean_var(last(binary_result.posteriors[:partial]))) rtol = 1e-10 atol = 1e-12
    @test manyplus_result.free_energy ≈ binary_result.free_energy rtol = 1e-10 atol = 1e-12
end

@testset "known-noise Bayesian linear regression" begin
    rng = Random.MersenneTwister(1234)
    x_data = Float64.(1:250)
    y_data = 0.5 .* x_data .+ 25.0 .+ randn(rng, 250)
    initialization = @initialization(μ(b) = NormalMeanVariance(0.0, 100.0))

    manyplus_result = infer(
        model = manyplus_linear_regression(),
        data = (x = x_data, y = y_data),
        initialization = initialization,
        returnvars = (a = KeepLast(), b = KeepLast()),
        iterations = 20,
        free_energy = true,
    )
    binary_result = infer(
        model = binary_linear_regression(),
        data = (x = x_data, y = y_data),
        initialization = initialization,
        returnvars = (a = KeepLast(), b = KeepLast()),
        iterations = 20,
        free_energy = true,
    )

    for variable in (:a, :b)
        @test collect(mean_var(manyplus_result.posteriors[variable])) ≈
            collect(mean_var(binary_result.posteriors[variable])) rtol = 1e-8 atol = 1e-10
    end
    @test length(manyplus_result.free_energy) == 20
    @test manyplus_result.free_energy ≈ binary_result.free_energy rtol = 1e-8 atol = 1e-10
end
