using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Distributions
using LinearAlgebra: I, Symmetric
using Random
using ReactiveMP
using RxInfer
using SurrogateModelling

import ReactiveMP: @call_marginalrule, @call_rule, Marginal

function benchmark_spd(rng, dimension; ridge = 1.0)
    factor = randn(rng, dimension, dimension)
    return Matrix(Symmetric(factor * factor' + ridge * I))
end

function benchmark_inputs(dimension)
    rng = MersenneTwister(4_200 + dimension)
    parameter_dimension = dimension^2
    q_a = MvNormalMeanCovariance(
        randn(rng, parameter_dimension),
        benchmark_spd(rng, parameter_dimension; ridge = 2.0),
    )
    q_y = MvNormalMeanCovariance(
        randn(rng, dimension), benchmark_spd(rng, dimension; ridge = 1.0)
    )
    q_x = MvNormalMeanCovariance(
        randn(rng, dimension), benchmark_spd(rng, dimension; ridge = 1.0)
    )
    q_y_x = MvNormalMeanCovariance(
        [mean(q_y); mean(q_x)], benchmark_spd(rng, 2dimension; ridge = 1.0)
    )
    m_y = MvNormalWeightedMeanPrecision(
        randn(rng, dimension), benchmark_spd(rng, dimension; ridge = 1.0)
    )
    m_x = MvNormalWeightedMeanPrecision(
        randn(rng, dimension), benchmark_spd(rng, dimension; ridge = 1.0)
    )
    q_W = PointMass(benchmark_spd(rng, dimension; ridge = 1.0))
    return (; q_a, q_y, q_x, q_y_x, m_y, m_x, q_W)
end

function benchmark_average_energy(meta, names, distributions)
    marginals = map(distribution -> Marginal(distribution, false, false), distributions)
    return score(AverageEnergy(), ContinuousTransition, names, marginals, meta)
end

function benchmark_rule_suite(meta, data)
    y_structured = @call_rule ContinuousTransition(:y, Marginalisation) (
        m_x = data.m_x, q_a = data.q_a, q_W = data.q_W, meta = meta
    )
    y_mean_field = @call_rule ContinuousTransition(:y, Marginalisation) (
        q_x = data.q_x, q_a = data.q_a, q_W = data.q_W, meta = meta
    )
    x_structured = @call_rule ContinuousTransition(:x, Marginalisation) (
        m_y = data.m_y, q_a = data.q_a, q_W = data.q_W, meta = meta
    )
    x_mean_field = @call_rule ContinuousTransition(:x, Marginalisation) (
        q_y = data.q_y, q_a = data.q_a, q_W = data.q_W, meta = meta
    )
    a_structured = @call_rule ContinuousTransition(:a, Marginalisation) (
        q_y_x = data.q_y_x, q_a = data.q_a, q_W = data.q_W, meta = meta
    )
    a_mean_field = @call_rule ContinuousTransition(:a, Marginalisation) (
        q_y = data.q_y,
        q_x = data.q_x,
        q_a = data.q_a,
        q_W = data.q_W,
        meta = meta,
    )
    W_structured = @call_rule ContinuousTransition(:W, Marginalisation) (
        q_y_x = data.q_y_x, q_a = data.q_a, meta = meta
    )
    W_mean_field = @call_rule ContinuousTransition(:W, Marginalisation) (
        q_y = data.q_y, q_x = data.q_x, q_a = data.q_a, meta = meta
    )
    joint = @call_marginalrule ContinuousTransition(:y_x) (
        m_y = data.m_y,
        m_x = data.m_x,
        q_a = data.q_a,
        q_W = data.q_W,
        meta = meta,
    )
    structured_energy = benchmark_average_energy(
        meta, Val{(:y_x, :a, :W)}(), (data.q_y_x, data.q_a, data.q_W)
    )
    mean_field_energy = benchmark_average_energy(
        meta,
        Val{(:y, :x, :a, :W)}(),
        (data.q_y, data.q_x, data.q_a, data.q_W),
    )

    # Force all results to remain live until the suite has completed.
    return sum(mean(y_structured)) + sum(mean(y_mean_field)) +
           sum(weightedmean(x_structured)) + sum(weightedmean(x_mean_field)) +
           sum(weightedmean(a_structured)) + sum(weightedmean(a_mean_field)) +
           sum(mean(W_structured)) + sum(mean(W_mean_field)) +
           sum(weightedmean(joint)) + structured_energy + mean_field_energy
end

function run_benchmarks(dimensions = (8, 16))
    println("dimension,implementation,seconds,allocated_mib,checksum")
    for dimension in dimensions
        data = benchmark_inputs(dimension)
        generic_meta = CTMeta(a -> reshape(a, dimension, dimension))
        specialized_meta = LinearReshapeMeta(dimension, dimension)

        # Compile both dispatch paths before measuring either one.
        benchmark_rule_suite(generic_meta, data)
        benchmark_rule_suite(specialized_meta, data)

        GC.gc()
        generic = @timed benchmark_rule_suite(generic_meta, data)
        GC.gc()
        specialized = @timed benchmark_rule_suite(specialized_meta, data)

        println(
            dimension,
            ",generic,",
            generic.time,
            ",",
            generic.bytes / 2.0^20,
            ",",
            generic.value,
        )
        println(
            dimension,
            ",specialized,",
            specialized.time,
            ",",
            specialized.bytes / 2.0^20,
            ",",
            specialized.value,
        )
        println("dimension $dimension speedup: ", generic.time / specialized.time, "x")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
