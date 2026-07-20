using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Distributions
using LinearAlgebra: Diagonal, I, Symmetric, cholesky
using Random
using ReactiveMP
using RxInfer
using SurrogateModelling

import ReactiveMP: @call_marginalrule, @call_rule, Marginal

function low_rank_benchmark_spd(rng, dimension; ridge = 1.0)
    factor = randn(rng, dimension, dimension)
    return Matrix(Symmetric(factor * factor' + ridge * I))
end

function low_rank_benchmark_common(rng, dimension)
    q_y = MvNormalMeanCovariance(
        randn(rng, dimension), low_rank_benchmark_spd(rng, dimension)
    )
    q_x = MvNormalMeanCovariance(
        randn(rng, dimension), low_rank_benchmark_spd(rng, dimension)
    )
    q_y_x = MvNormalMeanCovariance(
        [mean(q_y); mean(q_x)], low_rank_benchmark_spd(rng, 2dimension)
    )
    m_y = MvNormalWeightedMeanPrecision(
        randn(rng, dimension), low_rank_benchmark_spd(rng, dimension)
    )
    m_x = MvNormalWeightedMeanPrecision(
        randn(rng, dimension), low_rank_benchmark_spd(rng, dimension)
    )
    q_W = PointMass(low_rank_benchmark_spd(rng, dimension))
    return (; q_y, q_x, q_y_x, m_y, m_x, q_W)
end

function low_rank_benchmark_parameter_message(meta, q_a, common)
    return @call_rule ContinuousTransition(:a, Marginalisation) (
        q_y = common.q_y,
        q_x = common.q_x,
        q_a = q_a,
        q_W = common.q_W,
        meta = meta,
    )
end

function low_rank_benchmark_energy(meta, names, distributions)
    marginals = map(distribution -> Marginal(distribution, false, false), distributions)
    return score(AverageEnergy(), ContinuousTransition, names, marginals, meta)
end

function low_rank_benchmark_rule_suite(meta, q_a, common)
    y_structured = @call_rule ContinuousTransition(:y, Marginalisation) (
        m_x = common.m_x, q_a = q_a, q_W = common.q_W, meta = meta
    )
    x_mean_field = @call_rule ContinuousTransition(:x, Marginalisation) (
        q_y = common.q_y, q_a = q_a, q_W = common.q_W, meta = meta
    )
    a_mean_field = low_rank_benchmark_parameter_message(meta, q_a, common)
    W_structured = @call_rule ContinuousTransition(:W, Marginalisation) (
        q_y_x = common.q_y_x, q_a = q_a, meta = meta
    )
    joint = @call_marginalrule ContinuousTransition(:y_x) (
        m_y = common.m_y,
        m_x = common.m_x,
        q_a = q_a,
        q_W = common.q_W,
        meta = meta,
    )
    energy = low_rank_benchmark_energy(
        meta,
        Val{(:y, :x, :a, :W)}(),
        (common.q_y, common.q_x, q_a, common.q_W),
    )
    return sum(mean(y_structured)) + sum(weightedmean(x_mean_field)) +
           sum(weightedmean(a_mean_field)) + sum(mean(W_structured)) +
           sum(weightedmean(joint)) + energy
end

function low_rank_benchmark_factor(message)
    Lambda = Matrix(precision(message))
    @inbounds for i in axes(Lambda, 1)
        Lambda[i, i] += 1.0
    end
    factor = cholesky(Symmetric(Lambda))
    return sum(factor.L)
end

function low_rank_benchmark_measure(thunk, label, dimension, rank, parameter_dim)
    thunk()
    GC.gc()
    measurement = @timed thunk()
    println(
        dimension,
        ',',
        rank,
        ',',
        label,
        ',',
        parameter_dim,
        ',',
        measurement.time,
        ',',
        measurement.bytes / 2.0^20,
        ',',
        measurement.value,
    )
end

function run_low_rank_benchmarks(dimensions = (8, 16, 24); fixed_rank = 4)
    println("dimension,rank,operation,parameter_dim,seconds,allocated_mib,checksum")
    for dimension in dimensions
        rng = MersenneTwister(7_000 + dimension)
        common = low_rank_benchmark_common(rng, dimension)

        for rank in unique((min(fixed_rank, dimension), dimension))
            U = randn(rng, dimension, rank)
            V = randn(rng, dimension, rank)
            q_a = MvNormalMeanCovariance(
                randn(rng, rank), low_rank_benchmark_spd(rng, rank; ridge = 2.0)
            )
            A0 = zeros(dimension, dimension)
            generic_meta = CTMeta(a -> A0 + U * Diagonal(a) * V')
            specialized_meta = LinearLowRankMeta(U, V)

            for (name, meta) in
                (("generic_low_rank", generic_meta), ("specialized_low_rank", specialized_meta))
                low_rank_benchmark_measure(
                    "$(name)_parameter_message", dimension, rank, rank
                ) do
                    message = low_rank_benchmark_parameter_message(meta, q_a, common)
                    return sum(weightedmean(message)) + sum(precision(message))
                end
                low_rank_benchmark_measure("$(name)_rule_suite", dimension, rank, rank) do
                    return low_rank_benchmark_rule_suite(meta, q_a, common)
                end
            end

            specialized_message = low_rank_benchmark_parameter_message(
                specialized_meta, q_a, common
            )
            low_rank_benchmark_measure(
                "specialized_low_rank_posterior_factor", dimension, rank, rank
            ) do
                return low_rank_benchmark_factor(specialized_message)
            end
        end

        full_parameter_dim = dimension^2
        q_full = MvNormalMeanCovariance(
            randn(rng, full_parameter_dim), Diagonal(fill(2.0, full_parameter_dim))
        )
        reshape_meta = LinearReshapeMeta(dimension, dimension)
        low_rank_benchmark_measure(
            "reshape_parameter_message",
            dimension,
            dimension,
            full_parameter_dim,
        ) do
            message = low_rank_benchmark_parameter_message(reshape_meta, q_full, common)
            return sum(weightedmean(message)) + sum(precision(message))
        end
        reshape_message = low_rank_benchmark_parameter_message(reshape_meta, q_full, common)
        low_rank_benchmark_measure(
            "reshape_posterior_factor",
            dimension,
            dimension,
            full_parameter_dim,
        ) do
            return low_rank_benchmark_factor(reshape_message)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    dimensions = haskey(ENV, "DIMENSIONS") ?
                 Tuple(parse.(Int, split(ENV["DIMENSIONS"], ','))) : (8, 16, 24)
    fixed_rank = parse(Int, get(ENV, "FIXED_RANK", "4"))
    run_low_rank_benchmarks(dimensions; fixed_rank = fixed_rank)
end
