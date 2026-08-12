function build_gate(architecture::Symbol; n_features::Int = 65, n_experts::Int = 7)
    n_features == 65 || error("The paper VAE gate must receive 65 features, got $n_features")
    n_experts == 7 || error("The paper ensemble must contain seven experts, got $n_experts")
    if architecture === :moe
        return Lux.Chain(Lux.Dense(n_features => n_experts))
    elseif architecture === :moe_big
        return Lux.Chain(
            Lux.Dense(n_features => 192, Lux.relu),
            Lux.Dense(192 => n_experts),
        )
    end
    error("Unknown gate architecture: $architecture")
end

parameter_count(x::AbstractArray) = length(x)
parameter_count(x::NamedTuple) = sum(parameter_count, values(x); init = 0)
parameter_count(x::Tuple) = sum(parameter_count, x; init = 0)
parameter_count(::Any) = 0

expected_parameter_count(architecture::Symbol) =
    architecture === :moe ? 462 : architecture === :moe_big ? 14_023 :
    error("Unknown gate architecture: $architecture")

function validate_gate(architecture::Symbol, gate, ps, st)
    count = parameter_count(ps)
    count == expected_parameter_count(architecture) || error(
        "Unexpected parameter count for $architecture: $count",
    )
    x = zeros(Float32, 65)
    logits, _ = gate(x, ps, st)
    size(logits) == (7,) || error("Unexpected gate output shape: $(size(logits))")
    all(isfinite, logits) || error("Gate initialization produced non-finite logits")
    return true
end

function stable_softmax(logits)
    m = maximum(logits)
    weights = exp.(logits .- m)
    return weights ./ sum(weights)
end

function logsumexp(values)
    m = maximum(values)
    return m + log(sum(exp, values .- m))
end
