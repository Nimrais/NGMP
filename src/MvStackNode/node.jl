"""
    MvStack

A deterministic factor that gathers two or more scalar Gaussian inputs into a
single multivariate Gaussian edge,

```julia
hidden_vector ~ MvStack(inputs = [h1, h2, h3])
```

so that `out = [in_1, ..., in_H]`.

Why this node exists
--------------------
`ManyPlus` collapses `H` scalar neurons into a scalar sum, and its rules add
variances (`src/ManyPlusNode/rules.jl`). That is an independence assumption, and
it is fatal for uncertainty quantification: it makes the predictive variance a
non-negative combination of fixed non-negative basis functions, which can be
scaled down uniformly by data but can never dip at the observed inputs. Producing
GP-like contraction requires the *dense* covariance of the output weights, and
therefore requires the hidden layer to reach the output factor as a vector rather
than as a pre-summed scalar.

`MvStack` is the lossless alternative. Stacking is a measure-preserving bijection
`R^H -> R^H` with unit Jacobian, so both message rules and the free-energy
contribution are exact belief propagation, not an approximation:

  * toward `out`, the forward message is the joint Gaussian of the independent
    inbound scalars (diagonal covariance -- these are separate messages, so this
    is exact, not an assumption);
  * toward `in_k`, the message is the `k`-th coordinate marginal of
    `m_out * prod_{j != k} m_j`, which *does* carry back the correlations the
    downstream dense weight posterior has learned. This is the step `ManyPlus`
    cannot express.

Note on `collect_factorisation`
-------------------------------
Like `ManyPlus`, this node overrides `collect_factorisation`, because the
variadic-interface machinery needs a node-specific factorisation object. Unlike
`ManyPlus`, doing so costs nothing here: full sum-product over all edges is the
*exact* treatment of a deterministic copy, whereas for `ManyPlus` it silently
discarded the user's declared joint cluster in favour of an independence
assumption.
"""
struct MvStack end

ReactiveMP.as_node_symbol(::Type{MvStack}) = :MvStack
ReactiveMP.interfaces(::Type{MvStack}) = Val((:out, :inputs))
ReactiveMP.inputinterfaces(::Type{MvStack}) = Val((:inputs,))
ReactiveMP.alias_interface(::Type{MvStack}, ::Int64, name::Symbol) = name
ReactiveMP.is_predefined_node(::Type{MvStack}) =
    ReactiveMP.PredefinedNodeFunctionalForm()
ReactiveMP.sdtype(::Type{MvStack}) = ReactiveMP.Deterministic()

struct MvStackNodeFactorisation end

ReactiveMP.collect_factorisation(::Type{MvStack}, factorisation) =
    MvStackNodeFactorisation()

struct MvStackFactorNode{N} <: ReactiveMP.AbstractFactorNode
    out::ReactiveMP.NodeInterface
    inputs::NTuple{N, ReactiveMP.IndexedNodeInterface}
end

ReactiveMP.functionalform(::MvStackFactorNode) = MvStack
ReactiveMP.getinterfaces(node::MvStackFactorNode) = (node.out, node.inputs...)
ReactiveMP.getinboundinterfaces(node::MvStackFactorNode) = node.inputs
ReactiveMP.sdtype(::MvStackFactorNode) = ReactiveMP.Deterministic()

ReactiveMP.interfaceindices(node::MvStackFactorNode, name::Symbol) =
    (ReactiveMP.interfaceindex(node, name),)
ReactiveMP.interfaceindices(
    node::MvStackFactorNode, names::NTuple{N, Symbol}
) where {N} = map(name -> ReactiveMP.interfaceindex(node, name), names)

function ReactiveMP.interfaceindex(node::MvStackFactorNode, name::Symbol)
    if name === :out
        return 1
    elseif name === :inputs
        return 2
    end

    error("Unknown interface ':$(name)' for the [ $(ReactiveMP.functionalform(node)) ] node")
end

function ReactiveMP.factornode(::Type{MvStack}, interfaces, factorisation)
    out_index = findfirst(interface -> first(interface) === :out, interfaces)
    isnothing(out_index) && throw(ArgumentError("`MvStack` requires an `out` interface."))

    input_interfaces = filter(interface -> first(interface) === :inputs, interfaces)
    ninputs = length(input_interfaces)

    ninputs >= 2 || throw(
        ArgumentError("`MvStack` requires at least two inputs; got $(ninputs).")
    )

    return MvStackFactorNode(
        ReactiveMP.NodeInterface(interfaces[out_index]...),
        ntuple(
            index -> ReactiveMP.IndexedNodeInterface(
                index, ReactiveMP.NodeInterface(input_interfaces[index]...)
            ),
            ninputs,
        ),
    )
end

struct MvStackFunctionalDependencies <: ReactiveMP.FunctionalDependencies end

ReactiveMP.collect_functional_dependencies(::MvStackFactorNode, ::Nothing) =
    MvStackFunctionalDependencies()
ReactiveMP.collect_functional_dependencies(
    ::MvStackFactorNode, ::MvStackFunctionalDependencies
) = MvStackFunctionalDependencies()
ReactiveMP.collect_functional_dependencies(::MvStackFactorNode, dependencies) =
    error(
        "The functional dependencies for `MvStack` must be `nothing` or " *
        "`MvStackFunctionalDependencies`, got `$(typeof(dependencies))`."
    )

function ReactiveMP.activate!(
    node::MvStackFactorNode, options::ReactiveMP.FactorNodeActivationOptions
)
    dependencies = ReactiveMP.collect_functional_dependencies(
        node, ReactiveMP.getdependecies(options)
    )
    return ReactiveMP.activate!(dependencies, node, options)
end

function ReactiveMP.functional_dependencies(
    ::MvStackFunctionalDependencies,
    node::MvStackFactorNode{N},
    interface,
    interface_index::Int,
) where {N}
    message_dependencies = if interface_index === 1
        (node.inputs,)
    elseif 2 <= interface_index <= N + 1
        target_index = interface_index - 1
        other_inputs = ntuple(N - 1) do index
            node.inputs[index < target_index ? index : index + 1]
        end
        (node.out, other_inputs)
    else
        error("Bad interface index $(interface_index) for `MvStack`.")
    end

    return message_dependencies, ()
end

function ReactiveMP.collect_latest_messages(
    ::MvStackFunctionalDependencies,
    ::MvStackFactorNode{N},
    dependencies::Tuple{NTuple{N, ReactiveMP.IndexedNodeInterface}},
) where {N}
    inputs = dependencies[1]
    streams = map(ReactiveMP.get_stream_of_inbound_messages, inputs)

    names = Val{(:inputs,)}()
    observable = Rocket.combineLatest(streams, Rocket.PushNew()) |>
        Rocket.map_to((ReactiveMP.ManyOf(streams),))

    return names, observable
end

function ReactiveMP.collect_latest_messages(
    ::MvStackFunctionalDependencies,
    ::MvStackFactorNode,
    dependencies::Tuple{
        ReactiveMP.NodeInterface,
        NTuple{N, ReactiveMP.IndexedNodeInterface},
    },
) where {N}
    out = dependencies[1]
    inputs = dependencies[2]
    out_stream = ReactiveMP.get_stream_of_inbound_messages(out)
    input_streams = map(ReactiveMP.get_stream_of_inbound_messages, inputs)

    names = Val{(:out, :inputs)}()
    observable = Rocket.combineLatest(
        (out_stream, Rocket.combineLatest(input_streams, Rocket.PushNew())),
        Rocket.PushNew(),
    ) |> Rocket.map_to((out_stream, ReactiveMP.ManyOf(input_streams)))

    return names, observable
end

ReactiveMP.collect_latest_marginals(
    ::MvStackFunctionalDependencies, ::MvStackFactorNode, ::Tuple{}
) = (nothing, Rocket.of(nothing))
