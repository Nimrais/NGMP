"""
    ManyPlus

A deterministic factor that adds a collection of two or more scalar Gaussian
inputs with a single factor node.

Use it in an RxInfer model as

```julia
total := ManyPlus(inputs = [x1, x2, x3])
```
"""
struct ManyPlus end

ReactiveMP.as_node_symbol(::Type{ManyPlus}) = :ManyPlus
ReactiveMP.interfaces(::Type{ManyPlus}) = Val((:out, :inputs))
ReactiveMP.inputinterfaces(::Type{ManyPlus}) = Val((:inputs,))
ReactiveMP.alias_interface(::Type{ManyPlus}, ::Int64, name::Symbol) = name
ReactiveMP.is_predefined_node(::Type{ManyPlus}) =
    ReactiveMP.PredefinedNodeFunctionalForm()
ReactiveMP.sdtype(::Type{ManyPlus}) = ReactiveMP.Deterministic()

struct ManyPlusNodeFactorisation end

ReactiveMP.collect_factorisation(::Type{ManyPlus}, factorisation) =
    ManyPlusNodeFactorisation()

struct ManyPlusFactorNode{N} <: ReactiveMP.AbstractFactorNode
    out::ReactiveMP.NodeInterface
    inputs::NTuple{N, ReactiveMP.IndexedNodeInterface}
end

ReactiveMP.functionalform(::ManyPlusFactorNode) = ManyPlus
ReactiveMP.getinterfaces(node::ManyPlusFactorNode) = (node.out, node.inputs...)
ReactiveMP.getinboundinterfaces(node::ManyPlusFactorNode) = node.inputs
ReactiveMP.sdtype(::ManyPlusFactorNode) = ReactiveMP.Deterministic()

ReactiveMP.interfaceindices(node::ManyPlusFactorNode, name::Symbol) =
    (ReactiveMP.interfaceindex(node, name),)
ReactiveMP.interfaceindices(
    node::ManyPlusFactorNode, names::NTuple{N, Symbol}
) where {N} = map(name -> ReactiveMP.interfaceindex(node, name), names)

function ReactiveMP.interfaceindex(node::ManyPlusFactorNode, name::Symbol)
    if name === :out
        return 1
    elseif name === :inputs
        return 2
    end

    error("Unknown interface ':$(name)' for the [ $(ReactiveMP.functionalform(node)) ] node")
end

function ReactiveMP.factornode(
    ::Type{ManyPlus}, interfaces, factorisation
)
    out_index = findfirst(interface -> first(interface) === :out, interfaces)
    isnothing(out_index) && throw(ArgumentError("`ManyPlus` requires an `out` interface."))

    input_interfaces = filter(interface -> first(interface) === :inputs, interfaces)
    ninputs = length(input_interfaces)

    ninputs >= 2 || throw(
        ArgumentError("`ManyPlus` requires at least two inputs; got $(ninputs).")
    )

    return ManyPlusFactorNode(
        ReactiveMP.NodeInterface(interfaces[out_index]...),
        ntuple(
            index -> ReactiveMP.IndexedNodeInterface(
                index, ReactiveMP.NodeInterface(input_interfaces[index]...)
            ),
            ninputs,
        ),
    )
end

struct ManyPlusFunctionalDependencies <: ReactiveMP.FunctionalDependencies end

ReactiveMP.collect_functional_dependencies(::ManyPlusFactorNode, ::Nothing) =
    ManyPlusFunctionalDependencies()
ReactiveMP.collect_functional_dependencies(
    ::ManyPlusFactorNode, ::ManyPlusFunctionalDependencies
) = ManyPlusFunctionalDependencies()
ReactiveMP.collect_functional_dependencies(::ManyPlusFactorNode, dependencies) =
    error(
        "The functional dependencies for `ManyPlus` must be `nothing` or " *
        "`ManyPlusFunctionalDependencies`, got `$(typeof(dependencies))`."
    )

function ReactiveMP.activate!(
    node::ManyPlusFactorNode, options::ReactiveMP.FactorNodeActivationOptions
)
    dependencies = ReactiveMP.collect_functional_dependencies(
        node, ReactiveMP.getdependecies(options)
    )
    return ReactiveMP.activate!(dependencies, node, options)
end

function ReactiveMP.functional_dependencies(
    ::ManyPlusFunctionalDependencies,
    node::ManyPlusFactorNode{N},
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
        error("Bad interface index $(interface_index) for `ManyPlus`.")
    end

    return message_dependencies, ()
end

function ReactiveMP.collect_latest_messages(
    ::ManyPlusFunctionalDependencies,
    ::ManyPlusFactorNode{N},
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
    ::ManyPlusFunctionalDependencies,
    ::ManyPlusFactorNode,
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
    ::ManyPlusFunctionalDependencies, ::ManyPlusFactorNode, ::Tuple{}
) = (nothing, Rocket.of(nothing))
