"""
    NGMPDependencies(specification::NamedTuple)
    NGMPDependencies(; specification...)

A `ReactiveMP.FunctionalDependencies` policy for natural-gradient message passing.
Behaves as `DefaultFunctionalDependencies` (messages from the same factorization
cluster, marginals of the other clusters — so it composes with BP / structured /
mean-field factorization constraints), except that for every interface named in the
specification it additionally:

  - subscribes the outbound message to the *current marginal of the same edge*
    (delivered to the rule as `q_<edge>`),
  - dispatches the rule with the [`NaturalGradientMessage`](@ref) constraint tag
    instead of `Marginalisation`,
  - wraps the node meta into a fresh per-edge [`NGMPEdgeState`](@ref) so the rule
    can apply stateful damping/momentum.

Keys of the specification are the constrained interface names; values are optional
initial marginals used to break the message ↔ marginal cycle (pass `nothing` and use
`@initialization q(x) = ...` instead, which seeds the same stream).

```julia
y[k] ~ PoissonExp(z[k]) where { dependencies = NGMPDependencies(in = nothing), meta = DampingMeta(0.5, 0.2) }
```
"""
struct NGMPDependencies{S <: NamedTuple, P} <: ReactiveMP.FunctionalDependencies
    specification::S
    # Tangent-projection strategy carried into every `NaturalGradientMessage` this
    # policy dispatches (rules resolve it via `resolve_projection`); the sentinel
    # default resolves to `TangentProjection(type = ClosedForm)`.
    projection::P
    # Diagnostic registry of the per-edge states created at activation (one per
    # constrained interface per node); lets tests and users inspect damping state
    # and firing counts after inference.
    states::Vector{NGMPEdgeState}
end

NGMPDependencies(specification::NamedTuple; projection = ClosedFormDefault()) =
    NGMPDependencies(specification, projection, NGMPEdgeState[])
NGMPDependencies(; projection = ClosedFormDefault(), kwargs...) =
    NGMPDependencies((; kwargs...); projection = projection)

is_ngmp_interface(dependencies::NGMPDependencies, iname::Symbol) = iname ∈ keys(dependencies.specification)

# Port of `ReactiveMP.functional_dependencies(::RequireMarginalFunctionalDependencies, ...)`:
# default cluster-derived dependencies plus the same-edge marginal for constrained interfaces.
function ReactiveMP.functional_dependencies(dependencies::NGMPDependencies, factornode, interface, iindex)
    specification = dependencies.specification

    clusters = ReactiveMP.getlocalclusters(factornode)
    # Find the index of the cluster for the current interface
    cindex = ReactiveMP.clusterindex(clusters, iindex)
    # Fetch the actual cluster
    cluster = ReactiveMP.getfactorization(clusters, cindex)
    # Remove current edge index from the list of dependencies in the given cluster
    vdependencies = filter(ci -> ci !== iindex, cluster)
    # Map interface indices to the actual interfaces to get the messages dependencies
    message_dependencies = Iterators.map(inds -> map(i -> ReactiveMP.getinterface(factornode, i), inds), vdependencies)

    # For the marginal dependencies we need to skip the current cluster
    marginal_dependencies_default_clusters      = ReactiveMP.skipindex(ReactiveMP.get_node_local_marginals(clusters), cindex)
    marginal_dependencies_default_factorization = ReactiveMP.skipindex(ReactiveMP.getfactorization(clusters), cindex)

    marginal_dependencies = if is_ngmp_interface(dependencies, ReactiveMP.name(interface))
        # Auxiliary local marginal connected to the marginal stream of the variable on this edge:
        # the natural-gradient projection point q(x) for the message toward x
        extra_localmarginal = ReactiveMP.FactorNodeLocalMarginal(ReactiveMP.name(interface))
        extra_stream = ReactiveMP.MarginalObservable()
        ReactiveMP.connect!(extra_stream, ReactiveMP.get_stream_of_marginals(ReactiveMP.getvariable(interface)))
        ReactiveMP.set_stream_of_marginals!(extra_localmarginal, extra_stream)

        initialmarginal = specification[ReactiveMP.name(interface)]
        if !isnothing(initialmarginal)
            ReactiveMP.set_initial_marginal!(extra_stream, initialmarginal)
        end

        insertafter = sum(first(el) < iindex ? 1 : 0 for el in marginal_dependencies_default_factorization; init = 0)
        TupleTools.insertafter(marginal_dependencies_default_clusters, insertafter, (extra_localmarginal,))
    else
        marginal_dependencies_default_clusters
    end

    return message_dependencies, marginal_dependencies
end

# Port of the generic `ReactiveMP.activate!(::FunctionalDependencies, factornode, options)`,
# with two changes for NGMP-constrained interfaces: the rule dispatches on
# `NaturalGradientMessage()` instead of the hardcoded `Marginalisation()`, and the meta
# is wrapped into a fresh per-edge `NGMPEdgeState` (each interface owns its `MessageMapping`,
# which makes the damping state per node per edge).
function ReactiveMP.activate!(dependencies::NGMPDependencies, factornode, options)
    annotations          = ReactiveMP.getannotations(options)
    rulefallback         = ReactiveMP.getrulefallback(options)
    callbacks            = ReactiveMP.getcallbacks(options)
    fform                = ReactiveMP.functionalform(factornode)
    meta                 = ReactiveMP.collect_meta(fform, ReactiveMP.getmetadata(options))
    stream_postprocessor = ReactiveMP.getpostprocessor(options)

    foreach(enumerate(ReactiveMP.getinterfaces(factornode))) do (iindex, interface)
        if ReactiveMP.israndom(interface) || ReactiveMP.isdata(interface)
            ReactiveMP.with_functional_dependencies(dependencies, factornode, interface, iindex) do message_dependencies, marginal_dependencies
                messagestag, messages   = ReactiveMP.collect_latest_messages(dependencies, factornode, message_dependencies)
                marginalstag, marginals = ReactiveMP.collect_latest_marginals(dependencies, factornode, marginal_dependencies)

                vtag        = ReactiveMP.tag(interface)
                constrained = is_ngmp_interface(dependencies, ReactiveMP.name(interface))
                vconstraint = constrained ? NaturalGradientMessage(dependencies.projection) : ReactiveMP.Marginalisation()
                mappingmeta = constrained ? NGMPEdgeState(meta) : meta
                constrained && push!(dependencies.states, mappingmeta)

                stream_of_outbound_messages = Rocket.combineLatest((messages, marginals), Rocket.PushNew())

                mapping =
                    let messagemap = ReactiveMP.MessageMapping(
                            fform,
                            vtag,
                            vconstraint,
                            messagestag,
                            marginalstag,
                            mappingmeta,
                            annotations,
                            ReactiveMP.node_if_required(fform, factornode),
                            rulefallback,
                            callbacks
                        )
                        (dependencies) -> ReactiveMP.DeferredMessage(dependencies[1], dependencies[2], messagemap)
                    end

                stream_of_outbound_messages = stream_of_outbound_messages |> Rocket.map(ReactiveMP.AbstractMessage, mapping)
                stream_of_outbound_messages = ReactiveMP.postprocess_stream_of_outbound_messages(stream_postprocessor, stream_of_outbound_messages)
                ReactiveMP.set_stream_of_outbound_messages!(interface, stream_of_outbound_messages)
            end
        end
    end
end
