using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using ProbabilisticEnsembling
using RxInfer
using SurrogateModelling

# Two-level Softplus precision-gated PoE model.
#
# For every observation and hidden branch, `n_neurons` first-level experts share
# one `hidden_out`, so their Gaussian factors form a product of experts. The
# hidden output is transformed by `Softplus` into the precision of a final expert
# whose mean is computed directly from the original features. All final experts
# share `y`, forming a second product of experts.
#
# When `connect_hidden_to_y` is true, each completed hidden PoE also contributes
# one auxiliary Gaussian factor to `y`. That factor is deliberately created after
# the inner neuron loop, so it appears once per hidden branch rather than once per
# first-level expert.
#
# Expected entries in `priors`:
#
# - Scalars: `:τ_mean`, `:τ_gate`, `:τ_h`, and `:β_hidden`.
# - Per-hidden arrays: `:w_h`, `:β_out`, and, when requested, `:τ_hidden`.
# - Hidden-by-neuron arrays: `:w_mean` and `:w_a`.
@model function xor_softplus_hidden_poe(
    n_hidden,
    n_neurons,
    features,
    y,
    priors,
    softplus_dependencies,
    softplus_damping,
    normal_dependencies,
    normal_damping,
    connect_hidden_to_y,
)
    local w_mean, w_a, w_h
    local z_mean, za, γ_hidden, hidden_out
    local mean_contribution, gate_contribution
    local τ_mean, τ_gate, τ_h
    local β_hidden, β_out, τ_hidden

    τ_mean ~ priors[:τ_mean]
    τ_gate ~ priors[:τ_gate]
    τ_h ~ priors[:τ_h]
    β_hidden ~ priors[:β_hidden]

    for hidden in 1:n_hidden
        w_h[hidden] ~ priors[:w_h][hidden]
        β_out[hidden] ~ priors[:β_out][hidden]

        if connect_hidden_to_y
            τ_hidden[hidden] ~ priors[:τ_hidden][hidden]
        end

        for neuron in 1:n_neurons
            w_mean[hidden, neuron] ~ priors[:w_mean][hidden, neuron]
            w_a[hidden, neuron] ~ priors[:w_a][hidden, neuron]
        end
    end

    for observation in eachindex(y)
        for hidden in 1:n_hidden
            # The first-level Gaussian factors share this hidden output and
            # therefore form one PoE for each (hidden, observation) pair.
            for neuron in 1:n_neurons
                z_mean[hidden, neuron, observation] ~ softdot(
                    features[observation],
                    w_mean[hidden, neuron],
                    τ_mean,
                )

                za[hidden, neuron, observation] ~ softdot(
                    features[observation],
                    w_a[hidden, neuron],
                    τ_gate,
                ) where {
                    meta = LowRankMeta(),
                }

                γ_hidden[hidden, neuron, observation] ~
                    GammaShapeRate(1.0, β_hidden)

                γ_hidden[hidden, neuron, observation] ~ Softplus(
                    za[hidden, neuron, observation],
                ) where {
                    dependencies = softplus_dependencies,
                    meta = softplus_damping,
                }

                hidden_out[hidden, observation] ~ NormalMeanPrecision(
                    z_mean[hidden, neuron, observation],
                    γ_hidden[hidden, neuron, observation],
                ) where {
                    dependencies = normal_dependencies,
                    meta = normal_damping,
                }
            end

            # Optional deep supervision: exactly one factor per completed
            # hidden PoE, never one copy per contributing neuron.
            if connect_hidden_to_y
                y[observation] ~ NormalMeanPrecision(
                    hidden_out[hidden, observation],
                    τ_hidden[hidden],
                )
            end

            # The final expert mean depends on the original features, not on
            # hidden_out. The hidden output controls only its precision.
            mean_contribution[hidden, observation] ~ softdot(
                features[observation],
                w_h[hidden],
                τ_h,
            )

            gate_contribution[hidden, observation] ~
                GammaShapeRate(1.0, β_out[hidden])

            gate_contribution[hidden, observation] ~ Softplus(
                hidden_out[hidden, observation],
            ) where {
                dependencies = softplus_dependencies,
                meta = softplus_damping,
            }

            # Repeating this factor over `hidden` is intentional: the final
            # prediction is a PoE over the hidden-specific mean/gate pairs.
            y[observation] ~ NormalMeanPrecision(
                mean_contribution[hidden, observation],
                gate_contribution[hidden, observation],
            ) where {
                dependencies = normal_dependencies,
                meta = normal_damping,
            }
        end
    end
end

@constraints function xor_softplus_hidden_poe_training_constraints(connect_hidden_to_y)
    if connect_hidden_to_y
      q(
          w_mean,
          w_a,
          w_h,
          z_mean,
          za,
          γ_hidden,
          hidden_out,
          mean_contribution,
          gate_contribution,
          τ_mean,
          τ_gate,
          τ_h,
          β_hidden,
          β_out,
          τ_hidden,
      ) =
          q(w_mean) *
          q(w_a) *
          q(w_h) *
          q(
              z_mean,
              za,
              γ_hidden,
              hidden_out,
              mean_contribution,
              gate_contribution,
          ) *
          q(τ_mean) *
          q(τ_gate) *
          q(τ_h) *
          q(β_hidden) *
          q(β_out) *
          q(τ_hidden)
    else
        q(
          w_mean,
          w_a,
          w_h,
          z_mean,
          za,
          γ_hidden,
          hidden_out,
          mean_contribution,
          gate_contribution,
          τ_mean,
          τ_gate,
          τ_h,
          β_hidden,
          β_out
      ) =
          q(w_mean) *
          q(w_a) *
          q(w_h) *
          q(
              z_mean,
              za,
              γ_hidden,
              hidden_out,
              mean_contribution,
              gate_contribution,
          ) *
          q(τ_mean) *
          q(τ_gate) *
          q(τ_h) *
          q(β_hidden) *
          q(β_out)
    end 

    q(w_mean)::MomentForm()
    q(w_a)::MomentForm()
    q(w_h)::MomentForm()
end