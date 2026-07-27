include(joinpath(@__DIR__, "mnist_softplus_rxinfer_mlp_two_hidden.jl"))
include(joinpath(@__DIR__, "mnist_residual_sine_surrogate_common.jl"))

"""
Train the two-hidden-layer flattened ResidualSine classifier.

Both nonlinear layers use the same ResidualSine parameters. The forward
activation moments and backward Gaussian Fisher projections are closed form;
the categorical and uncertain-product sites remain Gaussian approximations.
"""
function train_residual_sine_mlp_two_hidden(
    ;
    residual_sine_rho=0.9,
    residual_sine_omega=1.0,
    kwargs...,
)
    hooks = residual_sine_activation_hooks(
        rho=residual_sine_rho,
        omega=residual_sine_omega,
    )
    return train_two_hidden_mlp_rxinfer_demo(
        ;
        kwargs...,
        activation_to_a_site=hooks.activation_to_a_site,
        activation_to_x_site=hooks.activation_to_x_site,
        activation_value=hooks.activation_value,
        activation_derivative=hooks.activation_derivative,
        activation_name="ResidualSine(ρ=$residual_sine_rho, ω=$residual_sine_omega)",
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_residual_sine_mlp_two_hidden()
end
