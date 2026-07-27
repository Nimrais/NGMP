using SurrogateModelling
import ExponentialFamily

"""
Activation hooks for the MNIST Gaussian-surrogate MLPs.

The forward site uses the exact first two moments of ResidualSine under a
Gaussian input. The backward site is the exact Gaussian Fisher projection of
the Gaussian output-cavity message at the current Gaussian input marginal.
Only the surrounding categorical/linear surrogate construction remains
approximate.
"""
function residual_sine_activation_hooks(; rho=0.9, omega=1.0)
    activation = ResidualSineMeta(; rho, omega)

    activation_value(value) =
        SurrogateModelling._residual_sine(value, activation)
    activation_derivative(value) =
        SurrogateModelling._residual_sine_prime(value, activation)

    function activation_to_x_site(mean_a, variance_a, activation_noise)
        transformed_mean, transformed_variance =
            SurrogateModelling._residual_sine_mean_var_1d(
                mean_a,
                max(variance_a, SITE_EPS),
                activation,
            )
        observation_variance =
            max(transformed_variance + activation_noise, SITE_EPS)
        return transformed_mean / observation_variance,
            inv(observation_variance)
    end

    function activation_to_a_site(
        mean_a,
        variance_a,
        mean_x_cavity,
        variance_x_cavity,
        activation_noise,
    )
        output_variance =
            max(variance_x_cavity + activation_noise, SITE_EPS)
        projection =
            SurrogateModelling._project_residual_sine_backward_1d(
                NormalMeanVariance(mean_a, max(variance_a, SITE_EPS)),
                SurrogateModelling.ResidualSineGaussianBackwardMessage(
                    mean_x_cavity / output_variance,
                    inv(output_variance),
                    activation,
                ),
            )
        natural = ExponentialFamily.getnaturalparameters(projection)
        precision = clamp(max(-2natural[2], 0.0), 0.0, 1e3)
        weighted_mean = isfinite(natural[1]) && precision > SITE_EPS ?
                        natural[1] : 0.0
        return weighted_mean, precision
    end

    return (;
        activation,
        activation_to_a_site,
        activation_to_x_site,
        activation_value,
        activation_derivative,
    )
end
