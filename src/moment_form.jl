import ReactiveMP
import BayesBase: GenericProd
import ExponentialFamily: MvNormalWeightedMeanPrecision, MvNormalMeanCovariance

export MomentForm

"""
    MomentForm

Marginal form constraint that converts a `MvNormalWeightedMeanPrecision` marginal to `MvNormalMeanCovariance.`
It can be usefull when some specific vmp/structured rule depends on mean and cov, and you don't want to convert actual marginal into the mean covariance form, in every rule call but only once.

Usage: `q(w)::MomentForm()` inside `@constraints`.
"""
struct MomentForm <: ReactiveMP.AbstractFormConstraint end

ReactiveMP.default_form_check_strategy(::MomentForm) = ReactiveMP.FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::MomentForm) = GenericProd()
ReactiveMP.constrain_form(::MomentForm, d::MvNormalWeightedMeanPrecision) =
    convert(MvNormalMeanCovariance, d)
ReactiveMP.constrain_form(::MomentForm, d) = d
