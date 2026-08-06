"""Examples of how the axiomatic approach can cover existing methods

These methods are defined by their (index x intervention x atomic x aggregator)
combinations. Can think of them as points in 4D space, we move along particular
axes.
"""

from .core import Explainer
from .index import FeatureCoalitions, PathSteps, UnitIndices, SinglePoint
from .intervention import BaselineMask, MarginalMask, ZeroMask
from .atomic import MarginalContribution, PathIntegratedGradient, AblationDelta, Gradient
from .aggregate import ShapleyWeights, Min, Mean, Identity


def shap(background, n_orderings=200, seed=0) -> Explainer:
    return Explainer(
        index=FeatureCoalitions(n_orderings, seed),
        intervention=MarginalMask(background, seed=seed),
        atomic=MarginalContribution(),
        aggregator=ShapleyWeights(),
    )


def marginalminshap(background, n_orderings=200, seed=0) -> Explainer:
    # Compared to shap, this replaces the aggregator ShapleyWeights -> Min
    return shap(background, n_orderings, seed).replace(aggregator=Min())


def baseline_shap(baseline, n_orderings=200, seed=0) -> Explainer:
    # Compared to shap, this replaces MarginalMask -> BaselineMask
    return Explainer(
        index=FeatureCoalitions(n_orderings, seed),
        intervention=BaselineMask(baseline),
        atomic=MarginalContribution(),
        aggregator=ShapleyWeights(),
    )


def integrated_gradients(baseline, n_steps=64, eps=1e-5) -> Explainer:
    """Integrated Gradients explainer.
    
    Args:
        baseline: baseline/reference point for integration
        n_steps: number of interpolation steps along the path
        eps: step size for finite-difference gradient approximation
    
    Returns:
        Explainer configured for integrated gradients
    """
    return Explainer(
        index=PathSteps(n_steps),
        intervention=BaselineMask(baseline),  # unused by the atomic
        atomic=PathIntegratedGradient(baseline, eps=eps),
        aggregator=Mean(),
    )


def saliency(eps=1e-5) -> Explainer:
    """Vanilla-gradient saliency map: score_j = df/dx_j at x.

    Args:
        eps: step size for finite-difference gradient approximation

    Returns:
        Explainer configured for saliency maps
    """
    return Explainer(
        index=SinglePoint(),
        intervention=ZeroMask(),  # unused by the atomic
        atomic=Gradient(eps=eps),
        aggregator=Identity(),
    )


def sae_attribution() -> Explainer:
    """Mechanistic methods apply feature removal to latent features

    Call .explain(f, z) where f = downstream decoded function value and z is the
    latent code.
    """
    return Explainer(
        index=UnitIndices(),
        intervention=ZeroMask(),
        atomic=AblationDelta(),
        aggregator=Identity(),
    )
