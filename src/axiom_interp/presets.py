"""Examples of how the axiomatic approach can cover existing methods

These methods are defined by their (index x intervention x atomic x aggregator)
combinations. Can think of them as points in 4D space, we move along particular
axes.
"""

from .core import Explainer
from .index import FeatureCoalitions, PathSteps, UnitIndices
from .intervention import BaselineMask, MarginalMask, ZeroMask
from .atomic import MarginalContribution, PathIntegratedGradient, AblationDelta
from .aggregate import ShapleyWeights, Min, Mean, Identity


def shap(background, n_orderings=200, seed=0) -> Explainer:
    return Explainer(
        index=FeatureCoalitions(n_orderings, seed),
        intervention=MarginalMask(background, seed=seed),
        atomic=MarginalContribution(),
        aggregator=ShapleyWeights(),
    )


def minshap(background, n_orderings=200, seed=0) -> Explainer:
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


def integrated_gradients(baseline, n_steps=64) -> Explainer:
    return Explainer(
        index=PathSteps(n_steps),
        intervention=BaselineMask(baseline),  # unused by the atomic
        atomic=PathIntegratedGradient(baseline),
        aggregator=Mean(),
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
