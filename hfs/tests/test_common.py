import networkx as nx
import pytest
from sklearn.utils.estimator_checks import check_estimator

from hfs import (
    TemplateClassifier,
    TemplateEstimator,
    TemplateTransformer,
    TreeBasedFeatureSelector,
)


@pytest.mark.parametrize(
    "estimator",
    [
        TreeBasedFeatureSelector(),
        TemplateEstimator(),
        TemplateTransformer(),
        TemplateClassifier(),
    ],
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
