import pytest
from sklearn.utils.estimator_checks import check_estimator

from hfs import (
    HierarchicalEstimator,
    HierarchicalFeatureSelector,
    HierarchicalPreprocessor,
    HillClimbingSelector,
    SHSELSelector,
    TemplateClassifier,
    TemplateEstimator,
    TemplateTransformer,
    TSELSelector,
)


@pytest.mark.parametrize(
    "estimator",
    [
        TSELSelector(),
        HierarchicalEstimator(),
        HierarchicalFeatureSelector(),
        HierarchicalPreprocessor(),
        HillClimbingSelector(),
        SHSELSelector(),
        TemplateEstimator(),
        TemplateTransformer(),
        TemplateClassifier(),
    ],
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
