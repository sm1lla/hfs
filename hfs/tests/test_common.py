import pytest
from sklearn.utils.estimator_checks import check_estimator

from hfs import (
    HierarchicalEstimator,
    HierarchicalFeatureSelector,
    HierarchicalPreprocessor,
    TemplateClassifier,
    TemplateEstimator,
    TemplateTransformer,
)

from ..gtd import GreedyTopDownSelector
from ..hill_climbing import BottomUpSelector, TopDownSelector
from ..hip import HIP
from ..hnb import HNB
from ..hnbs import HNBs
from ..mr import MR
from ..rnb import RNB
from ..shsel import SHSELSelector
from ..tsel import TSELSelector


@pytest.mark.parametrize(
    "estimator",
    [
        TSELSelector(),
        HierarchicalEstimator(),
        HierarchicalFeatureSelector(),
        HierarchicalPreprocessor(),
        TopDownSelector(),
        SHSELSelector(),
        TemplateEstimator(),
        TemplateTransformer(),
        TemplateClassifier(),
        HNB(),
        HNBs(),
        RNB(),
        MR(),
        HIP(),
        BottomUpSelector(),
        GreedyTopDownSelector(),
    ],
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
