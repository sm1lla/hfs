import pytest
from sklearn.utils.estimator_checks import check_estimator

from hfs import (
    EagerHierarchicalFeatureSelector,
    HierarchicalEstimator,
    HierarchicalPreprocessor,
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
        EagerHierarchicalFeatureSelector(),
        HierarchicalPreprocessor(),
        TopDownSelector(),
        SHSELSelector(),
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
