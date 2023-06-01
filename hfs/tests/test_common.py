import pytest
from sklearn.utils.estimator_checks import check_estimator

from hfs import TemplateClassifier, TemplateEstimator, TemplateTransformer, TSELSelector


@pytest.mark.parametrize(
    "estimator",
    [
        TSELSelector(),
        TemplateEstimator(),
        TemplateTransformer(),
        TemplateClassifier(),
    ],
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
