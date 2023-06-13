import pytest

from sklearn.utils.estimator_checks import check_estimator

from hfs import TemplateEstimator
from hfs import TemplateClassifier
from hfs import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
