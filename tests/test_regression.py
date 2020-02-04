from sklearn.datasets import load_diabetes
from skrules import SkopeRules


def test_regression_works():
    X, y = load_diabetes(return_X_y=True)
    clf = SkopeRules(
        regression=True,
        max_depth_duplication=2,
        n_estimators=30,
        precision_min=0.3,
        recall_min=0.1,
        feature_names=feature_names
    )
    clf.fit(X, y == idx)
    rules = clf.rules_[0:3]
    assert len(rules) > 0


