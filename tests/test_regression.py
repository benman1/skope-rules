from sklearn.datasets import load_diabetes
from skrules import SkopeRules


def test_regression_works():
    X, y = load_diabetes(return_X_y=True)
    feature_names = [
        'age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'
    ]
    clf = SkopeRules(
        regression=True,
        max_depth_duplication=2,
        n_estimators=30,
        precision_min=1.3,
        recall_min=1.1,
        feature_names=feature_names
    )
    clf.fit(X, y)
    rules = clf.rules_[0:3]
    assert len(rules) > 0
