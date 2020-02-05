from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skrules import SkopeRules


X, y = load_diabetes(return_X_y=True)
feature_names = [
    'age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'
]


def test_creates_rules():
    clf = SkopeRules(
        regression=True,
        max_depth_duplication=2,
        n_estimators=30,
        precision_min=0.0,
        recall_min=0.0,
        feature_names=feature_names
    )
    clf.fit(X, y)
    rules = clf.rules_
    assert len(rules) > 0


def test_cutoffs_still_produce_rules():
    clf = SkopeRules(
        regression=True,
        max_depth_duplication=2,
        n_estimators=30,
        precision_min=0.20,
        recall_min=0.20,
        feature_names=feature_names
    )
    clf.fit(X, y)
    rules = clf.rules_
    assert len(rules) > 0


def test_can_predict():
    clf = SkopeRules(
        regression=True,
        max_depth_duplication=2,
        n_estimators=30,
        precision_min=0.20,
        recall_min=0.20,
        feature_names=feature_names
    )
    clf.fit(X, y)
    clf.predict(X)


def test_performance_not_deteriorate():
    '''Compare the model performance to baselines.

    It's a bit unclear what to compare against since performance
    varies widely across models (in mse; vanilla settings):
    decision tree regressor: 6946
    random forest regressor: 2970
    linear model: 2820
    '''
    clf = SkopeRules(
        regression=True,
        max_depth_duplication=None,
        max_depth=X.shape[1] // 3,
        n_estimators=850,
        precision_min=0.,
        recall_min=.0,
        feature_names=feature_names
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    # comparing to a baseline from linear regression:
    assert mse < 2820
