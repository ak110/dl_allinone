
def test_xgboost():
    import sklearn.datasets
    import xgboost

    data = sklearn.datasets.load_boston()
    X, y = data.data, data.target

    xgb = xgboost.XGBClassifier(n_estimators=3)
    xgb.fit(X[:50], y[:50])
    assert xgb.predict(X[50:100]).shape == (50,)
