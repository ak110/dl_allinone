def test_run():
    import sklearn.datasets
    import xgboost

    data = sklearn.datasets.load_boston()
    X, y = data.data, data.target  # pylint: disable=no-member

    xgb = xgboost.XGBRegressor(n_estimators=3)
    xgb.fit(X[:100], y[:100])
    assert xgb.predict(X[100:]).shape == (len(X[100:]),)
