def test_run():
    import lightgbm as lgb
    import sklearn.datasets

    data = sklearn.datasets.load_boston()
    X, y = data.data, data.target  # pylint: disable=no-member

    lgb_train = lgb.Dataset(X[:100], y[:100])
    lgb_eval = lgb.Dataset(X[100:], y[100:], reference=lgb_train)

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"l2", "l1"},
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 0,
    }
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1,
        valid_sets=lgb_eval,
        early_stopping_rounds=1,
    )
    assert gbm.best_iteration == 1
