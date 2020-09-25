def test_run():
    import rgf.sklearn
    import sklearn.datasets

    data = sklearn.datasets.load_iris()
    X, y = data.data, data.target  # pylint: disable=no-member

    rgf = rgf.sklearn.RGFClassifier(
        max_leaf=100, algorithm="RGF_Sib", test_interval=100, n_iter=2
    )
    rgf.fit(X, y)
    assert rgf.predict(X).shape == (len(X),)
