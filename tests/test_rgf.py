
def test_rgf():
    import sklearn.datasets
    import rgf.sklearn
    data = sklearn.datasets.load_iris()
    X, y = data.data, data.target

    rgf = rgf.sklearn.RGFClassifier(max_leaf=100, algorithm="RGF_Sib", test_interval=100, n_iter=2)
    rgf.fit(X, y)
    assert rgf.predict(X).shape == (len(X),)
