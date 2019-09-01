def test_run(tmpdir):
    import joblib

    obj = {"Hello": "World!"}
    joblib.dump(obj, str(tmpdir / "obj.pkl"))
    obj2 = joblib.load(str(tmpdir / "obj.pkl"))
    assert tuple(obj.items()) == tuple(obj2.items())
