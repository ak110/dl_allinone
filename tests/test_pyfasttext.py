def test_run(tmpdir, data_dir):
    import pyfasttext

    model = pyfasttext.FastText()
    model.supervised(
        input=data_dir / "data.txt", output=tmpdir / "model", epoch=1, lr=0.7
    )
