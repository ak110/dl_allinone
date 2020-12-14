import pytest


@pytest.mark.xfail(reason="MXNet")
def test_run_nlp():
    import gluonnlp

    counter = gluonnlp.data.count_tokens(["alpha", "beta", "gamma", "beta"])
    vocab = gluonnlp.Vocab(counter)
    assert vocab["beta"] == 4


@pytest.mark.xfail(reason="MXNet")
def test_run_cv():
    import gluoncv
    import mxnet as mx

    img = mx.random.uniform(0, 255, (100, 100, 3)).astype("uint8")
    img = gluoncv.data.transforms.image.imresize(img, 200, 200)
    assert img.shape == (200, 200, 3)
