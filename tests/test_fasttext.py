def test_run():
    import fasttext

    assert tuple(fasttext.FastText.tokenize("Hello World")) == ("Hello", "World")
