
def test_fasttext():
    import fastText
    assert tuple(fastText.FastText.tokenize('Hello World')) == ('Hello', 'World')
