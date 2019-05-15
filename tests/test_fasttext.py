
def test_run():
    import fastText
    assert tuple(fastText.FastText.tokenize('Hello World')) == ('Hello', 'World')
