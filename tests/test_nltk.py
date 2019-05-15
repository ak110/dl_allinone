
def test_run():
    import nltk
    assert tuple(nltk.word_tokenize('Hello World')) == ('Hello', 'World')
