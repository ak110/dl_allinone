
def test_nltk():
    import nltk
    assert tuple(nltk.word_tokenize('Hello World')) == ('Hello', 'World')
