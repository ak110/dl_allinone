
def test_run():
    import allennlp.data.tokenizers
    tokenizer = allennlp.data.tokenizers.WordTokenizer(start_tokens=['<'], end_tokens=['>'])
    tokens = tuple([t.text for t in tokenizer.tokenize('Hello World')])
    assert tokens == ('<', 'Hello', 'World', '>')
