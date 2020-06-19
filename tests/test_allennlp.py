def test_run():
    import allennlp.data.tokenizers

    tokenizer = allennlp.data.tokenizers.WhitespaceTokenizer()
    tokens = [t.text for t in tokenizer.tokenize("Hello World")]
    assert tuple(tokens) == ("Hello", "World")
