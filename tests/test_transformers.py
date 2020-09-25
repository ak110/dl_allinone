def test_run():
    import ipadic
    import transformers

    tokenizer = transformers.BertJapaneseTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese", mecab_kwargs={"mecab_option": ipadic.MECAB_ARGS}
    )
    tokens = tokenizer.tokenize("すもももももももものうち")
    assert tuple(tokens) == ("す", "##も", "##も", "も", "もも", "も", "もも", "の", "うち")
