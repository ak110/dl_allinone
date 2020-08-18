def test_run():
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese"
    )
    tokens = tokenizer.tokenize("すもももももももものうち")
    assert tuple(tokens) == ("す", "##も", "##も", "も", "もも", "も", "もも", "の", "うち")
