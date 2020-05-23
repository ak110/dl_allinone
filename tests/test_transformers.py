
def test_run():
    from transformers.tokenization_bert_japanese import BertJapaneseTokenizer

    tokenizer = BertJapaneseTokenizer.from_pretrained(
        "bert-base-japanese-whole-word-masking"
    )
    tokens = tokenizer.tokenize("すもももももももものうち")
    assert tuple(tokens) == ("す", "##も", "##も", "も", "もも", "も", "もも", "の", "うち")
