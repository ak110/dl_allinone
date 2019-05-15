
def test_run(data_dir):
    import pytext
    featurizer = pytext.data.featurizer.SimpleFeaturizer.from_config(
        pytext.data.featurizer.SimpleFeaturizer.Config(),
        pytext.config.field_config.FeatureConfig()
    )
    data = pytext.data.featurizer.InputRecord(raw_text='Hello World')
    tokens = featurizer.featurize(data).tokens
    assert tuple(tokens) == ('hello', 'world')
