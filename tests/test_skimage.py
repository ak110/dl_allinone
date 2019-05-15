
def test_run(data_dir):
    import skimage.io
    assert skimage.io.imread(data_dir / 'data.jpg').shape == (20, 124, 3)
