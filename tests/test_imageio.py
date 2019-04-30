
def test_imageio(data_dir):
    import imageio
    assert imageio.imread(data_dir / 'data.jpg').shape == (20, 124, 3)
