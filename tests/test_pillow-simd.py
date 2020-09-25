def test_run():
    import numpy as np
    import PIL.Image

    # Pillow x.x.x → Pillow-SIMD x.x.x.postx
    assert ".post" in PIL.__version__

    image = np.zeros((32, 32), np.uint8)
    image = np.asarray(PIL.Image.fromarray(image).resize((15, 16)))
    assert image.shape == (16, 15)
    assert image.dtype == np.uint8
