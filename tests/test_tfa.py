import numpy as np


def test_run():
    import tensorflow_addons as tfa

    img = np.ones((20, 30, 3), dtype=np.uint8)
    r = tfa.image.rotate(img, np.pi / 2).numpy()
    assert r[0, 0, 0] == 0  # zero padding
    assert r[10, 15, 0] == 1  # rotated
