def test_run():
    import albumentations as A
    import numpy as np

    image = np.zeros((32, 32, 1))
    image = A.Resize(height=8, width=16)(image=image)["image"]
    assert image.shape[:2] == (8, 16)
