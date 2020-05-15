def test_run(data_dir):
    import cv2

    img = cv2.imread(str(data_dir / "data.jpg"), cv2.IMREAD_COLOR)
    assert img.shape == (20, 124, 3)
