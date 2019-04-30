
def test_cv2(data_dir):
    import cv2
    assert cv2.imread(str(data_dir / 'data.jpg'), cv2.IMREAD_COLOR).shape == (20, 124, 3)
