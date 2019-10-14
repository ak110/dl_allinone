def test_run(data_dir):
    import PIL.Image
    import pytesseract

    with PIL.Image.open(data_dir / "data.jpg") as img:
        text = pytesseract.image_to_string(img, lang="jpn")
        assert isinstance(text, str)
