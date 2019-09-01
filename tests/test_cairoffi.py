def test_run():
    import cairocffi as cairo
    import numpy as np

    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 100, 100)
    with cairo.Context(surface) as context:
        context.set_source_rgb(1, 1, 1)
        context.paint()
        context.select_font_face(
            "Courier", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
        )
        context.set_font_size(10)
        context.move_to(20, 20)
        context.set_source_rgb(0, 0, 0)
        context.show_text("abc")

    buf = surface.get_data()
    a = np.frombuffer(buf, np.uint8)
    assert len(a) == 100 * 100 * 4
