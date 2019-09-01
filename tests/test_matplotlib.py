def test_run(tmpdir):
    import matplotlib.pyplot as plt

    plt.plot([1, 2, 3, 4])
    plt.ylabel("some numbers")
    plt.savefig(str(tmpdir / "plot.pdf"))
    plt.savefig(str(tmpdir / "plot.png"))
    plt.savefig(str(tmpdir / "plot.svg"))
    plt.savefig(str(tmpdir / "plot.jpg"))
