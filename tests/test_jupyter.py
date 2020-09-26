import pathlib


def test_run(data_dir, tmpdir):
    import subprocess

    r = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--execute",
            "--to=pdf",
            f"--output-dir='{str(tmpdir)}'",
            data_dir / "jupyter.ipynb",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    assert r.returncode == 0
    assert (pathlib.Path(tmpdir) / "jupyter.pdf").exists()
