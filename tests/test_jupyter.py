def test_run(data_dir):
    import subprocess

    r = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--execute",
            "--to=pdf",
            "--stdout",
            data_dir / "jupyter.ipynb",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    assert r.returncode == 0
