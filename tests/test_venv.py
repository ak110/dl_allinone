import subprocess

import pytest


@pytest.mark.parametrize("system_site", [False, True])
def test_run(tmpdir, system_site):
    cmd = ["python3", "-m", "venv", str(tmpdir), "--symlinks"]
    if system_site:
        cmd.append("--system-site-packages")
    subprocess.run(cmd, check=True)

    cmd = [str(tmpdir / "bin" / "pip"), "-V"]
    c = subprocess.run(cmd, check=True, capture_output=True)
    stdout = c.stdout.decode("utf-8")
    # example of stdout:
    #   'pip 19.2.3 from /tmp/pytest-of-user/pytest-1/test_run_True_0/lib/python3.8/site-packages/pip (python 3.8)\n'
    assert str(tmpdir) in stdout
