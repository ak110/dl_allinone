import pathlib
import subprocess

import pytest


def test_run_tf():
    path = pathlib.Path(__file__).parent / "_horovod_tf.py"
    subprocess.run(["horovodrun", "-np", "1", "pytest", str(path)], check=True)


def test_run_torch():
    path = pathlib.Path(__file__).parent / "_horovod_torch.py"
    subprocess.run(["horovodrun", "-np", "1", "pytest", str(path)], check=True)


@pytest.mark.xfail(reason="MXNet")
def test_run_mxnet():
    path = pathlib.Path(__file__).parent / "_horovod_mxnet.py"
    subprocess.run(["horovodrun", "-np", "1", "pytest", str(path)], check=True)
