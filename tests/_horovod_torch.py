import numpy as np
import pytest


def test_run():
    import horovod.torch as hvd
    from torch import from_numpy  # pylint: disable=no-name-in-module

    hvd.init()

    x = np.ones((1, 2), dtype=np.float32)

    assert hvd.allreduce(from_numpy(x)).to("cpu").detach().numpy() == pytest.approx(x)
