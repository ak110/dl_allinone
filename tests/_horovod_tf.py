import numpy as np
import pytest


def test_run():
    import horovod.tensorflow as hvd

    hvd.init()

    x = np.ones((1, 2), dtype=np.float32)

    assert hvd.allreduce(x).numpy() == pytest.approx(x)
