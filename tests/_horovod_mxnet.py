import numpy as np
import pytest


def test_run():
    import horovod.mxnet as hvd
    import mxnet as mx

    hvd.init()

    x = np.ones((1, 2), dtype=np.float32)

    assert hvd.allreduce(mx.nd.array(x)).asnumpy() == pytest.approx(x)
