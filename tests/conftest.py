import pathlib

import pytest


@pytest.fixture
def keras():
    import tensorflow as tf
    import keras as keras_

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras_.backend.set_session(session)
    try:
        yield keras_
    finally:
        keras_.backend.clear_session()


@pytest.fixture
def data_dir():
    yield pathlib.Path(__file__).parent / "data"
