import numpy as np


def test_run(tmpdir):
    import tensorflow as tf

    X_train = np.random.rand(10, 28, 28, 3)
    y_train = np.random.rand(10)

    inputs = x = tf.keras.layers.Input((28, 28, 3))
    x = tf.keras.layers.Conv2D(1, 3, padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile("adam", "mse")

    model.summary()
    tf.keras.utils.plot_model(model, str(tmpdir / "model.png"))

    model.fit(X_train, y_train, batch_size=10, epochs=2)
    model.save(str(tmpdir / "model.h5"))

    model = tf.keras.models.load_model(str(tmpdir / "model.h5"))
    assert model.predict(X_train).shape == (len(X_train), 1)


def test_load_data():
    """<https://github.com/keras-team/keras/issues/12729>"""
    import tensorflow as tf

    tf.keras.datasets.imdb.load_data()
