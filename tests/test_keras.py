
import numpy as np


def test_keras(tmpdir, keras):
    X_train = np.random.rand(10, 28, 28, 3)
    y_train = np.random.rand(10)

    inputs = x = keras.layers.Input((28, 28, 3))
    x = keras.layers.Conv2D(1, 3, padding='same')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    model.compile('adam', 'mse')

    model.summary()
    keras.utils.plot_model(model, str(tmpdir / 'model.svg'))

    model.fit(X_train, y_train, batch_size=10, epochs=2)
    model.save(str(tmpdir / 'model.h5'))

    model = keras.models.load_model(str(tmpdir / 'model.h5'))
    assert model.predict(X_train).shape == (len(X_train), 1)
