
import numpy as np


def test_keras(tmpdir, keras):
    inputs = x = keras.layers.Input((28, 28, 3))
    x = keras.layers.Conv2D(1, 3, padding='same')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    model.compile('adam', 'mse')
    model.summary()
    keras.utils.plot_model(model, str(tmpdir / 'model.svg'))
    model.fit(np.random.rand(10, 28, 28, 3), np.random.rand(10), batch_size=10, epochs=2)
    model.save(str(tmpdir / 'model.h5'))
