import netron
import numpy as np

from tensorflow.lite.python import lite
from tensorflow.python import keras
from tensorflow.python.ops import array_ops, custom_gradient, math_ops


def get_model():
    def sign(x):
        return math_ops.sign(math_ops.sign(x) + 0.1)

    def _clipped_gradient(x, dy, clip_value):
        if clip_value is None:
            return dy

        zeros = array_ops.zeros_like(dy)
        mask = math_ops.less_equal(math_ops.abs(x), clip_value)
        return array_ops.wherev2(mask, dy, zeros)

    def ste_sign(x, clip_value=1.0):
        @custom_gradient.custom_gradient
        def _call(x):
            def grad(dy):
                return _clipped_gradient(x, dy, clip_value)

            return sign(x), grad

        return _call(x)

    class SteSign(keras.layers.Layer):
        precision = 1

        def __init__(self, clip_value: float = 1.0, **kwargs):
            self.clip_value = clip_value
            super().__init__(**kwargs)

        def call(self, inputs):
            return ste_sign(inputs, clip_value=self.clip_value)

        def get_config(self):
            return {**super().get_config(), "clip_value": self.clip_value}

    def init(shape, **kwargs):
        return np.random.choice([-1.0, 1.0], size=shape)

    def shortcut_block(x):
        shortcut = x
        # Larq quantized convolution layer
        x = SteSign()(x)
        x = keras.layers.Conv2D(
            64, 3, padding="same", kernel_initializer=init, use_bias=False,
        )(x)
        x = keras.layers.BatchNormalization(
            gamma_initializer=keras.initializers.RandomNormalV2(1.0),
            beta_initializer="uniform",
        )(x)
        return keras.layers.add([x, shortcut])

    img_input = keras.layers.Input(shape=(28, 28, 3))
    out = keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(
        img_input
    )
    out = keras.layers.BatchNormalization(
        gamma_initializer=keras.initializers.RandomNormalV2(1.0),
        beta_initializer="uniform",
    )(out)
    out = shortcut_block(out)
    out = shortcut_block(out)
    out = keras.layers.MaxPool2D(3, strides=2, padding="same")(out)
    out = keras.layers.GlobalAvgPool2D()(out)
    out = keras.layers.Dense(10, activation="softmax")(out)

    return keras.Model(inputs=img_input, outputs=out)


model = get_model()
model.save_weights("/tmp/model_weights.h5")


file_name = "/tmp/model.tflite"
converter = lite.TFLiteConverterV2.from_keras_model(model)
converter.allow_custom_ops = True
converter.experimental_new_converter = True
tflite_model = converter.convert()

with open(file_name, "wb") as file:
    file.write(tflite_model)
netron.start(file_name)
