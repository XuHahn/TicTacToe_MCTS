from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, Layer, Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy


class SL_Conv(Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides)
        self.pad = ZeroPadding2D((padding, padding))
        self.activation = Activation(activation=activation)

    def call(self, inputs, *args, **kwargs):
        inputs = self.pad(inputs)
        inputs = self.conv(inputs)
        inputs = self.activation(inputs)
        return inputs


def SL_policy_Network():
    return Sequential(
        [
            Input(shape=(19, 19, 48)),
            SL_Conv(filters=192, kernel_size=5, strides=1, padding=2, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=1, kernel_size=1, strides=1, padding=0, activation='softmax')
        ]
    )


def ValueNetwork():
    return Sequential(
        [
            Input(shape=(19, 19, 48)),
            SL_Conv(filters=192, kernel_size=5, strides=1, padding=2, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=192, kernel_size=3, strides=1, padding=1, activation='relu'),
            SL_Conv(filters=1, kernel_size=1, strides=1, padding=0, activation='relu'),
            Dense(256),
            Activation('tanh')
        ]
    )


model = ValueNetwork()
model.summary()
model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])
