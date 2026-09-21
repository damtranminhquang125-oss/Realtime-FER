import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dropout, GlobalAveragePooling2D,
    Dense, Add, Multiply, Reshape,
    GlobalMaxPooling2D
)
from tensorflow.keras.models import Model


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            use_bias=False
        )

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)

        attention_input = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(attention_input)

        return x * attention

def spatial_attention_module(x):
    return SpatialAttention(kernel_size=3)(x)


def channel_attention_module(x, reduction_ratio=8):
    channels = int(x.shape[-1])
    hidden = max(channels // reduction_ratio, 8)

    shared_1 = Dense(hidden, activation="relu", use_bias=False)
    shared_2 = Dense(channels, use_bias=False)

    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, channels))(avg_pool)
    avg_pool = shared_2(shared_1(avg_pool))

    max_pool = GlobalMaxPooling2D()(x)
    max_pool = Reshape((1, 1, channels))(max_pool)
    max_pool = shared_2(shared_1(max_pool))

    attention = Add()([avg_pool, max_pool])
    attention = Activation("sigmoid")(attention)

    return Multiply()([x, attention])


def cbam_block(x, reduction_ratio=8):
    x = channel_attention_module(x, reduction_ratio)
    return SpatialAttention(kernel_size=3)(x)


def conv_bn_relu(x, filters, kernel_size=3, dilation_rate=1):
    x = Conv2D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def residual_block(x, filters, dilation_rate=1):
    shortcut = x

    x = conv_bn_relu(x, filters, 3, dilation_rate=dilation_rate)

    x = Conv2D(
        filters,
        3,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)
    x = BatchNormalization()(x)

    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    return Activation("relu")(x)


def custom_cnn(input_shape=(48, 48, 1), num_classes=7):
    inputs = Input(shape=input_shape)

    # 48x48 -> 24x24
    x = conv_bn_relu(inputs, 64)
    x = conv_bn_relu(x, 64)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.20)(x)

    # 24x24 -> 12x12
    x = conv_bn_relu(x, 128)
    x = conv_bn_relu(x, 128)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    # 12x12 -> 6x6
    x = conv_bn_relu(x, 256)
    x = residual_block(x, 256, dilation_rate=2)
    x = cbam_block(x, reduction_ratio=8)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.30)(x)

    # 6x6
    x = conv_bn_relu(x, 512)
    x = conv_bn_relu(x, 512)
    x = conv_bn_relu(x, 512)
    x = Dropout(0.40)(x)

    # Classifier
    x = GlobalAveragePooling2D()(x)

    x = Dense(384, use_bias=False, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("swish")(x)
    x = Dropout(0.50)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name="FER_Custom_CNN")

if __name__ == "__main__":
    model = custom_cnn()
    model.summary()