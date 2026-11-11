"""Detection CNN from Houba et al. (arXiv:2405.11340), Sec. III.A / Fig. 6,
extended to take several TDI 2.0 channels at once (default: A and E).

Paper architecture (single A-channel input):
    2x Conv2D(32, 3x3, ReLU) -> MaxPool 2x2
    2x Conv2D(64, 3x3, ReLU) -> MaxPool 2x2
    2x Conv2D(64, 3x3, ReLU) -> MaxPool 2x2
    Flatten -> Dense 64 -> Dense 64 -> Dense 32 (ReLU) -> Dense 1 (linear)
The output is the "vertical pixel position" of the inspiral trace; a positive
output means detection, a negative one means no detection.

Multi-channel change: A and E spectrograms are stacked along the channel axis,
like the RGB planes of an image, so input shape is (H, W, n_channels). Only
the first Conv2D's kernel grows (3x3x1x32 -> 3x3x2x32); everything downstream
is unchanged. The first layer therefore sees A and E at the same time-frequency
pixel, which is what lets it combine them.

The paper does not give the reduced spectrogram resolution, so height/width
are arguments.
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

CHANNELS = ("A", "E")


def to_ae(x, y, z):
    """TDI X, Y, Z -> A, E (same definitions as plot_aet.py)."""
    return {
        "A": (z - x) / np.sqrt(2),
        "E": (x - 2 * y + z) / np.sqrt(6),
    }


def stack_channels(spectrograms, channels=CHANNELS):
    """Stack per-channel spectrograms into one (..., H, W, C) network input.

    spectrograms: dict like {"A": (N, H, W), "E": (N, H, W)}.
    Normalize each channel the same way you did for the single-channel
    network (A and E have similar noise PSDs, so a shared scheme is fine).
    """
    return np.stack([spectrograms[c] for c in channels], axis=-1)


def build_detection_cnn(height, width, n_channels=len(CHANNELS)):
    inputs = keras.Input(shape=(height, width, n_channels))
    x = inputs
    for filters in (32, 64, 64):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Flatten()(x)
    for units in (64, 64, 32):
        x = layers.Dense(units, activation="relu")(x)
    outputs = layers.Dense(1)(x)  # linear: sign = detection
    return keras.Model(inputs, outputs, name=f"detection_cnn_{n_channels}ch")


if __name__ == "__main__":
    # Smoke test: A+E model vs the paper's A-only model.
    build_detection_cnn(128, 128, n_channels=2).summary()
    build_detection_cnn(128, 128, n_channels=1).summary()
