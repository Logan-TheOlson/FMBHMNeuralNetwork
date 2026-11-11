"""Time-evolving TDI spectrograms for the detection CNN (paper Sec. III.A).

A window of `window_days` of data ending "now" is turned into a fixed-size
(height, width) image: log10 power on a log-spaced frequency axis, with the
stationary noise colour (and TDI nulls at 30/60 mHz) removed by subtracting
each row's median over time. A signal then appears as excess power.

Channels A and E are made separately and stacked as (H, W, 2) for the network.
The paper does not specify the exact spectrogram settings; these are choices.
"""
import numpy as np

DAY = 86400.0


def to_ae(x, y, z):
    return (z - x) / np.sqrt(2), (x - 2 * y + z) / np.sqrt(6)


def power_image(x, dt, height=128, width=128, nperseg=2048, fmin=3e-4, fmax=0.1):
    """x: 1-D NaN-free series -> (height, width) linear power on log-spaced rows."""
    hop = (len(x) - nperseg) // (width - 1)
    starts = np.arange(width) * hop
    segs = x[starts[:, None] + np.arange(nperseg)[None, :]]
    segs = segs - segs.mean(axis=1, keepdims=True)
    win = np.hanning(nperseg)
    power = np.abs(np.fft.rfft(segs * win, axis=1)) ** 2          # (width, nfreq)
    f = np.fft.rfftfreq(nperseg, dt)

    # log-spaced frequency rows: mean of FFT bins inside each row, or an
    # interpolation where a row is narrower than one FFT bin (low f).
    edges = np.geomspace(fmin, fmax, height + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    idx = np.searchsorted(f, edges)
    img = np.empty((height, width))
    for i in range(height):
        lo, hi = idx[i], idx[i + 1]
        if hi > lo:
            img[i] = power[:, lo:hi].mean(axis=1)
        else:
            j = np.clip(np.searchsorted(f, centres[i]), 1, len(f) - 1)
            w = (centres[i] - f[j - 1]) / (f[j] - f[j - 1])
            img[i] = (1 - w) * power[:, j - 1] + w * power[:, j]
    return img


def normalize(power):
    """Linear power image -> whitened log image, unit std (the network input)."""
    img = np.log10(power + 1e-300)
    img -= np.median(img, axis=1, keepdims=True)                   # whiten rows
    return (img / img.std()).astype(np.float32)


def spectrogram(x, dt, **kw):
    """x: 1-D NaN-free series -> (height, width) normalised log-power image."""
    return normalize(power_image(x, dt, **kw))


def ae_image(x, y, z, dt, **kw):
    """X, Y, Z windows -> (H, W, 2) image with channels [A, E]."""
    a, e = to_ae(x, y, z)
    return np.stack([spectrogram(a, dt, **kw), spectrogram(e, dt, **kw)], axis=-1)


def sliding_windows(t, x, y, z, dt, window_days=10.0, stride_days=0.5, **kw):
    """Yield (end_time_days, image) for windows ending every `stride_days`.

    Series must be NaN-free (use the `clean/tdi` group of the Spritz files).
    """
    n = int(round(window_days * DAY / dt))
    stride = int(round(stride_days * DAY / dt))
    for end in range(n, len(t) + 1, stride):
        s = slice(end - n, end)
        yield t[end - 1] / DAY, ae_image(x[s], y[s], z[s], dt, **kw)
