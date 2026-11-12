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


_ROWS_CACHE = {}


def _row_matrix(nperseg, dt, height, fmin, fmax, xp):
    """(nfreq, height) matrix that turns FFT power into log-spaced rows.

    A row averages the FFT bins inside it; where a row is narrower than one
    FFT bin (low f) it linearly interpolates between the two nearest bins.
    """
    key = (nperseg, dt, height, fmin, fmax, xp.__name__)
    if key not in _ROWS_CACHE:
        f = np.fft.rfftfreq(nperseg, dt)
        edges = np.geomspace(fmin, fmax, height + 1)
        centres = np.sqrt(edges[:-1] * edges[1:])
        idx = np.searchsorted(f, edges)
        m = np.zeros((len(f), height))
        for i in range(height):
            lo, hi = idx[i], idx[i + 1]
            if hi > lo:
                m[lo:hi, i] = 1.0 / (hi - lo)
            else:
                j = np.clip(np.searchsorted(f, centres[i]), 1, len(f) - 1)
                w = (centres[i] - f[j - 1]) / (f[j] - f[j - 1])
                m[j - 1, i] = 1 - w
                m[j, i] = w
        if xp is np:      # almost all zeros: sparse is ~10x faster than dense on CPU
            from scipy.sparse import csr_matrix
            _ROWS_CACHE[key] = csr_matrix(m.T)                    # (height, nfreq)
        else:             # GPU: a dense GEMM is fast and needs no sparse support
            _ROWS_CACHE[key] = xp.asarray(m)                      # (nfreq, height)
    return _ROWS_CACHE[key]


def power_images(x, dt, height=128, width=128, nperseg=2048, fmin=3e-4, fmax=0.1,
                 xp=np):
    """x: (B, n) NaN-free series -> (B, height, width) linear power on log-spaced rows.

    `xp` is numpy or cupy; x must already be an array of that module.
    """
    n = x.shape[1]
    hop = (n - nperseg) // (width - 1)
    starts = np.arange(width) * hop
    segs = x[:, starts[:, None] + np.arange(nperseg)[None, :]]     # (B, width, nperseg)
    segs = segs - segs.mean(axis=2, keepdims=True)
    win = xp.asarray(np.hanning(nperseg))
    power = xp.abs(xp.fft.rfft(segs * win, axis=2)) ** 2           # (B, width, nfreq)
    m = _row_matrix(nperseg, dt, height, fmin, fmax, xp)
    if xp is np:
        flat = power.reshape(-1, power.shape[-1])
        rows = np.asarray(m @ flat.T).T.reshape(power.shape[0], width, height)
    else:
        rows = power @ m                                           # (B, width, height)
    return rows.transpose(0, 2, 1)


def power_image(x, dt, **kw):
    """x: 1-D NaN-free series -> (height, width) linear power on log-spaced rows."""
    return power_images(x[None], dt, **kw)[0]


def normalize_batch(power, xp=np):
    """(B, H, W) linear power -> whitened log images, unit std each (network input)."""
    img = xp.log10(power + 1e-300)
    img = img - xp.median(img, axis=2, keepdims=True)              # whiten rows
    return (img / img.std(axis=(1, 2), keepdims=True)).astype(xp.float32)


def normalize(power):
    """Linear power image -> whitened log image, unit std (the network input)."""
    return normalize_batch(power[None])[0]


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
