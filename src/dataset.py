"""Simulated training set for the detection CNN (paper Sec. III.B, scaled down).

Per merger event: random IMRPhenomD source (bbhx, TDI 2.0 A/E), a long segment
of signal, then WINDOWS_PER_EVENT random 10-day windows ending between 30 days
and a few minutes before the merger. Each window gets fresh coloured noise.
For every merger window a merger-free window (noise only) is added, so the set
is balanced as in the paper.

Noise: Gaussian, with the A and E PSDs measured from the Spritz mbhb1 `clean`
data (first 25 days, where the signal is negligible). This includes the TDI-2
shape and the anti-aliasing filter. A and E noise are independent.

Labels (paper: vertical position of the dark trace in the NOISE-FREE
spectrogram, found with Canny edge detection; negative = no signal). Here, as
a simplification: trace pixels are those where the noise-free signal power in
the A+E image exceeds TRACE_THRESH x the window's per-row noise power; the
target is the mean row of those pixels / height (in (0, 1]), or -1 if fewer
than MIN_TRACE_PIX pixels qualify. Detection = network output > 0.

Source priors (my choices; the paper only states 1e5-1e7 Msun masses and
distances up to 100 Gpc): detector-frame component masses log-uniform in
[1e5, 1e7]; aligned spins uniform in [-0.9, 0.9]; isotropic sky and
orientation; luminosity distance uniform in [1, 100] Gpc.
"""
import sys
import time

import h5py
import numpy as np

from simulate import mbhb_aet
from spectrograms import DAY, normalize, power_image

DT = 5.0
WINDOW_DAYS = 10.0
SEG_PRE_DAYS = 41.0          # simulated segment starts this long before merger
SEG_POST_DAYS = 1.0          # ... and ends this long after it
WINDOWS_PER_EVENT = 10
MAX_PRE_MERGER_DAYS = 30.0   # window end drawn from [-30 d, -5 min]
TRACE_THRESH = 0.1
MIN_TRACE_PIX = 20
SPRITZ = "data/LDC2_spritz_mbhb1_training_v2.h5"


def noise_psds(path=SPRITZ, seg=2 ** 16):
    """Median-Welch one-sided PSDs [1/Hz] of Spritz clean A and E -> (f, S_A, S_E)."""
    with h5py.File(path) as f:
        d = f["clean/tdi"][: int(25 * DAY / DT)].ravel()
    a = (d["Z"] - d["X"]) / np.sqrt(2)
    e = (d["X"] - 2 * d["Y"] + d["Z"]) / np.sqrt(6)
    w = np.hanning(seg)
    out = []
    for y in (a, e):
        segs = y[: len(y) // seg * seg].reshape(-1, seg)
        p = 2 * DT * np.abs(np.fft.rfft(segs * w, axis=1)) ** 2 / np.sum(w ** 2)
        out.append(np.median(p, axis=0) / np.log(2))
    return np.fft.rfftfreq(seg, DT), out[0], out[1]


class Noise:
    def __init__(self, n):
        f, sa, se = noise_psds()
        fn = np.fft.rfftfreq(n, DT)
        lf = np.log(np.maximum(fn, f[1]))
        self.amp = [np.sqrt(np.exp(np.interp(lf, np.log(f[1:]), np.log(s[1:])))
                            * n / (4 * DT)) for s in (sa, se)]
        for a in self.amp:
            a[0] = 0.0
        self.n = n

    def draw(self, rng):
        out = []
        for a in self.amp:
            z = rng.standard_normal(len(a)) + 1j * rng.standard_normal(len(a))
            out.append(np.fft.irfft(a * z, n=self.n))
        return np.stack(out)             # (2, n): A, E


def random_source(rng):
    m = np.sort(10 ** rng.uniform(5, 7, 2))[::-1]
    return dict(m1=m[0], m2=m[1],
                chi1=rng.uniform(-0.9, 0.9), chi2=rng.uniform(-0.9, 0.9),
                dist_mpc=rng.uniform(1e3, 1e5),
                inc=np.arccos(rng.uniform(-1, 1)),
                lam=rng.uniform(0, 2 * np.pi), beta=np.arcsin(rng.uniform(-1, 1)),
                psi=rng.uniform(0, np.pi), phi_ref=rng.uniform(0, 2 * np.pi))


def label_and_image(sig, noise, height=128):
    """sig, noise: (2, n) windows -> (image (H, W, 2), target)."""
    noisy = sig + noise
    pn = [power_image(c, DT) for c in noisy]
    img = np.stack([normalize(p) for p in pn], axis=-1)
    ps = power_image(sig[0], DT) + power_image(sig[1], DT)
    floor = np.median(pn[0], axis=1, keepdims=True) + np.median(pn[1], axis=1, keepdims=True)
    trace = ps > TRACE_THRESH * floor
    if trace.sum() < MIN_TRACE_PIX:
        return img, -1.0
    rows = np.nonzero(trace)[0]
    return img, float(rows.mean() / height)


def make_dataset(n_events, seed=0, out="data/train_pilot.npz"):
    rng = np.random.default_rng(seed)
    nwin = int(WINDOW_DAYS * DAY / DT)
    nseg = int((SEG_PRE_DAYS + SEG_POST_DAYS) * DAY / DT)
    noise_win = Noise(nwin)
    imgs, targets, meta = [], [], []
    t0 = time.time()
    for ev in range(n_events):
        src = random_source(rng)
        tc = SEG_PRE_DAYS * DAY                       # coalescence, seg-relative
        a, e, _ = mbhb_aet(src["m1"], src["m2"], src["chi1"], src["chi2"],
                           src["dist_mpc"], src["inc"], src["lam"], src["beta"],
                           src["psi"], src["phi_ref"], tc, nseg, DT, 0.0)
        sig = np.stack([a, e])
        for _ in range(WINDOWS_PER_EVENT):
            pre = rng.uniform(300.0, MAX_PRE_MERGER_DAYS * DAY)   # s before merger
            end = int((tc - pre) / DT)
            s = sig[:, end - nwin:end]
            for with_signal in (True, False):
                x = s if with_signal else np.zeros_like(s)
                img, y = label_and_image(x, noise_win.draw(rng))
                imgs.append(img.astype(np.float16))
                targets.append(y)
                meta.append((ev, pre / DAY, src["m1"], src["m2"], src["dist_mpc"]))
        if (ev + 1) % 25 == 0:
            print(f"  {ev + 1}/{n_events} events, {time.time() - t0:.0f}s", flush=True)
    np.savez(out, x=np.stack(imgs), y=np.array(targets, dtype=np.float32),
                        meta=np.array(meta))
    y = np.array(targets)
    print(f"saved {out}: {len(y)} windows, positive {np.mean(y > 0):.2%}")


if __name__ == "__main__":
    make_dataset(int(sys.argv[1]) if len(sys.argv) > 1 else 10,
                 seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0,
                 out=sys.argv[3] if len(sys.argv) > 3 else "data/train_pilot.npz")
