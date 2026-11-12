"""Simulated dataset for the detection CNN (paper Sec. III.B, scaled up).

Per merger event: random IMRPhenomD source (bbhx, TDI 2.0 A/E), a long segment
of signal, then WINDOWS_PER_EVENT random 10-day windows ending between 30 days
and a few minutes before the merger. Each window gets fresh coloured noise.
For every merger window a merger-free window (noise only) is added, so the set
is balanced as in the paper (20 windows per event).

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

Output is a directory of shards (see shards.py), generated once and reused by
every training run, so hyperparameter setups are compared on identical data.

    python src/dataset.py --events 125000 --seed 0 --out data/train_125k
    python src/dataset.py --events 5000   --seed 1 --out data/test_5k

Resumable (finished shards are skipped). Run several processes at once with
--part I/N (process I of N takes shards I, I+N, ...). --device cuda moves the
waveform (needs bbhx-cuda12x + CuPy), noise and spectrograms onto the GPU.
"""
import argparse
import json
import math
import os
import time

import h5py
import numpy as np

import shards
from simulate import gpu_available, mbhb_aet
from spectrograms import DAY, normalize_batch, power_images

DT = 5.0
WINDOW_DAYS = 10.0
SEG_PRE_DAYS = 41.0          # simulated segment starts this long before merger
SEG_POST_DAYS = 1.0          # ... and ends this long after it
WINDOWS_PER_EVENT = 10
MAX_PRE_MERGER_DAYS = 30.0   # window end drawn from [-30 d, -5 min]
TRACE_THRESH = 0.1
MIN_TRACE_PIX = 20
HEIGHT = 128
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
    def __init__(self, n, xp=np):
        f, sa, se = noise_psds()
        fn = np.fft.rfftfreq(n, DT)
        lf = np.log(np.maximum(fn, f[1]))
        amp = np.stack([np.sqrt(np.exp(np.interp(lf, np.log(f[1:]), np.log(s[1:])))
                                * n / (4 * DT)) for s in (sa, se)])
        amp[:, 0] = 0.0
        self.amp = xp.asarray(amp)       # (2, nfreq)
        self.n, self.xp = n, xp

    def draw(self, rng, batch):
        """`batch` independent noise realisations -> (batch, 2, n): A, E."""
        shape = (batch,) + tuple(self.amp.shape)
        z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        return self.xp.fft.irfft(self.amp * z, n=self.n, axis=-1)


def random_source(rng):
    m = np.sort(10 ** rng.uniform(5, 7, 2))[::-1]
    return dict(m1=m[0], m2=m[1],
                chi1=rng.uniform(-0.9, 0.9), chi2=rng.uniform(-0.9, 0.9),
                dist_mpc=rng.uniform(1e3, 1e5),
                inc=np.arccos(rng.uniform(-1, 1)),
                lam=rng.uniform(0, 2 * np.pi), beta=np.arcsin(rng.uniform(-1, 1)),
                psi=rng.uniform(0, np.pi), phi_ref=rng.uniform(0, 2 * np.pi))


def label_and_images(sig, noise, has_signal, xp=np, height=HEIGHT):
    """Batched images + targets.

    sig, noise: (B, 2, n) windows (xp arrays); has_signal: (B,) bool numpy mask,
    False for noise-only windows (their noise-free image is empty, target -1).
    Returns images (B, H, W, 2) float32 and targets (B,) float32, on the device.
    """
    b, _, n = sig.shape
    pn = power_images((sig + noise).reshape(b * 2, n), DT, height=height, xp=xp)
    pn = pn.reshape(b, 2, *pn.shape[1:])                         # (B, 2, H, W)
    img = normalize_batch(pn.reshape(b * 2, *pn.shape[2:]), xp)
    img = img.reshape(pn.shape).transpose(0, 2, 3, 1)            # (B, H, W, 2)

    target = np.full(b, -1.0, dtype=np.float32)
    idx_host = np.nonzero(has_signal)[0]
    idx = xp.asarray(idx_host)
    if len(idx_host):
        ps = power_images(sig[idx].reshape(len(idx_host) * 2, n), DT, height=height, xp=xp)
        ps = ps.reshape(len(idx_host), 2, *ps.shape[1:]).sum(axis=1)  # A + E, (K, H, W)
        floor = (xp.median(pn[idx, 0], axis=2, keepdims=True)
                 + xp.median(pn[idx, 1], axis=2, keepdims=True))
        trace = ps > TRACE_THRESH * floor
        per_row = trace.sum(axis=2)                              # (K, H)
        count = per_row.sum(axis=1)
        mean_row = (per_row * xp.arange(per_row.shape[1])).sum(axis=1) / xp.maximum(count, 1)
        count, mean_row = to_host(count), to_host(mean_row)
        ok = count >= MIN_TRACE_PIX
        target[idx_host[ok]] = (mean_row[ok] / height).astype(np.float32)
    return img, target


def to_host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def build_config(args):
    n_shards = math.ceil(args.events / args.shard_events)
    return dict(format_version=2, n_events=args.events, seed=args.seed,
                shard_events=args.shard_events, n_shards=n_shards,
                windows_per_event=WINDOWS_PER_EVENT, dt=DT, window_days=WINDOW_DAYS,
                seg_pre_days=SEG_PRE_DAYS, seg_post_days=SEG_POST_DAYS,
                max_pre_merger_days=MAX_PRE_MERGER_DAYS, trace_thresh=TRACE_THRESH,
                min_trace_pix=MIN_TRACE_PIX, image_hw=[HEIGHT, 128],
                noise_psd_source=os.path.basename(SPRITZ), waveform="IMRPhenomD (bbhx), TDI2 AET")


def init_manifest(out, config):
    """Create manifest.json, or check an existing one describes the same dataset."""
    path = os.path.join(out, "manifest.json")
    os.makedirs(out, exist_ok=True)
    try:
        with open(path, "x") as f:
            json.dump(dict(config=config, created=time.strftime("%Y-%m-%d %H:%M:%S")),
                      f, indent=1)
    except FileExistsError:
        old = shards.read_manifest(out)["config"]
        if old != config:
            diff = {k: (old.get(k), config.get(k)) for k in set(old) | set(config)
                    if old.get(k) != config.get(k)}
            raise SystemExit(f"{out} already holds a different dataset (old, new): {diff}")


def generate_shard(i, config, noise, xp, wave_gpu):
    n_ev = min(config["shard_events"], config["n_events"] - i * config["shard_events"])
    seq = np.random.SeedSequence(config["seed"]).spawn(config["n_shards"])[i]
    rng = np.random.default_rng(seq)                     # sources, window times
    if xp is np:
        nrng = rng                                       # noise on the same stream
    else:
        nrng = xp.random.default_rng(int(seq.generate_state(1)[0]))
    nwin = int(WINDOW_DAYS * DAY / DT)
    nseg = int((SEG_PRE_DAYS + SEG_POST_DAYS) * DAY / DT)
    tc = SEG_PRE_DAYS * DAY                              # coalescence, seg-relative
    has_signal = np.arange(2 * WINDOWS_PER_EVENT) < WINDOWS_PER_EVENT
    imgs, targets, meta = [], [], []
    for ev in range(n_ev):
        src = random_source(rng)
        a, e, _ = mbhb_aet(src["m1"], src["m2"], src["chi1"], src["chi2"],
                           src["dist_mpc"], src["inc"], src["lam"], src["beta"],
                           src["psi"], src["phi_ref"], tc, nseg, DT, 0.0, use_gpu=wave_gpu)
        sig = xp.stack([a, e]) if wave_gpu else xp.asarray(np.stack([a, e]))
        pre = rng.uniform(300.0, MAX_PRE_MERGER_DAYS * DAY, WINDOWS_PER_EVENT)  # s
        ends = ((tc - pre) / DT).astype(int)
        s = xp.stack([sig[:, end - nwin:end] for end in ends])   # (10, 2, nwin)
        batch = xp.concatenate([s, xp.zeros_like(s)])            # + 10 noise-only
        img, y = label_and_images(batch, noise.draw(nrng, len(batch)), has_signal, xp)
        imgs.append(to_host(img.astype(xp.float16)))
        targets.append(y)
        gev = i * config["shard_events"] + ev
        for w in (1, 0):                                         # matches batch order
            for k in range(WINDOWS_PER_EVENT):
                meta.append((gev, pre[k] / DAY, src["m1"], src["m2"], src["dist_mpc"], w))
    return np.concatenate(imgs), np.concatenate(targets), np.array(meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=int, required=True, help="merger events (x20 windows)")
    ap.add_argument("--seed", type=int, default=0, help="use different seeds for train/test")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--shard-events", type=int, default=250,
                    help="events per shard (250 -> 5000 windows, ~330 MB)")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--bbhx-cpu", action="store_true",
                    help="with --device cuda: keep the bbhx waveform on the CPU "
                         "(use if only plain bbhx is installed)")
    ap.add_argument("--part", default="0/1",
                    help="I/N: this process handles shards with index %% N == I")
    args = ap.parse_args()

    config = build_config(args)
    init_manifest(args.out, config)
    part, nparts = (int(v) for v in args.part.split("/"))
    todo = [i for i in range(config["n_shards"])
            if i % nparts == part and not shards.shard_done(args.out, i)]

    use_cuda = args.device == "cuda" or (args.device == "auto" and gpu_available())
    xp, wave_gpu = np, False
    if use_cuda:
        import cupy as xp
        wave_gpu = gpu_available() and not args.bbhx_cpu
        if not wave_gpu:
            print("bbhx waveform on CPU (no bbhx CUDA build / --bbhx-cpu); "
                  "noise + spectrograms on GPU")
    print(f"{args.out}: {config['n_events']} events, {config['n_shards']} shards "
          f"({len(todo)} to do in part {args.part}), device={'cuda' if use_cuda else 'cpu'}, "
          f"waveform_on_gpu={wave_gpu}", flush=True)

    noise = Noise(int(WINDOW_DAYS * DAY / DT), xp)
    t0 = time.time()
    done_events = 0
    total_events = sum(min(config["shard_events"],
                           config["n_events"] - i * config["shard_events"]) for i in todo)
    for i in todo:
        x, y, meta = generate_shard(i, config, noise, xp, wave_gpu)
        shards.write_shard(args.out, i, x, y, meta)
        done_events += len(y) // (2 * WINDOWS_PER_EVENT)
        el = time.time() - t0
        eta = el / done_events * (total_events - done_events)
        print(f"  shard {i:5d} done: {done_events}/{total_events} events, "
              f"{el / 60:.1f} min elapsed, ~{eta / 60:.1f} min left, "
              f"positive {np.mean(y > 0):.1%}", flush=True)
    print("finished", args.out)


if __name__ == "__main__":
    main()
