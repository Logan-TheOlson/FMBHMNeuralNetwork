"""Build time-evolving A/E spectrogram windows from the Spritz mbhb1 event (the
one the paper validates on: m1 = 1.32e6, q = 0.463, merger at day 133.41) and
push them through the detection CNN.

Uses `clean/tdi` (no gaps, no glitches, as in the paper's validation). The
label "signal present" is a visibility threshold: a window is positive once
the median |A| of an hour of the noise-free A channel has exceeded RMS_THRESH x
the noise sigma (the noise-free series also holds short injected glitches at
days 110, 128 and 133, which a per-hour median ignores). The threshold is my choice,
not the paper's (the paper labels with Canny edge detection).

CAUTION: this only marks windows containing the merger itself (the inspiral is
far below the per-sample noise for ~23 days, yet the paper's network detects it
~20 days out). It is saved as `merger_in_window`, NOT as detection ground
truth. Proper labels need the paper's Canny-on-noise-free-spectrogram step.

NOTE: this is a *validation* file with a single merger. The network cannot be
trained from it; see the summary in the conversation.
"""
import sys

import h5py
import numpy as np

from detection_cnn import build_detection_cnn
from spectrograms import DAY, sliding_windows, to_ae

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/LDC2_spritz_mbhb1_training_v2.h5"
T_MERGER_DAYS = 133.41   # sky/cat coalescence time, 11526944.9 s
RMS_THRESH = 0.2
ROLL = 720   # samples per 1-hour block


def main():
    with h5py.File(PATH) as f:
        cl = f["clean/tdi"][:].ravel()
        nf = f["noisefree/tdi"][:].ravel()
        dt = float(f["clean/tdi"].attrs["dt"])
    t = cl["t"]
    sigma = np.std(to_ae(cl["X"], cl["Y"], cl["Z"])[0][: int(5 * DAY / dt)])
    a_sig = np.nan_to_num(to_ae(nf["X"], nf["Y"], nf["Z"])[0])
    nb = len(a_sig) // ROLL
    med = np.median(np.abs(a_sig[: nb * ROLL]).reshape(nb, ROLL), axis=1)
    t_vis = t[: nb * ROLL].reshape(nb, ROLL)[np.argmax(med > RMS_THRESH * sigma), -1] / DAY
    print(f"noise-free A first exceeds {RMS_THRESH} sigma at day {t_vis:.2f} "
          f"({T_MERGER_DAYS - t_vis:.2f} d before merger)")

    ends, imgs, labels = [], [], []
    for end_day, img in sliding_windows(t, cl["X"], cl["Y"], cl["Z"], dt):
        ends.append(end_day)
        imgs.append(img)
        labels.append(end_day > t_vis)
    x = np.stack(imgs)
    ends, labels = np.array(ends), np.array(labels)
    print(f"{len(x)} windows, image batch {x.shape}, signal-present: {labels.sum()}")
    np.savez_compressed("data/mbhb1_windows.npz", x=x, end_day=ends, merger_in_window=labels)

    # end-to-end shape check through both models (weights are untrained)
    for ch in (2, 1):
        m = build_detection_cnn(x.shape[1], x.shape[2], n_channels=ch)
        y = m.predict(x[..., :ch], verbose=0).ravel()
        print(f"{ch}-channel model output (untrained), min/max {y.min():.3f}/{y.max():.3f}")
    for e, l in zip(ends, labels):
        print(f"  window ends day {e:7.2f}  ({T_MERGER_DAYS - e:+6.2f} d to merger)  signal={l}")


if __name__ == "__main__":
    main()
