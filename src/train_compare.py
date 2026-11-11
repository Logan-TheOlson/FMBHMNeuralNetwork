"""Train the detection CNN on A only (paper) and on A+E, same data/seed/settings,
and compare on a held-out simulated test set.

Target/loss as in the paper: regress the trace's vertical pixel position
(positive) or -1 (no signal), MSE; detection = output > 0. Positives are
up-weighted so the (rarer) positive class carries equal total weight.
"""
import json
import sys

import numpy as np
import tensorflow as tf

from detection_cnn import build_detection_cnn

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SEED = 0


def auc(score, pos):
    """ROC AUC via rank statistic."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load(path):
    d = np.load(path)
    return d["x"], d["y"], d["meta"]     # meta cols: event, days_pre, m1, m2, dist_mpc


def run(name, channels, xtr, ytr, xte, yte, meta_te):
    tf.keras.utils.set_random_seed(SEED)
    model = build_detection_cnn(xtr.shape[1], xtr.shape[2], n_channels=len(channels))
    model.compile(tf.keras.optimizers.Adam(1e-3), loss="mse")
    w = np.where(ytr > 0, (ytr <= 0).sum() / max((ytr > 0).sum(), 1), 1.0).astype("float32")
    sl = list(channels)
    model.fit(xtr[..., sl].astype("float32"), ytr, sample_weight=w, batch_size=64,
              epochs=EPOCHS, validation_split=0.1, shuffle=True, verbose=2)
    out = model.predict(xte[..., sl].astype("float32"), batch_size=128, verbose=0).ravel()
    pos = yte > 0
    res = dict(model=name, auc=float(auc(out, pos)),
               tpr=float((out[pos] > 0).mean()), fpr=float((out[~pos] > 0).mean()))
    # detection rate of positives by days before merger
    bins = [0, 1, 3, 7, 15, 30.01]
    res["tpr_by_days_pre_merger"] = {
        f"{lo}-{hi}d": float((out[pos & (meta_te[:, 1] >= lo) & (meta_te[:, 1] < hi)] > 0).mean())
        for lo, hi in zip(bins[:-1], bins[1:])
        if (pos & (meta_te[:, 1] >= lo) & (meta_te[:, 1] < hi)).any()}
    print(json.dumps(res, indent=1), flush=True)
    return res


def main():
    xtr, ytr, _ = load("data/train_pilot.npz")
    xte, yte, mte = load("data/test_pilot.npz")
    print(f"train {xtr.shape}, positives {(ytr > 0).mean():.1%}; "
          f"test {xte.shape}, positives {(yte > 0).mean():.1%}", flush=True)
    results = [run("A only", [0], xtr, ytr, xte, yte, mte),
               run("A + E", [0, 1], xtr, ytr, xte, yte, mte)]
    json.dump(results, open("results/compare_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
