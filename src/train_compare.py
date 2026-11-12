"""Train the detection CNN on A only (paper) and on A+E and evaluate on a held-out
test set. Datasets come from dataset.py and are generated once, so every
hyperparameter setup below is trained and scored on identical data.

Target/loss as in the paper: regress the trace's vertical pixel position
(positive) or -1 (no signal), MSE; detection = output > 0. Positives are
up-weighted so the (rarer) positive class carries equal total weight.

    python src/train_compare.py --train data/train_125k --test data/test_5k --name baseline
    python src/train_compare.py --train data/train_125k --test data/test_5k --name lr3e-4 --lr 3e-4
    python src/train_compare.py ... --name wide --filters 64,128,128 --dropout 0.2

Each run writes results/<name>.json with the hyperparameters, dataset configs,
per-epoch loss history, timings and metrics, so runs can be compared afterwards:
    python src/compare_runs.py results/baseline.json results/lr3e-4.json

Validation uses the last --val-frac of the training SHARDS (whole events), so no
event's windows are split between training and validation.
"""
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

from detection_cnn import build_detection_cnn
from shards import ShardedDataset

MODELS = {"A": ("A only", [0]), "AE": ("A + E", [0, 1])}


def auc(score, pos):
    """ROC AUC via rank statistic."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def tpr_at_fpr(out, pos, fpr):
    """Detection rate when the threshold is set to give this false-alarm rate."""
    thr = np.quantile(out[~pos], 1 - fpr)
    return float((out[pos] > thr).mean())


def ints(text):
    return tuple(int(v) for v in text.split(",") if v)


def batches(ds, shard_ids, sl, batch_size, pos_weight, rng, shuffle, group):
    """Yield (x, y, w) batches, loading `group` shards into RAM at a time and
    shuffling across them (shard order is shuffled every epoch too)."""
    ids = list(shard_ids)
    if shuffle:
        rng.shuffle(ids)
    for g in range(0, len(ids), group):
        parts = [ds.load(i, ("x", "y")) for i in ids[g:g + group]]
        x = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        del parts
        order = rng.permutation(len(y)) if shuffle else np.arange(len(y))
        for k in range(0, len(order), batch_size):
            j = order[k:k + batch_size]
            yb = y[j]
            yield (x[j][..., sl].astype(np.float32), yb,
                   np.where(yb > 0, pos_weight, 1.0).astype(np.float32))


def make_tf_dataset(ds, shard_ids, sl, args, pos_weight, rng, shuffle):
    h, w = ds.manifest["config"]["image_hw"]
    sig = (tf.TensorSpec((None, h, w, len(sl)), tf.float32),
           tf.TensorSpec((None,), tf.float32), tf.TensorSpec((None,), tf.float32))
    if not shuffle:                                  # validation: plain (x, y)
        sig = sig[:2]
    gen = lambda: ((b[:2] if not shuffle else b) for b in
                   batches(ds, shard_ids, sl, args.batch_size, pos_weight, rng, shuffle,
                           args.shuffle_shards))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)


def evaluate(model, ds, sl, batch_size):
    outs, ys, metas = [], [], []
    for i in ds.shards:
        x, y, meta = ds.load(i)
        outs.append(model.predict(x[..., sl].astype(np.float32), batch_size=batch_size,
                                  verbose=0).ravel())
        ys.append(y)
        metas.append(meta)
    return np.concatenate(outs), np.concatenate(ys), np.concatenate(metas)


def metrics(out, y, meta):
    pos = y > 0
    res = dict(auc=float(auc(out, pos)), tpr=float((out[pos] > 0).mean()),
               fpr=float((out[~pos] > 0).mean()),
               tpr_at_fpr_1pct=tpr_at_fpr(out, pos, 0.01),
               tpr_at_fpr_0p1pct=tpr_at_fpr(out, pos, 0.001))
    bins = [0, 1, 3, 7, 15, 30.01]                   # days before merger
    res["tpr_by_days_pre_merger"] = {
        f"{lo}-{hi}d": float((out[pos & (meta[:, 1] >= lo) & (meta[:, 1] < hi)] > 0).mean())
        for lo, hi in zip(bins[:-1], bins[1:])
        if (pos & (meta[:, 1] >= lo) & (meta[:, 1] < hi)).any()}
    return res


def run(key, train, val_ids, train_ids, test, args, rng_seed):
    name, sl = MODELS[key]
    tf.keras.utils.set_random_seed(args.seed)
    h, w = train.manifest["config"]["image_hw"]
    model = build_detection_cnn(h, w, n_channels=len(sl), filters=args.filters,
                                convs_per_block=args.convs_per_block, dense=args.dense,
                                dropout=args.dropout)
    model.compile(tf.keras.optimizers.Adam(args.lr), loss="mse")

    y_train = train.labels(train_ids)
    pos_weight = float((y_train <= 0).sum() / max((y_train > 0).sum(), 1))
    rng = np.random.default_rng(rng_seed)            # same data order for every model
    train_ds = make_tf_dataset(train, train_ids, sl, args, pos_weight, rng, True)
    val_ds = (make_tf_dataset(train, val_ids, sl, args, pos_weight, rng, False)
              if val_ids else None)

    print(f"\n=== {name}: {model.count_params():,} params, {len(y_train):,} train windows "
          f"({(y_train > 0).mean():.1%} positive), pos_weight {pos_weight:.2f}", flush=True)
    t0 = time.perf_counter()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out, y, meta = evaluate(model, test, sl, args.batch_size * 4)
    eval_s = time.perf_counter() - t0

    res = dict(model=name, channels=sl, n_params=int(model.count_params()),
               train_seconds=train_s, eval_seconds=eval_s,
               history={k: [float(v) for v in vs] for k, vs in hist.history.items()},
               **metrics(out, y, meta))
    if args.save_model:
        path = os.path.splitext(args.out)[0] + f"_{key}.keras"
        model.save(path)
        res["saved_model"] = path
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="training dataset dir (dataset.py output)")
    ap.add_argument("--test", required=True, help="test dataset dir (use a different seed!)")
    ap.add_argument("--name", default=time.strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--out", help="results json (default results/<name>.json)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--models", default="A,AE", help="comma list from: A, AE")
    # hyperparameters
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--filters", type=ints, default=(32, 64, 64), help="conv filters per block")
    ap.add_argument("--convs-per-block", type=int, default=2)
    ap.add_argument("--dense", type=ints, default=(64, 64, 32), help="hidden Dense units")
    ap.add_argument("--dropout", type=float, default=0.0)
    # data / run
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of train SHARDS")
    ap.add_argument("--max-train-shards", type=int, help="use only the first N shards")
    ap.add_argument("--shuffle-shards", type=int, default=4,
                    help="shards held in RAM and shuffled together (4 ~ 1.3 GB)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mixed-precision", action="store_true",
                    help="float16 compute (faster on RTX tensor cores)")
    ap.add_argument("--require-gpu", action="store_true", help="abort if TF sees no GPU")
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()
    args.out = args.out or os.path.join("results", f"{args.name}.json")
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} exists (pass --overwrite or choose another --name)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print("GPUs:", [g.name for g in gpus] or "none (training on CPU)", flush=True)
    if args.require_gpu and not gpus:
        raise SystemExit("--require-gpu: TensorFlow sees no GPU")
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    train = ShardedDataset(args.train, args.max_train_shards)
    test = ShardedDataset(args.test)
    n_val = min(max(1, round(len(train.shards) * args.val_frac)), len(train.shards) - 1) \
        if len(train.shards) > 1 else 0
    train_ids = train.shards[:len(train.shards) - n_val]
    val_ids = train.shards[len(train.shards) - n_val:]
    print(f"train {len(train_ids)} shards, validation {len(val_ids)} shards, "
          f"test {len(test):,} windows ({(test.labels() > 0).mean():.1%} positive)", flush=True)

    results = [run(k, train, val_ids, train_ids, test, args, args.seed)
               for k in args.models.split(",")]
    record = dict(name=args.name, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                  hyperparameters={k: getattr(args, k) for k in (
                      "epochs", "batch_size", "lr", "filters", "convs_per_block", "dense",
                      "dropout", "val_frac", "max_train_shards", "shuffle_shards", "seed",
                      "mixed_precision")},
                  devices=[g.name for g in gpus] or ["cpu"], tensorflow=tf.__version__,
                  train_dataset=train.describe(), test_dataset=test.describe(),
                  n_train_shards=len(train_ids), n_val_shards=len(val_ids),
                  models=results)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
