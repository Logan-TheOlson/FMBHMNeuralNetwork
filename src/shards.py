"""On-disk layout of a generated dataset, shared by dataset.py and train_compare.py.

A dataset is a directory:

    data/<name>/manifest.json         generation config (seed, constants, ...)
    data/<name>/shard_00000_x.npy     (N, 128, 128, 2) float16 images, channels [A, E]
    data/<name>/shard_00000_y.npy     (N,) float32 targets (-1 = no trace, else (0, 1])
    data/<name>/shard_00000_meta.npy  (N, 6) float64: event, days_pre_merger,
                                      m1, m2, dist_mpc, merger_window (1/0)

A shard counts as complete once its meta file exists (it is written last), so an
interrupted generation can be resumed and a half-written shard is never read.
Shards hold whole events, so train/validation can be split by shard without
leaking an event's windows across the split.
"""
import json
import os

import numpy as np

META_COLS = ("event", "days_pre", "m1", "m2", "dist_mpc", "merger_window")


def shard_path(directory, i, kind):
    return os.path.join(directory, f"shard_{i:05d}_{kind}.npy")


def save_array(path, array):
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


def write_shard(directory, i, x, y, meta):
    save_array(shard_path(directory, i, "x"), x)
    save_array(shard_path(directory, i, "y"), y)
    save_array(shard_path(directory, i, "meta"), meta)        # completion marker


def shard_done(directory, i):
    return os.path.exists(shard_path(directory, i, "meta"))


def read_manifest(directory):
    with open(os.path.join(directory, "manifest.json")) as f:
        return json.load(f)


class ShardedDataset:
    """Read-only view of a generated dataset directory."""

    def __init__(self, directory, max_shards=None):
        self.dir = directory
        self.manifest = read_manifest(directory)
        n = self.manifest["config"]["n_shards"]
        self.shards = [i for i in range(n) if shard_done(directory, i)]
        if not self.shards:
            raise FileNotFoundError(f"no complete shards in {directory}")
        if max_shards is not None:
            self.shards = self.shards[:max_shards]
        self.sizes = [len(np.load(shard_path(directory, i, "y"), mmap_mode="r"))
                      for i in self.shards]

    def __len__(self):
        return sum(self.sizes)

    def load(self, i, parts=("x", "y", "meta")):
        return [np.load(shard_path(self.dir, i, p)) for p in parts]

    def labels(self, shards=None):
        return np.concatenate([self.load(i, ("y",))[0] for i in (shards or self.shards)])

    def describe(self):
        cfg = self.manifest["config"]
        return dict(path=self.dir, n_shards_used=len(self.shards),
                    n_shards_total=cfg["n_shards"], n_windows=len(self),
                    seed=cfg["seed"], config=cfg)
