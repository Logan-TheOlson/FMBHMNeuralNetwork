"""Print a side-by-side table of train_compare.py result files.

    python src/compare_runs.py results/*.json
"""
import json
import sys


def rows(path):
    with open(path) as f:
        r = json.load(f)
    if isinstance(r, list):                      # old format: list of model dicts
        r = dict(name=path, models=r, hyperparameters={}, train_dataset={})
    hp = r.get("hyperparameters", {})
    n = r.get("train_dataset", {}).get("n_windows", "?")
    for m in r["models"]:
        yield [r["name"], m["model"], n, hp.get("epochs", "?"), hp.get("lr", "?"),
               ",".join(map(str, hp.get("filters", []))) or "?", hp.get("dropout", "?"),
               m["auc"], m["tpr"], m.get("tpr_at_fpr_1pct"), m["fpr"],
               m.get("train_seconds")]


def main(paths):
    head = ["run", "model", "train_win", "epochs", "lr", "filters", "drop", "AUC", "TPR@0",
            "TPR@1%FPR", "FPR@0", "train_s"]
    table = [head] + [[f"{v:.4g}" if isinstance(v, float) else str(v) if v is not None else "-"
                       for v in row] for p in paths for row in rows(p)]
    widths = [max(len(r[i]) for r in table) for i in range(len(head))]
    for k, r in enumerate(table):
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))
        if k == 0:
            print("  ".join("-" * w for w in widths))


if __name__ == "__main__":
    main(sys.argv[1:])
