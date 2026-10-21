"""Entrena UN tamaño de modelo desde pesos aleatorios.

Uso:  python src/train.py --hidden 64
Guarda: models/mlp_h<hidden>.npz (pesos), models/mlp_h<hidden>.json (resumen),
        results/curves/h<hidden>.csv (curva de pérdida por época).
"""
import argparse
import csv
import json
import time

import numpy as np

from config import BATCH, DATA_CLEAN, EPOCHS, LR, MODELS, RESULTS, SEED, TOPK
from model import MLP, Adam, score


def train(hidden, clean_dir=DATA_CLEAN, models_dir=MODELS, results_dir=RESULTS,
          epochs=EPOCHS, batch=BATCH, lr=LR, seed=SEED, verbose=True):
    tr, va = np.load(clean_dir / "train.npz"), np.load(clean_dir / "val.npz")
    Xtr, ytr, Xva, yva = tr["X"], tr["y"], va["X"], va["y"]
    vocab = json.loads((clean_dir / "vocab.json").read_text(encoding="utf-8"))

    model = MLP(len(vocab), hidden, seed=seed)
    opt = Adam(model.p, lr)
    rng = np.random.default_rng(seed)
    curve, train_time = [], 0.0

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        perm = rng.permutation(len(ytr))
        total, nb = 0.0, 0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss, grads = model.loss_and_grads(Xtr[idx], ytr[idx])
            opt.step(model.p, grads)
            total += loss
            nb += 1
        train_time += time.perf_counter() - t0
        v = score(model, Xva, yva, TOPK)
        curve.append({"epoch": ep, "train_loss": round(total / nb, 4),
                      "val_loss": round(v["loss"], 4), "val_topk": round(v["topk"], 4)})
        if verbose:
            print(f"[h={hidden}] época {ep}/{epochs}  train {curve[-1]['train_loss']}  "
                  f"val {curve[-1]['val_loss']}  val_top{TOPK} {curve[-1]['val_topk']}", flush=True)

    models_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "curves").mkdir(parents=True, exist_ok=True)
    model.save(models_dir / f"mlp_h{hidden}.npz")
    with open(results_dir / "curves" / f"h{hidden}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    summary = {"hidden": hidden, "params": model.n_params, "epochs": epochs, "batch": batch,
               "lr": lr, "seed": seed, "train_time_s": round(train_time, 1),
               "final_train_loss": curve[-1]["train_loss"], "final_val_loss": curve[-1]["val_loss"]}
    (models_dir / f"mlp_h{hidden}.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    print(train(a.hidden, epochs=a.epochs, seed=a.seed))
