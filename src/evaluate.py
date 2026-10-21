"""Evalúa modelos entrenados sobre el TEST fijo y agrega una fila por tamaño a results/resultados.csv.

Uso:  python src/evaluate.py --hidden 16 32 64 128
Todas las métricas se calculan sobre posiciones del test cuya palabra siguiente está en el
vocabulario (se reporta qué % se excluyó). El baseline sugiere siempre las 3 palabras más
frecuentes del entrenamiento.
"""
import argparse
import json

import numpy as np

from config import DATA_CLEAN, MODELS, RESULTS, SIZES, TOPK
from model import MLP, score
from utils import upsert_csv

FIELDS = ["hidden", "params", "epochs", "seed", "train_time_s", "final_train_loss", "final_val_loss",
          "test_loss", "test_ppl", "test_top1", "test_top3", "baseline_top1", "baseline_top3",
          "n_test", "test_unk_excluded_pct", "test_id"]


def evaluate(hidden, clean_dir=DATA_CLEAN, models_dir=MODELS, results_dir=RESULTS):
    model = MLP.load(models_dir / f"mlp_h{hidden}.npz")
    meta = json.loads((models_dir / f"mlp_h{hidden}.json").read_text())
    te, tr = np.load(clean_dir / "test.npz"), np.load(clean_dir / "train.npz")
    Xte, yte = te["X"], te["y"]

    s = score(model, Xte, yte, TOPK)
    counts = np.bincount(tr["y"], minlength=model.vocab_size)
    counts[0] = 0
    top = np.argsort(-counts)[:TOPK]
    ykeep = yte[yte != 0]

    row = {
        "hidden": hidden, "params": model.n_params, "epochs": meta["epochs"], "seed": meta["seed"],
        "train_time_s": meta["train_time_s"], "final_train_loss": meta["final_train_loss"],
        "final_val_loss": meta["final_val_loss"],
        "test_loss": round(s["loss"], 4), "test_ppl": round(float(np.exp(s["loss"])), 1),
        "test_top1": round(s["topk"] if TOPK == 1 else s["top1"], 4), "test_top3": round(s["topk"], 4),
        "baseline_top1": round(float((ykeep == top[0]).mean()), 4),
        "baseline_top3": round(float(np.isin(ykeep, top).mean()), 4),
        "n_test": s["n"], "test_unk_excluded_pct": round(100 * (1 - len(ykeep) / len(yte)), 2),
        "test_id": (clean_dir / "test.sha256").read_text().strip()[:12],
    }
    upsert_csv(results_dir / "resultados.csv", row, key=["hidden"], fields=FIELDS)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="+", default=SIZES)
    for h in ap.parse_args().hidden:
        r = evaluate(h)
        print(f"h={h:>4}  params={r['params']:>8,}  test top-3 {r['test_top3']:.1%} "
              f"(baseline {r['baseline_top3']:.1%})  loss {r['test_loss']}  n={r['n_test']:,}")
