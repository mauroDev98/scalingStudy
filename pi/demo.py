"""Demo en vivo: escribe texto y mira las 3 sugerencias de cada tamaño de modelo.

Uso:  python3 pi/demo.py                 (usa todos los modelos que haya en models/)
      python3 pi/demo.py --hidden 16 128
Línea vacía o Ctrl-D para salir. Cada interacción se guarda en results/demo_log.jsonl
(todo lo que se muestre en la demo debe haberse producido en vivo o quedar registrado).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CONTEXT, DATA_CLEAN, MODELS, RESULTS  # noqa: E402
from model import MLP  # noqa: E402
from utils import tokenize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="*")
    a = ap.parse_args()
    sizes = a.hidden or sorted(int(p.stem.split("_h")[1]) for p in MODELS.glob("mlp_h*.npz"))
    if not sizes:
        raise SystemExit("No hay modelos en models/. Corre src/run_all.py primero.")
    models = {h: MLP.load(MODELS / f"mlp_h{h}.npz") for h in sizes}
    vocab = json.loads((DATA_CLEAN / "vocab.json").read_text(encoding="utf-8"))
    w2i = {w: i for i, w in enumerate(vocab)}
    log = RESULTS / "demo_log.jsonl"
    log.parent.mkdir(exist_ok=True)

    print("Autocompletado de siguiente palabra. Escribe una frase (línea vacía para salir).")
    print("Aviso: solo sugiere palabras probables según los textos de entrenamiento; no entiende lo que escribes.")
    while True:
        try:
            line = input("\nescribe> ")
        except EOFError:
            break
        if not line.strip():
            break
        ids = [w2i.get(w, 0) for w in tokenize(line)][-CONTEXT:]
        ids = [0] * (CONTEXT - len(ids)) + ids
        entry = {"time": datetime.now().isoformat(timespec="seconds"), "text": line, "suggestions": {}}
        for h, m in models.items():
            z = m.logits(np.array([ids]))[0]
            z[0] = -np.inf  # nunca sugerir <unk>
            top = [vocab[i] for i in np.argsort(-z)[:3]]
            entry["suggestions"][h] = top
            print(f"  {h:>4} neuronas ({m.n_params:>7,} parámetros): {', '.join(top)}")
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
