"""Prepara el dataset de siguiente palabra.

Uso:  python src/prepare_data.py

Entradas
  data/raw/train/*.txt   libros de entrenamiento (dominio público)
  data/raw/test/*.txt    texto de OTROS autores y/o escrito por el equipo. NUNCA se usa para entrenar.

Salidas (data/clean/)
  train.npz, val.npz, test.npz   ventanas (X = 3 palabras previas, y = palabra siguiente)
  vocab.json                      vocabulario, construido SOLO con el entrenamiento
  stats.json                      conteos para el reporte (qué se quitó, % de palabras desconocidas)
  test.sha256                     huella del test set: si cambia, el script se detiene

Val = el último 10% de cada libro de entrenamiento (sirve para vigilar el entrenamiento).
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from config import CONTEXT, DATA_CLEAN, DATA_RAW, UNK, VOCAB_SIZE
from utils import tokenize

import re

GUT_START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
GUT_END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", re.I)


def clean_text(raw):
    """Quita el encabezado y la licencia que Project Gutenberg agrega a cada libro."""
    m = GUT_START.search(raw)
    if m:
        raw = raw[m.end():]
    m = GUT_END.search(raw)
    if m:
        raw = raw[:m.start()]
    return raw


def load_dir(d):
    files = sorted(Path(d).glob("*.txt"))
    if not files:
        raise SystemExit(f"No hay archivos .txt en {d}. Ver data/SOURCES.md.")
    return {f.name: tokenize(clean_text(f.read_text(encoding="utf-8", errors="replace"))) for f in files}


def windows(ids, n=CONTEXT):
    ids = np.asarray(ids, dtype=np.int32)
    if len(ids) <= n:
        return np.zeros((0, n), np.int32), np.zeros(0, np.int32)
    idx = np.arange(n)[None, :] + np.arange(len(ids) - n)[:, None]
    return ids[idx], ids[n:]


def sha256_tokens(books):
    h = hashlib.sha256()
    for name in sorted(books):
        h.update(name.encode() + b":" + " ".join(books[name]).encode() + b"\n")
    return h.hexdigest()


def prepare(raw_dir=DATA_RAW, clean_dir=DATA_CLEAN, vocab_size=VOCAB_SIZE, val_frac=0.1, unfreeze=False):
    raw_dir, clean_dir = Path(raw_dir), Path(clean_dir)
    train_books = load_dir(raw_dir / "train")
    test_books = load_dir(raw_dir / "test")

    # Test congelado: si los textos de test cambian, hay que enterarse.
    test_hash = sha256_tokens(test_books)
    hash_path = clean_dir / "test.sha256"
    if hash_path.exists() and hash_path.read_text().strip() != test_hash and not unfreeze:
        raise SystemExit(
            "El TEST SET cambió respecto al que se congeló (data/clean/test.sha256).\n"
            "Si todos acordaron rehacerlo, corre con --unfreeze y reentrenen TODOS los tamaños."
        )

    train_tok, val_tok = {}, {}
    for name, toks in train_books.items():
        cut = int(len(toks) * (1 - val_frac))
        train_tok[name], val_tok[name] = toks[:cut], toks[cut:]

    counts = Counter(w for toks in train_tok.values() for w in toks)
    vocab = [UNK] + [w for w, _ in counts.most_common(vocab_size - 1)]
    w2i = {w: i for i, w in enumerate(vocab)}

    def encode(tokens):
        return np.array([w2i.get(w, 0) for w in tokens], dtype=np.int32)

    def build(books):
        Xs, ys, n_tok, n_unk = [], [], 0, 0
        for toks in books.values():
            ids = encode(toks)
            n_tok += len(ids)
            n_unk += int((ids == 0).sum())
            X, y = windows(ids)
            Xs.append(X)
            ys.append(y)
        X = np.concatenate(Xs) if Xs else np.zeros((0, CONTEXT), np.int32)
        y = np.concatenate(ys) if ys else np.zeros(0, np.int32)
        return X, y, n_tok, n_unk

    clean_dir.mkdir(parents=True, exist_ok=True)
    stats = {"context": CONTEXT, "vocab_size": len(vocab), "files": {}}
    for split, books in (("train", train_tok), ("val", val_tok), ("test", test_books)):
        X, y, n_tok, n_unk = build(books)
        if split != "train" and len(y) == 0:
            raise SystemExit(f"El conjunto {split} quedó vacío: faltan palabras en data/raw.")
        np.savez_compressed(clean_dir / f"{split}.npz", X=X, y=y)
        stats[split] = {
            "words": n_tok,
            "examples": int(len(y)),
            "unk_pct": round(100 * n_unk / max(n_tok, 1), 2),
            "files": {n: len(t) for n, t in books.items()},
        }
    (clean_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (clean_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    hash_path.write_text(test_hash + "\n")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--unfreeze", action="store_true", help="permite cambiar el test set congelado")
    a = ap.parse_args()
    s = prepare(vocab_size=a.vocab_size, unfreeze=a.unfreeze)
    for split in ("train", "val", "test"):
        print(f"{split:>5}: {s[split]['words']:>9,} palabras  {s[split]['examples']:>9,} ejemplos  "
              f"{s[split]['unk_pct']:>5}% desconocidas")
    print(f"vocabulario: {s['vocab_size']} palabras. Test congelado: {Path(DATA_CLEAN, 'test.sha256').read_text()[:12]}")
