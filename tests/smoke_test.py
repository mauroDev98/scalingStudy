"""Prueba de humo: verifica que TODO el pipeline funciona de punta a punta con un corpus de juguete.

Uso:  python tests/smoke_test.py
No usa los datos reales ni toca data/, models/ o results/ (todo va a una carpeta temporal).
Si pasa, el código sirve; los números reales dependen de los libros que elijan.
"""
import itertools
import math
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import evaluate  # noqa: E402
import prepare_data  # noqa: E402
import train  # noqa: E402


def toy_corpus(n, seed):
    """Cadena de 64 'palabras' donde la siguiente casi siempre es función de la anterior."""
    rnd = random.Random(seed)
    words = ["".join(p) for p in itertools.product("abcdefgh", repeat=2)]
    w, out = rnd.randrange(64), []
    for _ in range(n):
        out.append(words[w])
        w = (3 * w + 7) % 64 if rnd.random() < 0.8 else rnd.randrange(64)
    return " ".join(out)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "raw" / "train").mkdir(parents=True)
        (tmp / "raw" / "test").mkdir(parents=True)
        (tmp / "raw" / "train" / "a.txt").write_text(toy_corpus(20000, 1))
        (tmp / "raw" / "test" / "b.txt").write_text(toy_corpus(6000, 2))

        stats = prepare_data.prepare(tmp / "raw", tmp / "clean", vocab_size=80)
        print("datos:", {k: stats[k]["examples"] for k in ("train", "val", "test")})
        train.train(32, tmp / "clean", tmp / "models", tmp / "results", epochs=10, batch=64, lr=1e-2, verbose=False)
        r = evaluate.evaluate(32, tmp / "clean", tmp / "models", tmp / "results")
        print(f"top-3 {r['test_top3']:.1%} vs baseline {r['baseline_top3']:.1%}, loss {r['test_loss']}")

        assert r["test_top3"] > r["baseline_top3"] + 0.3, "el modelo no supera claramente al baseline"
        assert r["test_loss"] < math.log(stats["vocab_size"]), "la pérdida no baja del azar"

        # El test congelado debe detectar cambios.
        (tmp / "raw" / "test" / "b.txt").write_text(toy_corpus(6000, 3))
        try:
            prepare_data.prepare(tmp / "raw", tmp / "clean", vocab_size=80)
        except SystemExit:
            print("congelado del test: OK")
        else:
            raise AssertionError("cambiar el test no fue detectado")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
