"""Utilidades compartidas: tokenizador y escritura de CSV."""
import csv
import re
from pathlib import Path

# Solo letras (con acentos y ñ). Se ignoran números y puntuación.
WORD = re.compile(r"[a-záéíóúüñ]+")


def tokenize(text):
    return WORD.findall(text.lower())


def upsert_csv(path, row, key, fields):
    """Agrega `row` a un CSV; si ya hay una fila con los mismos valores en `key`, la reemplaza.
    Las filas quedan ordenadas por la columna `hidden`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    new = {k: row.get(k, "") for k in fields}
    rows = [r for r in rows if any(str(r.get(k)) != str(new[k]) for k in key)]
    rows.append(new)
    rows.sort(key=lambda r: (float(r["hidden"]), str(r.get("label", ""))))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)
