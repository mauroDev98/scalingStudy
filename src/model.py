"""MLP de siguiente palabra en NumPy puro (float32). Pesos iniciales aleatorios.

Arquitectura (idéntica en todos los tamaños, solo cambia `hidden`):
  3 palabras previas -> embedding (32 por palabra) -> concatenar (96)
  -> capa oculta (hidden, tanh) -> capa de salida (vocab) -> softmax

NumPy puro = cero problemas de instalación en la Raspberry Pi.
"""
import numpy as np

from config import CONTEXT, EMBED_DIM

F32 = np.float32


class MLP:
    def __init__(self, vocab_size, hidden, embed_dim=EMBED_DIM, context=CONTEXT, seed=0):
        rng = np.random.default_rng(seed)
        d_in = embed_dim * context
        self.vocab_size, self.hidden = vocab_size, hidden
        self.embed_dim, self.context = embed_dim, context
        self.p = {
            "E": (rng.standard_normal((vocab_size, embed_dim)) * 0.1).astype(F32),
            "W1": (rng.standard_normal((d_in, hidden)) * np.sqrt(1.0 / d_in)).astype(F32),
            "b1": np.zeros(hidden, F32),
            "W2": (rng.standard_normal((hidden, vocab_size)) * np.sqrt(1.0 / hidden)).astype(F32),
            "b2": np.zeros(vocab_size, F32),
        }

    @property
    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def logits(self, X):
        p = self.p
        e = p["E"][X].reshape(len(X), -1)
        h = np.tanh(e @ p["W1"] + p["b1"])
        return h @ p["W2"] + p["b2"]

    def top_k(self, X, k=3):
        """Índices de las k palabras más probables, de mayor a menor."""
        z = self.logits(X)
        idx = np.argpartition(-z, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(z, idx, 1), axis=1)
        return np.take_along_axis(idx, order, 1)

    def loss_and_grads(self, X, y):
        p = self.p
        B = len(y)
        e = p["E"][X].reshape(B, -1)
        h = np.tanh(e @ p["W1"] + p["b1"])
        z = h @ p["W2"] + p["b2"]
        z -= z.max(1, keepdims=True)
        ex = np.exp(z)
        pr = ex / ex.sum(1, keepdims=True)
        rows = np.arange(B)
        loss = float(-np.log(pr[rows, y] + 1e-9).mean())

        d = pr
        d[rows, y] -= 1
        d /= B
        g = {"W2": h.T @ d, "b2": d.sum(0)}
        dz = (d @ p["W2"].T) * (1 - h * h)
        g["W1"] = e.T @ dz
        g["b1"] = dz.sum(0)
        de = (dz @ p["W1"].T).reshape(B, self.context, self.embed_dim)
        gE = np.zeros_like(p["E"])
        np.add.at(gE, X, de)
        g["E"] = gE
        return loss, g

    def save(self, path):
        meta = np.array([self.vocab_size, self.hidden, self.embed_dim, self.context])
        np.savez(path, meta=meta, **self.p)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        vocab, hidden, embed, ctx = (int(v) for v in z["meta"])
        m = cls(vocab, hidden, embed, ctx)
        m.p = {k: z[k].astype(F32) for k in ("E", "W1", "b1", "W2", "b2")}
        return m


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        for k, w in params.items():
            g = grads[k]
            self.m[k] *= self.b1
            self.m[k] += (1 - self.b1) * g
            self.v[k] *= self.b2
            self.v[k] += (1 - self.b2) * g * g
            w -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + self.eps)


def score(model, X, y, k=3, batch=2048):
    """Métricas sobre los ejemplos cuya palabra objetivo está en el vocabulario (y != 0)."""
    keep = y != 0
    X, y = X[keep], y[keep]
    n = len(y)
    if n == 0:
        return {"n": 0, "loss": float("nan"), "top1": 0.0, "topk": 0.0}
    nll, top1, topk = 0.0, 0, 0
    for i in range(0, n, batch):
        xb, yb = X[i:i + batch], y[i:i + batch]
        z = model.logits(xb).astype(np.float64)
        z -= z.max(1, keepdims=True)
        nll += float((np.log(np.exp(z).sum(1)) - z[np.arange(len(yb)), yb]).sum())
        top1 += int((z.argmax(1) == yb).sum())
        idx = np.argpartition(-z, k - 1, axis=1)[:, :k]
        topk += int((idx == yb[:, None]).any(1).sum())
    return {"n": n, "loss": nll / n, "top1": top1 / n, "topk": topk / n}
