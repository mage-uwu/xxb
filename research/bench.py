"""BitNet LM on Tiny Shakespeare: {attention | Walsh long-conv} mixer x {ternary MLP | light DDLGN | none} FFN."""
import numpy as np, time, sys, json

sig = lambda x: 1 / (1 + np.exp(-x))
DTYPE = np.float32                                # gradcheck switches this to float64

class P:                                          # parameter + Adam state
    def __init__(self, w, lr_mul=1.0):
        w = np.asarray(w, DTYPE)
        self.w, self.g, self.m, self.v, self.lr_mul = w, np.zeros_like(w), np.zeros_like(w), np.zeros_like(w), lr_mul

def rms_fwd(x):
    r = 1 / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
    return x * r, r

def rms_bwd(dn, n, r):
    return r * (dn - n * (dn * n).mean(-1, keepdims=True))

def act_quant(x, eps=1e-5):
    s = 127 / np.abs(x).max(-1, keepdims=True).clip(min=eps)
    return np.clip(np.round(x * s), -128, 127) / s

def weight_quant(W, eps=1e-5):
    s = 1 / max(np.abs(W).mean(), eps)
    return np.clip(np.round(W * s), -1, 1) / s

class BitLinear:
    def __init__(self, d_in, d_out, rng):
        self.W = P(rng.standard_normal((d_in, d_out)) / np.sqrt(d_in))
        self.params = [self.W]

    def forward(self, x, hard=False):
        self.xn, self.r = rms_fwd(x)
        self.xq, self.Wq = act_quant(self.xn), weight_quant(self.W.w)
        return self.xq @ self.Wq

    def backward(self, dy):                       # STE through both quantizers
        d_in = self.xq.shape[-1]
        self.W.g += self.xq.reshape(-1, d_in).T @ dy.reshape(-1, dy.shape[-1])
        return rms_bwd(dy @ self.Wq.T, self.xn, self.r)

def rope(x, sign=1, base=10000.0):
    T, d = x.shape[-2:]
    ang = np.arange(T)[:, None] * base ** (-np.arange(0, d, 2) / d)
    cos, sin = np.cos(ang).astype(x.dtype), (sign * np.sin(ang)).astype(x.dtype)
    out = np.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
    return out

class BitAttention:
    def __init__(self, d, h, rng):
        self.h, self.dh = h, d // h
        self.layers = [BitLinear(d, d, rng) for _ in range(4)]
        self.params = [p for l in self.layers for p in l.params]

    def split(self, t): B, T, _ = t.shape; return t.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)
    def merge(self, t): B, h, T, dh = t.shape; return t.transpose(0, 2, 1, 3).reshape(B, T, h * dh)

    def forward(self, x, hard=False):
        q, k, v, o = self.layers
        T = x.shape[1]
        self.qr, self.kr = rope(self.split(q.forward(x))), rope(self.split(k.forward(x)))
        self.vh = self.split(v.forward(x))
        S = self.qr @ self.kr.swapaxes(-1, -2) / self.dh ** 0.5
        S = np.where(np.tril(np.ones((T, T), bool)), S, -np.inf)
        Pm = np.exp(S - S.max(-1, keepdims=True))
        self.P = Pm / Pm.sum(-1, keepdims=True)
        return o.forward(self.merge(self.P @ self.vh))

    def backward(self, dout):
        q, k, v, o = self.layers
        dY = self.split(o.backward(dout))
        dP, dv = dY @ self.vh.swapaxes(-1, -2), self.P.swapaxes(-1, -2) @ dY
        dS = self.P * (dP - (dP * self.P).sum(-1, keepdims=True)) / self.dh ** 0.5
        dq, dk = rope(dS @ self.kr, -1), rope(dS.swapaxes(-1, -2) @ self.qr, -1)
        return q.backward(self.merge(dq)) + k.backward(self.merge(dk)) + v.backward(self.merge(dv))

class WalshMix:
    """Hyena-style causal token mixer with a Walsh-Hadamard (dyadic) long conv instead of attention:
        [u | g] = BitLinear(x);  u <- causal short depthwise conv(u);
        v[t, c] = sum_{s <= t} h[c, t XOR s] * u[s, c]     (full-length learned kernel per channel)
        out = BitLinear(v * g)
    Without the s <= t mask this is exactly diag(WHT(h)) in the Walsh domain; with it, it is computable in
    O(T log^2 T) by divide and conquer (each half-split is one unmasked dyadic conv). Training uses the dense form."""
    def __init__(self, d, T, rng, short=3, lr_mul_h=1.0, long=True, kind="walsh"):
        self.d, self.short = d, short
        self.inp, self.out = BitLinear(d, 2 * d, rng), BitLinear(d, d, rng)
        t = np.arange(T)
        self.mask = (t[None, :] <= t[:, None]).astype(DTYPE)
        # walsh: dyadic lag t XOR s;  toeplitz (control): ordinary causal lag t - s, i.e. a Hyena-style long conv
        self.K = t[:, None] ^ t[None, :] if kind == "walsh" else np.maximum(t[:, None] - t[None, :], 0)
        self.onehot = np.eye(T, dtype=DTYPE)[self.K.ravel()]                 # (T*T, T) scatter plan for dh
        h0 = 0.02 * rng.standard_normal((d, T)); h0[:, 0] += 1.0             # identity-ish init
        if not long: h0, lr_mul_h = np.eye(1, T).repeat(d, 0), 0.0         # ablation: long conv frozen to identity
        self.h = P(h0, lr_mul_h)
        self.params = self.inp.params + self.out.params + [self.h]
        if short:
            w0 = 0.02 * rng.standard_normal((d, short)); w0[:, 0] += 1.0
            self.sw = P(w0); self.params.append(self.sw)

    def forward(self, x, hard=False):
        B, T, d = x.shape
        ug = self.inp.forward(x)
        self.u, self.g = ug[..., :d], ug[..., d:]
        us = self.u
        if self.short:                                                     # causal depthwise conv, taps t, t-1, ...
            pad = np.concatenate([np.zeros((B, self.short - 1, d), x.dtype), self.u], 1)
            us = sum(self.sw.w[:, j] * pad[:, self.short - 1 - j: self.short - 1 - j + T] for j in range(self.short))
            self.pad = pad
        self.us = us
        self.M = self.h.w[:, self.K] * self.mask                           # (d, T, T)
        self.v = (self.M @ us.transpose(2, 1, 0)).transpose(2, 1, 0)       # (B, T, d)
        return self.out.forward(self.v * self.g)

    def backward(self, dy):
        B, T, d = dy.shape
        dvg = self.out.backward(dy)
        dv, dg = dvg * self.g, dvg * self.v
        dvt, ust = dv.transpose(2, 1, 0), self.us.transpose(2, 1, 0)       # (d, T, B)
        dM = (dvt @ ust.transpose(0, 2, 1)) * self.mask                    # (d, T, T)
        self.h.g += dM.reshape(d, -1) @ self.onehot
        dus = (self.M.transpose(0, 2, 1) @ dvt).transpose(2, 1, 0)         # (B, T, d)
        du = dus
        if self.short:
            dpad = np.zeros_like(self.pad); k = self.short
            for j in range(k):
                sl = slice(k - 1 - j, k - 1 - j + T)
                dpad[:, sl] += self.sw.w[:, j] * dus
                self.sw.g[:, j] += (dus * self.pad[:, sl]).reshape(-1, d).sum(0)
            du = dpad[:, k - 1:]
        return self.inp.backward(np.concatenate([du, dg], -1))

class BitMLP:                                     # BitNet b1.58 FFN: BitLinear -> ReLU^2 -> BitLinear
    def __init__(self, d, hidden, rng):
        self.up, self.down = BitLinear(d, hidden, rng), BitLinear(hidden, d, rng)
        self.params = self.up.params + self.down.params
        self.ops = 2 * d * hidden                 # int8 x ternary add/subs per token

    def forward(self, x, hard=False):
        self.u = self.up.forward(x)
        return self.down.forward(np.maximum(self.u, 0) ** 2)

    def backward(self, dy):
        return self.up.backward(self.down.backward(dy) * 2 * np.maximum(self.u, 0))

class LogicLayer:
    """Light DLGN gate: 4 learnable truth-table entries per gate (sigmoid-relaxed),
    output = E[table[a, b]] for independent Bernoulli inputs a, b.
    Residual init: table ~ (0, 0, 1, 1), i.e. every gate starts as pass-through of input a."""
    def __init__(self, n_in, n_out, rng, z):
        self.ia, self.ib = rng.integers(0, n_in, n_out), rng.integers(0, n_in, n_out)
        self.n_in = n_in
        tgt = np.concatenate([self.ia, self.ib])      # precomputed scatter plan for backward
        self.order = np.argsort(tgt, kind="stable")
        self.uniq, self.starts = np.unique(tgt[self.order], return_index=True)
        self.th = P(np.tile(np.array([-z, -z, z, z]), (n_out, 1)) + 0.1 * rng.standard_normal((n_out, 4)))
        self.params = [self.th]

    def forward(self, x, hard=False):
        self.a, self.b = np.take(x, self.ia, axis=-1), np.take(x, self.ib, axis=-1)
        self.s = s = (self.th.w > 0).astype(x.dtype) if hard else sig(self.th.w)
        # expand the relaxed truth table into bilinear form: w0 + w1 a + w2 b + w3 ab
        self.w = (s[:, 0], s[:, 2] - s[:, 0], s[:, 1] - s[:, 0], s[:, 0] - s[:, 1] - s[:, 2] + s[:, 3])
        w0, w1, w2, w3 = self.w
        return w0 + w1 * self.a + (w2 + w3 * self.a) * self.b

    def backward(self, dy):
        a, b, s, (w0, w1, w2, w3) = self.a, self.b, self.s, self.w
        N, n = int(np.prod(dy.shape[:-1])), dy.shape[-1]
        dyf, af, bf = dy.reshape(N, n), a.reshape(N, n), b.reshape(N, n)
        dya = dyf * af
        g0, g1, g2, g3 = dyf.sum(0), dya.sum(0), np.einsum("nk,nk->k", dyf, bf), np.einsum("nk,nk->k", dya, bf)
        ds = np.stack([g0 - g1 - g2 + g3, g2 - g3, g1 - g3, g3], -1)
        self.th.g += ds * s * (1 - s)
        da = dyf * (w1 + w3 * bf)
        db = dyf * (w2 + w3 * af)
        g = np.take(np.concatenate([da, db], axis=1), self.order, axis=1)
        dx = np.zeros((N, self.n_in), dy.dtype)
        dx[:, self.uniq] = np.add.reduceat(g, self.starts, axis=1)   # scatter-add to wired inputs
        return dx.reshape(*dy.shape[:-1], self.n_in)

class LogicFFN:
    """RMSNorm -> thermometer soft bits sigmoid((x - t_j) / temp) -> L logic layers (fixed random wiring)
    -> GroupSum -> per-channel gain. n_th=1 reduces to plain sign bits sigmoid(x / temp)."""
    def __init__(self, d, width, depth, rng, z=3.0, lr_mul=1.0, n_th=1, temp=1.0):
        self.d, self.k, self.temp, self.off = d, width // d, temp, False
        q = (np.arange(n_th) + 0.5) / n_th                          # thresholds at N(0,1) quantiles
        self.t = np.quantile(np.random.default_rng(0).standard_normal(200000), q).astype(DTYPE) if n_th > 1 else np.zeros(1, DTYPE)
        self.layers = [LogicLayer(d * n_th if i == 0 else width, width, rng, z) for i in range(depth)]
        self.gain = P(np.ones(d))
        self.params = [p for l in self.layers for p in l.params] + [self.gain]
        for p in self.params[:-1]: p.lr_mul = lr_mul
        self.ops = width * depth                  # 2-input gate evaluations per token

    def forward(self, x, hard=False):
        if self.off: return np.zeros_like(x)
        self.xn, self.r = rms_fwd(x)
        u = (self.xn[..., None] - self.t) / self.temp               # (..., d, n_th)
        self.p = sig(u)
        h = (u > 0).astype(x.dtype) if hard else self.p
        h = h.reshape(*x.shape[:-1], -1)
        for l in self.layers: h = l.forward(h, hard)
        self.gs = h.reshape(*h.shape[:-1], self.d, self.k).mean(-1) - 0.5
        return self.gs * self.gain.w

    def backward(self, dy):
        self.gain.g += (dy * self.gs).reshape(-1, self.d).sum(0)
        dh = np.repeat(dy * self.gain.w / self.k, self.k, axis=-1)
        for l in reversed(self.layers): dh = l.backward(dh)
        dh = dh.reshape(self.p.shape) * self.p * (1 - self.p) / self.temp
        return rms_bwd(dh.sum(-1), self.xn, self.r)

    def diag(self):
        changed = np.mean([((l.th.w > 0) != np.array([0, 0, 1, 1], bool)).any(-1).mean() for l in self.layers])
        return dict(gates_changed=float(changed), gain_abs=float(np.abs(self.gain.w).mean()))

class NoFFN:
    params, ops = [], 0
    def forward(self, x, hard=False): return np.zeros_like(x)
    def backward(self, dy): return np.zeros_like(dy)

class Model:
    def __init__(self, V, d, h, n_layers, make_ffn, seed, make_mixer=None):
        rng = np.random.default_rng(seed)
        make_mixer = make_mixer or (lambda rng: BitAttention(d, h, rng))
        self.emb = P(rng.standard_normal((V, d)))
        self.blocks = [(make_mixer(rng), make_ffn(rng)) for _ in range(n_layers)]
        self.head = P(rng.standard_normal((d, V)) / np.sqrt(d))
        self.params = [self.emb, self.head] + [p for a, f in self.blocks for p in a.params + f.params]
        self.ffn_params = sum(p.w.size for _, f in self.blocks for p in f.params)
        self.ffn_ops = sum(f.ops for _, f in self.blocks)

    def loss(self, idx, tgt, hard=False, train=True):
        self.idx = idx
        x = self.emb.w[idx]
        for attn, ffn in self.blocks:
            x = x + attn.forward(x, hard)
            x = x + ffn.forward(x, hard)
        self.xn, self.r = rms_fwd(x)
        logits = self.xn @ self.head.w
        logits -= logits.max(-1, keepdims=True)
        lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        B, T = tgt.shape
        loss = -np.take_along_axis(lp, tgt[..., None], -1).mean()
        if train:
            dlog = np.exp(lp); np.put_along_axis(dlog, tgt[..., None], np.take_along_axis(dlog, tgt[..., None], -1) - 1, -1)
            dlog /= B * T
            self.head.g += self.xn.reshape(-1, self.xn.shape[-1]).T @ dlog.reshape(-1, dlog.shape[-1])
            dx = rms_bwd(dlog @ self.head.w.T, self.xn, self.r)
            for attn, ffn in reversed(self.blocks):
                dx = dx + ffn.backward(dx)
                dx = dx + attn.backward(dx)
            np.add.at(self.emb.g, idx, dx)
        return loss

def adam(params, t, lr, b1=0.9, b2=0.95, eps=1e-8):
    for p in params:
        p.m = b1 * p.m + (1 - b1) * p.g
        p.v = b2 * p.v + (1 - b2) * p.g ** 2
        p.w -= lr * p.lr_mul * (p.m / (1 - b1 ** t)) / (np.sqrt(p.v / (1 - b2 ** t)) + eps)
        p.g[...] = 0

def run(cfg):
    text = open("shakespeare.txt").read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(data)); train, val = data[:n], data[n:]
    d, h, L, T, B = 64, 4, 2, 64, cfg.get("batch", 16)
    kind = cfg["ffn"]
    make = {"mlp": lambda rng: BitMLP(d, cfg.get("hidden", 4 * d), rng),
            "dlgn": lambda rng: LogicFFN(d, cfg["width"], cfg["depth"], rng, cfg.get("z", 3.0), cfg.get("lr_mul", 1.0), cfg.get("n_th", 1), cfg.get("temp", 1.0)),
            "none": lambda rng: NoFFN()}[kind]
    mixer = {"attn": lambda rng: BitAttention(d, h, rng),
             "walsh": lambda rng: WalshMix(d, T, rng, cfg.get("short", 3), cfg.get("lr_mul_h", 1.0), cfg.get("long", True), cfg.get("kind", "walsh"))}[cfg.get("mixer", "attn")]
    model = Model(len(chars), d, h, L, make, cfg["seed"], mixer)
    rng = np.random.default_rng(1000 + cfg["seed"])
    vrng = np.random.default_rng(12345)
    vstarts = vrng.integers(0, len(val) - T - 1, 256)
    vx = np.stack([val[s:s + T] for s in vstarts]); vy = np.stack([val[s + 1:s + T + 1] for s in vstarts])
    def evaluate(hard):
        return np.mean([model.loss(vx[i:i + 64], vy[i:i + 64], hard, train=False) for i in range(0, 256, 64)])
    steps, lr, t0, hist = cfg["steps"], cfg["lr"], time.time(), []
    for it in range(1, steps + 1):
        s = rng.integers(0, len(train) - T - 1, B)
        x = np.stack([train[j:j + T] for j in s]); y = np.stack([train[j + 1:j + T + 1] for j in s])
        tl = model.loss(x, y)
        warm = min(1.0, it / 100)
        adam(model.params, it, lr * warm * (1 - it / (steps + 1)))
        if it % cfg.get("eval_every", 500) == 0 or it == steps:
            hist.append((it, float(tl), float(evaluate(False))))
    res = dict(cfg, ffn_params=model.ffn_params, ffn_ops=model.ffn_ops,
               val_soft=float(evaluate(False)), val_hard=float(evaluate(True)), secs=round(time.time() - t0, 1), hist=hist)
    if cfg.get("mixer") == "walsh":                       # share of kernel energy off the identity tap
        res["h_offdiag_energy"] = [float((a.h.w[:, 1:] ** 2).sum() / (a.h.w ** 2).sum()) for a, _ in model.blocks]
    if kind == "dlgn":
        res["diag"] = [f.diag() for _, f in model.blocks]
        for _, f in model.blocks: f.off = True
        res["val_ffn_off"] = float(evaluate(False))
    return res

if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    print(json.dumps(run(cfg)), flush=True)
