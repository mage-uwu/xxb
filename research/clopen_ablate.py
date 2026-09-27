import torch, statistics, sys
import clopen_p4_fast as C   # the original script, unchanged
torch.set_num_threads(4)
dev = torch.device("cpu")

class NoPatch(torch.nn.Module):                       # skip P4: keep all T positions
    def forward(self, x): return x

def changed(model):                                   # ternary weights that differ from their identity init
    ch = tot = 0; th = []
    for blk in model.blocks:
        for mod, init in ((blk.temporal, lambda w: (torch.arange(w.shape[2]) == w.shape[2] // 2).float().expand_as(w)),
                          (blk.point, None)):
            q = mod.weight.detach().round().clamp(-1, 1)
            if init is not None: ref = init(q)
            else:
                ref = torch.zeros_like(q)
                for o in range(q.shape[0]): ref[o, o % q.shape[1], 0] = 1
            ch += (q != ref).sum().item(); tot += q.numel(); th.append(mod.theta.detach().abs().max().item())
    return ch, tot, max(th) if th else 0

def run(depth, patch=True, steps=300, seed=0, blocks_lr=None):
    torch.manual_seed(seed)
    m = C.FastP4Clopen(32, 8, 8, depth)
    if not patch: m.patch = NoPatch()
    if blocks_lr is not None:                         # monkeypatch optimizer lr for blocks
        orig = torch.optim.Adam
        def adam(groups, **kw):
            groups[0]["lr"] = blocks_lr; return orig(groups, **kw)
        C.torch.optim.Adam = adam
    acc, ex = C.train(m, dev, steps=steps, batch_size=128, latent_positions=15, channels=32, outputs=8, noise=0.12)
    if blocks_lr is not None: C.torch.optim.Adam = orig
    return acc, ex, changed(m) if depth else (0, 0, 0)

def oracle(n=4096):                                    # perfect P4 decode + teacher weights
    t = C.HierarchicalTeacher(32, 8); torch.manual_seed(99)
    x, y = C.make_batch(n, 15, 32, t, dev, 0.12)
    lat = C.hsign(x.reshape(n, 15, 4, 32).sum(2))     # P4 majority (ties -> +1)
    pred = C.hsign(C.hsign(lat.sum(1)) @ t.weight.T)
    return (pred == y).float().mean().item(), (pred == y).all(1).float().mean().item()

print("oracle (P4 majority + true teacher weights): bit %.4f exact %.4f" % oracle())
cfgs = [("depth 3 (default)", dict(depth=3)), ("depth 0 (no CLOPEN blocks)", dict(depth=0)),
        ("depth 3, no P4", dict(depth=3, patch=False)), ("depth 0, no P4", dict(depth=0, patch=False)),
        ("depth 3, 2000 steps", dict(depth=3, steps=2000)), ("depth 0, 2000 steps", dict(depth=0, steps=2000)),
        ("depth 3, blocks lr 3e-2, 2000 steps", dict(depth=3, steps=2000, blocks_lr=3e-2))]
for name, kw in cfgs:
    rs = [run(seed=s, **kw) for s in range(3)]
    a = [r[0] for r in rs]; e = [r[1] for r in rs]; ch = rs[0][2]
    print(f"{name:38s} bit {statistics.mean(a):.4f} ±{statistics.pstdev(a):.4f}  exact {statistics.mean(e):.4f}"
          + (f"  | block weights changed {ch[0]}/{ch[1]}, max|theta| {ch[2]:.3f}" if kw.get('depth', 3) else ""))
