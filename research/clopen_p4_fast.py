#!/usr/bin/env python3
"""
Fast P4 Latent CLOPEN

This is the minimal architecture that gave the useful ablation result:
    full-resolution sequence
        -> fixed P4 pure-logic reduction
        -> deep CLOPEN stack at T/4
        -> hard ternary logic head

No dynamic routing.
No entropy model.
No cross-attention.
No local/global decoder.
No ragged tensors.

Forward hidden state is {-1,+1}.
Forward learned weights are {-1,0,+1}.

Example:
    python clopen_p4_fast.py
    python clopen_p4_fast.py --device cuda --steps 500 --batch 512
"""

import argparse
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Straight-through discrete ops

class TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        return w.round().clamp(-1, 1)

    @staticmethod
    def backward(ctx, grad):
        return grad


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad):
        return grad


def q3(w):
    return TernarySTE.apply(w)


def hsign(x):
    return SignSTE.apply(x)


# ---------------------------------------------------------------------
# CLOPEN primitives

class PointClopen(nn.Module):
    """
    Grouped 1x1 ternary threshold layer.
    x: [B,T,C]
    """
    def __init__(self, channels, group_size=8):
        super().__init__()
        assert channels % group_size == 0
        self.channels = channels
        self.group_size = group_size
        self.groups = channels // group_size

        self.weight = nn.Parameter(torch.zeros(channels, group_size, 1))
        self.theta = nn.Parameter(torch.zeros(channels))

        # Identity-like initialization keeps deep logic stable.
        with torch.no_grad():
            for o in range(channels):
                self.weight[o, o % group_size, 0] = 1.0

    def forward(self, x):
        y = F.conv1d(
            x.transpose(1, 2),
            q3(self.weight),
            bias=-self.theta,
            groups=self.groups,
        )
        return hsign(y.transpose(1, 2))


class TemporalClopen(nn.Module):
    """
    Depthwise ternary temporal logic convolution.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1
        self.channels = channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel_size))
        self.theta = nn.Parameter(torch.zeros(channels))

        with torch.no_grad():
            self.weight[:, 0, kernel_size // 2] = 1.0

    def forward(self, x):
        p = self.kernel_size // 2
        z = F.pad(x.transpose(1, 2), (p, p), mode="circular")
        y = F.conv1d(
            z,
            q3(self.weight),
            bias=-self.theta,
            groups=self.channels,
        )
        return hsign(y.transpose(1, 2))


class ClopenBlock(nn.Module):
    def __init__(self, channels, group_size=8):
        super().__init__()
        self.temporal = TemporalClopen(channels, 3)
        self.point = PointClopen(channels, group_size)

    def forward(self, x):
        return self.point(self.temporal(x))


# ---------------------------------------------------------------------
# The BLT idea we keep: fixed P4 latentization

class P4MajorityPatch(nn.Module):
    """
    Pure-logic aligned P4 reduction, independently per channel.

    This deliberately uses reshape+sum rather than a depthwise Conv1d:
    P=4 is fixed, so there is no reason to pay convolution dispatch overhead.

        [B,T,C] -> [B,T/4,4,C] -> sum over the 4 local bits -> hard sign

    At inference this is just a 4-input majority/threshold gate (LUT4).
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        B, T, C = x.shape
        assert T % 4 == 0, "Sequence length must be divisible by 4."
        y = x.reshape(B, T // 4, 4, C).sum(dim=2)
        return hsign(y)


class LogicHead(nn.Module):
    def __init__(self, channels, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outputs, channels))
        self.theta = nn.Parameter(torch.zeros(outputs))
        nn.init.normal_(self.weight, mean=0.0, std=0.4)

    def scores(self, x):
        # Global hard majority over latent positions.
        pooled = hsign(x.sum(dim=1))
        return pooled @ q3(self.weight).T - self.theta

    def forward(self, x):
        return hsign(self.scores(x))


class FastP4Clopen(nn.Module):
    """
    x [B,T,C]
      -> P4 pure-logic compression [B,T/4,C]
      -> deep CLOPEN stack
      -> hard logic head
    """
    def __init__(
        self,
        channels=32,
        group_size=8,
        outputs=8,
        depth=3,
    ):
        super().__init__()
        self.patch = P4MajorityPatch(channels)
        self.blocks = nn.ModuleList(
            [ClopenBlock(channels, group_size) for _ in range(depth)]
        )
        self.head = LogicHead(channels, outputs)

    def features(self, x):
        x = self.patch(x)
        for block in self.blocks:
            x = block(x)
        return x

    def scores(self, x):
        return self.head.scores(self.features(x))

    def forward(self, x):
        return hsign(self.scores(x))


# ---------------------------------------------------------------------
# Synthetic benchmark used to validate P4 latentization

class HierarchicalTeacher:
    def __init__(self, channels=32, outputs=8, fanin=6, seed=123):
        g = torch.Generator().manual_seed(seed)
        self.channels = channels
        self.outputs = outputs

        self.weight = torch.zeros(outputs, channels)
        for o in range(outputs):
            idx = torch.randperm(channels, generator=g)[:fanin]
            signs = torch.where(
                torch.rand(fanin, generator=g) > 0.5,
                torch.ones(fanin),
                -torch.ones(fanin),
            )
            self.weight[o, idx] = signs

    def labels(self, latent, device):
        # target depends on the latent/global state
        global_state = hsign(latent.sum(dim=1))
        return hsign(global_state @ self.weight.to(device).T)


def make_batch(
    batch_size,
    latent_positions,
    channels,
    teacher,
    device,
    noise=0.12,
):
    # One latent bit-vector repeated four times, then locally corrupted.
    latent = torch.where(
        torch.rand(batch_size, latent_positions, channels, device=device) > 0.5,
        1.0,
        -1.0,
    )

    x = latent.repeat_interleave(4, dim=1)

    if noise > 0:
        flips = torch.where(
            torch.rand_like(x) < noise,
            -1.0,
            1.0,
        )
        x = x * flips

    y = teacher.labels(latent, device)
    return x, y


def train(
    model,
    device,
    steps=300,
    batch_size=128,
    latent_positions=15,
    channels=32,
    outputs=8,
    noise=0.12,
):
    teacher = HierarchicalTeacher(channels, outputs)

    opt = torch.optim.Adam(
        [
            {"params": model.blocks.parameters(), "lr": 1e-3},
            {"params": model.head.parameters(), "lr": 3e-2},
        ]
    )

    for step in range(steps):
        x, y = make_batch(
            batch_size,
            latent_positions,
            channels,
            teacher,
            device,
            noise,
        )

        s = model.scores(x)
        loss = F.relu(1.0 - y * s).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        x, y = make_batch(
            4096,
            latent_positions,
            channels,
            teacher,
            device,
            noise,
        )
        pred = model(x)
        bit_acc = (pred == y).float().mean().item()
        exact = (pred == y).all(dim=1).float().mean().item()

    return bit_acc, exact


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(
    model,
    device,
    batch_size,
    latent_positions,
    channels,
    outputs,
    noise,
    iters=50,
):
    teacher = HierarchicalTeacher(channels, outputs)
    x, y = make_batch(
        batch_size,
        latent_positions,
        channels,
        teacher,
        device,
        noise,
    )

    # Warmup.
    for _ in range(8):
        s = model.scores(x)
        loss = F.relu(1.0 - y * s).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()

    sync(device)

    samples = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()

        s = model.scores(x)
        loss = F.relu(1.0 - y * s).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()

        sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)

    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--latent-positions", type=int, default=15)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--outputs", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--bench-iters", type=int, default=50)
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    if device.type == "cpu":
        torch.set_num_threads(min(5, torch.get_num_threads()))

    T = args.latent_positions * 4

    model = FastP4Clopen(
        channels=args.channels,
        group_size=args.group_size,
        outputs=args.outputs,
        depth=args.depth,
    ).to(device)

    acc, exact = train(
        model,
        device,
        steps=args.steps,
        batch_size=args.batch,
        latent_positions=args.latent_positions,
        channels=args.channels,
        outputs=args.outputs,
        noise=args.noise,
    )

    ms = benchmark(
        model,
        device,
        args.batch,
        args.latent_positions,
        args.channels,
        args.outputs,
        args.noise,
        args.bench_iters,
    )

    print("FAST P4 LATENT CLOPEN")
    print(f"device             : {device}")
    print(f"sequence length    : {T}")
    print(f"deep positions     : {args.latent_positions}")
    print(f"compression        : 4.0x")
    print(f"channels           : {args.channels}")
    print(f"group size         : {args.group_size}")
    print(f"depth              : {args.depth}")
    print(f"bit accuracy       : {acc:.4f}")
    print(f"exact sample acc   : {exact:.4f}")
    print(f"forward+backward   : {ms:.3f} ms")


if __name__ == "__main__":
    main()
