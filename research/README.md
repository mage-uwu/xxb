# research scripts

These are the prototypes and benchmarks behind `walshnet/`.

- **`bench.py`:** NumPy BitNet LM with manual backward passes. It covers:
  - mixers: attention, Walsh long conv, Toeplitz long conv, or short conv only;
  - FFNs: ternary MLP, light DDLGN, or none.

  Run it as `python3 bench.py '{"ffn":"mlp","mixer":"walsh","lr":0.03,"seed":0,"steps":3000}'` (it expects `shakespeare.txt` in the current directory).
- **`sweep.py`:** runs a JSON list of `bench.py` configs, 4 at a time with 1 thread each.
- **`kern_bench.c`:** inference micro-benchmarks for a BitNet layer: VNNI BitLinear, fp32 attention, and bit-sliced DDLGN.
- **`walsh_mixer_bench.c`:** end-to-end layer throughput of the Walsh / short-conv / Toeplitz mixers against attention, each combined with a ternary MLP or DDLGN.

  Build it with `gcc -O3 -march=native -ffast-math walsh_mixer_bench.c -lm`.

## Findings (Tiny Shakespeare, d=64, 2 layers, T=64, 3000 steps, 2 seeds)

| Mixer | + ternary MLP | + DDLGN |
|---|---|---|
| attention (BitNet) | 1.814 | 1.991 |
| Walsh long conv + short conv + gate | 1.775 | 1.880 |
| short conv + gate only | 1.773 | — |
| Toeplitz (ordinary causal) long conv + short conv + gate | 1.760 | — |

- **Walsh long conv:** the learned Walsh kernels stay near identity (6–21% of their energy off the identity tap), so at this context length the gain over attention comes from the short conv and the gate.
- **Inference speed:** at BERT-tiny shapes (d=128, seq 128), Walsh + bit-sliced DDLGN is about 3.5× the layer throughput of attention + ternary MLP. At seq 512 it's about 13×.

## CLOPEN (`clopen_p4_fast.py`, `clopen_ablate.py`, `clopen_gate_bench.c`)

`clopen_p4_fast.py` is the original P4-latent CLOPEN script. `clopen_ablate.py` ablates it over 3 seeds:

| Configuration | Bit accuracy | Exact match |
|---|---|---|
| Oracle: P4 majority + true teacher weights | 0.842 | 0.287 |
| Depth 3 (default, 300 steps) | 0.821 | 0.232 |
| Depth 0 | 0.821 (identical) | 0.232 |
| No P4, depth 3 or 0 | 0.792 | 0.173 |
| Depth 3, 2,000 steps | 0.615 | 0.019 |
| Depth 0, 2,000 steps | 0.827 | 0.243 |

- **The CLOPEN stack is a no-op in the default run.** 0 of 1,056 block weights leave their identity init: at lr 1e-3, 300 steps can't move a latent weight across the ±0.5 rounding boundary. Depth 3 is bit-identical to depth 0.
- **The gain comes from P4 majority denoising.** The task is built around it: the input is a latent repeated 4× with 12% bit flips.
- **When the blocks do train, accuracy collapses.** Two causes:
  - the identity STE passes gradients through gates far from their threshold;
  - `PointClopen` is block-diagonal over groups and `TemporalClopen` is depthwise, so channel groups never mix before the head.
- **Circular padding** in `TemporalClopen` wraps the sequence end onto its start, and is non-causal.

**As an LM FFN** (`walshnet --ffn clopen`), with random fan-in wiring and a clipped STE, CLOPEN-3 matches DDLGN at an equal gate count:

| FFN | causal val loss | masked-LM val loss |
|---|---|---|
| CLOPEN-3 | 1.874 | 1.727 |
| DDLGN | 1.880 | 1.712 |

- **Neither beats having no FFN** in the Walsh model (1.875), and quality is flat from 256 to 8K gates.
- **Clipped vs identity STE:** the clipped STE beats the identity STE by 0.015–0.045.
- **Fan-in:** fan-in 8 is no better than fan-in 3.
- **Hardened speed:** CLOPEN-3 costs one `vpternlogd` per gate like DDLGN, but is about 12% slower per gate because of the extra load.
