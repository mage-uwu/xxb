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
