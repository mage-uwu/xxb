# walshnet

Pure C (C11 + AVX-512 + OpenMP, no dependencies) pretraining and inference for **Walsh-mixer BitNet** language models:

- **Token mixer:** Hyena-style block with a **Walsh–Hadamard (dyadic) long convolution**: `[u|g] = BitLinear(x)`, then a causal 3-tap depthwise conv on `u`, then `v[t] = Σ h[t⊕s]·u[s]`, then `BitLinear(v ⊙ g)`.
  - **causal** (`s ≤ t`, for language modelling): O(T log²T) divide and conquer. Each half-split is one full dyadic conv, done with FWHTs.
  - **bidirectional** (masked LM, BERT-style): exactly `WHT⁻¹(WHT(h) ⊙ WHT(u))`, O(T log T), additions only in the transforms.
- **FFN**, one of:
  - **ternary MLP:** BitNet b1.58, ReLU².
  - **light DDLGN:** differentiable logic-gate network. 4 truth-table logits per 2-input gate, fixed random wiring, residual (pass-through) init, thermometer-encoded inputs, GroupSum readout.
  - **CLOPEN:** ternary threshold gates over ±1 states, `sign(Σ q3(w)·x − θ)`, with fan-in G, fixed random wiring, pass-through init, thermometer sign inputs and a clipped straight-through estimator. The training forward pass *is* the inference function, with no relaxation.
- **BitLinear** (BitNet b1.58): per-token RMSNorm, then int8 absmax activations × ternary absmean weights. The forward pass is the *exact* int8 × ternary product on AVX-512 VNNI (`vpdpbusd`). Training uses the straight-through estimator with fp32 backward GEMMs.

## Build and test

```sh
make            # train, infer, test   (needs gcc/clang with AVX-512 VNNI + OpenMP)
./test          # kernels vs scalar refs, finite-difference gradients (4 model variants), causality
```

## Train

```sh
./train --data input.txt --mode causal --ffn mlp   --out causal_mlp.bin
./train --data input.txt --mode causal --ffn ddlgn --out causal_ddlgn.bin
./train --data input.txt --mode bidir  --ffn mlp   --out bidir_mlp.bin      # masked LM, 15% masking (80/10/10)
./train --data input.txt --mode bidir  --ffn ddlgn --out bidir_ddlgn.bin
./train --data input.txt --mode causal --ffn clopen --cl-lrmul 3 --out causal_clopen.bin
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--d` | 64 | model width (multiple of 16) |
| `--layers` | 2 | number of blocks |
| `--T` | 64 | sequence length (power of 2) |
| `--hidden` | 4d | MLP hidden size |
| `--batch` | 16 | sequences per step |
| `--steps` | 3000 | training steps |
| `--lr` | 3e-2 | Adam learning rate (linear warmup, then linear decay) |
| `--dl-width` | 2048 | gates per DDLGN layer |
| `--dl-depth` | 4 | DDLGN layers |
| `--dl-nth` | 8 | thermometer thresholds per input channel |
| `--dl-temp` | 0.5 | thermometer sigmoid temperature |
| `--dl-z` | 1 | residual-init strength (truth-table logits start at ±z) |
| `--dl-lrmul` | 10 | gate learning-rate multiplier |
| `--cl-width` / `--cl-depth` / `--cl-fanin` | 2048 / 4 / 3 | CLOPEN gates per layer, layers, and inputs per gate |
| `--cl-nth` / `--cl-temp` | 8 / 0.5 | CLOPEN thermometer thresholds and STE window scale |
| `--cl-ste` | clip | `clip` passes gradients only when \|pre-activation\| ≤ 1; `id` always passes them |
| `--cl-lrmul` | 1 (3 works best) | CLOPEN learning-rate multiplier |
| `--threads` | all cores | OpenMP threads |

Adam uses β = (0.9, 0.95). The data is any text file, tokenised to a character vocabulary.

## Inference

```sh
./infer bench    --model causal_ddlgn.bin --tokens 8192     # batched throughput (+ bit-sliced vs float check)
./infer generate --model causal_mlp.bin --prompt "ROMEO:" --n 400 --temp 0.8
./infer fill     --model bidir_mlp.bin  --text "Wh_t is th_ m_tter, my g_od lord?"
```

At load time, a DDLGN FFN is **hardened**: each gate's truth table becomes one of 16 two-input functions. Gates are sorted by type (with the wiring remapped) and evaluated **bit-sliced**, 512 tokens per `__m512i`, one `vpternlogd` per gate. GroupSum is a carry-save vertical popcount. `infer bench` checks that the bit-sliced engine reproduces the float hard path exactly.

## Results (Tiny Shakespeare, d=64, 2 layers, T=64, batch 16, 3000 steps; 4-core Xeon @ 2.1 GHz, AVX-512)

| Model | val loss | hardened | train tok/s | inference tok/s (8192-token batches) |
|---|---|---|---|---|
| causal Walsh + ternary MLP | 1.775 | — | 266K | 2.2M |
| causal Walsh + DDLGN | 1.880 | 1.880 | 134K | 2.9M (bit-sliced) |
| bidir Walsh + ternary MLP (masked LM) | 1.620 | — | 256K | 2.2M |
| bidir Walsh + DDLGN (masked LM) | 1.690 | 1.690 | 136K | 2.9M (bit-sliced) |

- **Loss:** causal models use next-char loss in nats. Masked-LM models use loss on masked characters only, so the two aren't directly comparable.
- **Agreement with the NumPy prototype** (`../research/bench.py`, same configs, 2 seeds): 1.771 / 1.778 (MLP) and 1.877 / 1.883 (DDLGN).
- **Training speed:** the C trainer is about 7× faster than NumPy for the MLP model and about 95× faster for DDLGN, which NumPy can't vectorise well.
- **Hardening:** the "hardened" column uses fixed boolean gates and binary inputs, i.e. what the bit-sliced engine runs. It costs nothing.

### Logic FFNs vs no FFN (causal, same setup, mean of 2 seeds)

| FFN | 3,000 steps | 10,000 steps (1 seed) |
|---|---|---|
| none | 1.875 | 1.849 |
| DDLGN, 8K gates | 1.880 | 1.850 |
| CLOPEN-3, 8K gates | 1.874 | 1.845 |
| ternary MLP | **1.766** | **1.715** |

With this Walsh mixer, which already contains a multiplicative gate, neither logic-gate FFN measurably beats having no FFN. That holds from 256 to 8K gates and out to 10K steps. Only the ternary MLP adds quality. In an attention model, DDLGN did help (1.991 vs 2.071 with no FFN).

**Hardened inference cost:** at equal gate count, CLOPEN-3 is about 12% slower per gate than DDLGN (1.95 vs 1.73 ns per gate per 512 tokens). Both evaluate as a single `vpternlogd`; CLOPEN needs a third load. See `../research/clopen_gate_bench.c`.

## Performance notes

- **GEMM:** 8×32 register-blocked AVX-512 kernel with packed B panels (avoids 4K aliasing). About 130 GFLOP/s per core, and it scales linearly to 4 cores (~515 GFLOP/s).
- **Weight gradients** (`xᵀ·dy`) use an in-register 16×16 AVX-512 transpose followed by the NN kernel.
- **DDLGN training** runs in token tiles of 32 through all gate layers, so the working set stays in L2. The backward pass recomputes the tile's forward pass instead of storing activations.
- **Profiling:** build with `-DWN_PROF` to get per-component timers (`g_prof`).

## Limitations

- **Hardware:** requires AVX-512F/BW/VNNI.
- **Shapes:** `T` must be a power of 2 and `d` a multiple of 16.
- **Generation** recomputes a sliding window of `T` tokens per new token. There is no incremental decode cache yet.
- **Bidirectional mode** still uses the causal 3-tap short conv. A centred short conv would probably help masked-LM quality.
- **Checkpoints** store weights only, not optimizer state, so training can't be resumed. The format is `WNT2`; the Config changed when CLOPEN was added.
- **CLOPEN inference** uses the float path. A bit-sliced CLOPEN engine exists only in `../research/clopen_gate_bench.c`.
