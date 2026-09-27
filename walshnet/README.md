# walshnet

A deep stack of **gated short-conv BitNet blocks**, pretrained and served in pure C: C11, AVX-512, AMX and OpenMP, with no dependencies.

```
block(x) = BitLinear_out( conv_K(u) ⊙ g ),   [u | g] = BitLinear_in(x),   x ← x + block(x)
```

- **BitLinear** (BitNet b1.58): per-token RMSNorm, then int8 absmax activations × ternary {−1, 0, +1} absmean weights. Training uses the straight-through estimator.
- **conv_K:** depthwise K-tap convolution. It's **causal** (taps t, t−1, …) for language modelling and **centred** for masked LM.
- **No attention, no FFN, no long convolution.** The ablations in `../research` found this simplest stack gave the best quality per parameter on our benchmark (see below).

## Standalone reference: `dilated_walshnet.c`

Everything below (training, inference, streaming, masked infill, benchmarks and the full self-test) is also in **one self-contained C file**, generated from these sources. It trains byte-identical checkpoints:

```sh
gcc -O3 -march=native -mamx-tile -mamx-int8 -mamx-bf16 -fopenmp dilated_walshnet.c -lm -o dilated_walshnet
./dilated_walshnet test
./dilated_walshnet train --data input.txt --dilate 3 --out model.bin            # causal
./dilated_walshnet train --data input.txt --mode bidir --dilate 3 --out mlm.bin # bidirectional (masked LM)
./dilated_walshnet generate --model model.bin --prompt "ROMEO:"
./dilated_walshnet fill --model mlm.bin --text "Wh_t is th_ m_tter"
./dilated_walshnet bench --model model.bin
```

## Why it's fast on a CPU

| | How |
|---|---|
| **Forward** | int8 × ternary on **AMX tiles** (`tdpbusd`, about 2 TMAC/s per core), with a VNNI fallback. Bit-exact against an fp32 reference. |
| **Backward** | Both GEMMs run on **AMX bf16 tiles** (`tdpbf16ps`). Ternary weights and int8 activation codes are exact in bf16; only the gradients are rounded (relative error 1.7e-3 vs fp32). |
| **Training parallelism** | **Data-parallel replicas:** each core takes its slice of the batch through the whole network with no inner barriers. The weights and packed ternary forms are shared, and gradients are reduced once per step. |
| **Glue kernels** | In-register 16×16 transposes, fused bf16 packing, and a per-sequence fused conv/gate backward. |
| **Decoding** | **Constant-state streaming:** each layer keeps only the last K−1 conv inputs, so each new token costs O(layers · d²) regardless of stream length. There's no KV cache and no window recompute. It's verified to match the batch forward bit for bit. |

## Build, test, run

```sh
make                 # gcc/clang with AVX-512 (+ AMX optional; auto-detected at runtime, WN_NO_AMX=1 disables)
./test               # gemm/BitLinear/AMX/bf16 vs references, finite-difference grads, receptive field,
                     # streaming == batch, data-parallel == single-model gradients
./train --data input.txt --out model.bin                        # causal LM, 7 layers, d=64
./train --data input.txt --mode bidir --out mlm.bin             # masked LM (centred conv, 15% masking)
./train --data input.txt --d 128 --layers 12 --batch 32 --steps 8000 --lr 2e-2 --out big.bin
./infer bench    --model model.bin
./infer generate --model model.bin --prompt "ROMEO:" --n 400 --temp 0.7
./infer fill     --model mlm.bin   --text "Wh_t is th_ m_tter, my g_od lord?"
```

`train` options:

| Flag | Default |
|---|---|
| `--mode` | `causal` (or `bidir`) |
| `--d` | 64 |
| `--layers` | 7 |
| `--T` | 64 |
| `--K` | 3 (conv taps) |
| `--dilate` | 0 (off); with c, layer l's taps are spaced 2^(l mod c) apart |
| `--save-best` | off; keeps the best-validation checkpoint |
| `--batch` | 16 |
| `--steps` | 3000 |
| `--lr` | 3e-2 |
| `--warmup` | 100 |
| `--wd` | 0 |
| `--mask` | 0.15 |
| `--seed` | 0 |
| `--threads` | all cores |

Adam uses β = (0.9, 0.95) with linear warmup, then linear decay. The data is any text file, tokenised to a character vocabulary.

## Results

**Setup:** Tiny Shakespeare (1.1 MB, 65 characters, 90/10 split). One 4-core Xeon @ 2.1 GHz with AVX-512 and AMX.

| Model | Params | Val loss | Train time | Train tok/s |
|---|---|---|---|---|
| 7 × d64, causal, 3K steps × 16 × 64 | 96K | 1.718 | **5.8 s** | 530K (630K at batch 64) |
| 7 × d64, masked LM (centred conv) | 96K | 1.329 (masked chars) | 5.3 s | 577K |
| 12 × d128, causal, 8K steps × 32 × 64 | 611K | **1.523** | 125 s | 131K |

| Inference | 7 × d64 | 12 × d128 |
|---|---|---|
| Batched forward | 2.8–3.1M tok/s | 0.63M tok/s |
| Streaming, 1 stream (1 core) | 280–310K tok/s (3.2–3.6 µs/token) | 58K tok/s (17 µs/token) |
| Streaming, 64–1024 streams (4 cores) | **2.9M tok/s** | **1.0M tok/s** |

### Ablations that led here

Same data, d=64, 3,000 steps, 2 seeds; the scripts are in `../research`.

| Causal val loss | Params | Val loss |
|---|---|---|
| Attention + ternary MLP (BitNet baseline) | 107K | 1.814 |
| Walsh long conv + short conv, + ternary MLP | 107K | 1.766 |
| 7 × (Walsh long conv + short conv), no FFN | 124K | 1.734 |
| **7 × short conv only (this)** | 96K | **1.715** |

- **Walsh long conv:** it mixes positions by XOR distance and didn't help. At T=256 it hurt more (1.768 vs 1.734).
- **Logic-gate FFNs:** DDLGN and CLOPEN matched having no FFN at all.
- **Masked LM:** with a centred conv, masked-LM loss improved from 1.604 (Walsh bidirectional) to 1.329.

## Limitations

- **Context:** receptive field = layers × (K−1) + 1 tokens (15 for 7 layers with K=3). Long-range dependencies need more depth, a bigger K, or a shift-invariant long convolution.
- **Evaluation:** one small single-domain corpus, character level. Nothing here is tested at scale.
- **Hardware:** requires AVX-512 (VNNI, BF16). AMX is used when present.
- **Shapes:** `d` must be a multiple of 16, or 32 for the AMX path. The bidirectional mode needs an odd K.
- **Checkpoints** hold weights only, not optimizer state.
