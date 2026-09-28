# moth

**Moth** is a Monarch Mixer language model with ternary weights, in one short, dependency-free C file (`moth.c`, about 360 lines, nanoGPT style). It trains a character-level LM and then samples from it. It is built for CPU-native speed: no data moves unless it has to.

```
block(x):  x += Mon_o( Mon_a(n) ⊙ conv_T(Mon_v(n)) ),  n = rms(x)     # sequence mixer: gated causal long conv
           x += Mon_d( relu(Mon_u(n))² ),               n = rms(x)     # channel mixer
Mon(x)  =  P · L · Pᵀ · R · x      L, R block-diagonal (√d blocks of √d×√d), 2·d^1.5 params
```

## Quantisation

| | Weights | Activations | Arithmetic |
|---|---|---|---|
| **Forward** (Monarch) | ternary {−1,0,+1}, absmean scale | int8, per-token absmax | int16 accumulate. A ternary dot over 16 terms is at most 16·127 in magnitude. |
| **Forward** (long conv) | ternary per-channel kernel over the full context | int8, static EMA-calibrated scale | int16 accumulate (at most 128·127), strictly causal |
| **Backward** | the same ternary codes (STE to the fp32 master weights) | gradients int8: delayed per-tensor scale, stochastic rounding | `dx = Wᵀ·g8` in int16, `dW = g8ᵀ·x8` in per-thread int32 slabs |

The following stay in fp32, as in BitNet: the embedding and tied head, the RMSNorm statistics, the gate multiply, the scale factors and AdamW.

## The best part is no part: data movement that was removed

Each change below removes O(tokens) memory traffic. The only transpositions left touch weights (O(params), about 4 KB, once per step).

| Removed | How |
|---|---|
| **Monarch permutations** (strided gathers in every L-factor loop, a transposed weight copy) | View a width-d vector as an M×M tile. R mixes along rows. L mixes along columns, which on a row-major tile is lane-wise vector math with weights stored `[o][i][lane]`. P and Pᵀ never exist, and every loop streams contiguous int8. |
| **Float mid tensor between R and L** | Monarch runs fused per token in registers/L1. R's int16 output is requantised straight to int8 (absmax codes are scale-invariant), and only those codes are stored, for the backward. |
| **RMSNorm output tensor** | `codes(x·r) = codes(x)`, so the norm folds into the per-token scale. The backward recomputes `x·r` from the residual stream. |
| **Duplicate quantisation** | Mon_a and Mon_v share one set of input codes. |
| **Gate, relu², conv-input, residual-add passes** | All fused into two per-token loops per layer, with no intermediate float tensors. |
| **Backward requantisation of float inputs** (and storing those floats) | The input's per-token scale is folded into the gradient before quantising, `g' = dy·s_x[t]`, so `dW = Σ g8'ᵀ·x8` reuses the forward's int8 codes directly. `dx` takes the scale back per token. |
| **Per-tensor gradient absmax pre-passes and barriers** | Delayed scaling (last step's amax, as in FP8 training) with saturation and stochastic rounding. Each Monarch's backward is one fused per-token pass. Steps −1 and 0 only calibrate the scales. |
| **Horizontal reductions in Rᵀ** | A 4 KB ternary transpose of R once per step. The dR slab is stored `[b][o][i]` and transposed only at reduce time. |
| **int32 widening** | Ternary dot products fit in int16, which doubles the SIMD lanes. |

Speed on 4 cores (AVX-512), per training step at the default size (0.31M params, 2048 tokens per step):

| | 1 thread | 4 threads |
|---|---|---|
| v1: separate passes, strided permutations | 679 ms | 255 ms |
| v2: fused, permutation-free, delayed int8 grads | 442 ms | 135 ms |
| v2 + int16 accumulators | **307 ms** | **105 ms** |

The loss curves match (val 2.17 at step 500 for both). The int16 change is bit-identical.

## Result

The full default run takes 3000 steps at about 105 ms/step, around 5.5 minutes on 4 cores. It reaches **val loss 1.97** (train 1.75) on Tiny Shakespeare:

```
LEONTES:
If the deadset on father with and ray thou to not
That sham is namping and my more,
Whild de wer charl I am your come, made with a on my
soody o's parsh in he nou a grough hen and forse!
```

## Run

```sh
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
cc -O3 -march=native -fopenmp moth.c -o moth -lm
./moth input.txt          # with no input file, it trains on its own source
```

The defaults are d=256 (M=16), 4 layers, T=128, batch 16 and 3000 steps. Change the `#define`s at the top of the file to resize the model. Two limits apply:

- The int16 accumulators need `M·127` and `T·127` below 2¹⁵, so M ≤ 256 and T ≤ 256.
- The int32 dW slabs need `B·T·127²` below 2³¹.

**Next steps for more speed:** VNNI (`vpdpbusd`) wants the reduction dimension interleaved by 4. The fp32 tied head is now a visible share of the step.
