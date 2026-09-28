# moth

**Moth** is a Monarch Mixer language model with ternary weights, in one short, dependency-free C file (`moth.c`, about 330 lines, nanoGPT style). It trains a character-level LM and then samples from it.

```
block(x):  u = rms(x);  x += Mon_o( Mon_a(u) ⊙ conv_T(Mon_v(u)) )     # sequence mixer: gated causal long conv
           u = rms(x);  x += Mon_d( relu(Mon_u(u))² )                 # channel mixer
Mon(x)  =  P · L · Pᵀ · R · x      L, R block-diagonal (√d blocks of √d×√d), 2·d^1.5 params
```

## Quantisation

| | Weights | Activations | Arithmetic |
|---|---|---|---|
| **Forward** (Monarch factors) | ternary {−1,0,+1}, absmean scale | int8, per-token absmax | int32 accumulate. Weights are ternary, so it is adds and subtracts only. |
| **Forward** (long conv) | ternary per-channel kernel over the full context | int8, static EMA-calibrated scale | int32 accumulate, strictly causal |
| **Backward** | the same ternary codes (STE to the fp32 master weights) | gradients stochastically rounded to int8, per tensor; inputs requantised to int8, per tensor | `dx = Wᵀ·g8` (ternary × int8) and `dW = g8ᵀ·x8` (int8 × int8), both int32 |

The following stay in fp32, as in BitNet: the token embedding and tied output head, the RMSNorm, the gate multiply, the scale factors and the AdamW update.

Some design notes:

- **Monarch permutation is free.** The Pᵀ between the two block-diagonal factors is only a stride. Each factor reads and writes through `IX(tr,b,j)`, so the activations are never transposed.
- **The conv is causal after quantisation.** A per-token scale can't be factored out of a sum over time, and a per-sequence scale would leak the future's absmax. So the conv input uses a scale calibrated by an EMA of previous batches: it's used first, then updated.
- **Per-tensor scales in the backward pass** let the int32 reduction over tokens in `dW` run with no rescaling inside the loop. Stochastic rounding keeps the int8 gradients unbiased.

## Run

```sh
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
cc -O3 -march=native -fopenmp moth.c -o moth -lm
./moth input.txt          # with no input file, it trains on its own source
```

The defaults are d=256 (M=16), 4 layers, T=128, batch 16 and 3000 steps: 0.31M parameters, 0.29M of them ternary. Change the `#define`s at the top of the file to resize the model. Keep `B*T*127²` below 2³¹, so the int32 accumulators can't overflow.
