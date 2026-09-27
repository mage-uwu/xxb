// walshnet: pure C (C11 + AVX-512 + OpenMP) pretraining and inference for Walsh-mixer BitNet models.
#pragma once
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ utils */
#ifdef WN_PROF                                  // build with -DWN_PROF for per-component timings
enum { P_BLF, P_BLB_DW, P_BLB_DX, P_BLB_RMS, P_PREP, P_SHORT, P_LONGF, P_LONGB, P_GATE, P_DLF, P_DLB, P_HEAD, P_EMB, P_ADAM, P_N };
extern double g_prof[P_N];
#define PROF(id, stmt) do { double _t0 = now_sec(); stmt; g_prof[id] += now_sec() - _t0; } while (0)
#else
#define PROF(id, stmt) do { stmt; } while (0)
#endif
void *xmalloc(size_t bytes);                    // 64-byte aligned, zeroed; aborts on failure
double now_sec(void);

typedef struct { uint64_t s; } Rng;
static inline uint64_t rng_u64(Rng *r) { r->s ^= r->s << 13; r->s ^= r->s >> 7; r->s ^= r->s << 17; return r->s; }
static inline float rng_unif(Rng *r) { return (rng_u64(r) >> 40) * (1.0f / 16777216.0f); }   // [0, 1)
float rng_normal(Rng *r);
static inline Rng rng_seed(uint64_t seed) { Rng r = {seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull}; if (!r.s) r.s = 1; for (int i = 0; i < 8; i++) rng_u64(&r); return r; }

static inline __m512 exp512(__m512 x) {            // AVX-512 exp, rel err ~1e-7 (x clamped to [-87, 88])
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.0f)), _mm512_set1_ps(88.0f));
    const __m512 t = _mm512_mul_ps(x, _mm512_set1_ps(1.44269504f)), n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512 f = _mm512_sub_ps(t, n);
    __m512 p = _mm512_set1_ps(1.535336188e-4f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.339887440e-3f)); p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.618437357e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.550332471e-2f)); p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.402264791e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.931472028e-1f)); p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}
static inline __m512 sigm512(__m512 x) { return _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_add_ps(_mm512_set1_ps(1.0f), exp512(_mm512_sub_ps(_mm512_setzero_ps(), x)))); }

/* trainable parameter with Adam state */
typedef struct { float *w, *g, *m, *v; size_t n; float lr_mul; } Param;
void param_init(Param *p, size_t n, float lr_mul);

/* ------------------------------------------------------------------ gemm (fp32, row-major) */
// C[M][N] (+)= A[M][K] * B[K][N]
void sgemm(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc, int accumulate);
void transpose(const float *src, int rows, int cols, float *dst);   // dst[cols][rows]
// C[M][N] (+)= A^T * B with A stored [K][M] (lda) -- weight gradients without explicit transposes
void sgemm_tn(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc, int accumulate);

/* ------------------------------------------------------------------ BitLinear (BitNet b1.58) */
// y = RMSNorm(x) -> int8 absmax per token -> x ternary(absmean) W. Training: STE; forward is exact int8 x ternary on VNNI.
extern int g_noquant;                            // tests: bypass both quantizers (fp32 path) for finite-difference checks
typedef struct {
    int K, M;                                    // in, out   (K % 16 == 0, M % 16 == 0)
    Param W;                                     // latent fp32 weights [K][M]
    int8_t *Wt; int32_t *colsum; float ws;       // packed ternary [K/4][M][4], column sums, weight scale
    float *WqT;                                  // dequantized ternary, transposed [M][K] (backward)
    // per-forward cache (capacity N tokens)
    int cap, N; const float *x; uint8_t *xu; float *sx, *r, *xq; // xq: dequantized activations [N][K] (fp32 path)
    float *tmpT, *dxn;                           // tmpT: [K][M] prepare scratch
} BitLinear;
void bl_init(BitLinear *l, int K, int M, int cap, Rng *rng);
void bl_prepare(BitLinear *l);                   // quantize + pack weights (call after every optimizer step)
void bl_forward(BitLinear *l, const float *x, int N, float *y);
void bl_backward(BitLinear *l, const float *dy, float *dx);  // accumulates W.g; writes dx

/* ------------------------------------------------------------------ Walsh mixer */
enum { MODE_CAUSAL = 0, MODE_BIDIR = 1 };
void fwht_rows(float *x, int n, int d);         // in-place unnormalized FWHT over n rows of d floats (d % 16 == 0)

typedef struct {
    int d, T, mode, cap;
    BitLinear inp, out;                          // d -> 2d (u | g), d -> d
    Param sw;                                    // short conv taps [3][d]
    Param h;                                     // long-conv kernel [T][d] (dyadic lag, channel)
    float *Hs;                                   // bidir: WHT(h)/T  [T][d];  causal: level spectra WHT(h[n/2..n))/(n/2)
    int lv_off[32];
    // caches
    float *ug, *us, *v, *z, *U;                  // U: bidir cache of WHT(us) per sequence
    float *dz, *dug, *dus, *dv;
    float *dh_acc;                               // [threads][T][d] partial kernel grads
    float *scratch;                              // [threads][4][T][d]
} WalshMix;
void wm_init(WalshMix *m, int d, int T, int mode, int cap, Rng *rng);
void wm_prepare(WalshMix *m);                    // per step: quantize projections, spectra of h
void wm_forward(WalshMix *m, const float *x, int N, float *y);
void wm_backward(WalshMix *m, const float *dy, float *dx);

/* ------------------------------------------------------------------ FFNs */
typedef struct {
    int d, hid, cap; BitLinear up, down; float *u, *a, *da;
} MLP;
void mlp_init(MLP *f, int d, int hid, int cap, Rng *rng);
void mlp_prepare(MLP *f);
void mlp_forward(MLP *f, const float *x, int N, float *y);
void mlp_backward(MLP *f, const float *dy, float *dx);

// Light DDLGN: 4 learnable truth-table logits per 2-input gate, fixed random wiring, residual (pass-through) init,
// thermometer-encoded inputs, GroupSum readout with per-channel gain. Soft (relaxed) for training, hard for inference.
typedef struct {
    int d, nth, width, depth, k, cap; float temp;
    float th[16];                                // thermometer thresholds
    int32_t **ia, **ib;                          // wiring per layer [width]
    Param *theta;                                // per layer [width][4]
    Param gain;                                  // [d]
    // caches (gate-major: [feature][N])
    int N; float *xn, *r; float **act;          // act[0] = input bits [d*nth][cap], act[i+1] = layer i out [width][cap]
    float *gs;                                   // [N][d]
    float **dact; float *dxn, *xnT;
    float *s4;                                   // per layer sigmoid(theta) scratch
    float *tpart;                                // per-thread partial theta grads
} DDLGN;
void dl_init(DDLGN *f, int d, int nth, float temp, int width, int depth, float z, float lr_mul, int cap, Rng *rng);
void dl_forward(DDLGN *f, const float *x, int N, float *y, int hard);
void dl_backward(DDLGN *f, const float *dy, float *dx);
float dl_gates_changed(const DDLGN *f);         // fraction of gates whose hard truth table != pass-through A

// CLOPEN FFN: layers of ternary threshold gates over {-1,+1} states. The training forward pass IS the inference
// function (no relaxation). Gate: out = sign(sum_j q3(w_j) x[idx_j] - theta), q3 = round+clamp to {-1,0,+1}, fixed
// random wiring with fan-in G, thermometer sign inputs, GroupSum readout. Backward: straight-through estimator,
// identity (ste=0) or clipped to |pre-activation| <= 1 (ste=1).
typedef struct {
    int d, nth, width, depth, fanin, k, cap, ste; float temp;
    float th[16];
    int32_t **idx;                               // per layer [width][fanin]
    Param *w, *theta;                            // per layer latent weights [width][fanin], thresholds [width]
    Param gain;                                  // [d]
    float *wq;                                   // quantized weights [depth][width][fanin]
    int N; float *xn, *r, *xnT, *in0, *gs, *dxn; // in0: input sign bits, gate-major [d*nth][cap]
    float *acts, *pre, *gbuf, *wpart;            // per-thread tile buffers and partial grads
} Clopen;
void cl_init(Clopen *f, int d, int nth, float temp, int width, int depth, int fanin, int ste, float lr_mul, int cap, Rng *rng);
void cl_forward(Clopen *f, const float *x, int N, float *y);
void cl_backward(Clopen *f, const float *dy, float *dx);
float cl_gates_changed(const Clopen *f);         // fraction of gates no longer computing pass-through of input 0

/* ------------------------------------------------------------------ model */
enum { FFN_MLP = 0, FFN_DDLGN = 1, FFN_NONE = 2, FFN_CLOPEN = 3 };
typedef struct {
    int vocab;                                   // model vocab (bidir adds +1 [MASK] token internally)
    int d, n_layers, T, mode, ffn, hidden;
    int dl_width, dl_depth, dl_nth; float dl_temp, dl_z, dl_lr_mul;
    uint64_t seed;
    int cl_width, cl_depth, cl_fanin, cl_nth, cl_ste; float cl_temp, cl_lr_mul;
} Config;

typedef struct {
    Config c; int V_in, cap;                     // V_in = embedding rows
    Param emb, head;                             // [V_in][d], [d][vocab]
    WalshMix *mix; MLP *mlp; DDLGN *dl; Clopen *cl;
    Param **params; int n_params;
    // activations
    float **xs;                                  // residual stream per sub-layer: xs[0..2L] each [cap][d]
    float *tmp, *dx, *dtmp, *xn, *r, *logits;
    int N; const int *tok;
} Model;
void model_init(Model *m, const Config *c, int cap);
void model_prepare(Model *m);                   // after each optimizer step
// forward + loss (targets < 0 are ignored). If grad: also backward (accumulates param grads). Returns mean loss.
double model_step(Model *m, const int *tok, const int *tgt, int N, int grad, int hard);
void adam_step(Model *m, float lr, int t, float b1, float b2, float wd);
void zero_grads(Model *m);
int model_save(const Model *m, const char *path, const unsigned char *vocab_chars);
int model_load(Model *m, const char *path, unsigned char *vocab_chars, int cap);
size_t model_n_params(const Model *m);
