// walshnet: a deep stack of gated short-conv BitNet blocks, pretrained and served in pure C (C11 + AVX-512 + OpenMP).
//
//   block(x) = BitLinear_out( conv_K(u) * g ),   [u | g] = BitLinear_in(x),   x <- x + block(x)
//
// BitLinear = RMSNorm -> int8 absmax activations -> ternary {-1,0,+1} weights (BitNet b1.58), exact on AVX-512 VNNI.
// conv_K is a depthwise K-tap conv: causal (taps t, t-1, ...) for language modelling, centred for masked LM.
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
enum { P_BLF, P_BLB_DW, P_BLB_DX, P_BLB_RMS, P_PREP, P_CONV, P_HEAD, P_ADAM, P_QT, P_PACKDY, P_N };
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

typedef struct { float *w, *g, *m, *v; size_t n; float lr_mul; } Param;   // parameter + Adam state
void param_init(Param *p, size_t n, float lr_mul);

static inline __attribute__((always_inline)) void tr16(__m512 r[16]) {                  // in-register 16x16 transpose (4 permute stages)
    static const int lo8[16] = {0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23}, hi8[16] = {8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31};
    static const int lo4[16] = {0,1,2,3,16,17,18,19,8,9,10,11,24,25,26,27}, hi4[16] = {4,5,6,7,20,21,22,23,12,13,14,15,28,29,30,31};
    static const int lo2[16] = {0,1,16,17,4,5,20,21,8,9,24,25,12,13,28,29}, hi2[16] = {2,3,18,19,6,7,22,23,10,11,26,27,14,15,30,31};
    static const int lo1[16] = {0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30}, hi1[16] = {1,17,3,19,5,21,7,23,9,25,11,27,13,29,15,31};
    const int *los[4] = {lo8, lo4, lo2, lo1}, *his[4] = {hi8, hi4, hi2, hi1};
    for (int st = 0, b = 8; st < 4; st++, b >>= 1) {
        const __m512i il = _mm512_loadu_si512(los[st]), ih = _mm512_loadu_si512(his[st]);
        __m512 t[16];
        for (int i = 0; i < 16; i++) t[i] = r[i];
        for (int i = 0; i < 16; i++) if (!(i & b)) {
            r[i] = _mm512_permutex2var_ps(t[i], il, t[i + b]); r[i + b] = _mm512_permutex2var_ps(t[i], ih, t[i + b]);
        }
    }
}

/* ------------------------------------------------------------------ gemm (fp32, row-major) */
void sgemm(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc, int accumulate);   // C (+)= A B
void sgemm_tn(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc, int accumulate); // C (+)= A^T B, A [K][M]
void transpose(const float *src, int rows, int cols, float *dst);   // dst[cols][rows]
// bf16 GEMM (AVX512-BF16 vdpbf16ps, fp32 accumulate): C[M][N] (+)= alpha * A[M][K] Bp, K even.
// A: bf16 row-major (lda); Bp: B packed as pairs along K, [K/2][N][2] (see pack_pairs_bf16).
typedef uint16_t bf16;
void gemm_bf16(int M, int N, int K, const bf16 *A, int lda, const bf16 *Bp, float *C, int ldc, float alpha, int accumulate);
void pack_pairs_bf16(const float *B, int K, int N, int ldb, const float *rowscale, bf16 *dst);   // dst[k/2][n][k&1] = B[k][n]*rowscale[k]
void to_bf16(const float *src, bf16 *dst, size_t n);
// Intel AMX (tile) paths, used automatically when available and shapes fit (see amx.c). WN_NO_AMX=1 disables.
extern int g_amx;
int amx_init(void);                              // request tile permission; returns 1 if AMX is usable
void amx_bitlinear_fwd(int N, int M, int K, const uint8_t *xu, const int8_t *Wt, const int32_t *colsum, const float *sx, float ws, float *y);
void amx_gemm_bf16(int M, int N, int K, const bf16 *A, int lda, const bf16 *Bp, float *C, int ldc, float alpha, int accumulate);

/* ------------------------------------------------------------------ BitLinear (BitNet b1.58) */
extern int g_noquant;                            // tests: bypass both quantizers (fp32 path) for finite differences
extern int g_bf16_bwd;                           // backward GEMMs in bf16 (default 1; tests use fp32)
typedef struct {
    int K, M;                                    // in, out   (K % 16 == 0, M % 16 == 0)
    Param W;                                     // latent fp32 weights [K][M]
    int8_t *Wt; int32_t *colsum; float ws;       // packed ternary [K/4][M][4], column sums, weight scale
    float *WqT;                                  // dequantized ternary, transposed [M][K] (backward)
    int cap, N; const float *x; uint8_t *xu; float *sx, *r, *xq;   // per-forward cache (capacity cap tokens)
    float *tmpT, *dxn;                           // tmpT: [K][M] prepare scratch
    bf16 *Wb, *qT, *dyA, *dyB;                   // bf16 backward: packed ternary W^T, q^T [K][N], dy row-major, sx*dy pairs
    float *tT;                                   // [M][K] prepare scratch
    int store_xq;                                // keep fp32 dequantized activations (tests / fp32 backward)
} BitLinear;
void bl_init(BitLinear *l, int K, int M, int cap, Rng *rng);
void bl_prepare(BitLinear *l);                   // quantize + pack weights (after every optimizer step)
void bl_forward(BitLinear *l, const float *x, int N, float *y);
void bl_backward(BitLinear *l, const float *dy, float *dx);  // accumulates W.g; writes dx

/* ------------------------------------------------------------------ gated short-conv block */
enum { MODE_CAUSAL = 0, MODE_BIDIR = 1 };
typedef struct {
    int d, T, K, mode, cap;
    BitLinear inp, out;                          // d -> 2d (u | g), d -> d
    Param cw;                                    // depthwise conv taps [K][d]; tap j reads position t + off(j)
    float *ug, *us, *z, *dz, *dug;               // caches [cap][2d] / [cap][d]
} Block;
static inline int tap_off(int mode, int K, int j) { return mode == MODE_CAUSAL ? -j : j - K / 2; }
void blk_init(Block *b, int d, int T, int K, int mode, int cap, Rng *rng);
void blk_prepare(Block *b);
void blk_forward(Block *b, const float *x, int N, float *y);
void blk_backward(Block *b, const float *dy, float *dx);

/* ------------------------------------------------------------------ model */
typedef struct {
    int vocab, d, n_layers, T, K, mode;          // bidir adds a [MASK] token to the embedding table
    uint64_t seed;
} Config;

typedef struct {
    Config c; int V_in, cap;
    Param emb, head;                             // [V_in][d], [d][vocab]
    Block *blk;
    Param **params; int n_params;
    float **xs;                                  // residual stream: xs[0..L] each [cap][d]
    float *tmp, *dx, *dtmp, *xn, *r, *logits, *headT;
    int N;
    float loss_norm;                             // if > 0: normalise grads by this count instead of the local one
} Model;
void model_init(Model *m, const Config *c, int cap);
void model_prepare(Model *m);                    // after each optimizer step
// forward + loss over targets >= 0; if grad, also backward (accumulates param grads). Returns mean loss.
double model_step(Model *m, const int *tok, const int *tgt, int N, int grad);
void model_logits(Model *m, const int *tok, int N);   // forward only -> m->logits
// Data-parallel training: R replicas share the master's weights and packed ternary forms; each thread runs its slice
// of the batch through the whole network (L2-resident, no inner barriers); gradients are reduced into the master.
void model_init_replica(Model *r, Model *master, int cap);
void replicas_sync(Model *reps, int R, const Model *master);   // after each master prepare (copies weight scales)
double train_step_dp(Model *master, Model *reps, int R, const int *tok, const int *tgt, int N, int T);
void adam_step(Model *m, float lr, int t, float b1, float b2, float wd);
void zero_grads(Model *m);
size_t model_n_params(const Model *m);
int model_save(const Model *m, const char *path, const unsigned char *vocab_chars);
int model_load(Model *m, const char *path, unsigned char *vocab_chars, int cap);

/* ------------------------------------------------------------------ streaming decode (causal models) */
// Constant-state incremental decoding: per layer only the last K-1 conv inputs are kept, so each new token costs
// O(layers * d^2) regardless of how long the stream has been running. Runs S independent streams in lock-step.
typedef struct {
    Model *m; int S, pos;
    float *x, *y, *ug, *z, *hist;                // hist: [layers][K-1][S][d] past u values (ring, newest first)
} Stream;
void stream_init(Stream *s, Model *m, int S);    // requires m->cap >= S and a causal model
void stream_reset(Stream *s);
void stream_step(Stream *s, const int *tok);     // tok[S] -> logits in s->m->logits [S][vocab]
