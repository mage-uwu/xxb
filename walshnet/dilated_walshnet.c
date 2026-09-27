/*
 * dilated_walshnet.c: standalone reference implementation of walshnet, in one file.
 *
 * Model: a deep stack of gated, dilated short-conv BitNet blocks
 *
 *     block(x) = BitLinear_out( conv_K,dil(u) * g ),   [u | g] = BitLinear_in(x),   x <- x + block(x)
 *
 *   - BitLinear (BitNet b1.58): per-token RMSNorm -> int8 absmax activations x ternary {-1,0,+1} absmean
 *     weights; trained with the straight-through estimator on latent fp32 weights.
 *   - conv: depthwise K-tap conv, dilation 2^(l mod c) at layer l.
 *       causal        taps t, t-dil, t-2dil, ...    (language modelling, streaming generation)
 *       bidirectional taps centred on t              (masked LM, BERT-style 80/10/10 masking)
 *   - no attention, no FFN: final RMSNorm + fp32 head.
 *
 * CPU engine (x86-64, AVX-512; AMX used automatically when present, WN_NO_AMX=1 disables):
 *   - forward  int8 x ternary: AMX int8 tiles (tdpbusd) or AVX-512 VNNI (vpdpbusd), exact
 *   - backward both GEMMs in bf16 on AMX tiles (tdpbf16ps) / AVX-512 BF16; ternary weights and int8
 *              activation codes are exact in bf16, only gradients are rounded (fp32 path for tests)
 *   - data-parallel replicas (one core = one batch slice through the whole network, shared weights)
 *   - constant-state streaming decode: per layer a ring of the last (K-1)*dil conv inputs
 *
 * Build:   gcc -O3 -march=native -mamx-tile -mamx-int8 -mamx-bf16 -fopenmp dilated_walshnet.c -lm -o dilated_walshnet
 * Usage:   ./dilated_walshnet test
 *          ./dilated_walshnet train --data input.txt [--mode causal|bidir] [--d 64 --layers 7 --T 64 --K 3 --dilate 0]
 *                                   [--batch 16 --steps 3000 --lr 3e-2 --save-best --out model.bin] (see: train --help)
 *          ./dilated_walshnet bench    --model model.bin [--tokens 8192]
 *          ./dilated_walshnet generate --model model.bin --prompt "ROMEO:" [--n 400 --temp 0.8 --seed 1]   (causal)
 *          ./dilated_walshnet fill     --model mlm.bin --text "Wh_t is th_ m_tter"                     (bidirectional)
 */
#define _GNU_SOURCE
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

/* ==================== core declarations ===================================================== */

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
    int d, T, K, mode, cap, dil;                 // dil: tap spacing (dilation) of this layer's conv
    BitLinear inp, out;                          // d -> 2d (u | g), d -> d
    Param cw;                                    // depthwise conv taps [K][d]; tap j reads position t + off(j)
    float *ug, *us, *z, *dz, *dug;               // caches [cap][2d] / [cap][d]
} Block;
static inline int model_dil(int dil_cycle, int l) { return dil_cycle ? 1 << (l % dil_cycle) : 1; }
static inline int tap_off(int mode, int K, int j) { return mode == MODE_CAUSAL ? -j : j - K / 2; }
static inline int blk_off(const Block *b, int j) { return tap_off(b->mode, b->K, j) * b->dil; }
void blk_init(Block *b, int d, int T, int K, int mode, int dil, int cap, Rng *rng);
void blk_prepare(Block *b);
void blk_forward(Block *b, const float *x, int N, float *y);
void blk_backward(Block *b, const float *dy, float *dx);

/* ------------------------------------------------------------------ model */
typedef struct {
    int vocab, d, n_layers, T, K, mode;          // bidir adds a [MASK] token to the embedding table
    int dil_cycle;                               // layer l dilation = 2^(l mod dil_cycle); 0 = no dilation
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
    float *x, *y, *ug, *z, *hist;                // hist: per layer a ring of (K-1)*dil past u values, [slot][S][d]
    size_t *hoff; int *hlen;                     // per-layer ring offset (floats) and length (slots)
} Stream;
void stream_init(Stream *s, Model *m, int S);    // requires m->cap >= S and a causal model
void stream_reset(Stream *s);
void stream_step(Stream *s, const int *tok);     // tok[S] -> logits in s->m->logits [S][vocab]

/* ==================== utils + fp32 / bf16 GEMM ============================================== */

#ifdef WN_PROF
double g_prof[P_N];
#endif

void *xmalloc(size_t bytes) {
    size_t n = (bytes + 63) / 64 * 64; if (!n) n = 64;
    void *p = aligned_alloc(64, n);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", bytes); exit(1); }
    memset(p, 0, n); return p;
}

double now_sec(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

float rng_normal(Rng *r) {
    float u1 = rng_unif(r) + 1e-7f, u2 = rng_unif(r);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

void param_init(Param *p, size_t n, float lr_mul) {
    p->n = n; p->lr_mul = lr_mul;
    p->w = xmalloc(4 * n); p->g = xmalloc(4 * n); p->m = xmalloc(4 * n); p->v = xmalloc(4 * n);
}

void transpose(const float *src, int rows, int cols, float *dst) {
    const int R16 = rows / 16 * 16, C16 = cols / 16 * 16;
    #pragma omp parallel for schedule(static) if ((size_t)rows * cols > 65536)
    for (int i0 = 0; i0 < R16; i0 += 16) {
        for (int j0 = 0; j0 < C16; j0 += 16) {
            __m512 r[16];
            for (int i = 0; i < 16; i++) r[i] = _mm512_loadu_ps(src + (size_t)(i0 + i) * cols + j0);
            tr16(r);
            for (int j = 0; j < 16; j++) _mm512_storeu_ps(dst + (size_t)(j0 + j) * rows + i0, r[j]);
        }
        for (int i = i0; i < i0 + 16; i++) for (int j = C16; j < cols; j++) dst[(size_t)j * rows + i] = src[(size_t)i * cols + j];
    }
    for (int i = R16; i < rows; i++) for (int j = 0; j < cols; j++) dst[(size_t)j * rows + i] = src[(size_t)i * cols + j];
}

/* 8 x 32 register-blocked micro-kernel; mr <= 8 rows, nr <= 32 cols (masked) */
static inline __attribute__((always_inline)) void kern(int mr, int nr, int K, const float *A, size_t ars, size_t aks, const float *B, int ldb,
                                                       float *C, int ldc, int accumulate) {
    __m512 c0[8], c1[8];
    const __mmask16 m0 = nr >= 16 ? 0xFFFF : (__mmask16)((1u << nr) - 1);
    const __mmask16 m1 = nr >= 32 ? 0xFFFF : nr > 16 ? (__mmask16)((1u << (nr - 16)) - 1) : 0;
    for (int r = 0; r < 8; r++) { c0[r] = _mm512_setzero_ps(); c1[r] = _mm512_setzero_ps(); }
    for (int k = 0; k < K; k++) {
        const float *b = B + (size_t)k * ldb;
        __m512 b0 = _mm512_maskz_loadu_ps(m0, b), b1 = _mm512_maskz_loadu_ps(m1, b + 16);
        for (int r = 0; r < mr; r++) {
            __m512 a = _mm512_set1_ps(A[r * ars + k * aks]);
            c0[r] = _mm512_fmadd_ps(a, b0, c0[r]); c1[r] = _mm512_fmadd_ps(a, b1, c1[r]);
        }
    }
    for (int r = 0; r < mr; r++) {
        float *c = C + (size_t)r * ldc;
        if (accumulate) { c0[r] = _mm512_add_ps(c0[r], _mm512_maskz_loadu_ps(m0, c)); c1[r] = _mm512_add_ps(c1[r], _mm512_maskz_loadu_ps(m1, c + 16)); }
        _mm512_mask_storeu_ps(c, m0, c0[r]); _mm512_mask_storeu_ps(c + 16, m1, c1[r]);
    }
}

// Tasks = (group of IG row-blocks) x (32-col block). Each task packs its B panel (KC x 32) once per K chunk
// into a contiguous strip (avoids 4K-aliasing on large ldb) and reuses it across its row blocks.
static void gemm_core(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc, int accumulate) {
    enum { KC = 256, IG = 8 };
    const int mb = (M + 7) / 8, nb = (N + 31) / 32, ng = (mb + IG - 1) / IG;
    #pragma omp parallel if ((double)M * N * K > 2e5)
    {
        float Bp[KC * 32] __attribute__((aligned(64)));
        #pragma omp for collapse(2) schedule(static)
        for (int gi = 0; gi < ng; gi++)
            for (int jb = 0; jb < nb; jb++) {
                const int j = jb * 32, nr = N - j < 32 ? N - j : 32;
                for (int k0 = 0; k0 < K; k0 += KC) {
                    const int kc = K - k0 < KC ? K - k0 : KC, acc = accumulate || k0 > 0;
                    const __mmask16 m0 = nr >= 16 ? 0xFFFF : (__mmask16)((1u << nr) - 1);
                    const __mmask16 m1 = nr >= 32 ? 0xFFFF : nr > 16 ? (__mmask16)((1u << (nr - 16)) - 1) : 0;
                    for (int k = 0; k < kc; k++) {
                        const float *b = B + (size_t)(k0 + k) * ldb + j;
                        _mm512_store_ps(Bp + k * 32, _mm512_maskz_loadu_ps(m0, b));
                        _mm512_store_ps(Bp + k * 32 + 16, _mm512_maskz_loadu_ps(m1, b + 16));
                    }
                    for (int ib = gi * IG; ib < mb && ib < (gi + 1) * IG; ib++) {
                        const int i = ib * 8, mr = M - i < 8 ? M - i : 8;
                        const float *a = A + (size_t)i * lda + k0; float *c = C + (size_t)i * ldc + j;
                        if (mr == 8) kern(8, nr, kc, a, lda, 1, Bp, 32, c, ldc, acc);
                        else         kern(mr, nr, kc, a, lda, 1, Bp, 32, c, ldc, acc);
                    }
                }
            }
    }
}

void sgemm(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc, int accumulate) {
    gemm_core(M, N, K, A, lda, B, ldb, C, ldc, accumulate);
}

// A^T B: one fast AVX-512 transpose into a reusable scratch buffer, then the NN kernel (not reentrant)
void sgemm_tn(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc, int accumulate) {
    static __thread float *buf = NULL; static __thread size_t cap = 0;   // per-thread (replicas run concurrently)
    const size_t need = (size_t)M * K;
    if (need > cap) { free(buf); cap = need; buf = xmalloc(4 * cap); }
    if (lda != M) { fprintf(stderr, "sgemm_tn: lda must equal M\n"); exit(1); }
    transpose(A, K, M, buf);
    gemm_core(M, N, K, buf, K, B, ldb, C, ldc, accumulate);
}

/* ---------------- bf16 GEMM ---------------- */
void to_bf16(const float *src, bf16 *dst, size_t n) {
    #pragma omp parallel for schedule(static) if (n > 65536)
    for (size_t i = 0; i < n; i += 16) {
        if (i + 16 <= n) _mm256_storeu_si256((__m256i *)(dst + i), (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src + i)));
        else for (size_t j = i; j < n; j++) { __m512 v = _mm512_set1_ps(src[j]); __m256i b = (__m256i)_mm512_cvtneps_pbh(v); dst[j] = (bf16)_mm256_extract_epi16(b, 0); }
    }
}

void pack_pairs_bf16(const float *B, int K, int N, int ldb, const float *rowscale, bf16 *dst) {
    static const short il[32] = {0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23,8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31};
    const __m512i idx = _mm512_loadu_si512(il);
    #pragma omp parallel for schedule(static) if ((size_t)K * N > 65536)
    for (int kp = 0; kp < K / 2; kp++) {
        const float *r0 = B + (size_t)(2 * kp) * ldb, *r1 = r0 + ldb;
        const __m512 s0 = _mm512_set1_ps(rowscale ? rowscale[2 * kp] : 1.0f), s1 = _mm512_set1_ps(rowscale ? rowscale[2 * kp + 1] : 1.0f);
        bf16 *o = dst + (size_t)kp * N * 2;
        int n = 0;
        for (; n + 16 <= N; n += 16) {
            const __m256i a = (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(_mm512_loadu_ps(r0 + n), s0));
            const __m256i b = (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(_mm512_loadu_ps(r1 + n), s1));
            _mm512_storeu_si512(o + 2 * n, _mm512_permutexvar_epi16(idx, _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1)));
        }
        for (; n < N; n++) {
            float t[2] = {r0[n] * (rowscale ? rowscale[2 * kp] : 1.0f), r1[n] * (rowscale ? rowscale[2 * kp + 1] : 1.0f)};
            bf16 bb[2]; to_bf16(t, bb, 2); o[2 * n] = bb[0]; o[2 * n + 1] = bb[1];
        }
    }
}

static inline __attribute__((always_inline)) void kern_bf16(int mr, int nr, int K, const bf16 *A, int lda, const bf16 *Bp,
                                                            float *C, int ldc, float alpha, int accumulate) {
    __m512 c0[8], c1[8];
    const __mmask16 m0 = nr >= 16 ? 0xFFFF : (__mmask16)((1u << nr) - 1);
    const __mmask16 m1 = nr >= 32 ? 0xFFFF : nr > 16 ? (__mmask16)((1u << (nr - 16)) - 1) : 0;
    for (int r = 0; r < 8; r++) { c0[r] = _mm512_setzero_ps(); c1[r] = _mm512_setzero_ps(); }
    for (int kp = 0; kp < K / 2; kp++) {
        const __m512bh b0 = (__m512bh)_mm512_load_si512(Bp + kp * 64), b1 = (__m512bh)_mm512_load_si512(Bp + kp * 64 + 32);
        for (int r = 0; r < mr; r++) {
            const __m512bh a = (__m512bh)_mm512_set1_epi32(*(const int32_t *)(A + (size_t)r * lda + 2 * kp));
            c0[r] = _mm512_dpbf16_ps(c0[r], a, b0); c1[r] = _mm512_dpbf16_ps(c1[r], a, b1);
        }
    }
    const __m512 al = _mm512_set1_ps(alpha);
    for (int r = 0; r < mr; r++) {
        float *c = C + (size_t)r * ldc;
        __m512 v0 = _mm512_mul_ps(c0[r], al), v1 = _mm512_mul_ps(c1[r], al);
        if (accumulate) { v0 = _mm512_add_ps(v0, _mm512_maskz_loadu_ps(m0, c)); v1 = _mm512_add_ps(v1, _mm512_maskz_loadu_ps(m1, c + 16)); }
        _mm512_mask_storeu_ps(c, m0, v0); _mm512_mask_storeu_ps(c + 16, m1, v1);
    }
}

// Tasks = (group of IG row-blocks) x (32-col block); each packs its B strip [KC/2][32][2] contiguously once per K chunk.
void gemm_bf16(int M, int N, int K, const bf16 *A, int lda, const bf16 *Bp, float *C, int ldc, float alpha, int accumulate) {
    if (g_amx && M % 32 == 0 && N % 32 == 0 && K % 32 == 0) { amx_gemm_bf16(M, N, K, A, lda, Bp, C, ldc, alpha, accumulate); return; }
    enum { KC = 512, IG = 8 };
    const int mb = (M + 7) / 8, nb = (N + 31) / 32, ng = (mb + IG - 1) / IG;
    #pragma omp parallel if ((double)M * N * K > 2e5)
    {
        bf16 Bs[(KC / 2) * 64] __attribute__((aligned(64)));
        #pragma omp for collapse(2) schedule(static)
        for (int gi = 0; gi < ng; gi++)
            for (int jb = 0; jb < nb; jb++) {
                const int j = jb * 32, nr = N - j < 32 ? N - j : 32;
                const __mmask32 mk = nr >= 16 ? 0xFFFFFFFFu : (__mmask32)((1ull << (2 * nr)) - 1);
                const __mmask32 mk1 = nr >= 32 ? 0xFFFFFFFFu : nr > 16 ? (__mmask32)((1ull << (2 * (nr - 16))) - 1) : 0;
                for (int k0 = 0; k0 < K; k0 += KC) {
                    const int kc = K - k0 < KC ? K - k0 : KC, acc = accumulate || k0 > 0;
                    for (int kp = 0; kp < kc / 2; kp++) {
                        const bf16 *src = Bp + ((size_t)(k0 / 2 + kp) * N + j) * 2;
                        _mm512_store_si512(Bs + kp * 64, _mm512_maskz_loadu_epi16(mk, src));
                        _mm512_store_si512(Bs + kp * 64 + 32, _mm512_maskz_loadu_epi16(mk1, src + 32));
                    }
                    for (int ib = gi * IG; ib < mb && ib < (gi + 1) * IG; ib++) {
                        const int i = ib * 8, mr = M - i < 8 ? M - i : 8;
                        const bf16 *a = A + (size_t)i * lda + k0; float *c = C + (size_t)i * ldc + j;
                        if (mr == 8) kern_bf16(8, nr, kc, a, lda, Bs, c, ldc, alpha, acc);
                        else         kern_bf16(mr, nr, kc, a, lda, Bs, c, ldc, alpha, acc);
                    }
                }
            }
    }
}

/* ==================== AMX tiles ============================================================= */


/* Intel AMX paths (Sapphire Rapids+):
     int8:  TDPBUSD   C[16x16 i32] += A[16 x 64 u8] . B[16 x (16 cols x 4 k) s8]    (BitLinear forward)
     bf16:  TDPBF16PS C[16x16 f32] += A[16 x 32 bf16] . B[16 x (16 cols x 2 k) bf16] (BitLinear backward)
   B operands are exactly the layouts already used by the VNNI / bf16 code: [K/4][N][4] bytes and [K/2][N][2] bf16.
   Each task computes a 32 x 32 output block from four accumulator tiles (tmm0-3), A tiles tmm4-5, B tiles tmm6-7. */

int g_amx = 0;

typedef struct { uint8_t palette, start_row, rsv[14]; uint16_t colsb[16]; uint8_t rows[16]; } __attribute__((packed)) TileCfg;

static void tile_config(void) {
    static __thread int configured = 0;
    if (configured) return;
    TileCfg cfg; memset(&cfg, 0, sizeof cfg); cfg.palette = 1;
    for (int i = 0; i < 8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
    _tile_loadconfig(&cfg); configured = 1;
}

int amx_init(void) {
    if (getenv("WN_NO_AMX")) return g_amx = 0;
    #ifndef ARCH_REQ_XCOMP_PERM
    #define ARCH_REQ_XCOMP_PERM 0x1023
    #endif
    if (!__builtin_cpu_supports("amx-int8") || !__builtin_cpu_supports("amx-bf16")) return g_amx = 0;
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, 18 /* XFEATURE_XTILEDATA */)) return g_amx = 0;
    return g_amx = 1;
}

/* y[N][M] = ((xu[N][K] . Wt) - 128 colsum) * sx[n] * ws     (N % 32, M % 32, K % 64 == 0) */
void amx_bitlinear_fwd(int N, int M, int K, const uint8_t *xu, const int8_t *Wt, const int32_t *colsum, const float *sx, float ws, float *y) {
    const int nb = N / 32, mb = M / 32;
    #pragma omp parallel if (N >= 128)
    {
        tile_config();
        int32_t Cb[4][16 * 16] __attribute__((aligned(64)));
        #pragma omp for collapse(2) schedule(static)
        for (int ib = 0; ib < nb; ib++)
            for (int jb = 0; jb < mb; jb++) {
                const int n0 = ib * 32, m0 = jb * 32;
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                for (int k0 = 0; k0 < K; k0 += 64) {
                    _tile_loadd(4, xu + (size_t)n0 * K + k0, K);
                    _tile_loadd(5, xu + (size_t)(n0 + 16) * K + k0, K);
                    _tile_loadd(6, Wt + ((size_t)(k0 / 4) * M + m0) * 4, M * 4);
                    _tile_loadd(7, Wt + ((size_t)(k0 / 4) * M + m0 + 16) * 4, M * 4);
                    _tile_dpbusd(0, 4, 6); _tile_dpbusd(1, 4, 7); _tile_dpbusd(2, 5, 6); _tile_dpbusd(3, 5, 7);
                }
                _tile_stored(0, Cb[0], 64); _tile_stored(1, Cb[1], 64); _tile_stored(2, Cb[2], 64); _tile_stored(3, Cb[3], 64);
                for (int t = 0; t < 4; t++) {
                    const int rn = n0 + (t >> 1) * 16, cm = m0 + (t & 1) * 16;
                    const __m512i cs = _mm512_slli_epi32(_mm512_loadu_si512(colsum + cm), 7);
                    for (int r = 0; r < 16; r++) {
                        const __m512 v = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_load_si512(Cb[t] + r * 16), cs));
                        _mm512_storeu_ps(y + (size_t)(rn + r) * M + cm, _mm512_mul_ps(v, _mm512_set1_ps(sx[rn + r] * ws)));
                    }
                }
            }
    }
}

/* C[M][N] (+)= alpha * A[M][K] . Bp,  A bf16 row-major (lda), Bp pairs [K/2][N][2]   (M % 32, N % 32, K % 32 == 0) */
void amx_gemm_bf16(int M, int N, int K, const bf16 *A, int lda, const bf16 *Bp, float *C, int ldc, float alpha, int accumulate) {
    const int ib_n = M / 32, jb_n = N / 32;
    #pragma omp parallel if ((double)M * N * K > 1e6)
    {
        tile_config();
        float Cb[4][16 * 16] __attribute__((aligned(64)));
        #pragma omp for collapse(2) schedule(static)
        for (int ib = 0; ib < ib_n; ib++)
            for (int jb = 0; jb < jb_n; jb++) {
                const int i0 = ib * 32, j0 = jb * 32;
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                for (int k0 = 0; k0 < K; k0 += 32) {
                    _tile_loadd(4, A + (size_t)i0 * lda + k0, lda * 2);
                    _tile_loadd(5, A + (size_t)(i0 + 16) * lda + k0, lda * 2);
                    _tile_loadd(6, Bp + ((size_t)(k0 / 2) * N + j0) * 2, N * 4);
                    _tile_loadd(7, Bp + ((size_t)(k0 / 2) * N + j0 + 16) * 2, N * 4);
                    _tile_dpbf16ps(0, 4, 6); _tile_dpbf16ps(1, 4, 7); _tile_dpbf16ps(2, 5, 6); _tile_dpbf16ps(3, 5, 7);
                }
                _tile_stored(0, Cb[0], 64); _tile_stored(1, Cb[1], 64); _tile_stored(2, Cb[2], 64); _tile_stored(3, Cb[3], 64);
                const __m512 al = _mm512_set1_ps(alpha);
                for (int t = 0; t < 4; t++) {
                    const int ri = i0 + (t >> 1) * 16, cj = j0 + (t & 1) * 16;
                    for (int r = 0; r < 16; r++) {
                        float *c = C + (size_t)(ri + r) * ldc + cj;
                        __m512 v = _mm512_mul_ps(_mm512_load_ps(Cb[t] + r * 16), al);
                        if (accumulate) v = _mm512_add_ps(v, _mm512_loadu_ps(c));
                        _mm512_storeu_ps(c, v);
                    }
                }
            }
    }
}

/* ==================== BitLinear ============================================================= */

int g_noquant = 0;
int g_bf16_bwd = 1;

void bl_init(BitLinear *l, int K, int M, int cap, Rng *rng) {
    if (K % 16 || M % 16) { fprintf(stderr, "BitLinear: need K%%16==0, M%%16==0 (K=%d M=%d)\n", K, M); exit(1); }
    l->K = K; l->M = M; l->cap = cap;
    param_init(&l->W, (size_t)K * M, 1.0f);
    for (size_t i = 0; i < (size_t)K * M; i++) l->W.w[i] = rng_normal(rng) / sqrtf((float)K);
    l->Wt = xmalloc((size_t)K * M); l->colsum = xmalloc(4 * (size_t)M); l->WqT = xmalloc(4 * (size_t)K * M);
    l->xu = xmalloc((size_t)cap * K); l->sx = xmalloc(4 * (size_t)cap); l->r = xmalloc(4 * (size_t)cap);
    l->xq = xmalloc(4 * (size_t)cap * K); l->tmpT = xmalloc(4 * (size_t)K * (M > cap ? M : cap));
    l->tT = xmalloc(4 * (size_t)K * M); l->Wb = xmalloc(2 * (size_t)K * M); l->qT = xmalloc(2 * (size_t)K * cap); l->dyA = xmalloc(2 * (size_t)cap * M); l->dyB = xmalloc(2 * (size_t)cap * M); l->dxn = xmalloc(4 * (size_t)cap * K);
}

static void bl_prepare_impl(BitLinear *l) {
    const int K = l->K, M = l->M; const float *W = l->W.w;
    if (g_noquant) { l->ws = 1.0f; transpose(W, K, M, l->WqT); return; }
    double s = 0;
    #pragma omp parallel for reduction(+:s) schedule(static) if ((size_t)K * M > 16384)
    for (size_t i = 0; i < (size_t)K * M; i++) s += fabsf(W[i]);
    l->ws = (float)(s / ((double)K * M)); if (l->ws < 1e-5f) l->ws = 1e-5f;
    const __m512 inv = _mm512_set1_ps(1.0f / l->ws), one = _mm512_set1_ps(1.0f), mone = _mm512_set1_ps(-1.0f);
    float *t = l->tmpT;                                                    // ternary values t in {-1,0,1}, [K][M]
    #pragma omp parallel for schedule(static) if ((size_t)K * M > 16384)
    for (size_t i = 0; i < (size_t)K * M; i += 16) {
        const __m512 q = _mm512_roundscale_ps(_mm512_mul_ps(_mm512_loadu_ps(W + i), inv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm512_storeu_ps(t + i, _mm512_min_ps(one, _mm512_max_ps(mone, q)));
    }
    #pragma omp parallel for schedule(static) if ((size_t)K * M > 16384)
    for (int m = 0; m < M; m += 16) {                                      // VNNI pack [K/4][M][4] + column sums
        __m512i cs = _mm512_setzero_si512();
        for (int kq = 0; kq < K / 4; kq++) {
            __m128i r[4];
            for (int e = 0; e < 4; e++) {
                const __m512i v = _mm512_cvtps_epi32(_mm512_loadu_ps(t + (size_t)(4 * kq + e) * M + m));
                cs = _mm512_add_epi32(cs, v); r[e] = _mm512_cvtepi32_epi8(v);
            }
            const __m128i ab0 = _mm_unpacklo_epi8(r[0], r[1]), ab1 = _mm_unpackhi_epi8(r[0], r[1]);
            const __m128i cd0 = _mm_unpacklo_epi8(r[2], r[3]), cd1 = _mm_unpackhi_epi8(r[2], r[3]);
            int8_t *dst = l->Wt + ((size_t)kq * M + m) * 4;
            _mm_storeu_si128((__m128i *)dst, _mm_unpacklo_epi16(ab0, cd0));
            _mm_storeu_si128((__m128i *)(dst + 16), _mm_unpackhi_epi16(ab0, cd0));
            _mm_storeu_si128((__m128i *)(dst + 32), _mm_unpacklo_epi16(ab1, cd1));
            _mm_storeu_si128((__m128i *)(dst + 48), _mm_unpackhi_epi16(ab1, cd1));
        }
        _mm512_storeu_si512(l->colsum + m, cs);
    }
    transpose(t, K, M, l->tT);                                             // [M][K]
    pack_pairs_bf16(l->tT, M, K, K, NULL, l->Wb);                        // bf16 pairs along M (exact)
    const __m512 ws = _mm512_set1_ps(l->ws);                               // fp32 dequantized W^T (fp32 backward path)
    #pragma omp parallel for schedule(static) if ((size_t)K * M > 16384)
    for (size_t i = 0; i < (size_t)K * M; i += 16) _mm512_storeu_ps(l->WqT + i, _mm512_mul_ps(_mm512_loadu_ps(l->tT + i), ws));
}

/* int8 (u8 + 128 offset) x ternary via VNNI: y[n][m] = (acc - 128 colsum[m]) * sx[n] * ws */
static inline __attribute__((always_inline)) void vnni_tile(const BitLinear *l, int n0, int nr, int m, int nc, float *y) {
    const int K = l->K, M = l->M;
    __m512i acc[4][4];
    for (int r = 0; r < 4; r++) for (int j = 0; j < 4; j++) acc[r][j] = _mm512_setzero_si512();
    for (int kq = 0; kq < K / 4; kq++) {
        const int8_t *wp = l->Wt + ((size_t)kq * M + m) * 4;
        __m512i w[4];
        for (int j = 0; j < nc; j++) w[j] = _mm512_loadu_si512(wp + 64 * j);
        for (int r = 0; r < nr; r++) {
            __m512i a = _mm512_set1_epi32(*(const int32_t *)(l->xu + (size_t)(n0 + r) * K + kq * 4));
            for (int j = 0; j < nc; j++) acc[r][j] = _mm512_dpbusd_epi32(acc[r][j], a, w[j]);
        }
    }
    for (int r = 0; r < nr; r++) {
        __m512 sc = _mm512_set1_ps(l->sx[n0 + r] * l->ws);
        for (int j = 0; j < nc; j++) {
            __m512i cs = _mm512_slli_epi32(_mm512_loadu_si512(l->colsum + m + 16 * j), 7);
            _mm512_storeu_ps(y + (size_t)(n0 + r) * M + m + 16 * j, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc[r][j], cs)), sc));
        }
    }
}

static void vnni_gemm(const BitLinear *l, int N, float *y) {
    const int M = l->M, nmb = (M + 63) / 64;
    if (N < 64) {                                                          // small batch: serial, no OpenMP overhead
        for (int n0 = 0; n0 < N; n0 += 4)
            for (int mb = 0; mb < nmb; mb++) {
                const int m = mb * 64, nr = N - n0 < 4 ? N - n0 : 4, nc = (M - m) / 16 < 4 ? (M - m) / 16 : 4;
                if (nr == 4 && nc == 4) vnni_tile(l, n0, 4, m, 4, y); else vnni_tile(l, n0, nr, m, nc, y);
            }
        return;
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int n0 = 0; n0 < N; n0 += 4)
        for (int mb = 0; mb < nmb; mb++) {
            const int m = mb * 64, nr = N - n0 < 4 ? N - n0 : 4, nc = (M - m) / 16 < 4 ? (M - m) / 16 : 4;
            if (nr == 4 && nc == 4) vnni_tile(l, n0, 4, m, 4, y);            // hot path: fully unrolled
            else vnni_tile(l, n0, nr, m, nc, y);
        }
}

static inline void quant_row(BitLinear *l, const float *x, int n, int store_xq) {
    const int K = l->K;
                                          // RMSNorm folded into absmax int8 quant
        const float *xr = x + (size_t)n * K;
        __m512 vmax = _mm512_set1_ps(1e-5f), vss = _mm512_setzero_ps();
        for (int k = 0; k < K; k += 16) {
            const __m512 v = _mm512_loadu_ps(xr + k);
            vmax = _mm512_max_ps(vmax, _mm512_abs_ps(v)); vss = _mm512_fmadd_ps(v, v, vss);
        }
        const float amax = _mm512_reduce_max_ps(vmax), r = 1.0f / sqrtf(_mm512_reduce_add_ps(vss) / K + 1e-6f);
        l->r[n] = r;
        float *xq = l->xq + (size_t)n * K;
        if (g_noquant) { for (int k = 0; k < K; k++) xq[k] = xr[k] * r; return; }
        const float sx = amax * r / 127.0f; l->sx[n] = sx;
        const __m512 s = _mm512_set1_ps(127.0f / amax), vsx = _mm512_set1_ps(sx);
        uint8_t *xu = l->xu + (size_t)n * K;
        for (int k = 0; k < K; k += 16) {
            const __m512 q = _mm512_roundscale_ps(_mm512_mul_ps(_mm512_loadu_ps(xr + k), s), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            if (store_xq) _mm512_storeu_ps(xq + k, _mm512_mul_ps(q, vsx));
            _mm_storeu_si128((__m128i *)(xu + k), _mm_xor_si128(_mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(q)), _mm_set1_epi8((char)0x80)));
        }
    }

static void bl_forward_impl(BitLinear *l, const float *x, int N, float *y) {
    const int K = l->K;
    if (N > l->cap) { fprintf(stderr, "BitLinear: N=%d > cap=%d\n", N, l->cap); exit(1); }
    l->x = x; l->N = N;
    const int store_xq = !g_bf16_bwd || l->store_xq;                     // fp32 copy only needed by the fp32 backward path
    if (N >= 64) {                                                         // (no OpenMP region at all for small N: decode latency)
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) quant_row(l, x, n, store_xq);
    } else for (int n = 0; n < N; n++) quant_row(l, x, n, store_xq);
    if (g_noquant) { sgemm(N, l->M, K, l->xq, K, l->W.w, l->M, y, l->M, 0); return; }
    if (g_amx && N % 32 == 0 && l->M % 32 == 0 && K % 64 == 0) amx_bitlinear_fwd(N, l->M, K, l->xu, l->Wt, l->colsum, l->sx, l->ws, y);
    else vnni_gemm(l, N, y);
}

static void build_qT(BitLinear *l) {                                      // qT[k][n] = q[n][k] (bf16, exact), from xu
    const int K = l->K, N = l->N;
    #pragma omp parallel for collapse(2) schedule(static) if ((size_t)K * N > 65536)
    for (int n0 = 0; n0 < N; n0 += 16)
        for (int k0 = 0; k0 < K; k0 += 16) {
            __m512 r[16];
            for (int i = 0; i < 16; i++)
                r[i] = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(l->xu + (size_t)(n0 + i) * K + k0))), _mm512_set1_epi32(128)));
            tr16(r);
            for (int j = 0; j < 16; j++) _mm256_storeu_si256((__m256i *)(l->qT + (size_t)(k0 + j) * N + n0), (__m256i)_mm512_cvtneps_pbh(r[j]));
        }
}

static void pack_dy(BitLinear *l, const float *dy) {                     // dyA = bf16(dy) row-major; dyB = pairs of sx*dy
    const int N = l->N, M = l->M;
    static const short il[32] = {0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23,8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31};
    const __m512i idx = _mm512_loadu_si512(il);
    #pragma omp parallel for schedule(static) if ((size_t)N * M > 65536)
    for (int np = 0; np < N / 2; np++) {
        const float *r0 = dy + (size_t)(2 * np) * M, *r1 = r0 + M;
        const __m512 s0 = _mm512_set1_ps(l->sx[2 * np]), s1 = _mm512_set1_ps(l->sx[2 * np + 1]);
        for (int m = 0; m < M; m += 16) {
            const __m512 a = _mm512_loadu_ps(r0 + m), b = _mm512_loadu_ps(r1 + m);
            _mm256_storeu_si256((__m256i *)(l->dyA + (size_t)(2 * np) * M + m), (__m256i)_mm512_cvtneps_pbh(a));
            _mm256_storeu_si256((__m256i *)(l->dyA + (size_t)(2 * np + 1) * M + m), (__m256i)_mm512_cvtneps_pbh(b));
            const __m256i pa = (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(a, s0)), pb = (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(b, s1));
            _mm512_storeu_si512(l->dyB + ((size_t)np * M + m) * 2, _mm512_permutexvar_epi16(idx, _mm512_inserti64x4(_mm512_castsi256_si512(pa), pb, 1)));
        }
    }
}

static void rms_backward(const BitLinear *l, float *dx) {                   // dx = r (dxn - xn mean(dxn xn))
    const int K = l->K, N = l->N;
    #pragma omp parallel for schedule(static) if (N >= 64)
    for (int n = 0; n < N; n++) {
        const float r = l->r[n], *xr = l->x + (size_t)n * K, *g = l->dxn + (size_t)n * K;
        float dot = 0; for (int k = 0; k < K; k++) dot += g[k] * xr[k] * r;
        dot /= K;
        float *o = dx + (size_t)n * K;
        for (int k = 0; k < K; k++) o[k] = r * (g[k] - xr[k] * r * dot);
    }
}

void bl_backward(BitLinear *l, const float *dy, float *dx) {
    const int K = l->K, M = l->M, N = l->N;
    if (g_bf16_bwd && !g_noquant && N % 16 == 0) {
        PROF(P_PACKDY, pack_dy(l, dy));
        PROF(P_BLB_DX, gemm_bf16(N, K, M, l->dyA, M, l->Wb, l->dxn, K, l->ws, 0));                    // dx = dy . ternary^T
        PROF(P_QT, build_qT(l));
        PROF(P_BLB_DW, gemm_bf16(K, M, N, l->qT, N, l->dyB, l->W.g, M, 1.0f, 1));                     // dW = q^T . (sx dy)
        PROF(P_BLB_RMS, rms_backward(l, dx));
        return;
    }
    PROF(P_BLB_DW, sgemm_tn(K, M, N, l->xq, K, dy, M, l->W.g, M, 1));      // dW += xq^T dy   (STE: dL/dW := dL/dWq)
    PROF(P_BLB_DX, sgemm(N, K, M, dy, M, l->WqT, K, l->dxn, K, 0));        // dxq = dy Wq^T   (STE: dL/dxn := dL/dxq)
    PROF(P_BLB_RMS, rms_backward(l, dx));
}

void bl_forward(BitLinear *l, const float *x, int N, float *y) { PROF(P_BLF, bl_forward_impl(l, x, N, y)); }
void bl_prepare(BitLinear *l) { PROF(P_PREP, bl_prepare_impl(l)); }

/* ==================== gated dilated short-conv block ======================================== */

/* Gated short-conv block:  [u | g] = BitLinear_in(x);  us[t] = sum_j cw[j] u[t + off(j)];  y = BitLinear_out(us * g)
   Taps that fall outside the current sequence of T tokens read zeros. Layout [token][channel]; loops over channels. */

void blk_init(Block *b, int d, int T, int K, int mode, int dil, int cap, Rng *rng) {
    if (d % 16) { fprintf(stderr, "block: d must be a multiple of 16\n"); exit(1); }
    if (mode == MODE_BIDIR && !(K & 1)) { fprintf(stderr, "block: bidirectional mode needs an odd K\n"); exit(1); }
    b->d = d; b->T = T; b->K = K; b->mode = mode; b->cap = cap; b->dil = dil;
    bl_init(&b->inp, d, 2 * d, cap, rng); bl_init(&b->out, d, d, cap, rng);
    param_init(&b->cw, (size_t)K * d, 1.0f);
    for (int j = 0; j < K; j++)                                           // identity-ish init: centre / current tap
        for (int c = 0; c < d; c++) b->cw.w[j * d + c] = (tap_off(mode, K, j) == 0) + 0.02f * rng_normal(rng);
    const size_t nd = (size_t)cap * d;
    b->ug = xmalloc(8 * nd); b->us = xmalloc(4 * nd); b->z = xmalloc(4 * nd); b->dz = xmalloc(4 * nd); b->dug = xmalloc(8 * nd);
}

void blk_prepare(Block *b) { bl_prepare(&b->inp); bl_prepare(&b->out); }

static void conv_gate_fwd(Block *b, int N) {
    const int d = b->d, d2 = 2 * d, T = b->T, K = b->K; const float *cw = b->cw.w;
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const int t = n % T; const float *g = b->ug + (size_t)n * d2 + d;
        for (int c = 0; c < d; c += 16) {
            __m512 a = _mm512_setzero_ps();
            for (int j = 0; j < K; j++) {
                const int o = blk_off(b, j);
                if (t + o < 0 || t + o >= T) continue;
                a = _mm512_fmadd_ps(_mm512_loadu_ps(cw + j * d + c), _mm512_loadu_ps(b->ug + (size_t)(n + o) * d2 + c), a);
            }
            _mm512_storeu_ps(b->us + (size_t)n * d + c, a);
            _mm512_storeu_ps(b->z + (size_t)n * d + c, _mm512_mul_ps(a, _mm512_loadu_ps(g + c)));
        }
    }
}

void blk_forward(Block *b, const float *x, int N, float *y) {
    if (N % b->T) { fprintf(stderr, "block: N must be a multiple of T\n"); exit(1); }
    bl_forward(&b->inp, x, N, b->ug);
    PROF(P_CONV, conv_gate_fwd(b, N));
    bl_forward(&b->out, b->z, N, y);
}

// Per-sequence fused backward (sequence stays in L1/L2): pass 1 gate grads, pass 2 conv input grads + tap grads.
static void conv_gate_bwd(Block *b, int N) {
    const int d = b->d, d2 = 2 * d, T = b->T, K = b->K, nseq = N / T, nt = omp_get_max_threads();
    const float *cw = b->cw.w; float *dus = b->z;                          // z is dead after out-backward: reuse
    float *acc = xmalloc(4 * (size_t)nt * K * d);                          // per-thread tap-grad partials
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < nseq; s++) {
        float *tg = acc + (size_t)omp_get_thread_num() * K * d;
        const size_t base = (size_t)s * T;
        for (int t = 0; t < T; t++) {                                     // gate: dus = dz * g,  dg = dz * us
            const size_t n = base + t;
            for (int c = 0; c < d; c += 16) {
                const __m512 dz = _mm512_loadu_ps(b->dz + n * d + c);
                _mm512_storeu_ps(dus + n * d + c, _mm512_mul_ps(dz, _mm512_loadu_ps(b->ug + n * d2 + d + c)));
                _mm512_storeu_ps(b->dug + n * d2 + d + c, _mm512_mul_ps(dz, _mm512_loadu_ps(b->us + n * d + c)));
            }
        }
        for (int t = 0; t < T; t++) {                                     // du[t] = sum_j cw[j] dus[t - off(j)];  dcw[j] += dus[t] u[t + off(j)]
            const size_t n = base + t;
            for (int c = 0; c < d; c += 16) {
                __m512 a = _mm512_setzero_ps();
                const __m512 g = _mm512_loadu_ps(dus + n * d + c);
                for (int j = 0; j < K; j++) {
                    const int o = blk_off(b, j);
                    if (t - o >= 0 && t - o < T) a = _mm512_fmadd_ps(_mm512_loadu_ps(cw + j * d + c), _mm512_loadu_ps(dus + (n - o) * d + c), a);
                    if (t + o >= 0 && t + o < T) {
                        float *p = tg + j * d + c;
                        _mm512_storeu_ps(p, _mm512_fmadd_ps(g, _mm512_loadu_ps(b->ug + (n + o) * d2 + c), _mm512_loadu_ps(p)));
                    }
                }
                _mm512_storeu_ps(b->dug + n * d2 + c, a);
            }
        }
    }
    for (int t = 0; t < nt; t++)
        for (int i = 0; i < K * d; i += 16) {
            float *g = b->cw.g + i;
            _mm512_storeu_ps(g, _mm512_add_ps(_mm512_loadu_ps(g), _mm512_loadu_ps(acc + (size_t)t * K * d + i)));
        }
    free(acc);
}

void blk_backward(Block *b, const float *dy, float *dx) {
    bl_backward(&b->out, dy, b->dz);
    PROF(P_CONV, conv_gate_bwd(b, b->inp.N));
    bl_backward(&b->inp, b->dug, dx);
}

/* ==================== model, loss, Adam, checkpoints, data-parallel replicas ================ */

void model_init(Model *m, const Config *c, int cap) {
    memset(m, 0, sizeof(*m));
    static int amx_checked = 0; if (!amx_checked) { amx_init(); amx_checked = 1; }
    m->c = *c; m->cap = cap;
    const int d = c->d, L = c->n_layers, V = c->vocab;
    m->V_in = V + (c->mode == MODE_BIDIR);                                 // + [MASK]
    Rng rng = rng_seed(c->seed);
    param_init(&m->emb, (size_t)m->V_in * d, 1.0f);
    for (size_t i = 0; i < m->emb.n; i++) m->emb.w[i] = rng_normal(&rng);
    param_init(&m->head, (size_t)d * V, 1.0f);
    for (size_t i = 0; i < m->head.n; i++) m->head.w[i] = rng_normal(&rng) / sqrtf((float)d);
    m->blk = calloc(L, sizeof(Block));
    m->params = calloc(2 + 3 * L, sizeof(Param *));
    m->params[m->n_params++] = &m->emb; m->params[m->n_params++] = &m->head;
    for (int i = 0; i < L; i++) {
        blk_init(&m->blk[i], d, c->T, c->K, c->mode, model_dil(c->dil_cycle, i), cap, &rng);
        m->params[m->n_params++] = &m->blk[i].inp.W; m->params[m->n_params++] = &m->blk[i].out.W; m->params[m->n_params++] = &m->blk[i].cw;
    }
    m->xs = malloc(sizeof(float *) * (L + 1));
    for (int i = 0; i <= L; i++) m->xs[i] = xmalloc(4 * (size_t)cap * d);
    m->tmp = xmalloc(4 * (size_t)cap * d); m->dx = xmalloc(4 * (size_t)cap * d); m->dtmp = xmalloc(4 * (size_t)cap * d);
    m->xn = xmalloc(4 * (size_t)cap * d); m->r = xmalloc(4 * (size_t)cap); m->logits = xmalloc(4 * (size_t)cap * V);
    m->headT = xmalloc(4 * (size_t)V * d);
    model_prepare(m);
}

void model_prepare(Model *m) { for (int i = 0; i < m->c.n_layers; i++) blk_prepare(&m->blk[i]); }

size_t model_n_params(const Model *m) { size_t s = 0; for (int i = 0; i < m->n_params; i++) s += m->params[i]->n; return s; }

void zero_grads(Model *m) { for (int i = 0; i < m->n_params; i++) memset(m->params[i]->g, 0, 4 * m->params[i]->n); }

static void forward(Model *m, const int *tok, int N) {
    const int d = m->c.d, L = m->c.n_layers, V = m->c.vocab;
    if (N > m->cap) { fprintf(stderr, "model: N=%d > cap=%d\n", N, m->cap); exit(1); }
    m->N = N;
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) memcpy(m->xs[0] + (size_t)n * d, m->emb.w + (size_t)tok[n] * d, 4 * (size_t)d);
    for (int i = 0; i < L; i++) {
        blk_forward(&m->blk[i], m->xs[i], N, m->tmp);
        float *xo = m->xs[i + 1]; const float *xi = m->xs[i], *t = m->tmp;
        #pragma omp parallel for schedule(static)
        for (size_t e = 0; e < (size_t)N * d; e += 16) _mm512_storeu_ps(xo + e, _mm512_add_ps(_mm512_loadu_ps(xi + e), _mm512_loadu_ps(t + e)));
    }
    const float *xf = m->xs[L];
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {                                          // final RMSNorm
        __m512 ss = _mm512_setzero_ps();
        for (int c = 0; c < d; c += 16) { const __m512 v = _mm512_loadu_ps(xf + (size_t)n * d + c); ss = _mm512_fmadd_ps(v, v, ss); }
        const float r = 1.0f / sqrtf(_mm512_reduce_add_ps(ss) / d + 1e-6f); m->r[n] = r;
        for (int c = 0; c < d; c += 16) _mm512_storeu_ps(m->xn + (size_t)n * d + c, _mm512_mul_ps(_mm512_loadu_ps(xf + (size_t)n * d + c), _mm512_set1_ps(r)));
    }
    PROF(P_HEAD, sgemm(N, V, d, m->xn, d, m->head.w, V, m->logits, V, 0));
}

void model_logits(Model *m, const int *tok, int N) { forward(m, tok, N); }

double model_step(Model *m, const int *tok, const int *tgt, int N, int grad) {
    const int d = m->c.d, L = m->c.n_layers, V = m->c.vocab;
    forward(m, tok, N);
    int count = 0; for (int n = 0; n < N; n++) count += tgt[n] >= 0;
    if (!count) return 0;
    double loss = 0;
    #pragma omp parallel for schedule(static) reduction(+:loss)
    for (int n = 0; n < N; n++) {                                          // softmax cross-entropy
        float *lg = m->logits + (size_t)n * V;
        if (tgt[n] < 0) { if (grad) memset(lg, 0, 4 * (size_t)V); continue; }
        float mx = lg[0]; for (int v = 1; v < V; v++) mx = fmaxf(mx, lg[v]);
        double s = 0; for (int v = 0; v < V; v++) s += expf(lg[v] - mx);
        loss += log(s) - (lg[tgt[n]] - mx);
        if (grad) {
            const float nrm = m->loss_norm > 0 ? m->loss_norm : (float)count;
            const float inv = (float)(1.0 / s) / nrm;
            for (int v = 0; v < V; v++) lg[v] = expf(lg[v] - mx) * inv;
            lg[tgt[n]] -= 1.0f / nrm;
        }
    }
    loss /= count;
    if (!grad) return loss;
    PROF(P_HEAD, sgemm_tn(d, V, N, m->xn, d, m->logits, V, m->head.g, V, 1));
    transpose(m->head.w, d, V, m->headT);
    PROF(P_HEAD, sgemm(N, d, V, m->logits, V, m->headT, d, m->tmp, d, 0));
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {                                          // final RMSNorm backward
        const float r = m->r[n], *xn = m->xn + (size_t)n * d, *g = m->tmp + (size_t)n * d;
        float dot = 0; for (int c = 0; c < d; c++) dot += g[c] * xn[c];
        dot /= d;
        for (int c = 0; c < d; c++) m->dx[(size_t)n * d + c] = r * (g[c] - xn[c] * dot);
    }
    for (int i = L - 1; i >= 0; i--) {                                     // residual: dx += block^T(dx)
        blk_backward(&m->blk[i], m->dx, m->dtmp);
        float *dx = m->dx; const float *t = m->dtmp;
        #pragma omp parallel for schedule(static)
        for (size_t e = 0; e < (size_t)N * d; e += 16) _mm512_storeu_ps(dx + e, _mm512_add_ps(_mm512_loadu_ps(dx + e), _mm512_loadu_ps(t + e)));
    }
    for (int n = 0; n < N; n++) {                                          // embedding grad
        float *g = m->emb.g + (size_t)tok[n] * d; const float *s = m->dx + (size_t)n * d;
        for (int c = 0; c < d; c++) g[c] += s[c];
    }
    return loss;
}

void adam_step(Model *m, float lr, int t, float b1, float b2, float wd) {
#ifdef WN_PROF
    const double t0 = now_sec();
#endif
    const float c1 = 1.0f / (1.0f - powf(b1, t)), c2 = 1.0f / (1.0f - powf(b2, t));
    for (int i = 0; i < m->n_params; i++) {
        Param *p = m->params[i]; const float plr = lr * p->lr_mul;
        #pragma omp parallel for schedule(static) if (p->n > 65536)
        for (size_t j = 0; j < p->n; j++) {
            const float g = p->g[j];
            p->m[j] = b1 * p->m[j] + (1 - b1) * g;
            p->v[j] = b2 * p->v[j] + (1 - b2) * g * g;
            p->w[j] -= plr * ((p->m[j] * c1) / (sqrtf(p->v[j] * c2) + 1e-8f) + wd * p->w[j]);
            p->g[j] = 0;
        }
    }
#ifdef WN_PROF
    g_prof[P_ADAM] += now_sec() - t0;
#endif
    model_prepare(m);
}

/* checkpoint: magic "WSC2", Config, vocab chars[256], params (weights only) */
int model_save(const Model *m, const char *path, const unsigned char *vocab_chars) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    fwrite("WSC2", 1, 4, f); fwrite(&m->c, sizeof(Config), 1, f); fwrite(vocab_chars, 1, 256, f);
    for (int i = 0; i < m->n_params; i++) fwrite(m->params[i]->w, 4, m->params[i]->n, f);
    fclose(f); return 0;
}

int model_load(Model *m, const char *path, unsigned char *vocab_chars, int cap) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    char magic[4]; Config c;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "WSC2", 4) || fread(&c, sizeof(Config), 1, f) != 1 || fread(vocab_chars, 1, 256, f) != 256) { fclose(f); return -2; }
    model_init(m, &c, cap);
    for (int i = 0; i < m->n_params; i++) if (fread(m->params[i]->w, 4, m->params[i]->n, f) != m->params[i]->n) { fclose(f); return -3; }
    fclose(f); model_prepare(m); return 0;
}

/* ---- data-parallel replicas ---- */
static void share_bl(BitLinear *r, const BitLinear *m) {
    free(r->Wt); free(r->colsum); free(r->WqT); free(r->Wb); free(r->tmpT); free(r->tT);
    r->Wt = m->Wt; r->colsum = m->colsum; r->WqT = m->WqT; r->Wb = m->Wb; r->tmpT = r->tT = NULL; r->ws = m->ws;
}

void model_init_replica(Model *r, Model *master, int cap) {
    model_init(r, &master->c, cap);
    for (int i = 0; i < r->n_params; i++) { free(r->params[i]->w); free(r->params[i]->m); free(r->params[i]->v); r->params[i]->w = master->params[i]->w; r->params[i]->m = r->params[i]->v = NULL; }
    for (int l = 0; l < r->c.n_layers; l++) { share_bl(&r->blk[l].inp, &master->blk[l].inp); share_bl(&r->blk[l].out, &master->blk[l].out); }
}

void replicas_sync(Model *reps, int R, const Model *master) {
    for (int i = 0; i < R; i++) for (int l = 0; l < master->c.n_layers; l++) {
        reps[i].blk[l].inp.ws = master->blk[l].inp.ws; reps[i].blk[l].out.ws = master->blk[l].out.ws;
    }
}

double train_step_dp(Model *master, Model *reps, int R, const int *tok, const int *tgt, int N, int T) {
    const int nseq = N / T, per = (nseq + R - 1) / R;
    int total = 0; for (int n = 0; n < N; n++) total += tgt[n] >= 0;
    double loss = 0;
    const int saved = omp_get_max_active_levels(); omp_set_max_active_levels(1);   // inner regions run on 1 thread
    #pragma omp parallel num_threads(R) reduction(+:loss)
    {
        const int t = omp_get_thread_num(), s0 = t * per, s1 = s0 + per < nseq ? s0 + per : nseq;
        if (s0 < s1) {
            Model *r = &reps[t]; const int n0 = s0 * T, nn = (s1 - s0) * T;
            int cnt = 0; for (int n = n0; n < n0 + nn; n++) cnt += tgt[n] >= 0;
            r->loss_norm = (float)total;
            loss += model_step(r, tok + n0, tgt + n0, nn, 1) * cnt;
        }
    }
    omp_set_max_active_levels(saved);
    for (int i = 0; i < master->n_params; i++) {                           // reduce grads into the master
        Param *p = master->params[i];
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < p->n; j++) {
            float s = 0; for (int t = 0; t < R; t++) { s += reps[t].params[i]->g[j]; reps[t].params[i]->g[j] = 0; }
            p->g[j] += s;
        }
    }
    return total ? loss / total : 0;
}

/* ==================== constant-state streaming decode ======================================= */

/* Constant-state streaming decode. Layer l keeps a ring of its last (K-1)*dil_l conv inputs u (per stream);
   the tap reading u[t - j*dil] is slot (pos - j*dil) mod ring_len, which is zero until written. */

void stream_init(Stream *s, Model *m, int S) {
    if (m->c.mode != MODE_CAUSAL) { fprintf(stderr, "stream: needs a causal model\n"); exit(1); }
    if (S > m->cap) { fprintf(stderr, "stream: S > model capacity\n"); exit(1); }
    const int d = m->c.d, L = m->c.n_layers;
    s->m = m; s->S = S;
    s->x = xmalloc(4 * (size_t)S * d); s->y = xmalloc(4 * (size_t)S * d); s->ug = xmalloc(8 * (size_t)S * d); s->z = xmalloc(4 * (size_t)S * d);
    s->hoff = malloc(sizeof(size_t) * L); s->hlen = malloc(sizeof(int) * L);
    size_t tot = 0;
    for (int l = 0; l < L; l++) {
        s->hlen[l] = (m->c.K - 1) * m->blk[l].dil; s->hoff[l] = tot;
        tot += (size_t)(s->hlen[l] > 0 ? s->hlen[l] : 1) * S * d;
    }
    s->hist = xmalloc(4 * tot);
    stream_reset(s);
}

void stream_reset(Stream *s) {
    const Model *m = s->m; const int L = m->c.n_layers;
    const size_t tot = s->hoff[L - 1] + (size_t)(s->hlen[L - 1] > 0 ? s->hlen[L - 1] : 1) * s->S * m->c.d;
    memset(s->hist, 0, 4 * tot);
    s->pos = 0;
}

void stream_step(Stream *s, const int *tok) {
    Model *m = s->m; const int d = m->c.d, L = m->c.n_layers, K = m->c.K, S = s->S, V = m->c.vocab, d2 = 2 * d;
    for (int i = 0; i < S; i++) memcpy(s->x + (size_t)i * d, m->emb.w + (size_t)tok[i] * d, 4 * (size_t)d);
    for (int l = 0; l < L; l++) {
        Block *b = &m->blk[l]; float *H = s->hist + s->hoff[l]; const int hl = s->hlen[l], dil = b->dil;
        bl_forward(&b->inp, s->x, S, s->ug);
        const float *cw = b->cw.w;
        for (int i = 0; i < S; i++)
            for (int c = 0; c < d; c += 16) {
                __m512 a = _mm512_mul_ps(_mm512_loadu_ps(cw + c), _mm512_loadu_ps(s->ug + (size_t)i * d2 + c));
                for (int j = 1; j < K; j++) {
                    const int slot = (int)(((long)s->pos - (long)j * dil) % hl + hl) % hl;
                    a = _mm512_fmadd_ps(_mm512_loadu_ps(cw + j * d + c), _mm512_loadu_ps(H + ((size_t)slot * S + i) * d + c), a);
                }
                _mm512_storeu_ps(s->z + (size_t)i * d + c, _mm512_mul_ps(a, _mm512_loadu_ps(s->ug + (size_t)i * d2 + d + c)));
            }
        if (hl > 0) {                                                      // push u_t into the ring
            const int slot = s->pos % hl;
            for (int i = 0; i < S; i++) memcpy(H + ((size_t)slot * S + i) * d, s->ug + (size_t)i * d2, 4 * (size_t)d);
        }
        bl_forward(&b->out, s->z, S, s->y);
        for (size_t e = 0; e < (size_t)S * d; e += 16) _mm512_storeu_ps(s->x + e, _mm512_add_ps(_mm512_loadu_ps(s->x + e), _mm512_loadu_ps(s->y + e)));
    }
    for (int i = 0; i < S; i++) {                                          // final RMSNorm
        __m512 ss = _mm512_setzero_ps();
        for (int c = 0; c < d; c += 16) { const __m512 v = _mm512_loadu_ps(s->x + (size_t)i * d + c); ss = _mm512_fmadd_ps(v, v, ss); }
        const __m512 r = _mm512_set1_ps(1.0f / sqrtf(_mm512_reduce_add_ps(ss) / d + 1e-6f));
        for (int c = 0; c < d; c += 16) _mm512_storeu_ps(m->xn + (size_t)i * d + c, _mm512_mul_ps(_mm512_loadu_ps(s->x + (size_t)i * d + c), r));
    }
    for (int i = 0; i < S; i++) {                                         // head GEMV (vocab-wide vectors, masked tail)
        const float *xn = m->xn + (size_t)i * d; float *lg = m->logits + (size_t)i * V;
        for (int v0 = 0; v0 < V; v0 += 64) {
            __m512 a[4]; __mmask16 mk[4];
            for (int q = 0; q < 4; q++) { const int rem = V - v0 - 16 * q; mk[q] = rem >= 16 ? 0xFFFF : rem > 0 ? (__mmask16)((1u << rem) - 1) : 0; a[q] = _mm512_setzero_ps(); }
            for (int c = 0; c < d; c++) {
                const __m512 xc = _mm512_set1_ps(xn[c]); const float *h = m->head.w + (size_t)c * V + v0;
                for (int q = 0; q < 4; q++) if (mk[q]) a[q] = _mm512_fmadd_ps(xc, _mm512_maskz_loadu_ps(mk[q], h + 16 * q), a[q]);
            }
            for (int q = 0; q < 4; q++) if (mk[q]) _mm512_mask_storeu_ps(lg + v0 + 16 * q, mk[q], a[q]);
        }
    }
    s->pos++;
}

/* ==================== train ================================================================= */

// walshnet pretraining: causal LM (causal short conv) or masked LM (centred short conv) on a byte/char corpus.

typedef struct { const char *data, *out; int steps, batch, eval_every, eval_windows, warmup, save_best; float lr, wd, mask_p; } Opts;

static void train_usage(void) {
    fprintf(stderr,
        "usage: train --data FILE [options]\n"
        "  --mode causal|bidir  --d 64 --layers 7 --T 64 --K 3 --dilate 0 (cycle c: layer l spacing 2^(l mod c))\n"
        "  --batch 16 --steps 3000 --lr 3e-2 --wd 0 --warmup 100 --seed 0 --mask 0.15\n"
        "  --eval-every 500 --eval-windows 256 --out model.bin --save-best --threads N\n");
    exit(1);
}

static int *load_corpus(const char *path, size_t *n_out, unsigned char *vocab, int *V) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *raw = malloc(n); if (fread(raw, 1, n, f) != n) { perror("read"); exit(1); } fclose(f);
    int used[256] = {0}, map[256]; for (size_t i = 0; i < n; i++) used[raw[i]] = 1;
    *V = 0; for (int c = 0; c < 256; c++) if (used[c]) { map[c] = *V; vocab[(*V)++] = (unsigned char)c; }
    int *tok = malloc(sizeof(int) * n); for (size_t i = 0; i < n; i++) tok[i] = map[raw[i]];
    free(raw); *n_out = n; return tok;
}

// fill one batch: B windows of length T. causal: next-token targets. bidir: BERT-style masking (80/10/10).
static void make_batch(const int *data, size_t len, const Config *c, int B, float mask_p, Rng *r, int *tok, int *tgt) {
    const int T = c->T;
    for (int b = 0; b < B; b++) {
        const size_t s = rng_u64(r) % (len - T - 1);
        for (int t = 0; t < T; t++) {
            const int n = b * T + t, x = data[s + t];
            if (c->mode == MODE_CAUSAL) { tok[n] = x; tgt[n] = data[s + t + 1]; continue; }
            tok[n] = x; tgt[n] = -1;
            if (rng_unif(r) < mask_p) {
                tgt[n] = x; const float u = rng_unif(r);
                tok[n] = u < 0.8f ? c->vocab : u < 0.9f ? (int)(rng_u64(r) % c->vocab) : x;
            }
        }
    }
}

static double evaluate(Model *m, const int *val, size_t len, const Opts *o, int B) {
    Rng r = rng_seed(12345); int *tok = malloc(sizeof(int) * B * m->c.T), *tgt = malloc(sizeof(int) * B * m->c.T);
    double s = 0; int nb = 0;
    for (int w = 0; w < o->eval_windows; w += B, nb++) {
        make_batch(val, len, &m->c, B, o->mask_p, &r, tok, tgt);
        s += model_step(m, tok, tgt, B * m->c.T, 0);
    }
    free(tok); free(tgt); return s / nb;
}

static int cmd_train(int argc, char **argv) {
    Config c = {.d = 64, .n_layers = 7, .T = 64, .K = 3, .mode = MODE_CAUSAL, .seed = 0};
    Opts o = {.data = NULL, .out = "model.bin", .steps = 3000, .batch = 16, .eval_every = 500, .eval_windows = 256,
              .warmup = 100, .lr = 3e-2f, .wd = 0, .mask_p = 0.15f};
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        #define ARG(name) (!strcmp(a, name) && v && ++i)
        if      ARG("--data") o.data = v;
        else if ARG("--out") o.out = v;
        else if ARG("--mode") c.mode = !strcmp(v, "bidir") ? MODE_BIDIR : MODE_CAUSAL;
        else if ARG("--d") c.d = atoi(v);
        else if ARG("--layers") c.n_layers = atoi(v);
        else if ARG("--T") c.T = atoi(v);
        else if ARG("--K") c.K = atoi(v);
        else if ARG("--dilate") c.dil_cycle = atoi(v);
        else if (!strcmp(a, "--save-best")) o.save_best = 1;
        else if ARG("--seed") c.seed = strtoull(v, 0, 10);
        else if ARG("--batch") o.batch = atoi(v);
        else if ARG("--steps") o.steps = atoi(v);
        else if ARG("--lr") o.lr = atof(v);
        else if ARG("--wd") o.wd = atof(v);
        else if ARG("--warmup") o.warmup = atoi(v);
        else if ARG("--mask") o.mask_p = atof(v);
        else if ARG("--eval-every") o.eval_every = atoi(v);
        else if ARG("--eval-windows") o.eval_windows = atoi(v);
        else if ARG("--threads") omp_set_num_threads(atoi(v));
        else train_usage();
    }
    if (!o.data) train_usage();
    unsigned char vocab[256] = {0}; size_t len;
    int *data = load_corpus(o.data, &len, vocab, &c.vocab);
    const size_t n_train = (size_t)(0.9 * len);
    const int *train = data, *val = data + n_train; const size_t n_val = len - n_train;
    const int B = o.batch, N = B * c.T;
    Model m; model_init(&m, &c, N);
    printf("walshnet | %s LM | %d x gated %s short-conv (K=%d) BitNet blocks | d=%d T=%d | vocab %d | %zu params | %d threads | AMX %s\n",
           c.mode ? "masked" : "causal", c.n_layers, c.mode ? "centred" : "causal", c.K, c.d, c.T, c.vocab, model_n_params(&m), omp_get_max_threads(), g_amx ? "on" : "off");
    int rf = 1; for (int l = 0; l < c.n_layers; l++) rf += (c.mode ? c.K / 2 : c.K - 1) * model_dil(c.dil_cycle, l);
    printf("receptive field: %d tokens%s | training window T=%d\n", rf, c.mode ? " each side" : "", c.T);
    int R = omp_get_max_threads(); if (R > B) R = B;                       // data-parallel replicas (one per thread)
    Model *reps = NULL;
    if (R > 1) { reps = calloc(R, sizeof(Model)); for (int i = 0; i < R; i++) model_init_replica(&reps[i], &m, ((B + R - 1) / R) * c.T); }
    int *tok = malloc(sizeof(int) * N), *tgt = malloc(sizeof(int) * N);
    Rng r = rng_seed(1000 + c.seed);
    double t_train = 0, run_loss = 0, best_val = 1e9; int run_n = 0;
    for (int it = 1; it <= o.steps; it++) {
        make_batch(train, n_train, &c, B, o.mask_p, &r, tok, tgt);
        const double t0 = now_sec();
        run_loss += R > 1 ? train_step_dp(&m, reps, R, tok, tgt, N, c.T) : model_step(&m, tok, tgt, N, 1); run_n++;
        const float warm = it < o.warmup ? (float)it / o.warmup : 1.0f;
        adam_step(&m, o.lr * warm * (1.0f - (float)it / (o.steps + 1)), it, 0.9f, 0.95f, o.wd);
        if (R > 1) replicas_sync(reps, R, &m);
        t_train += now_sec() - t0;
        if (it % o.eval_every == 0 || it == o.steps) {
            const double vl = evaluate(&m, val, n_val, &o, B);
            const int improved = vl < best_val; if (improved) best_val = vl;
            printf("step %5d | train %.4f | val %.4f%s | %.0f tok/s | %.1fs\n", it, run_loss / run_n, vl, improved && o.save_best ? " *" : "", (double)it * N / t_train, t_train);
            if (improved && o.save_best) model_save(&m, o.out, vocab);
            fflush(stdout); run_loss = 0; run_n = 0;
        }
    }
    const double fv = evaluate(&m, val, n_val, &o, B);
    printf("final val %.4f | best val %.4f | train throughput %.0f tok/s | %.1fs\n", fv, fv < best_val ? fv : best_val, (double)o.steps * N / t_train, t_train);
    if (o.save_best && fv >= best_val) printf("kept best checkpoint in %s (val %.4f)\n", o.out, best_val);
    else if (model_save(&m, o.out, vocab)) fprintf(stderr, "could not save %s\n", o.out); else printf("saved %s\n", o.out);
    return 0;
}

/* ==================== inference: bench / generate / fill ==================================== */

// walshnet inference
//   infer bench    --model M [--tokens 8192]               batched forward + streaming decode throughput
//   infer generate --model M --prompt TEXT [--n 400] [--temp 0.8] [--seed 1]    constant-state streaming (causal)
//   infer fill     --model M --text "Wh_t is th_ m_tter"                        masked infill (bidirectional)

static const char *arg(int argc, char **argv, const char *name, const char *def) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}

static int sample(const float *lg, int V, float temp, Rng *r) {
    float mx = lg[0]; for (int v = 1; v < V; v++) mx = fmaxf(mx, lg[v]);
    double sum = 0; float p[256];
    for (int v = 0; v < V; v++) { p[v] = expf((lg[v] - mx) / temp); sum += p[v]; }
    double u = rng_unif(r) * sum;
    for (int v = 0; v < V; v++) { u -= p[v]; if (u <= 0) return v; }
    return V - 1;
}

static int cmd_infer(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: infer bench|generate|fill --model M ...\n"); return 1; }
    const char *cmd = argv[1], *path = arg(argc, argv, "--model", "model.bin");
    if (arg(argc, argv, "--threads", NULL)) omp_set_num_threads(atoi(arg(argc, argv, "--threads", "4")));
    const int cap = atoi(arg(argc, argv, "--tokens", "8192"));
    Model m; unsigned char vocab[256]; int stoi[256];
    const int rc = model_load(&m, path, vocab, cap);
    if (rc) { fprintf(stderr, "cannot load %s (%d)\n", path, rc); return 1; }
    for (int i = 0; i < 256; i++) stoi[i] = -1;
    for (int i = 0; i < m.c.vocab; i++) stoi[vocab[i]] = i;
    const Config *c = &m.c; const int V = c->vocab, T = c->T;

    if (!strcmp(cmd, "bench")) {
        printf("model: %d x gated %s short-conv (K=%d) BitNet blocks | d=%d T=%d | %zu params | %d threads | AMX %s\n",
               c->n_layers, c->mode ? "centred" : "causal", c->K, c->d, T, model_n_params(&m), omp_get_max_threads(), g_amx ? "on" : "off");
        const int N = cap / T * T; int *tok = malloc(sizeof(int) * N); Rng r = rng_seed(3);
        for (int n = 0; n < N; n++) tok[n] = rng_u64(&r) % V;
        for (int w = 0; w < 3; w++) model_logits(&m, tok, N);
        double best = 1e9;
        for (int it = 0; it < 20; it++) { const double t0 = now_sec(); model_logits(&m, tok, N); best = fmin(best, now_sec() - t0); }
        printf("  batched forward (%d tokens)       : %10.0f tok/s\n", N, N / best);
        if (c->mode == MODE_CAUSAL) {
            const int streams[] = {1, 16, 64, 256, 1024};
            const int R = omp_get_max_threads();
            for (int si = 0; si < 5; si++) {
                const int S = streams[si]; if (S > cap) break;
                if (S >= 64 && R > 1) {                                   // independent streams: one replica per core, no sync
                    Model *reps = calloc(R, sizeof(Model)); Stream *sts = calloc(R, sizeof(Stream));
                    for (int i = 0; i < R; i++) { model_init_replica(&reps[i], &m, S / R); stream_init(&sts[i], &reps[i], S / R); }
                    const int steps = S <= 256 ? 2000 : 500;
                    const int saved = omp_get_max_active_levels(); omp_set_max_active_levels(1);
                    double t = 0;
                    #pragma omp parallel num_threads(R)
                    {
                        Stream *st = &sts[omp_get_thread_num()]; int *tk = calloc(st->S, sizeof(int));
                        for (int i = 0; i < 50; i++) stream_step(st, tk);
                        #pragma omp barrier
                        const double t0 = now_sec();
                        for (int i = 0; i < steps; i++) stream_step(st, tk);
                        #pragma omp barrier
                        #pragma omp master
                        t = now_sec() - t0;
                    }
                    omp_set_max_active_levels(saved);
                    printf("  streaming decode, %4d streams   : %10.0f tok/s  (%d cores x %d streams, %.2f us per step)\n", S, (double)S * steps / t, R, S / R, t / steps * 1e6);
                    continue;
                }
                Stream st; stream_init(&st, &m, S); int *st_tok = malloc(sizeof(int) * S);
                for (int i = 0; i < S; i++) st_tok[i] = rng_u64(&r) % V;
                const int steps = S == 1 ? 20000 : S <= 16 ? 4000 : 1000;
                for (int i = 0; i < 100; i++) stream_step(&st, st_tok);
                const double t0 = now_sec();
                for (int i = 0; i < steps; i++) stream_step(&st, st_tok);
                const double t = now_sec() - t0;
                printf("  streaming decode, %4d stream%s   : %10.0f tok/s  (1 core, %.2f us per step)\n", S, S > 1 ? "s" : " ", (double)S * steps / t, t / steps * 1e6);
            }
        }
        return 0;
    }
    if (!strcmp(cmd, "generate")) {
        if (c->mode != MODE_CAUSAL) { fprintf(stderr, "generate needs a causal model\n"); return 1; }
        const char *prompt = arg(argc, argv, "--prompt", "\n"); const int n_new = atoi(arg(argc, argv, "--n", "400"));
        const float temp = atof(arg(argc, argv, "--temp", "0.8")); Rng r = rng_seed(atoi(arg(argc, argv, "--seed", "1")));
        Stream st; stream_init(&st, &m, 1);
        int cur = 0; fputs(prompt, stdout);
        for (const char *p = prompt; *p; p++) {                            // feed the prompt through the stream state
            const int id = stoi[(unsigned char)*p]; if (id < 0) continue;
            cur = id; if (p[1]) stream_step(&st, &cur);
        }
        const double t0 = now_sec();
        for (int i = 0; i < n_new; i++) {                                 // constant work per token, unbounded length
            stream_step(&st, &cur);
            cur = sample(m.logits, V, temp, &r);
            putchar(vocab[cur]);
        }
        const double t = now_sec() - t0;
        fprintf(stderr, "\n[%d tokens in %.3f s: %.0f tok/s, single stream]\n", n_new, t, n_new / t);
        return 0;
    }
    if (!strcmp(cmd, "fill")) {
        if (c->mode != MODE_BIDIR) { fprintf(stderr, "fill needs a bidirectional model\n"); return 1; }
        const char *text = arg(argc, argv, "--text", "Wh_t is th_ m_tter"); const int L = strlen(text);
        if (L > T) { fprintf(stderr, "text longer than T=%d\n", T); return 1; }
        int *win = malloc(sizeof(int) * T); const int sp = stoi[' '] >= 0 ? stoi[' '] : 0;
        for (int t = 0; t < T; t++) win[t] = t >= L ? sp : text[t] == '_' ? V : (stoi[(unsigned char)text[t]] >= 0 ? stoi[(unsigned char)text[t]] : sp);
        model_logits(&m, win, T);
        printf("input : %s\noutput: ", text);
        for (int t = 0; t < L; t++) {
            if (text[t] != '_') { putchar(text[t]); continue; }
            const float *lg = m.logits + (size_t)t * V; int best = 0;
            for (int v = 1; v < V; v++) if (lg[v] > lg[best]) best = v;
            putchar(vocab[best]);
        }
        putchar('\n');
        return 0;
    }
    fprintf(stderr, "unknown command %s\n", cmd); return 1;
}

/* ==================== self-test ============================================================= */

// walshnet tests: kernels vs references, finite-difference gradients, causality/receptive field, streaming == batch.

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void test_gemm(void) {
    int shapes[][3] = {{13, 45, 7}, {64, 65, 100}, {100, 256, 300}, {8, 32, 1}, {5, 3, 600}};
    Rng r = rng_seed(1); double worst = 0;
    for (int s = 0; s < 5; s++) {
        const int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2];
        float *A = xmalloc(4 * M * K), *At = xmalloc(4 * M * K), *B = xmalloc(4 * K * N), *C = xmalloc(4 * M * N), *C2 = xmalloc(4 * M * N);
        for (int i = 0; i < M * K; i++) A[i] = rng_normal(&r);
        for (int i = 0; i < K * N; i++) B[i] = rng_normal(&r);
        for (int i = 0; i < M * N; i++) C[i] = 1.0f;
        sgemm(M, N, K, A, K, B, N, C, N, 1);
        transpose(A, M, K, At);                                            // At [K][M]
        sgemm_tn(M, N, K, At, M, B, N, C2, N, 0);
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
            double ref = 0; for (int k = 0; k < K; k++) ref += (double)A[i * K + k] * B[k * N + j];
            worst = fmax(worst, fabs(ref + 1 - C[i * N + j]) / (1 + fabs(ref)));
            worst = fmax(worst, fabs(ref - C2[i * N + j]) / (1 + fabs(ref)));
        }
        free(A); free(At); free(B); free(C); free(C2);
    }
    CHECK(worst < 1e-4, "gemm err %g", worst);
    printf("sgemm / sgemm_tn / transpose : match naive references (max rel err %.1e)\n", worst);
}

static void test_bitlinear(void) {
    Rng r = rng_seed(2); BitLinear l; const int K = 64, M = 48, N = 37;
    bl_init(&l, K, M, N, &r); bl_prepare(&l); l.store_xq = 1;
    float *x = xmalloc(4 * N * K), *y = xmalloc(4 * N * M);
    for (int i = 0; i < N * K; i++) x[i] = rng_normal(&r) * 3;
    bl_forward(&l, x, N, y);
    double err = 0;
    for (int n = 0; n < N; n++) for (int m = 0; m < M; m++) {
        double ref = 0; for (int k = 0; k < K; k++) ref += (double)l.xq[n * K + k] * l.WqT[m * K + k];
        err = fmax(err, fabs(ref - y[n * M + m]));
    }
    CHECK(err < 1e-3, "bitlinear err %g", err);
    printf("bitlinear                    : int8 x ternary VNNI exact vs fp32 reference (max err %.1e)\n", err);
}

static void test_bf16_backward(void) {
    Rng r = rng_seed(8); const int K = 64, M = 128, N = 96; BitLinear l;
    bl_init(&l, K, M, N, &r); bl_prepare(&l);
    float *x = xmalloc(4 * N * K), *dy = xmalloc(4 * N * M), *dx32 = xmalloc(4 * N * K), *dx16 = xmalloc(4 * N * K), *g32 = xmalloc(4 * K * M);
    for (int i = 0; i < N * K; i++) x[i] = rng_normal(&r);
    for (int i = 0; i < N * M; i++) dy[i] = rng_normal(&r);
    float *y = xmalloc(4 * N * M);
    g_bf16_bwd = 0; bl_forward(&l, x, N, y); memset(l.W.g, 0, 4 * K * M); bl_backward(&l, dy, dx32); memcpy(g32, l.W.g, 4 * K * M);
    g_bf16_bwd = 1; bl_forward(&l, x, N, y); memset(l.W.g, 0, 4 * K * M); bl_backward(&l, dy, dx16);
    double ex = 0, nx = 0, ew = 0, nw = 0;
    for (int i = 0; i < N * K; i++) { ex += (dx16[i] - dx32[i]) * (dx16[i] - dx32[i]); nx += dx32[i] * dx32[i]; }
    for (int i = 0; i < K * M; i++) { ew += (l.W.g[i] - g32[i]) * (l.W.g[i] - g32[i]); nw += g32[i] * g32[i]; }
    ex = sqrt(ex / nx); ew = sqrt(ew / nw);
    CHECK(ex < 1e-2 && ew < 1e-2, "bf16 backward rel err dx %g dW %g", ex, ew);
    printf("bf16 backward                : vs fp32 backward, relative L2 error dx %.1e, dW %.1e\n", ex, ew);
}

static void test_amx(void) {
    if (!amx_init()) { printf("amx                          : not available, skipped\n"); return; }
    Rng r = rng_seed(12); BitLinear l; const int K = 128, M = 96, N = 64;
    bl_init(&l, K, M, N, &r); bl_prepare(&l);
    float *x = xmalloc(4 * N * K), *y1 = xmalloc(4 * N * M), *y2 = xmalloc(4 * N * M);
    for (int i = 0; i < N * K; i++) x[i] = rng_normal(&r);
    g_amx = 0; bl_forward(&l, x, N, y1); g_amx = 1; bl_forward(&l, x, N, y2);
    double ef = 0; for (int i = 0; i < N * M; i++) ef = fmax(ef, fabs(y1[i] - y2[i]));
    const int Mg = 64, Ng = 96, Kg = 160;
    bf16 *A = xmalloc(2 * Mg * Kg), *Bp = xmalloc(2 * Kg * Ng); float *Af = xmalloc(4 * Mg * Kg), *Bf = xmalloc(4 * Kg * Ng), *C1 = xmalloc(4 * Mg * Ng), *C2 = xmalloc(4 * Mg * Ng);
    for (int i = 0; i < Mg * Kg; i++) Af[i] = rng_normal(&r);
    for (int i = 0; i < Kg * Ng; i++) Bf[i] = rng_normal(&r);
    to_bf16(Af, A, Mg * Kg); pack_pairs_bf16(Bf, Kg, Ng, Ng, NULL, Bp);
    for (int i = 0; i < Mg * Ng; i++) C1[i] = C2[i] = 1.0f;
    g_amx = 0; gemm_bf16(Mg, Ng, Kg, A, Kg, Bp, C1, Ng, 0.5f, 1); g_amx = 1; gemm_bf16(Mg, Ng, Kg, A, Kg, Bp, C2, Ng, 0.5f, 1);
    double eg = 0; for (int i = 0; i < Mg * Ng; i++) eg = fmax(eg, fabs(C1[i] - C2[i]) / (1 + fabs(C1[i])));
    CHECK(ef < 1e-4 && eg < 1e-5, "amx fwd err %g, bf16 gemm err %g", ef, eg);
    printf("amx                          : int8 BitLinear fwd == VNNI (max err %.1e); bf16 tiles == AVX-512 bf16 (rel err %.1e)\n", ef, eg);
}

static void test_gradients(int mode, int K, int dc) {
    g_noquant = 1;
    Config c = {.vocab = 11, .d = 16, .n_layers = 3, .T = 16, .K = K, .mode = mode, .dil_cycle = dc, .seed = 7};
    const int N = 3 * c.T; Model m; model_init(&m, &c, N);
    Rng r = rng_seed(4);
    int *tok = malloc(4 * N), *tgt = malloc(4 * N);
    for (int n = 0; n < N; n++) { tok[n] = rng_u64(&r) % m.V_in; tgt[n] = (mode == MODE_BIDIR && n % 3) ? -1 : (int)(rng_u64(&r) % c.vocab); }
    zero_grads(&m); model_step(&m, tok, tgt, N, 1);
    double worst = 0; int checked = 0;
    for (int p = 0; p < m.n_params; p++) {
        Param *P = m.params[p];
        for (int trial = 0; trial < 8; trial++) {
            const size_t i = rng_u64(&r) % P->n; const float e = 1e-2f, w0 = P->w[i];
            P->w[i] = w0 + e; model_prepare(&m); const double lp = model_step(&m, tok, tgt, N, 0);
            P->w[i] = w0 - e; model_prepare(&m); const double lm = model_step(&m, tok, tgt, N, 0);
            P->w[i] = w0; model_prepare(&m);
            const double num = (lp - lm) / (2 * e), ana = P->g[i];
            if (fabs(num) + fabs(ana) < 3e-4) continue;
            worst = fmax(worst, fabs(num - ana) / (fabs(num) + fabs(ana))); checked++;
        }
    }
    CHECK(worst < 2e-2, "gradcheck mode=%d K=%d dil=%d worst %g", mode, K, dc, worst);
    printf("gradcheck %-6s K=%d dil=%d    : worst rel err %.1e over %d entries\n", mode ? "bidir" : "causal", K, dc, worst, checked);
    g_noquant = 0;
}

// changing token p may only move logits inside the receptive field [p - L*right, p + L*left]
static void test_receptive_field(int mode, int K, int dc) {
    Config c = {.vocab = 11, .d = 32, .n_layers = 4, .T = 64, .K = K, .mode = mode, .dil_cycle = dc, .seed = 9};
    const int N = c.T, p = 30; Model m; model_init(&m, &c, N);
    int tok[64]; Rng r = rng_seed(5);
    for (int n = 0; n < N; n++) tok[n] = rng_u64(&r) % 11;
    float *a = xmalloc(4 * N * 11); model_logits(&m, tok, N); memcpy(a, m.logits, 4 * N * 11);
    tok[p] = (tok[p] + 1) % 11; model_logits(&m, tok, N);
    int sd = 0; for (int l = 0; l < c.n_layers; l++) sd += model_dil(dc, l);
    const int back = mode == MODE_CAUSAL ? 0 : sd * (K / 2), fwd = mode == MODE_CAUSAL ? sd * (K - 1) : sd * (K / 2);
    double outside = 0, inside = 0;
    for (int n = 0; n < N; n++) for (int v = 0; v < 11; v++) {
        const double dl = fabs(a[n * 11 + v] - m.logits[n * 11 + v]);
        if (n < p - back || n > p + fwd) outside = fmax(outside, dl); else inside = fmax(inside, dl);
    }
    CHECK(outside == 0 && inside > 0, "receptive field mode=%d: outside %g inside %g", mode, outside, inside);
    printf("receptive field %-6s K=%d dil=%d: edit at t=%d moves logits only in [%d, %d] (outside %g, inside %.3f)\n",
           mode ? "bidir" : "causal", K, dc, p, p - back, p + fwd, outside, inside);
}

static void test_stream(int K, int dc) {
    Config c = {.vocab = 11, .d = 32, .n_layers = 4, .T = 64, .K = K, .mode = MODE_CAUSAL, .dil_cycle = dc, .seed = 3};
    const int T = c.T, S = 3; Model m; model_init(&m, &c, S * T);
    Rng r = rng_seed(6);
    for (int l = 0; l < c.n_layers; l++) for (size_t i = 0; i < m.blk[l].cw.n; i++) m.blk[l].cw.w[i] += 0.3f * rng_normal(&r);
    int *tok = malloc(sizeof(int) * S * T);
    for (int i = 0; i < S * T; i++) tok[i] = rng_u64(&r) % 11;
    model_logits(&m, tok, S * T);                                          // batch: S sequences of T
    float *ref = xmalloc(4 * S * T * 11); memcpy(ref, m.logits, 4 * S * T * 11);
    Stream s; stream_init(&s, &m, S); double err = 0;
    for (int t = 0; t < T; t++) {
        int step_tok[3]; for (int i = 0; i < S; i++) step_tok[i] = tok[i * T + t];
        stream_step(&s, step_tok);
        for (int i = 0; i < S; i++) for (int v = 0; v < 11; v++) err = fmax(err, fabs(m.logits[i * 11 + v] - ref[(i * T + t) * 11 + v]));
    }
    CHECK(err < 1e-4, "stream vs batch K=%d err %g", K, err);
    printf("streaming decode K=%d dil=%d   : %d streams x %d steps == batch forward (max |logit diff| %.1e)\n", K, dc, S, T, err);
}

static void test_data_parallel(void) {
    Config c = {.vocab = 13, .d = 64, .n_layers = 3, .T = 32, .K = 3, .mode = MODE_CAUSAL, .seed = 21};
    const int N = 8 * c.T, R = 4;
    Model a, m; model_init(&a, &c, N); model_init(&m, &c, N);             // identical weights (same seed)
    Model *reps = calloc(R, sizeof(Model)); for (int i = 0; i < R; i++) model_init_replica(&reps[i], &m, N / R);
    Rng r = rng_seed(22); int *tok = malloc(4 * N), *tgt = malloc(4 * N);
    for (int n = 0; n < N; n++) { tok[n] = rng_u64(&r) % 13; tgt[n] = (n % 5 == 0) ? -1 : (int)(rng_u64(&r) % 13); }
    zero_grads(&a); zero_grads(&m);
    const double la = model_step(&a, tok, tgt, N, 1), lm = train_step_dp(&m, reps, R, tok, tgt, N, c.T);
    double num = 0, den = 0;
    for (int i = 0; i < a.n_params; i++) for (size_t j = 0; j < a.params[i]->n; j++) {
        const double e = a.params[i]->g[j] - m.params[i]->g[j]; num += e * e; den += (double)a.params[i]->g[j] * a.params[i]->g[j];
    }
    const double rel = sqrt(num / den);
    CHECK(fabs(la - lm) < 1e-5 && rel < 1e-5, "data parallel: loss %g vs %g, grad rel err %g", la, lm, rel);
    printf("data-parallel step (%d thr)   : loss %.6f == %.6f, gradients match (rel L2 err %.1e)\n", R, lm, la, rel);
}

static int cmd_test(void) {
    printf("threads: %d\n", omp_get_max_threads());
    test_gemm(); test_bitlinear(); test_bf16_backward(); test_amx();
    for (int mode = 0; mode < 2; mode++) for (int K = 3; K <= 5; K += 2) test_gradients(mode, K, 0);
    for (int mode = 0; mode < 2; mode++) test_gradients(mode, 3, 3);
    for (int mode = 0; mode < 2; mode++) { test_receptive_field(mode, 3, 0); test_receptive_field(mode, 3, 3); }
    test_receptive_field(MODE_CAUSAL, 4, 0);
    test_stream(3, 0); test_stream(4, 0); test_stream(3, 3); test_stream(4, 4);
    test_data_parallel();
    printf(fails ? "\n%d FAILURES\n" : "\nall tests passed\n", fails);
    return fails != 0;
}

/* ==================== entry point ==================== */

int main(int argc, char **argv) {
    const char *cmd = argc > 1 ? argv[1] : "";
    if (!strcmp(cmd, "train")) return cmd_train(argc - 1, argv + 1);
    if (!strcmp(cmd, "bench") || !strcmp(cmd, "generate") || !strcmp(cmd, "fill")) return cmd_infer(argc, argv);
    if (!strcmp(cmd, "test")) return cmd_test();
    fprintf(stderr, "usage: %s train|bench|generate|fill|test [options]   (see the comment at the top of the source)\n", argv[0]);
    return 1;
}
