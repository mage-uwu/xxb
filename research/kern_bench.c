// Inference kernels: BitLinear (int8 x ternary via AVX-512 VNNI), fp32 attention, bit-sliced DDLGN.
// Single-threaded timing at BERT-tiny shapes. Build: gcc -O3 -march=native -ffast-math kern.c -lm
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *amalloc(size_t n) { void *p = aligned_alloc(64, (n + 63) / 64 * 64); memset(p, 0, (n + 63) / 64 * 64); return p; }
static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)rs; }
static float frnd(void) { return (rnd() >> 8) * (1.0f / 16777216.0f) * 2 - 1; }

/* ---------------- BitLinear: absmax int8 activations (RMSNorm folded in) x ternary weights ---------------- */
typedef struct { int K, M; int8_t *Wp; int8_t *W; int32_t *colsum; float ws; } BitLinear;   // Wp: [K/4][M][4] VNNI layout

static void bl_init(BitLinear *l, int K, int M) {
    l->K = K; l->M = M; l->ws = 0.05f;
    l->W = amalloc((size_t)K * M); l->Wp = amalloc((size_t)K * M); l->colsum = amalloc(4 * (size_t)M);
    for (size_t i = 0; i < (size_t)K * M; i++) l->W[i] = (int8_t)(rnd() % 3) - 1;       // W[k][m]
    for (int k = 0; k < K; k++) for (int m = 0; m < M; m++) {
        l->Wp[((size_t)(k / 4) * M + m) * 4 + k % 4] = l->W[(size_t)k * M + m];
        l->colsum[m] += l->W[(size_t)k * M + m];
    }
}

static void quant_rows(const float *x, int N, int K, uint8_t *xu, float *sx) {   // K % 16 == 0
    for (int n = 0; n < N; n++) {
        const float *r = x + (size_t)n * K;
        __m512 amax = _mm512_set1_ps(1e-5f), ss = _mm512_setzero_ps();
        for (int k = 0; k < K; k += 16) { __m512 v = _mm512_loadu_ps(r + k); amax = _mm512_max_ps(amax, _mm512_abs_ps(v)); ss = _mm512_fmadd_ps(v, v, ss); }
        float am = _mm512_reduce_max_ps(amax), s2 = _mm512_reduce_add_ps(ss);
        __m512 s = _mm512_set1_ps(127.0f / am);
        for (int k = 0; k < K; k += 16) {
            __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(r + k), s));      // round-to-nearest
            _mm_storeu_si128((__m128i *)(xu + (size_t)n * K + k), _mm_xor_si128(_mm512_cvtsepi32_epi8(q), _mm_set1_epi8((char)0x80)));
        }
        sx[n] = (am / 127.0f) / sqrtf(s2 / K + 1e-6f);          // dequant * RMSNorm factor
    }
}

// y[N][M] = BitLinear(x[N][K]); N % 4 == 0, M % 64 == 0, K % 4 == 0
static void bl_forward(const BitLinear *l, const float *x, int N, float *y, uint8_t *xu, float *sx, int requant) {
    const int K = l->K, M = l->M;
    if (requant) quant_rows(x, N, K, xu, sx);
    for (int m = 0; m < M; m += 64)
        for (int n = 0; n < N; n += 4) {
            __m512i acc[4][4];
            for (int r = 0; r < 4; r++) for (int j = 0; j < 4; j++) acc[r][j] = _mm512_setzero_si512();
            for (int kq = 0; kq < K / 4; kq++) {
                const int8_t *wp = l->Wp + ((size_t)kq * M + m) * 4;
                __m512i w0 = _mm512_load_si512(wp), w1 = _mm512_load_si512(wp + 64),
                        w2 = _mm512_load_si512(wp + 128), w3 = _mm512_load_si512(wp + 192);
                for (int r = 0; r < 4; r++) {
                    __m512i a = _mm512_set1_epi32(*(const int32_t *)(xu + (size_t)(n + r) * K + kq * 4));
                    acc[r][0] = _mm512_dpbusd_epi32(acc[r][0], a, w0);
                    acc[r][1] = _mm512_dpbusd_epi32(acc[r][1], a, w1);
                    acc[r][2] = _mm512_dpbusd_epi32(acc[r][2], a, w2);
                    acc[r][3] = _mm512_dpbusd_epi32(acc[r][3], a, w3);
                }
            }
            for (int r = 0; r < 4; r++) {
                __m512 sc = _mm512_set1_ps(sx[n + r] * l->ws);
                for (int j = 0; j < 4; j++) {
                    __m512i cs = _mm512_slli_epi32(_mm512_load_si512(l->colsum + m + 16 * j), 7);   // remove +128 offset
                    __m512 v = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[r][j], cs));
                    _mm512_storeu_ps(y + (size_t)(n + r) * M + m + 16 * j, _mm512_mul_ps(v, sc));
                }
            }
        }
}

/* ---------------- fp32 softmax attention (bidirectional, BERT-style), AVX-512 ---------------- */
static inline __m512 exp512(__m512 x) {                                  // exp for x <= 0, rel err ~1e-7
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
    __m512 t = _mm512_mul_ps(x, _mm512_set1_ps(1.44269504f)), n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT);
    __m512 f = _mm512_sub_ps(t, n), p = _mm512_set1_ps(1.535336188e-4f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.339887440e-3f)); p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.618437357e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.550332471e-2f)); p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.402264791e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.931472028e-1f)); p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}

// Register-blocked core: QB queries at a time, NZ = T/16 score vectors, NW = dh/16 output vectors (compile-time)
#define QB 4   /* max queries per block (scratch sizing) */
static inline __attribute__((always_inline)) void attn_head(const float *base, float *out, int d, int h, int dh, int T,
                                                             const int NZ, const int NW, const int QB_, float *Kt, float *s) {
#define QB QB_
    const float scale = 1.0f / sqrtf((float)dh);
    for (int j = 0; j < T; j++) for (int e = 0; e < dh; e++) Kt[e * T + j] = base[(size_t)j * 3 * d + d + h * dh + e];
    for (int i0 = 0; i0 < T; i0 += QB) {
        __m512 sv[QB][NZ];
        for (int r = 0; r < QB; r++) for (int z = 0; z < NZ; z++) sv[r][z] = _mm512_setzero_ps();
        for (int e = 0; e < dh; e++) {
            __m512 kz[NZ];
            for (int z = 0; z < NZ; z++) kz[z] = _mm512_load_ps(Kt + e * T + 16 * z);
            for (int r = 0; r < QB; r++) {
                __m512 qe = _mm512_set1_ps(base[(size_t)(i0 + r) * 3 * d + h * dh + e] * scale);
                for (int z = 0; z < NZ; z++) sv[r][z] = _mm512_fmadd_ps(qe, kz[z], sv[r][z]);
            }
        }
        float inv[QB];
        for (int r = 0; r < QB; r++) {
            __m512 mx = sv[r][0]; for (int z = 1; z < NZ; z++) mx = _mm512_max_ps(mx, sv[r][z]);
            __m512 m = _mm512_set1_ps(_mm512_reduce_max_ps(mx)), sum = _mm512_setzero_ps();
            for (int z = 0; z < NZ; z++) { __m512 p = exp512(_mm512_sub_ps(sv[r][z], m)); sum = _mm512_add_ps(sum, p); _mm512_store_ps(s + r * T + 16 * z, p); }
            inv[r] = 1.0f / _mm512_reduce_add_ps(sum);
        }
        __m512 o[QB][NW];
        for (int r = 0; r < QB; r++) for (int w = 0; w < NW; w++) o[r][w] = _mm512_setzero_ps();
        for (int j = 0; j < T; j++) {
            const float *v = base + (size_t)j * 3 * d + 2 * d + h * dh;
            __m512 vw[NW]; for (int w = 0; w < NW; w++) vw[w] = _mm512_loadu_ps(v + 16 * w);
            for (int r = 0; r < QB; r++) { __m512 p = _mm512_set1_ps(s[r * T + j]); for (int w = 0; w < NW; w++) o[r][w] = _mm512_fmadd_ps(p, vw[w], o[r][w]); }
        }
        for (int r = 0; r < QB; r++) {
            float *op = out + (size_t)(i0 + r) * d + h * dh;
            for (int w = 0; w < NW; w++) _mm512_storeu_ps(op + 16 * w, _mm512_mul_ps(o[r][w], _mm512_set1_ps(inv[r])));
        }
    }
#undef QB
#define QB 4
}

// qkv: [N][3d] (q | k | v), out: [N][d]; sequences of length T
static void attention(const float *qkv, float *out, int N, int T, int H, int dh, float *Kt, float *s) {
    const int d = H * dh;
    for (int b = 0; b < N / T; b++)
        for (int h = 0; h < H; h++) {
            const float *base = qkv + (size_t)b * T * 3 * d; float *ob = out + (size_t)b * T * d;
            if (T == 64 && dh == 16)       attn_head(base, ob, d, h, dh, T, 4, 1, 4, Kt, s);
            else if (T == 64 && dh == 32)  attn_head(base, ob, d, h, dh, T, 4, 2, 4, Kt, s);
            else if (T == 128 && dh == 64) attn_head(base, ob, d, h, dh, T, 8, 4, 4, Kt, s);
            else if (T == 512 && dh == 64) attn_head(base, ob, d, h, dh, T, 32, 4, 1, Kt, s);
            else { fprintf(stderr, "unsupported attention shape\n"); exit(1); }
        }
}

/* ---------------- DDLGN, hardened + bit-sliced: one __m512i = one gate's output for 512 tokens ---------------- */
typedef struct { int n_in, n_out; int32_t *ia, *ib; int bounds[17]; uint8_t *type; } LLayer;   // gates sorted by type
typedef struct { int d, nth, width, depth, k; float *th, *gain; LLayer *L; } DLGN;

static void ll_init(LLayer *l, int n_in, int n_out) {
    l->n_in = n_in; l->n_out = n_out;
    l->ia = amalloc(4 * (size_t)n_out); l->ib = amalloc(4 * (size_t)n_out); l->type = amalloc(n_out);
    int cnt[16] = {0};
    uint8_t *t = malloc(n_out); for (int g = 0; g < n_out; g++) { t[g] = rnd() % 16; cnt[t[g]]++; }
    l->bounds[0] = 0; for (int i = 0; i < 16; i++) l->bounds[i + 1] = l->bounds[i] + cnt[i];
    for (int g = 0, pos[16], init = 0; g < n_out; g++) {             // counting sort by gate type
        if (!init) { for (int i = 0; i < 16; i++) pos[i] = l->bounds[i]; init = 1; }
        int p = pos[t[g]]++; l->type[p] = t[g]; l->ia[p] = rnd() % n_in; l->ib[p] = rnd() % n_in;
    }
    free(t);
}

static void dlgn_init(DLGN *m, int d, int nth, int width, int depth) {
    m->d = d; m->nth = nth; m->width = width; m->depth = depth; m->k = width / d;
    m->th = amalloc(4 * (size_t)nth); m->gain = amalloc(4 * (size_t)d);
    for (int j = 0; j < nth; j++) m->th[j] = -1.5f + 3.0f * (j + 0.5f) / nth;
    for (int c = 0; c < d; c++) m->gain[c] = 1.0f;
    m->L = malloc(sizeof(LLayer) * depth);
    for (int i = 0; i < depth; i++) ll_init(&m->L[i], i ? width : d * nth, width);
}

// imm8 for vpternlogd(a, b, a): truth-table bit (2a+b) of gate type t
#define IMM(t) ((((t) >> 0 & 1) * 0x03) | (((t) >> 1 & 1) * 0x0C) | (((t) >> 2 & 1) * 0x30) | (((t) >> 3 & 1) * 0xC0))
#define GCASE(T) case T: for (int g = lo; g < hi; g++) { \
        __m512i a = _mm512_load_si512(in + ia[g]), b = _mm512_load_si512(in + ib[g]); \
        _mm512_store_si512(out + g, _mm512_ternarylogic_epi32(a, b, a, IMM(T))); } break;

static void ll_forward(const LLayer *l, const __m512i *in, __m512i *out) {
    const int32_t *ia = l->ia, *ib = l->ib;
    for (int t = 0; t < 16; t++) {
        const int lo = l->bounds[t], hi = l->bounds[t + 1];
        switch (t) { GCASE(0) GCASE(1) GCASE(2) GCASE(3) GCASE(4) GCASE(5) GCASE(6) GCASE(7)
                     GCASE(8) GCASE(9) GCASE(10) GCASE(11) GCASE(12) GCASE(13) GCASE(14) GCASE(15) }
    }
}

// vertical popcount of k bitplanes via carry-save adder tree -> P count planes (weights 1,2,4,..)
static int csa_count(const __m512i *in, int k, __m512i *cnt) {
    __m512i buf[2][256]; int n = k, P = 0, cur = 0;
    memcpy(buf[0], in, sizeof(__m512i) * k);
    while (n > 0) {
        __m512i *src = buf[cur], *car = buf[cur ^ 1]; int nc = 0, ns = n;
        while (ns >= 3) {
            __m512i a = src[--ns], b = src[--ns], c = src[--ns];
            src[ns++] = _mm512_ternarylogic_epi32(a, b, c, 0x96);          // sum  = a ^ b ^ c
            car[nc++] = _mm512_ternarylogic_epi32(a, b, c, 0xE8);          // carry = maj(a, b, c)
        }
        if (ns == 2) { __m512i a = src[0], b = src[1]; src[0] = _mm512_xor_si512(a, b); car[nc++] = _mm512_and_si512(a, b); ns = 1; }
        cnt[P++] = ns ? src[0] : _mm512_setzero_si512();
        cur ^= 1; n = nc;
    }
    return P;
}

// in-register 16x16 float transpose (4 stages of block swaps via permutex2var)
static inline void tr16(__m512 r[16]) {
    for (int b = 8; b >= 1; b >>= 1) {
        int lo[16], hi[16];
        for (int l = 0; l < 16; l++) { lo[l] = (l & b) ? 16 + l - b : l; hi[l] = (l & b) ? 16 + l : l + b; }
        __m512i ilo = _mm512_loadu_si512(lo), ihi = _mm512_loadu_si512(hi);
        for (int i = 0; i < 16; i++) if (!(i & b)) {
            __m512 a = r[i], c = r[i + b];
            r[i] = _mm512_permutex2var_ps(a, ilo, c); r[i + b] = _mm512_permutex2var_ps(a, ihi, c);
        }
    }
}

// x: [512][d] residual stream (fp32); adds DLGN(x) in place. d % 16 == 0. Scratch: planes A/B [max(width, d*nth)], ot [d][512]
static void dlgn_forward(const DLGN *m, float *x, float *xt, __m512i *A, __m512i *B, float *ot) {
    const int d = m->d, N = 512, nth = m->nth; (void)xt;
    float rr[512];
    for (int n = 0; n < N; n++) {                                           // per-token RMSNorm factor
        __m512 ss = _mm512_setzero_ps();
        for (int c = 0; c < d; c += 16) { __m512 v = _mm512_loadu_ps(x + n * d + c); ss = _mm512_fmadd_ps(v, v, ss); }
        rr[n] = 1.0f / sqrtf(_mm512_reduce_add_ps(ss) / d + 1e-6f);
    }
    for (int q = 0; q < N / 16; q++) {                                      // 16 tokens x 16 channels per tile
        __m512 rv = _mm512_loadu_ps(rr + 16 * q);
        for (int c0 = 0; c0 < d; c0 += 16) {
            __m512 t[16];
            for (int i = 0; i < 16; i++) t[i] = _mm512_loadu_ps(x + (size_t)(16 * q + i) * d + c0);
            tr16(t);                                                        // t[c] = channel c0+c over 16 tokens
            for (int c = 0; c < 16; c++) {
                __m512 v = _mm512_mul_ps(t[c], rv);
                for (int j = 0; j < nth; j++)
                    ((uint16_t *)(A + (c0 + c) * nth + j))[q] = _mm512_cmp_ps_mask(v, _mm512_set1_ps(m->th[j]), _CMP_GT_OQ);
            }
        }
    }
    __m512i *in = A, *out = B;
    for (int i = 0; i < m->depth; i++) { ll_forward(&m->L[i], in, out); __m512i *t = in; in = out; out = t; }
    for (int c = 0; c < d; c++) {                                           // GroupSum -> per-token counts
        __m512i cnt[16]; int P = csa_count(in + (size_t)c * m->k, m->k, cnt);
        const float s = m->gain[c] / m->k, off = -0.5f * m->gain[c];
        for (int q = 0; q < 8; q++) {                                       // 64 tokens at a time
            __m512i bytes = _mm512_setzero_si512();
            for (int p = 0; p < P; p++) {
                __mmask64 mk = ((const uint64_t *)&cnt[p])[q];
                bytes = _mm512_or_si512(bytes, _mm512_maskz_mov_epi8(mk, _mm512_set1_epi8((char)(1 << p))));
            }
            for (int u = 0; u < 4; u++) {
                __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bytes, u)));
                _mm512_store_ps(ot + (size_t)c * N + q * 64 + u * 16, _mm512_fmadd_ps(v, _mm512_set1_ps(s), _mm512_set1_ps(off)));
            }
        }
    }
    for (int q = 0; q < N / 16; q++)                                        // transposed residual add
        for (int c0 = 0; c0 < d; c0 += 16) {
            __m512 t[16];
            for (int c = 0; c < 16; c++) t[c] = _mm512_load_ps(ot + (size_t)(c0 + c) * N + 16 * q);
            tr16(t);
            for (int i = 0; i < 16; i++) {
                float *px = x + (size_t)(16 * q + i) * d + c0;
                _mm512_storeu_ps(px, _mm512_add_ps(_mm512_loadu_ps(px), t[i]));
            }
        }
}

/* ---------------- correctness checks against scalar references ---------------- */
static int check(void) {
    int bad = 0;
    { // BitLinear vs scalar
        BitLinear l; bl_init(&l, 128, 128); int N = 8;
        float *x = amalloc(4 * N * 128), *y = amalloc(4 * N * 128), *sx = amalloc(4 * N); uint8_t *xu = amalloc(N * 128);
        for (int i = 0; i < N * 128; i++) x[i] = frnd();
        bl_forward(&l, x, N, y, xu, sx, 1);
        for (int n = 0; n < N; n++) for (int m = 0; m < 128; m++) {
            int32_t acc = 0; for (int k = 0; k < 128; k++) acc += ((int)xu[n * 128 + k] - 128) * l.W[k * 128 + m];
            if (fabsf(acc * sx[n] * l.ws - y[n * 128 + m]) > 1e-3f) bad++;
        }
        printf("check bitlinear: %s\n", bad ? "FAIL" : "ok");
    }
    { // DLGN vs scalar bit-by-bit
        DLGN m; dlgn_init(&m, 16, 4, 64, 3); int N = 512, d = 16, b2 = 0;
        float *x = amalloc(4 * N * d), *x0 = amalloc(4 * N * d), *xt = amalloc(4 * N * d), *ot = amalloc(4 * N * d);
        __m512i *A = amalloc(64 * 64), *B = amalloc(64 * 64);
        for (int i = 0; i < N * d; i++) x0[i] = x[i] = frnd() * 2;
        dlgn_forward(&m, x, xt, A, B, ot);
        for (int n = 0; n < N; n++) {
            uint8_t h[64], h2[64]; float ss = 0; for (int c = 0; c < d; c++) ss += x0[n * d + c] * x0[n * d + c];
            float r = 1.0f / sqrtf(ss / d + 1e-6f);
            for (int c = 0; c < d; c++) for (int j = 0; j < 4; j++) h[c * 4 + j] = x0[n * d + c] * r > m.th[j];
            for (int L = 0; L < 3; L++) {
                for (int g = 0; g < 64; g++) { int a = h[m.L[L].ia[g]], b = h[m.L[L].ib[g]]; h2[g] = m.L[L].type[g] >> (2 * a + b) & 1; }
                memcpy(h, h2, 64);
            }
            for (int c = 0; c < d; c++) {
                int cnt = 0; for (int q = 0; q < m.k; q++) cnt += h[c * m.k + q];
                float ref = x0[n * d + c] + ((float)cnt / m.k - 0.5f) * m.gain[c];
                if (fabsf(ref - x[n * d + c]) > 1e-5f) b2++;
            }
        }
        printf("check dlgn: %s\n", b2 ? "FAIL" : "ok"); bad += b2;
    }
    { // attention vs scalar
        int N = 128, T = 64, H = 2, dh = 32, d = 64, b3 = 0;
        float *qkv = amalloc(4 * N * 3 * d), *o = amalloc(4 * N * d), *Kt = amalloc(4 * dh * T), *s = amalloc(4 * QB * T);
        for (int i = 0; i < N * 3 * d; i++) qkv[i] = frnd() * 2;
        attention(qkv, o, N, T, H, dh, Kt, s);
        for (int b = 0; b < N / T; b++) for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) {
            double sc[512], mx = -1e30, sum = 0;
            for (int j = 0; j < T; j++) { double a = 0; for (int e = 0; e < dh; e++) a += qkv[((b*T+i)*3*d) + h*dh + e] * qkv[((b*T+j)*3*d) + d + h*dh + e]; sc[j] = a / sqrt(dh); if (sc[j] > mx) mx = sc[j]; }
            for (int j = 0; j < T; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
            for (int e = 0; e < dh; e++) { double a = 0; for (int j = 0; j < T; j++) a += sc[j] / sum * qkv[((b*T+j)*3*d) + 2*d + h*dh + e];
                if (fabs(a - o[(b*T+i)*d + h*dh + e]) > 1e-4) b3++; }
        }
        printf("check attention: %s\n", b3 ? "FAIL" : "ok"); bad += b3;
    }
    return bad;
}

/* ---------------- timing ---------------- */
#define BENCH(label, reps, stmt) ({ double best = 1e9; for (int _r = 0; _r < 5; _r++) { double t0 = now(); \
        for (int _i = 0; _i < reps; _i++) { stmt; } double t = (now() - t0) / reps; if (t < best) best = t; } \
        printf("  %-38s %9.1f us / 512 tok  %8.3f us/tok\n", label, best * 1e6, best * 1e6 / 512); best; })

int main(int argc, char **argv) {
    if (check()) return 1;
    const int N = 512;
    int configs[][4] = {{64, 64, 2, 4}, {128, 128, 2, 2}};     // {d, T, n_layers_unused, heads}
    for (int ci = 0; ci < 2; ci++) {
        const int d = configs[ci][0], T = configs[ci][1], H = configs[ci][3], dh = d / H, hid = 4 * d;
        printf("\n== d=%d  heads=%d  seq=%d  ffn_hidden=%d  (512 tokens per batch, 1 thread) ==\n", d, H, T, hid);
        BitLinear qkv, o, up, down; bl_init(&qkv, d, 3 * d); bl_init(&o, d, d); bl_init(&up, d, hid); bl_init(&down, hid, d);
        float *x = amalloc(4 * (size_t)N * d), *y3 = amalloc(4 * (size_t)N * 3 * d), *att = amalloc(4 * (size_t)N * d),
              *h = amalloc(4 * (size_t)N * hid), *y = amalloc(4 * (size_t)N * d), *sx = amalloc(4 * N),
              *Kt = amalloc(4 * (size_t)dh * T), *s = amalloc(4 * (size_t)QB * T);
        uint8_t *xu = amalloc((size_t)N * hid);
        for (int i = 0; i < N * d; i++) x[i] = frnd();
        for (int i = 0; i < N * 3 * d; i++) y3[i] = frnd();
        double t_proj = BENCH("attn proj  (QKV + O BitLinear)", 200,
                              bl_forward(&qkv, x, N, y3, xu, sx, 1); bl_forward(&o, att, N, y, xu, sx, 1));
        double t_core = BENCH("attn core  (QK^T, softmax, PV fp32)", 50, attention(y3, att, N, T, H, dh, Kt, s));
        double t_mlp = BENCH("ternary MLP (up, ReLU^2, down)", 100,
                             bl_forward(&up, x, N, h, xu, sx, 1);
                             for (int i = 0; i < N * hid; i++) { float v = h[i] > 0 ? h[i] : 0; h[i] = v * v; }
                             bl_forward(&down, h, N, y, xu, sx, 1));
        double t_attn = t_proj + t_core;
        printf("  -> BitNet layer (attn + MLP): %.1f us / 512 tok\n", (t_attn + t_mlp) * 1e6);
        int widths[] = {d * 16, d * 32, d * 64};
        for (int wi = 0; wi < 3; wi++) {
            DLGN m; dlgn_init(&m, d, 8, widths[wi], 4);
            int maxw = widths[wi] > d * 8 ? widths[wi] : d * 8;
            __m512i *A = amalloc(64 * (size_t)maxw), *B = amalloc(64 * (size_t)maxw);
            float *xt = amalloc(4 * (size_t)N * d), *ot = amalloc(4 * (size_t)N * d);
            char label[64]; snprintf(label, 64, "DDLGN %dx4 = %d gates (%dk params)", widths[wi], widths[wi] * 4, widths[wi] * 16 / 1024);
            double t_dl = BENCH(label, 200, dlgn_forward(&m, x, xt, A, B, ot));
            printf("     FFN speedup vs MLP: %5.1fx   layer speedup: %4.2fx\n", t_mlp / t_dl, (t_attn + t_mlp) / (t_attn + t_dl));
        }
    }
    return 0;
}
