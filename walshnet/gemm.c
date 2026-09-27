#include "wn.h"
#include <time.h>
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

static inline void tr16(__m512 r[16]) {                  // in-register 16x16 transpose (4 permute stages)
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
    static float *buf = NULL; static size_t cap = 0;
    const size_t need = (size_t)M * K;
    if (need > cap) { free(buf); cap = need; buf = xmalloc(4 * cap); }
    if (lda != M) { fprintf(stderr, "sgemm_tn: lda must equal M\n"); exit(1); }
    transpose(A, K, M, buf);
    gemm_core(M, N, K, buf, K, B, ldb, C, ldc, accumulate);
}
