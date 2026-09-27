#include "wn.h"

int g_noquant = 0;

void bl_init(BitLinear *l, int K, int M, int cap, Rng *rng) {
    if (K % 16 || M % 16) { fprintf(stderr, "BitLinear: need K%%16==0, M%%16==0 (K=%d M=%d)\n", K, M); exit(1); }
    l->K = K; l->M = M; l->cap = cap;
    param_init(&l->W, (size_t)K * M, 1.0f);
    for (size_t i = 0; i < (size_t)K * M; i++) l->W.w[i] = rng_normal(rng) / sqrtf((float)K);
    l->Wt = xmalloc((size_t)K * M); l->colsum = xmalloc(4 * (size_t)M); l->WqT = xmalloc(4 * (size_t)K * M);
    l->xu = xmalloc((size_t)cap * K); l->sx = xmalloc(4 * (size_t)cap); l->r = xmalloc(4 * (size_t)cap);
    l->xq = xmalloc(4 * (size_t)cap * K); l->tmpT = xmalloc(4 * (size_t)K * M); l->dxn = xmalloc(4 * (size_t)cap * K);
}

static void bl_prepare_impl(BitLinear *l) {
    const int K = l->K, M = l->M; const float *W = l->W.w;
    if (g_noquant) { l->ws = 1.0f; transpose(W, K, M, l->WqT); return; }
    double s = 0;
    #pragma omp parallel for reduction(+:s) schedule(static) if ((size_t)K * M > 16384)
    for (size_t i = 0; i < (size_t)K * M; i++) s += fabsf(W[i]);
    l->ws = (float)(s / ((double)K * M)); if (l->ws < 1e-5f) l->ws = 1e-5f;
    const float inv = 1.0f / l->ws, ws = l->ws;
    float *Wq = l->tmpT;                                                   // scratch: row-major dequantized ternary [K][M]
    #pragma omp parallel for schedule(static) if ((size_t)K * M > 16384)
    for (int k = 0; k < K; k++)
        for (int m = 0; m < M; m++) {
            float q = rintf(W[(size_t)k * M + m] * inv); q = q > 1 ? 1 : q < -1 ? -1 : q;
            Wq[(size_t)k * M + m] = q * ws;
            l->Wt[((size_t)(k / 4) * M + m) * 4 + k % 4] = (int8_t)q;
        }
    #pragma omp parallel for schedule(static) if ((size_t)K * M > 16384)
    for (int m = 0; m < M; m++) { int cs = 0; for (int k = 0; k < K; k++) cs += l->Wt[((size_t)(k / 4) * M + m) * 4 + k % 4]; l->colsum[m] = cs; }
    transpose(Wq, K, M, l->WqT);
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
    #pragma omp parallel for collapse(2) schedule(static) if (N >= 64)
    for (int n0 = 0; n0 < N; n0 += 4)
        for (int mb = 0; mb < nmb; mb++) {
            const int m = mb * 64, nr = N - n0 < 4 ? N - n0 : 4, nc = (M - m) / 16 < 4 ? (M - m) / 16 : 4;
            if (nr == 4 && nc == 4) vnni_tile(l, n0, 4, m, 4, y);            // hot path: fully unrolled
            else vnni_tile(l, n0, nr, m, nc, y);
        }
}

static void bl_forward_impl(BitLinear *l, const float *x, int N, float *y) {
    const int K = l->K;
    if (N > l->cap) { fprintf(stderr, "BitLinear: N=%d > cap=%d\n", N, l->cap); exit(1); }
    l->x = x; l->N = N;
    #pragma omp parallel for schedule(static) if (N >= 64)
    for (int n = 0; n < N; n++) {                                          // RMSNorm folded into absmax int8 quant
        const float *xr = x + (size_t)n * K;
        __m512 vmax = _mm512_set1_ps(1e-5f), vss = _mm512_setzero_ps();
        for (int k = 0; k < K; k += 16) {
            const __m512 v = _mm512_loadu_ps(xr + k);
            vmax = _mm512_max_ps(vmax, _mm512_abs_ps(v)); vss = _mm512_fmadd_ps(v, v, vss);
        }
        const float amax = _mm512_reduce_max_ps(vmax), r = 1.0f / sqrtf(_mm512_reduce_add_ps(vss) / K + 1e-6f);
        l->r[n] = r;
        float *xq = l->xq + (size_t)n * K;
        if (g_noquant) { for (int k = 0; k < K; k++) xq[k] = xr[k] * r; continue; }
        const float sx = amax * r / 127.0f; l->sx[n] = sx;
        const __m512 s = _mm512_set1_ps(127.0f / amax), vsx = _mm512_set1_ps(sx);
        uint8_t *xu = l->xu + (size_t)n * K;
        for (int k = 0; k < K; k += 16) {
            const __m512 q = _mm512_roundscale_ps(_mm512_mul_ps(_mm512_loadu_ps(xr + k), s), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm512_storeu_ps(xq + k, _mm512_mul_ps(q, vsx));
            _mm_storeu_si128((__m128i *)(xu + k), _mm_xor_si128(_mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(q)), _mm_set1_epi8((char)0x80)));
        }
    }
    if (g_noquant) { sgemm(N, l->M, K, l->xq, K, l->W.w, l->M, y, l->M, 0); return; }
    vnni_gemm(l, N, y);
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
    PROF(P_BLB_DW, sgemm_tn(K, M, N, l->xq, K, dy, M, l->W.g, M, 1));      // dW += xq^T dy   (STE: dL/dW := dL/dWq)
    PROF(P_BLB_DX, sgemm(N, K, M, dy, M, l->WqT, K, l->dxn, K, 0));        // dxq = dy Wq^T   (STE: dL/dxn := dL/dxq)
    PROF(P_BLB_RMS, rms_backward(l, dx));
}

void bl_forward(BitLinear *l, const float *x, int N, float *y) { PROF(P_BLF, bl_forward_impl(l, x, N, y)); }
void bl_prepare(BitLinear *l) { PROF(P_PREP, bl_prepare_impl(l)); }
