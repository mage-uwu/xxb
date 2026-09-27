#include "wn.h"

/* Gated short-conv block:  [u | g] = BitLinear_in(x);  us[t] = sum_j cw[j] u[t + off(j)];  y = BitLinear_out(us * g)
   Taps that fall outside the current sequence of T tokens read zeros. Layout [token][channel]; loops over channels. */

void blk_init(Block *b, int d, int T, int K, int mode, int cap, Rng *rng) {
    if (d % 16) { fprintf(stderr, "block: d must be a multiple of 16\n"); exit(1); }
    if (mode == MODE_BIDIR && !(K & 1)) { fprintf(stderr, "block: bidirectional mode needs an odd K\n"); exit(1); }
    b->d = d; b->T = T; b->K = K; b->mode = mode; b->cap = cap;
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
                const int o = tap_off(b->mode, K, j);
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
                    const int o = tap_off(b->mode, K, j);
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
