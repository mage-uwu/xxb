#include "wn.h"

/* Walsh mixer (Hyena-style block, dyadic long conv):
     [u | g] = BitLinear(x);  us = causal 3-tap depthwise conv(u);
     causal: v[t] = sum_{s<=t} h[t^s] us[s]        bidir: v[t] = sum_s h[t^s] us[s] = WHT^-1(WHT(h) * WHT(us))
     y = BitLinear(v * g)
   Layout: [token][channel], sequences of T (power of 2) contiguous tokens; every inner loop is over channels. */

#define BASE 8          /* causal divide & conquer: blocks of <= BASE tokens are done directly */

void fwht_rows(float *x, int n, int d) {
    for (int h = 1; h < n; h <<= 1)
        for (int i = 0; i < n; i += 2 * h)
            for (int j = i; j < i + h; j++) {
                float *a = x + (size_t)j * d, *b = x + (size_t)(j + h) * d;
                for (int c = 0; c < d; c += 16) {
                    __m512 va = _mm512_loadu_ps(a + c), vb = _mm512_loadu_ps(b + c);
                    _mm512_storeu_ps(a + c, _mm512_add_ps(va, vb)); _mm512_storeu_ps(b + c, _mm512_sub_ps(va, vb));
                }
            }
}

static inline void vmul(float *o, const float *a, const float *b, size_t n) {       // o = a * b
    for (size_t i = 0; i < n; i += 16) _mm512_storeu_ps(o + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
}
static inline void vfma(float *o, const float *a, const float *b, size_t n) {       // o += a * b
    for (size_t i = 0; i < n; i += 16)
        _mm512_storeu_ps(o + i, _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), _mm512_loadu_ps(o + i)));
}
static inline void vadd(float *o, const float *a, size_t n) {
    for (size_t i = 0; i < n; i += 16) _mm512_storeu_ps(o + i, _mm512_add_ps(_mm512_loadu_ps(o + i), _mm512_loadu_ps(a + i)));
}

static int ilog2(int n) { int l = 0; while ((1 << l) < n) l++; return l; }

void wm_init(WalshMix *m, int d, int T, int mode, int cap, Rng *rng) {
    if (T & (T - 1)) { fprintf(stderr, "WalshMix: T must be a power of 2\n"); exit(1); }
    if (d % 16) { fprintf(stderr, "WalshMix: d must be a multiple of 16\n"); exit(1); }
    m->d = d; m->T = T; m->mode = mode; m->cap = cap;
    bl_init(&m->inp, d, 2 * d, cap, rng); bl_init(&m->out, d, d, cap, rng);
    param_init(&m->sw, 3 * (size_t)d, 1.0f);
    for (int j = 0; j < 3; j++) for (int c = 0; c < d; c++) m->sw.w[j * d + c] = (j == 0) + 0.02f * rng_normal(rng);
    param_init(&m->h, (size_t)T * d, 1.0f);
    for (int k = 0; k < T; k++) for (int c = 0; c < d; c++) m->h.w[k * d + c] = (k == 0) + 0.02f * rng_normal(rng);
    m->Hs = xmalloc(4 * (size_t)T * d);
    for (int n = 2 * BASE, lv = 0, off = 0; n <= T; n <<= 1, lv++) { m->lv_off[lv] = off; off += n / 2; }
    const size_t nd = (size_t)cap * d; const int nt = omp_get_max_threads();
    m->ug = xmalloc(8 * nd); m->us = xmalloc(4 * nd); m->v = xmalloc(4 * nd); m->z = xmalloc(4 * nd); m->U = xmalloc(4 * nd);
    m->dz = xmalloc(4 * nd); m->dug = xmalloc(8 * nd); m->dus = xmalloc(4 * nd); m->dv = xmalloc(4 * nd);
    m->dh_acc = xmalloc(4 * (size_t)nt * 2 * T * d); m->scratch = xmalloc(4 * (size_t)nt * 4 * T * d);
}

void wm_prepare(WalshMix *m) {
    const int T = m->T, d = m->d;
    bl_prepare(&m->inp); bl_prepare(&m->out);
    if (m->mode == MODE_BIDIR) {
        memcpy(m->Hs, m->h.w, 4 * (size_t)T * d); fwht_rows(m->Hs, T, d);
        for (size_t i = 0; i < (size_t)T * d; i++) m->Hs[i] /= T;
    } else {
        for (int n = 2 * BASE, lv = 0; n <= T; n <<= 1, lv++) {
            const int mh = n / 2; float *S = m->Hs + (size_t)m->lv_off[lv] * d;
            memcpy(S, m->h.w + (size_t)mh * d, 4 * (size_t)mh * d); fwht_rows(S, mh, d);
            for (size_t i = 0; i < (size_t)mh * d; i++) S[i] /= mh;
        }
    }
}

/* ---- causal dyadic conv, divide and conquer ---- */
static void causal_fwd(const WalshMix *m, const float *u, float *y, int n, int lv, float *scr) {
    const int d = m->d;
    if (n <= BASE) {
        for (int t = 0; t < n; t++)
            for (int s = 0; s <= t; s++) vfma(y + (size_t)t * d, m->h.w + (size_t)(t ^ s) * d, u + (size_t)s * d, d);
        return;
    }
    const int mh = n / 2; const size_t md = (size_t)mh * d;
    causal_fwd(m, u, y, mh, lv - 1, scr);
    causal_fwd(m, u + md, y + md, mh, lv - 1, scr);
    memcpy(scr, u, 4 * md); fwht_rows(scr, mh, d);                       // y_R += WHT(Hhi * WHT(u_L))
    vmul(scr, scr, m->Hs + (size_t)m->lv_off[lv] * d, md); fwht_rows(scr, mh, d);
    vadd(y + md, scr, md);
}

// du += causal^T(dy);  spec[lv] += WHT(u_L) * WHT(dy_R) per level;  dhd[k] (k < BASE) += direct-part kernel grads
static void causal_bwd(const WalshMix *m, const float *u, const float *dy, float *du, int n, int lv, float *spec, float *dhd, float *scr) {
    const int d = m->d;
    if (n <= BASE) {
        for (int t = 0; t < n; t++)
            for (int s = 0; s <= t; s++) {
                vfma(du + (size_t)s * d, m->h.w + (size_t)(t ^ s) * d, dy + (size_t)t * d, d);
                vfma(dhd + (size_t)(t ^ s) * d, dy + (size_t)t * d, u + (size_t)s * d, d);
            }
        return;
    }
    const int mh = n / 2; const size_t md = (size_t)mh * d;
    causal_bwd(m, u, dy, du, mh, lv - 1, spec, dhd, scr);
    causal_bwd(m, u + md, dy + md, du + md, mh, lv - 1, spec, dhd, scr);
    float *F1 = scr, *F2 = scr + md;
    memcpy(F1, u, 4 * md); fwht_rows(F1, mh, d);                          // WHT(u_L)
    memcpy(F2, dy + md, 4 * md); fwht_rows(F2, mh, d);                    // WHT(dy_R)
    vfma(spec + (size_t)m->lv_off[lv] * d, F1, F2, md);                   // kernel-grad spectrum for h[n/2 .. n)
    vmul(F2, F2, m->Hs + (size_t)m->lv_off[lv] * d, md); fwht_rows(F2, mh, d);
    vadd(du, F2, md);                                                     // du_L += WHT(Hhi * WHT(dy_R))
}

static void short_fwd(WalshMix *m, int N) {                                  // us = causal 3-tap depthwise conv(u)
    const int d = m->d, T = m->T, d2 = 2 * d; const float *sw = m->sw.w;
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const int t = n % T; float *o = m->us + (size_t)n * d;
        for (int c = 0; c < d; c += 16) {
            __m512 a = _mm512_mul_ps(_mm512_loadu_ps(sw + c), _mm512_loadu_ps(m->ug + (size_t)n * d2 + c));
            if (t >= 1) a = _mm512_fmadd_ps(_mm512_loadu_ps(sw + d + c), _mm512_loadu_ps(m->ug + (size_t)(n - 1) * d2 + c), a);
            if (t >= 2) a = _mm512_fmadd_ps(_mm512_loadu_ps(sw + 2 * d + c), _mm512_loadu_ps(m->ug + (size_t)(n - 2) * d2 + c), a);
            _mm512_storeu_ps(o + c, a);
        }
    }
}

static void long_fwd(WalshMix *m, int N) {                                   // v = dyadic long conv(us), per sequence
    const int d = m->d, T = m->T, top = ilog2(T) - ilog2(2 * BASE); const size_t Td = (size_t)T * d;
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < N / T; b++) {
        const size_t off = (size_t)b * Td;
        if (m->mode == MODE_BIDIR) {
            memcpy(m->U + off, m->us + off, 4 * Td); fwht_rows(m->U + off, T, d);
            vmul(m->v + off, m->U + off, m->Hs, Td); fwht_rows(m->v + off, T, d);
        } else {
            float *scr = m->scratch + (size_t)omp_get_thread_num() * 4 * Td;
            memset(m->v + off, 0, 4 * Td);
            causal_fwd(m, m->us + off, m->v + off, T, top, scr);
        }
    }
}

static void gate_fwd(WalshMix *m, int N) {
    const int d = m->d, d2 = 2 * d;
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) vmul(m->z + (size_t)n * d, m->v + (size_t)n * d, m->ug + (size_t)n * d2 + d, d);
}

void wm_forward(WalshMix *m, const float *x, int N, float *y) {
    if (N % m->T) { fprintf(stderr, "WalshMix: N must be a multiple of T\n"); exit(1); }
    bl_forward(&m->inp, x, N, m->ug);
    PROF(P_SHORT, short_fwd(m, N));
    PROF(P_LONGF, long_fwd(m, N));
    PROF(P_GATE, gate_fwd(m, N));
    bl_forward(&m->out, m->z, N, y);
}

static void gate_bwd(WalshMix *m, int N) {
    const int d = m->d, d2 = 2 * d;
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        vmul(m->dv + (size_t)n * d, m->dz + (size_t)n * d, m->ug + (size_t)n * d2 + d, d);
        vmul(m->dug + (size_t)n * d2 + d, m->dz + (size_t)n * d, m->v + (size_t)n * d, d);
    }
}

static void long_bwd(WalshMix *m, int N) {                                   // dus, and kernel grads into h.g
    const int d = m->d, T = m->T, nt = omp_get_max_threads(), top = ilog2(T) - ilog2(2 * BASE);
    const size_t Td = (size_t)T * d;
    memset(m->dh_acc, 0, 4 * (size_t)nt * 2 * Td);
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < N / T; b++) {
        const size_t off = (size_t)b * Td; const int th = omp_get_thread_num();
        float *acc = m->dh_acc + (size_t)th * 2 * Td, *scr = m->scratch + (size_t)th * 4 * Td;
        if (m->mode == MODE_BIDIR) {
            float *DV = scr; memcpy(DV, m->dv + off, 4 * Td); fwht_rows(DV, T, d);
            vfma(acc, m->U + off, DV, Td);                                    // spectrum of dL/dh
            vmul(m->dus + off, DV, m->Hs, Td); fwht_rows(m->dus + off, T, d);  // symmetric operator
        } else {
            memset(m->dus + off, 0, 4 * Td);
            causal_bwd(m, m->us + off, m->dv + off, m->dus + off, T, top, acc, acc + Td, scr);
        }
    }
    float *S = m->dh_acc;                                                     // reduce thread partials
    for (int t = 1; t < nt; t++) vadd(S, m->dh_acc + (size_t)t * 2 * Td, 2 * Td);
    if (m->mode == MODE_BIDIR) {
        fwht_rows(S, T, d);
        for (size_t i = 0; i < Td; i++) m->h.g[i] += S[i] / T;
    } else {
        for (size_t i = 0; i < (size_t)BASE * d && i < Td; i++) m->h.g[i] += S[Td + i];
        for (int n = 2 * BASE, lv = 0; n <= T; n <<= 1, lv++) {
            const int mh = n / 2; float *Sl = S + (size_t)m->lv_off[lv] * d;
            fwht_rows(Sl, mh, d);
            for (size_t i = 0; i < (size_t)mh * d; i++) m->h.g[(size_t)mh * d + i] += Sl[i] / mh;
        }
    }
}

static void short_bwd(WalshMix *m, int N) {
    const int d = m->d, T = m->T, d2 = 2 * d; const float *sw = m->sw.w;
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {                                             // du
        const int t = n % T; float *o = m->dug + (size_t)n * d2;
        for (int c = 0; c < d; c += 16) {
            __m512 a = _mm512_mul_ps(_mm512_loadu_ps(sw + c), _mm512_loadu_ps(m->dus + (size_t)n * d + c));
            if (t + 1 < T) a = _mm512_fmadd_ps(_mm512_loadu_ps(sw + d + c), _mm512_loadu_ps(m->dus + (size_t)(n + 1) * d + c), a);
            if (t + 2 < T) a = _mm512_fmadd_ps(_mm512_loadu_ps(sw + 2 * d + c), _mm512_loadu_ps(m->dus + (size_t)(n + 2) * d + c), a);
            _mm512_storeu_ps(o + c, a);
        }
    }
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < d; c += 16) {                                         // taps
        __m512 g0 = _mm512_setzero_ps(), g1 = g0, g2 = g0;
        for (int n = 0; n < N; n++) {
            const int t = n % T; __m512 gd = _mm512_loadu_ps(m->dus + (size_t)n * d + c);
            g0 = _mm512_fmadd_ps(gd, _mm512_loadu_ps(m->ug + (size_t)n * d2 + c), g0);
            if (t >= 1) g1 = _mm512_fmadd_ps(gd, _mm512_loadu_ps(m->ug + (size_t)(n - 1) * d2 + c), g1);
            if (t >= 2) g2 = _mm512_fmadd_ps(gd, _mm512_loadu_ps(m->ug + (size_t)(n - 2) * d2 + c), g2);
        }
        float *gw = m->sw.g;
        _mm512_storeu_ps(gw + c, _mm512_add_ps(_mm512_loadu_ps(gw + c), g0));
        _mm512_storeu_ps(gw + d + c, _mm512_add_ps(_mm512_loadu_ps(gw + d + c), g1));
        _mm512_storeu_ps(gw + 2 * d + c, _mm512_add_ps(_mm512_loadu_ps(gw + 2 * d + c), g2));
    }
}

void wm_backward(WalshMix *m, const float *dy, float *dx) {
    const int N = m->inp.N;
    bl_backward(&m->out, dy, m->dz);
    PROF(P_GATE, gate_bwd(m, N));
    PROF(P_LONGB, long_bwd(m, N));
    PROF(P_SHORT, short_bwd(m, N));
    bl_backward(&m->inp, m->dug, dx);
}
