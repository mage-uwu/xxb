#include "wn.h"

/* CLOPEN FFN (training + float inference path).
   States are exactly {-1,+1}; weights are exactly {-1,0,+1}; the forward pass is the deployed function.
   Activations are gate-major ([feature][token]); each thread runs token tiles of TT through all layers so the
   working set stays in L2, and the backward pass recomputes the tile instead of storing activations. */

#define TT 32

static float norm_quantile(double q) {
    double lo = -8, hi = 8;
    for (int i = 0; i < 80; i++) { double mid = 0.5 * (lo + hi); if (0.5 * erfc(-mid / sqrt(2.0)) < q) lo = mid; else hi = mid; }
    return (float)(0.5 * (lo + hi));
}

void cl_init(Clopen *f, int d, int nth, float temp, int width, int depth, int fanin, int ste, float lr_mul, int cap, Rng *rng) {
    if (width % d || nth > 16 || fanin < 1 || fanin > 16) { fprintf(stderr, "CLOPEN: width %% d, nth <= 16, 1 <= fanin <= 16\n"); exit(1); }
    f->d = d; f->nth = nth; f->temp = temp; f->width = width; f->depth = depth; f->fanin = fanin; f->ste = ste;
    f->k = width / d; f->cap = cap;
    for (int j = 0; j < nth; j++) f->th[j] = nth > 1 ? norm_quantile((j + 0.5) / nth) : 0.0f;
    f->idx = malloc(sizeof(int32_t *) * depth); f->w = malloc(sizeof(Param) * depth); f->theta = malloc(sizeof(Param) * depth);
    for (int i = 0; i < depth; i++) {
        const int n_in = i ? width : d * nth;
        f->idx[i] = xmalloc(4 * (size_t)width * fanin);
        for (size_t e = 0; e < (size_t)width * fanin; e++) f->idx[i][e] = rng_u64(rng) % n_in;
        param_init(&f->w[i], (size_t)width * fanin, lr_mul);
        param_init(&f->theta[i], width, lr_mul);
        for (int g = 0; g < width; g++)                                   // identity init: pass-through of input 0
            for (int j = 0; j < fanin; j++) f->w[i].w[g * fanin + j] = (j == 0) + 0.1f * rng_normal(rng);
    }
    param_init(&f->gain, d, 1.0f);
    for (int c = 0; c < d; c++) f->gain.w[c] = 1.0f;
    const size_t nt = omp_get_max_threads(), rows = width > d * nth ? width : d * nth;
    f->wq = xmalloc(4 * (size_t)depth * width * fanin);
    f->xn = xmalloc(4 * (size_t)cap * d); f->r = xmalloc(4 * (size_t)cap); f->xnT = xmalloc(4 * (size_t)cap * d);
    f->in0 = xmalloc(4 * (size_t)d * nth * cap); f->gs = xmalloc(4 * (size_t)cap * d); f->dxn = xmalloc(4 * (size_t)cap * d);
    f->acts = xmalloc(4 * nt * depth * width * TT); f->pre = xmalloc(4 * nt * depth * width * TT);
    f->gbuf = xmalloc(4 * nt * 2 * rows * TT);
    f->wpart = xmalloc(4 * nt * depth * width * (fanin + 1));
}

static inline float q3(float w) { const float q = rintf(w); return q > 1 ? 1 : q < -1 ? -1 : q; }

/* all layers for tokens [n0, n0+tt) -> acts/pre tile buffers [depth][W][TT] */
static void tile_forward(const Clopen *f, int n0, int tt, int N, float *acts, float *pre) {
    const int W = f->width, G = f->fanin;
    for (int i = 0; i < f->depth; i++) {
        const float *in = i ? acts + (size_t)(i - 1) * W * TT : f->in0 + n0;
        const size_t ls = i ? TT : (size_t)N;
        float *out = acts + (size_t)i * W * TT, *pr = pre + (size_t)i * W * TT;
        const int32_t *idx = f->idx[i]; const float *wq = f->wq + (size_t)i * W * G, *th = f->theta[i].w;
        for (int g = 0; g < W; g++) {
            float *o = out + (size_t)g * TT, *p = pr + (size_t)g * TT;
            if (tt == TT) {
                __m512 a0 = _mm512_set1_ps(-th[g]), a1 = a0;
                for (int j = 0; j < G; j++) {
                    const float wj = wq[g * G + j]; if (wj == 0) continue;      // ternary sparsity: skip zeros
                    const float *x = in + idx[g * G + j] * ls; const __m512 wv = _mm512_set1_ps(wj);
                    a0 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(x), a0); a1 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(x + 16), a1);
                }
                _mm512_store_ps(p, a0); _mm512_store_ps(p + 16, a1);
                const __m512 one = _mm512_set1_ps(1.0f), mone = _mm512_set1_ps(-1.0f);
                _mm512_store_ps(o, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(a0, _mm512_setzero_ps(), _CMP_GE_OQ), mone, one));
                _mm512_store_ps(o + 16, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(a1, _mm512_setzero_ps(), _CMP_GE_OQ), mone, one));
            } else {
                for (int t = 0; t < tt; t++) {
                    float y = -th[g];
                    for (int j = 0; j < G; j++) y += wq[g * G + j] * in[idx[g * G + j] * ls + t];
                    p[t] = y; o[t] = y >= 0 ? 1.0f : -1.0f;
                }
            }
        }
    }
}

void cl_forward(Clopen *f, const float *x, int N, float *y) {
    const int d = f->d, nth = f->nth, W = f->width, k = f->k;
    if (N > f->cap) { fprintf(stderr, "CLOPEN: N > cap\n"); exit(1); }
    f->N = N;
    for (int i = 0; i < f->depth; i++)
        for (size_t e = 0; e < (size_t)W * f->fanin; e++) f->wq[(size_t)i * W * f->fanin + e] = q3(f->w[i].w[e]);
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {                                         // RMSNorm
        const float *xr = x + (size_t)n * d; float ss = 0;
        for (int c = 0; c < d; c++) ss += xr[c] * xr[c];
        const float r = 1.0f / sqrtf(ss / d + 1e-6f); f->r[n] = r;
        for (int c = 0; c < d; c++) f->xn[(size_t)n * d + c] = xr[c] * r;
    }
    transpose(f->xn, N, d, f->xnT);
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < d; c++)                                           // thermometer sign bits
        for (int j = 0; j < nth; j++) {
            float *o = f->in0 + (size_t)(c * nth + j) * N; const float *xc = f->xnT + (size_t)c * N, t = f->th[j];
            for (int n = 0; n < N; n++) o[n] = xc[n] > t ? 1.0f : -1.0f;
        }
    const int ntile = (N + TT - 1) / TT;
    #pragma omp parallel for schedule(static)
    for (int ti = 0; ti < ntile; ti++) {
        const int n0 = ti * TT, tt = N - n0 < TT ? N - n0 : TT, th = omp_get_thread_num();
        float *acts = f->acts + (size_t)th * f->depth * W * TT, *pre = f->pre + (size_t)th * f->depth * W * TT;
        tile_forward(f, n0, tt, N, acts, pre);
        const float *last = acts + (size_t)(f->depth - 1) * W * TT, sc = 0.5f / k;
        for (int c = 0; c < d; c++) {                                     // GroupSum: 0.5 * mean of k signs
            float sum[TT] __attribute__((aligned(64)));
            for (int t = 0; t < TT; t++) sum[t] = 0;
            for (int q = 0; q < k; q++) { const float *row = last + (size_t)(c * k + q) * TT; for (int t = 0; t < TT; t++) sum[t] += row[t]; }
            for (int t = 0; t < tt; t++) {
                const float gs = sum[t] * sc; f->gs[(size_t)(n0 + t) * d + c] = gs; y[(size_t)(n0 + t) * d + c] = gs * f->gain.w[c];
            }
        }
    }
}

void cl_backward(Clopen *f, const float *dy, float *dx) {
    const int d = f->d, nth = f->nth, W = f->width, k = f->k, N = f->N, G = f->fanin, depth = f->depth, nt = omp_get_max_threads();
    const int n_in0 = d * nth, rows = W > n_in0 ? W : n_in0, clip = f->ste;
    for (int c = 0; c < d; c++) {
        double s = 0; for (int n = 0; n < N; n++) s += dy[(size_t)n * d + c] * f->gs[(size_t)n * d + c];
        f->gain.g[c] += (float)s;
    }
    memset(f->wpart, 0, 4 * (size_t)nt * depth * W * (G + 1));
    const float it = 1.0f / f->temp;
    const int ntile = (N + TT - 1) / TT;
    #pragma omp parallel for schedule(static)
    for (int ti = 0; ti < ntile; ti++) {
        const int n0 = ti * TT, tt = N - n0 < TT ? N - n0 : TT, th = omp_get_thread_num();
        float *acts = f->acts + (size_t)th * depth * W * TT, *pre = f->pre + (size_t)th * depth * W * TT;
        float *gA = f->gbuf + (size_t)th * 2 * rows * TT, *gB = gA + (size_t)rows * TT;
        tile_forward(f, n0, tt, N, acts, pre);                            // recompute (L2-resident)
        for (int c = 0; c < d; c++) {                                     // GroupSum backward
            const float sc = f->gain.w[c] * 0.5f / k;
            for (int q = 0; q < k; q++) {
                float *row = gA + (size_t)(c * k + q) * TT;
                for (int t = 0; t < TT; t++) row[t] = t < tt ? dy[(size_t)(n0 + t) * d + c] * sc : 0.0f;
            }
        }
        float *dout = gA, *din = gB;
        for (int i = depth - 1; i >= 0; i--) {
            const int n_in = i ? W : n_in0;
            const float *in = i ? acts + (size_t)(i - 1) * W * TT : f->in0 + n0;
            const size_t ls = i ? TT : (size_t)N;
            const int32_t *idx = f->idx[i]; const float *wq = f->wq + (size_t)i * W * G, *pr = pre + (size_t)i * W * TT;
            float *wp = f->wpart + ((size_t)th * depth + i) * W * (G + 1);
            memset(din, 0, 4 * (size_t)n_in * TT);
            for (int g = 0; g < W; g++) {
                const float *go = dout + (size_t)g * TT, *p = pr + (size_t)g * TT;
                if (tt == TT) {                                           // vectorized full tile (2 x 16 tokens)
                    __m512 g0 = _mm512_load_ps(go), g1 = _mm512_load_ps(go + 16);
                    if (clip) {                                           // STE window |pre| <= 1
                        const __m512 one = _mm512_set1_ps(1.0f);
                        g0 = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_load_ps(p)), one, _CMP_LE_OQ), g0);
                        g1 = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_load_ps(p + 16)), one, _CMP_LE_OQ), g1);
                    }
                    wp[g * (G + 1) + G] -= _mm512_reduce_add_ps(_mm512_add_ps(g0, g1));
                    for (int j = 0; j < G; j++) {
                        const int src = idx[g * G + j]; const float *x = in + src * ls; const float wj = wq[g * G + j];
                        wp[g * (G + 1) + j] += _mm512_reduce_add_ps(_mm512_fmadd_ps(g0, _mm512_loadu_ps(x), _mm512_mul_ps(g1, _mm512_loadu_ps(x + 16))));
                        if (wj != 0) {
                            float *dxr = din + (size_t)src * TT; const __m512 wv = _mm512_set1_ps(wj);
                            _mm512_store_ps(dxr, _mm512_fmadd_ps(g0, wv, _mm512_load_ps(dxr)));
                            _mm512_store_ps(dxr + 16, _mm512_fmadd_ps(g1, wv, _mm512_load_ps(dxr + 16)));
                        }
                    }
                    continue;
                }
                float gy[TT] __attribute__((aligned(64)));
                float st = 0;
                for (int t = 0; t < TT; t++) {                            // STE through sign
                    gy[t] = t < tt ? go[t] * (clip ? (fabsf(p[t]) <= 1.0f) : 1.0f) : 0.0f;
                    st += gy[t];
                }
                wp[g * (G + 1) + G] -= st;                                // d theta
                for (int j = 0; j < G; j++) {
                    const int src = idx[g * G + j]; const float *x = in + src * ls; float *dxr = din + (size_t)src * TT;
                    float sw = 0; const float wj = wq[g * G + j];
                    for (int t = 0; t < tt; t++) { sw += gy[t] * x[t]; dxr[t] += gy[t] * wj; }
                    wp[g * (G + 1) + j] += sw;                            // d latent weight (STE through q3)
                }
            }
            float *tmp = dout; dout = din; din = tmp;
        }
        for (int c = 0; c < d; c++)                                       // thermometer sign backward (STE)
            for (int t = 0; t < tt; t++) {
                float acc = 0;
                for (int j = 0; j < nth; j++) {
                    const float u = (f->xn[(size_t)(n0 + t) * d + c] - f->th[j]) * it;
                    acc += dout[(size_t)(c * nth + j) * TT + t] * (clip ? (fabsf(u) <= 1.0f) : 1.0f);
                }
                f->dxn[(size_t)(n0 + t) * d + c] = acc * it;
            }
    }
    for (int i = 0; i < depth; i++)                                       // reduce partial grads
        for (int g = 0; g < W; g++) {
            for (int j = 0; j <= G; j++) {
                float s = 0; for (int t = 0; t < nt; t++) s += f->wpart[(((size_t)t * depth + i) * W + g) * (G + 1) + j];
                if (j < G) f->w[i].g[g * G + j] += s; else f->theta[i].g[g] += s;
            }
        }
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {                                         // RMSNorm backward
        const float r = f->r[n], *xn = f->xn + (size_t)n * d, *g = f->dxn + (size_t)n * d;
        float dot = 0; for (int c = 0; c < d; c++) dot += g[c] * xn[c];
        dot /= d;
        for (int c = 0; c < d; c++) dx[(size_t)n * d + c] = r * (g[c] - xn[c] * dot);
    }
}

float cl_gates_changed(const Clopen *f) {
    size_t ch = 0, tot = 0; const int G = f->fanin;
    for (int i = 0; i < f->depth; i++)
        for (int g = 0; g < f->width; g++, tot++) {
            int same = fabsf(f->theta[i].w[g]) < 1.0f && q3(f->w[i].w[g * G]) == 1;
            for (int j = 1; j < G && same; j++) same = q3(f->w[i].w[g * G + j]) == 0;
            ch += !same;
        }
    return (float)ch / tot;
}
