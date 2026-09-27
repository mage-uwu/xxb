#include "wn.h"

/* Light DDLGN (training path). Gate g computes the relaxed truth table
     out = s00 (1-a)(1-b) + s01 (1-a) b + s10 a (1-b) + s11 a b  =  w0 + w1 a + w2 b + w3 a b,   s = sigmoid(theta)
   Residual init: theta ~ (-z, -z, z, z) so every gate starts as pass-through of input a.
   Activations are stored gate-major ([feature][token]) so each gate reads two contiguous rows. */

static inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

#define TT 32          /* token tile: all gate layers of a tile stay resident in L2 */

static float norm_quantile(double q) {                        // inverse normal CDF by bisection on erf
    double lo = -8, hi = 8;
    for (int i = 0; i < 80; i++) { double mid = 0.5 * (lo + hi); if (0.5 * erfc(-mid / sqrt(2.0)) < q) lo = mid; else hi = mid; }
    return (float)(0.5 * (lo + hi));
}

void dl_init(DDLGN *f, int d, int nth, float temp, int width, int depth, float z, float lr_mul, int cap, Rng *rng) {
    if (width % d || nth > 16) { fprintf(stderr, "DDLGN: width must be a multiple of d, nth <= 16\n"); exit(1); }
    f->d = d; f->nth = nth; f->temp = temp; f->width = width; f->depth = depth; f->k = width / d; f->cap = cap;
    for (int j = 0; j < nth; j++) f->th[j] = nth > 1 ? norm_quantile((j + 0.5) / nth) : 0.0f;
    f->ia = malloc(sizeof(int32_t *) * depth); f->ib = malloc(sizeof(int32_t *) * depth);
    f->theta = malloc(sizeof(Param) * depth);
    f->act = malloc(sizeof(float *) * (depth + 1)); f->dact = malloc(sizeof(float *) * (depth + 1));
    for (int i = 0; i < depth; i++) {
        const int n_in = i ? width : d * nth;
        f->ia[i] = xmalloc(4 * (size_t)width); f->ib[i] = xmalloc(4 * (size_t)width);
        for (int g = 0; g < width; g++) { f->ia[i][g] = rng_u64(rng) % n_in; f->ib[i][g] = rng_u64(rng) % n_in; }
        param_init(&f->theta[i], 4 * (size_t)width, lr_mul);
        for (int g = 0; g < width; g++)
            for (int e = 0; e < 4; e++) f->theta[i].w[g * 4 + e] = (e < 2 ? -z : z) + 0.1f * rng_normal(rng);
    }
    param_init(&f->gain, d, 1.0f);
    for (int c = 0; c < d; c++) f->gain.w[c] = 1.0f;
    f->act[0] = xmalloc(4 * (size_t)d * nth * cap);                    // input soft bits, gate-major [d*nth][cap]
    const size_t nt = omp_get_max_threads(), rows = width > d * nth ? width : d * nth;
    f->act[1] = xmalloc(4 * nt * depth * width * TT);                    // per-thread tile activations [depth][width][TT]
    f->dact[0] = xmalloc(4 * nt * 2 * rows * TT);                        // per-thread ping-pong tile grads
    f->xn = xmalloc(4 * (size_t)cap * d); f->r = xmalloc(4 * (size_t)cap); f->gs = xmalloc(4 * (size_t)cap * d);
    f->dxn = xmalloc(4 * (size_t)cap * d); f->xnT = xmalloc(4 * (size_t)cap * d);
    f->s4 = xmalloc(4 * (size_t)depth * width * 4);
    f->tpart = xmalloc(4 * (size_t)omp_get_max_threads() * depth * width * 4);
}

static inline void chunk(int N, int *n0, int *n1) {          // this thread's token range (multiples of 16)
    const int nt = omp_get_num_threads(), t = omp_get_thread_num();
    int c = (N + nt - 1) / nt; c = (c + 15) / 16 * 16;
    *n0 = t * c < N ? t * c : N; *n1 = *n0 + c < N ? *n0 + c : N;
}

/* forward all gate layers for tokens [n0, n0+tt) into the thread's tile buffers acts[depth][W][TT] */
static void tile_forward(const DDLGN *f, int n0, int tt, int N, float *acts) {
    const int W = f->width;
    for (int i = 0; i < f->depth; i++) {
        const float *s4 = f->s4 + (size_t)i * W * 4;
        const float *in = i ? acts + (size_t)(i - 1) * W * TT : f->act[0] + n0;
        const size_t ls = i ? TT : (size_t)N;
        float *out = acts + (size_t)i * W * TT;
        const int32_t *ia = f->ia[i], *ib = f->ib[i];
        for (int g = 0; g < W; g++) {
            const float *s = s4 + 4 * g;
            const float w0 = s[0], w1 = s[2] - s[0], w2 = s[1] - s[0], w3 = s[0] - s[1] - s[2] + s[3];
            const float *a = in + ia[g] * ls, *b = in + ib[g] * ls; float *o = out + (size_t)g * TT;
            if (tt == TT) {
                const __m512 W0 = _mm512_set1_ps(w0), W1 = _mm512_set1_ps(w1), W2 = _mm512_set1_ps(w2), W3 = _mm512_set1_ps(w3);
                for (int v = 0; v < TT; v += 16) {
                    const __m512 av = _mm512_loadu_ps(a + v), bv = _mm512_loadu_ps(b + v);
                    _mm512_store_ps(o + v, _mm512_fmadd_ps(_mm512_fmadd_ps(W3, av, W2), bv, _mm512_fmadd_ps(W1, av, W0)));
                }
            } else for (int t = 0; t < tt; t++) o[t] = w0 + w1 * a[t] + (w2 + w3 * a[t]) * b[t];
        }
    }
}

void dl_forward(DDLGN *f, const float *x, int N, float *y, int hard) {
    const int d = f->d, nth = f->nth, W = f->width, k = f->k;
    if (N > f->cap) { fprintf(stderr, "DDLGN: N > cap\n"); exit(1); }
    f->N = N;
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {                                         // RMSNorm
        const float *xr = x + (size_t)n * d; float ss = 0;
        for (int c = 0; c < d; c++) ss += xr[c] * xr[c];
        const float r = 1.0f / sqrtf(ss / d + 1e-6f); f->r[n] = r;
        for (int c = 0; c < d; c++) f->xn[(size_t)n * d + c] = xr[c] * r;
    }
    transpose(f->xn, N, d, f->xnT);
    const float it = 1.0f / f->temp;
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < d; c++)                                           // thermometer soft bits
        for (int j = 0; j < nth; j++) {
            float *o = f->act[0] + (size_t)(c * nth + j) * N; const float *xc = f->xnT + (size_t)c * N, t = f->th[j];
            if (hard) for (int n = 0; n < N; n++) o[n] = xc[n] > t ? 1.0f : 0.0f;
            else {
                int n = 0;
                for (; n + 16 <= N; n += 16)
                    _mm512_storeu_ps(o + n, sigm512(_mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(xc + n), _mm512_set1_ps(t)), _mm512_set1_ps(it))));
                for (; n < N; n++) o[n] = sigm((xc[n] - t) * it);
            }
        }
    for (int i = 0; i < f->depth; i++) {
        float *s4 = f->s4 + (size_t)i * W * 4; const float *th = f->theta[i].w;
        for (int e = 0; e < 4 * W; e++) s4[e] = hard ? (th[e] > 0) : sigm(th[e]);
    }
    const int ntile = (N + TT - 1) / TT;
    #pragma omp parallel for schedule(static)
    for (int ti = 0; ti < ntile; ti++) {
        const int n0 = ti * TT, tt = N - n0 < TT ? N - n0 : TT;
        float *acts = f->act[1] + (size_t)omp_get_thread_num() * f->depth * W * TT;
        tile_forward(f, n0, tt, N, acts);
        const float *last = acts + (size_t)(f->depth - 1) * W * TT, inv = 1.0f / k;
        for (int c = 0; c < d; c++) {                                     // GroupSum (sum k rows of the tile, vectorized)
            const float gn = f->gain.w[c];
            float sum[TT] __attribute__((aligned(64)));
            for (int t = 0; t < TT; t++) sum[t] = 0;
            for (int q = 0; q < k; q++) {
                const float *row = last + (size_t)(c * k + q) * TT;
                for (int t = 0; t < TT; t++) sum[t] += row[t];
            }
            for (int t = 0; t < tt; t++) {
                const float gs = sum[t] * inv - 0.5f;
                f->gs[(size_t)(n0 + t) * d + c] = gs; y[(size_t)(n0 + t) * d + c] = gs * gn;
            }
        }
    }
}

void dl_backward(DDLGN *f, const float *dy, float *dx) {
    const int d = f->d, nth = f->nth, W = f->width, k = f->k, N = f->N, nt = omp_get_max_threads(), depth = f->depth;
    const int n_in0 = d * nth, rows = W > n_in0 ? W : n_in0;
    for (int c = 0; c < d; c++) {                                         // gain grad
        double s = 0; for (int n = 0; n < N; n++) s += dy[(size_t)n * d + c] * f->gs[(size_t)n * d + c];
        f->gain.g[c] += (float)s;
    }
    memset(f->tpart, 0, 4 * (size_t)nt * depth * W * 4);
    const float it = 1.0f / f->temp;
    const int ntile = (N + TT - 1) / TT;
    #pragma omp parallel for schedule(static)
    for (int ti = 0; ti < ntile; ti++) {
        const int n0 = ti * TT, tt = N - n0 < TT ? N - n0 : TT, th = omp_get_thread_num();
        float *acts = f->act[1] + (size_t)th * depth * W * TT;
        float *gA = f->dact[0] + (size_t)th * 2 * rows * TT, *gB = gA + (size_t)rows * TT;
        tile_forward(f, n0, tt, N, acts);                                 // recompute this tile's activations (L2-resident)
        for (int c = 0; c < d; c++) {                                     // GroupSum backward
            const float sc = f->gain.w[c] / k;
            for (int q = 0; q < k; q++) for (int t = 0; t < tt; t++) gA[(size_t)(c * k + q) * TT + t] = dy[(size_t)(n0 + t) * d + c] * sc;
        }
        float *dout = gA, *din = gB;
        for (int i = depth - 1; i >= 0; i--) {
            const int n_in = i ? W : n_in0;
            const float *s4 = f->s4 + (size_t)i * W * 4;
            const float *in = i ? acts + (size_t)(i - 1) * W * TT : f->act[0] + n0;
            const size_t ls = i ? TT : (size_t)N;                          // row stride of the layer input
            const int32_t *ia = f->ia[i], *ib = f->ib[i];
            float *tp = f->tpart + ((size_t)th * depth + i) * W * 4;
            memset(din, 0, 4 * (size_t)n_in * TT);
            for (int g = 0; g < W; g++) {
                const float *s = s4 + 4 * g;
                const float w1 = s[2] - s[0], w2 = s[1] - s[0], w3 = s[0] - s[1] - s[2] + s[3];
                const float *a = in + ia[g] * ls, *b = in + ib[g] * ls, *go = dout + (size_t)g * TT;
                float *da = din + (size_t)ia[g] * TT, *db = din + (size_t)ib[g] * TT;
                float g0 = 0, g1 = 0, g2 = 0, g3 = 0;
                if (tt == TT && ia[g] != ib[g]) {
                    const __m512 W1 = _mm512_set1_ps(w1), W2 = _mm512_set1_ps(w2), W3 = _mm512_set1_ps(w3);
                    __m512 G0 = _mm512_setzero_ps(), G1 = G0, G2 = G0, G3 = G0;
                    for (int v = 0; v < TT; v += 16) {
                        const __m512 gv = _mm512_load_ps(go + v), an = _mm512_loadu_ps(a + v), bn = _mm512_loadu_ps(b + v), ga = _mm512_mul_ps(gv, an);
                        G0 = _mm512_add_ps(G0, gv); G1 = _mm512_add_ps(G1, ga);
                        G2 = _mm512_fmadd_ps(gv, bn, G2); G3 = _mm512_fmadd_ps(ga, bn, G3);
                        _mm512_store_ps(da + v, _mm512_fmadd_ps(gv, _mm512_fmadd_ps(W3, bn, W1), _mm512_load_ps(da + v)));
                        _mm512_store_ps(db + v, _mm512_fmadd_ps(gv, _mm512_fmadd_ps(W3, an, W2), _mm512_load_ps(db + v)));
                    }
                    g0 = _mm512_reduce_add_ps(G0); g1 = _mm512_reduce_add_ps(G1); g2 = _mm512_reduce_add_ps(G2); g3 = _mm512_reduce_add_ps(G3);
                } else if (ia[g] != ib[g]) {
                    for (int t = 0; t < tt; t++) {
                        const float gv = go[t], an = a[t], bn = b[t], ga = gv * an;
                        g0 += gv; g1 += ga; g2 += gv * bn; g3 += ga * bn;
                        da[t] += gv * (w1 + w3 * bn); db[t] += gv * (w2 + w3 * an);
                    }
                } else {
                    for (int t = 0; t < tt; t++) {
                        const float gv = go[t], an = a[t], ga = gv * an;
                        g0 += gv; g1 += ga; g2 += ga; g3 += ga * an;
                        da[t] += gv * (w1 + w2 + 2 * w3 * an);
                    }
                }
                tp[4 * g] += g0; tp[4 * g + 1] += g1; tp[4 * g + 2] += g2; tp[4 * g + 3] += g3;
            }
            float *tmp = dout; dout = din; din = tmp;
        }
        for (int c = 0; c < d; c++)                                       // thermometer backward -> dxn rows of this tile
            for (int t = 0; t < tt; t++) {
                float acc = 0;
                for (int j = 0; j < nth; j++) {
                    const float p = f->act[0][(size_t)(c * nth + j) * N + n0 + t];
                    acc += dout[(size_t)(c * nth + j) * TT + t] * p * (1 - p);
                }
                f->dxn[(size_t)(n0 + t) * d + c] = acc * it;
            }
    }
    for (int i = 0; i < depth; i++) {                                     // reduce theta grads over threads
        const float *s4 = f->s4 + (size_t)i * W * 4; float *tg = f->theta[i].g;
        for (int g = 0; g < W; g++) {
            float G[4] = {0, 0, 0, 0};
            for (int t = 0; t < nt; t++) for (int e = 0; e < 4; e++) G[e] += f->tpart[(((size_t)t * depth + i) * W + g) * 4 + e];
            // chain rule from bilinear coefficients (w0..w3) to table entries (s00, s01, s10, s11)
            const float ds[4] = {G[0] - G[1] - G[2] + G[3], G[2] - G[3], G[1] - G[3], G[3]};
            for (int e = 0; e < 4; e++) { const float sv = s4[4 * g + e]; tg[4 * g + e] += ds[e] * sv * (1 - sv); }
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

float dl_gates_changed(const DDLGN *f) {
    size_t changed = 0, total = 0;
    for (int i = 0; i < f->depth; i++)
        for (int g = 0; g < f->width; g++, total++) {
            const float *t = f->theta[i].w + 4 * g;
            changed += !((t[0] <= 0) && (t[1] <= 0) && (t[2] > 0) && (t[3] > 0));
        }
    return (float)changed / total;
}
