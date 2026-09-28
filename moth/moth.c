// moth.c: Moth, a tiny ternary Monarch Mixer char-level LM in one file of pure C (nanoGPT style).
//
//   block(x):  x += Mon_o( Mon_a(n) * conv(Mon_v(n)) ),  n = rms(x)       sequence mixer (gated causal long conv)
//              x += Mon_d( relu(Mon_u(n))^2 ),           n = rms(x)       channel mixer
//   Mon: view a width-d vector as an MxM tile (d = M^2). R mixes along each row, L along each column.
//        That is P L P^T R. The permutations are never materialised: a column op on a row-major tile is
//        lane-wise vector math with weights stored [o][i][lane], so every loop streams contiguous int8.
//
// forward : ternary {-1,0,1} weights (absmean) x int8 activations (per-token absmax), STE. A ternary dot
//           over 16 (Monarch) or 128 (conv) int8 terms fits in int16, so those accumulators use 32 lanes, not 16.
//           Absmax codes are scale-invariant, so RMSNorm, R->L requantisation and the gate all fold into
//           per-token scales. No float tensor is written that the backward doesn't need. The conv input uses a
//           static EMA scale, so it stays strictly causal.
// backward: gradients are int8 with a delayed per-tensor scale (last step's amax, as in FP8 training) and
//           stochastic rounding. There is no amax pre-pass and no barrier inside a Monarch. The input's per-token
//           scale is folded into the gradient, so dW = g8^T x8 reuses the forward's int8 codes.
//           dx = W^T g8 (ternary x int8, int16) and dW (int8 x int8) accumulate in per-thread int32 slabs.
//
// cc -O3 -march=native -fopenmp moth.c -o moth -lm && ./moth input.txt
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#define TID omp_get_thread_num()
#else
#define TID 0
#define omp_get_max_threads() 1
#endif

#define M 16            // monarch block size
#define D (M * M)       // model width
#define W3 (M * M * M)  // weights per monarch factor
#define L 4             // layers
#define T 128           // context = long-conv length
#define B 16            // batch size
#define N (B * T)       // tokens per batch
#define STEPS 3000
#define LR 3e-3f

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static float urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 40) / 16777216.f; }
static float randn(void) { return sqrtf(-2 * logf(urand() + 1e-9f)) * cosf(6.2831853f * urand()); }
static uint32_t hash(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16); }
static float *fa(size_t n) { return calloc(n, sizeof(float)); }
static int8_t *ia(size_t n) { return calloc(n, 1); }
static int NT;          // threads
static uint32_t seed;   // per-step stochastic rounding seed

typedef struct { float *w, *g, *m, *v; int n; } P;   // master weight, grad, adam moments
static P ps[64]; static int np;
static P *param(int n, float sd) {
    P *p = &ps[np++]; p->n = n; p->w = fa(n); p->g = fa(n); p->m = fa(n); p->v = fa(n);
    for (int i = 0; i < n; i++) p->w[i] = sd * randn();
    return p;
}

// ---- quantizers ------------------------------------------------------------------------------
static float tern(const float *w, int8_t *q, int n, int st) {        // absmean ternary, returns scale
    float s = 1e-8f; for (int i = 0; i < n; i++) s += fabsf(w[i * st]); s /= n;
    for (int i = 0; i < n; i++) { float r = roundf(w[i * st] / s); q[i * st] = r > 1 ? 1 : r < -1 ? -1 : r; }
    return s;
}
static float q8t(const float *x, int8_t *q, float mul) {              // per-token absmax int8; scale*mul
    float m = 1e-8f; for (int i = 0; i < D; i++) m = fmaxf(m, fabsf(x[i]));
    float is = 127 / m; for (int i = 0; i < D; i++) q[i] = (int8_t)rintf(x[i] * is);
    return m * mul / 127;
}
static int8_t sr8(float v, uint32_t idx) {                            // stochastic round + saturate
    float r = floorf(v + (hash(idx ^ seed) >> 8) * 0x1p-24f);
    return r > 127 ? 127 : r < -127 ? -127 : (int8_t)r;
}
typedef struct { float prev, cur[64]; } Amax;                          // delayed gradient scale
static float gscale(Amax *a) { return (a->prev > 0 ? a->prev : 1e-3f) / 127; }
static void roll(Amax *a) { a->prev = 0; for (int t = 0; t < NT; t++) { a->prev = fmaxf(a->prev, a->cur[t]); a->cur[t] = 0; } }

// ---- Monarch: R (row blocks, weights [b][i][o]) then L (column blocks, weights [o][i][lane]) ----
typedef struct { P *r, *l; int8_t rq[W3], rt[W3], lq[W3], *mq; float rs, ls, *ms; Amax gl, gr; int32_t *dw; uint32_t salt; } Mon;
static void mon_init(Mon *m, float sd, uint32_t salt) {
    m->r = param(W3, sd / sqrtf(M)); m->l = param(W3, 1 / sqrtf(M));
    m->mq = ia(N * D); m->ms = fa(N); m->dw = calloc((size_t)NT * 2 * W3, 4); m->salt = salt * 0x9E3779B9u;
}
static void mon_prep(Mon *m) {                                              // once per step, O(params):
    m->rs = tern(m->r->w, m->rq, W3, 1); m->ls = tern(m->l->w, m->lq, W3, 1);  // a 4KB ternary transpose so R^T
    for (int b = 0; b < M; b++) for (int i = 0; i < M; i++) for (int o = 0; o < M; o++)  // is broadcast-accumulate too
        m->rt[(b * M + o) * M + i] = m->rq[(b * M + i) * M + o];
}
// token t: x = int8 codes with scale sx -> y (float). Stores the int8 mid codes for the backward.
static void mon_fwd(Mon *m, const int8_t *x, float sx, float *y, int t) {
    int16_t z[D] = {0}, acc[D] = {0}; int mx = 1; int8_t *q = m->mq + t * D;  // |ternary.int8| <= 16*127: int16
    for (int b = 0; b < M; b++) for (int i = 0; i < M; i++) {                 // R: z_b += x_bi * R_bi.
        int xi = x[b * M + i]; const int8_t *w = m->rq + (b * M + i) * M; int16_t *zb = z + b * M;
        for (int o = 0; o < M; o++) zb[o] += xi * w[o];
    }
    for (int k = 0; k < D; k++) mx = abs(z[k]) > mx ? abs(z[k]) : mx;        // requantise int16 -> int8 directly
    float qs = 127.f / mx, sm = m->ms[t] = sx * m->rs * mx / 127;
    for (int k = 0; k < D; k++) q[k] = (int8_t)rintf(z[k] * qs);
    for (int o = 0; o < M; o++) for (int i = 0; i < M; i++) {                 // L: row o += w_oi (.) row i, lane-wise
        const int8_t *w = m->lq + (o * M + i) * M, *qi = q + i * M; int16_t *a = acc + o * M;
        for (int b = 0; b < M; b++) a[b] += w[b] * qi[b];
    }
    for (int k = 0; k < D; k++) y[k] = acc[k] * sm * m->ls;
}
// token t: dy (float) -> dx (float); int8 x int8 dW into thread tid's int32 slab
static void mon_bwd(Mon *m, const int8_t *x, float sx, const float *dy, float *dx, int t, int tid) {
    const int8_t *q = m->mq + t * D; float sm = m->ms[t], gl = gscale(&m->gl), gr = gscale(&m->gr), am = 0;
    int8_t g[D]; int16_t dm[D] = {0}; int32_t *dwr = m->dw + (size_t)tid * 2 * W3, *dwl = dwr + W3;
    for (int k = 0; k < D; k++) { float v = dy[k] * sm; am = fmaxf(am, fabsf(v)); g[k] = sr8(v / gl, (t * D + k) ^ m->salt); }
    m->gl.cur[tid] = fmaxf(m->gl.cur[tid], am);
    for (int o = 0; o < M; o++) for (int i = 0; i < M; i++) {                 // L^T g and dL, lane-wise
        const int8_t *w = m->lq + (o * M + i) * M, *go = g + o * M, *qi = q + i * M;
        int16_t *a = dm + i * M; int32_t *dl = dwl + (o * M + i) * M;
        for (int b = 0; b < M; b++) { a[b] += w[b] * go[b]; dl[b] += go[b] * qi[b]; }
    }
    float c = gl * m->ls / sm * sx / gr; am = 0;                             // fold sx so dR reuses x's codes
    for (int k = 0; k < D; k++) { float v = dm[k] * c; am = fmaxf(am, fabsf(v)); g[k] = sr8(v, (t * D + k) ^ ~m->salt); }
    m->gr.cur[tid] = fmaxf(m->gr.cur[tid], am * gr);
    float cx = gr * m->rs / sx; int16_t ax[D] = {0};
    for (int b = 0; b < M; b++) for (int o = 0; o < M; o++) {                 // R^T g and dR, broadcast g_bo
        int go = g[b * M + o]; const int8_t *w = m->rt + (b * M + o) * M, *xb = x + b * M;
        int16_t *a = ax + b * M; int32_t *dr = dwr + (b * M + o) * M;           // dR slab is [b][o][i]
        for (int i = 0; i < M; i++) { a[i] += go * w[i]; dr[i] += xb[i] * go; }
    }
    for (int k = 0; k < D; k++) dx[k] = ax[k] * cx;
}
static void mon_reduce(Mon *m) {                                             // sum thread slabs, O(params)
    float gr = gscale(&m->gr), gl = gscale(&m->gl);
    #pragma omp parallel for
    for (int k = 0; k < W3; k++) {
        int32_t sr = 0, sl = 0;
        for (int t = 0; t < NT; t++) { int32_t *s = m->dw + (size_t)t * 2 * W3; sr += s[k]; sl += s[W3 + k]; s[k] = s[W3 + k] = 0; }
        int b = k / (M * M), o = k / M % M, i = k % M;
        m->r->g[(b * M + i) * M + o] += sr * gr; m->l->g[k] += sl * gl;
    }
    roll(&m->gr); roll(&m->gl);
}

// ---- RMSNorm (no gain). Forward is folded into q8t's scale; backward recomputes y = x r ---------
static float rinv(const float *x) { float s = 0; for (int i = 0; i < D; i++) s += x[i] * x[i]; return 1 / sqrtf(s / D + 1e-5f); }
static void rms_bwd(const float *x, float r, const float *dy, float *dx) {   // dx += d rms(x)
    float dot = 0; for (int i = 0; i < D; i++) dot += dy[i] * x[i]; dot *= r * r / D;
    for (int i = 0; i < D; i++) dx[i] += (dy[i] - x[i] * dot) * r;
}

// ---- model -------------------------------------------------------------------------------------
typedef struct { P *k; int8_t kq[T * D]; float ks[D], alpha, s; Amax ga; } Conv;   // ternary kernel [T][D]
typedef struct {
    Mon a, v, o, u, d; Conv cv;
    int8_t *q1, *qo, *q2, *qd, *vq, *gc;      // int8 codes: Monarch inputs, conv input, conv grad
    float *s1, *so, *s2, *sd, *r1, *r2;       // per-token scales and rms
    float *av, *cc, *x1, *h;                  // the only float activations kept for backward
} Layer;
static Layer ly[L];
static P *E;                                  // token embedding, tied with the output head (fp32)
static int V;
static float *X[L + 1], *NF, *RF, *LG, *DX, *S1;

static void build(void) {
    NT = omp_get_max_threads(); if (NT > 64) NT = 64;
    E = param(V * D, 0.02f);
    for (int l = 0; l < L; l++) {
        Layer *y = &ly[l]; float ro = 1 / sqrtf(2 * L);
        mon_init(&y->a, 1, 10 * l + 1); mon_init(&y->v, 1, 10 * l + 2); mon_init(&y->o, ro, 10 * l + 3);
        mon_init(&y->u, 1, 10 * l + 4); mon_init(&y->d, ro, 10 * l + 5);
        y->cv.k = param(T * D, 1);
        for (int ch = 0; ch < D; ch++) {         // decaying init: far taps quantize to 0 -> local prior
            float tau = 1 + urand() * T / 4;
            for (int j = 0; j < T; j++) y->cv.k->w[j * D + ch] *= expf(-j / tau);
        }
        int8_t **q[] = {&y->q1, &y->qo, &y->q2, &y->qd, &y->vq, &y->gc};
        for (int i = 0; i < 6; i++) *q[i] = ia(N * D);
        float **s[] = {&y->s1, &y->so, &y->s2, &y->sd, &y->r1, &y->r2};
        for (int i = 0; i < 6; i++) *s[i] = fa(N);
        y->av = fa(N * D); y->cc = fa(N * D); y->x1 = fa(N * D); y->h = fa(N * D);
    }
    for (int l = 0; l <= L; l++) X[l] = fa(N * D);
    NF = fa(N * D); RF = fa(N); DX = fa(N * D); S1 = fa(N * D); LG = fa(N * V);
}

static void forward(const int *tok, int nb, int train) {
    int n = nb * T;
    for (int t = 0; t < n; t++) memcpy(X[0] + t * D, E->w + tok[t] * D, D * sizeof(float));
    for (int l = 0; l < L; l++) {
        Layer *y = &ly[l]; Conv *c = &y->cv; float mx = 0;
        mon_prep(&y->a); mon_prep(&y->v); mon_prep(&y->o); mon_prep(&y->u); mon_prep(&y->d);
        for (int ch = 0; ch < D; ch++) c->ks[ch] = tern(c->k->w + ch, c->kq + ch, T, D);
        c->s = (c->alpha > 0 ? c->alpha : 1) / 127;
        #pragma omp parallel for reduction(max:mx)
        for (int t = 0; t < n; t++) {           // phase 1 (per token): rms+quant, Mon_a, Mon_v -> conv codes
            const float *x = X[l] + t * D; int8_t *q = y->q1 + t * D, *vq = y->vq + t * D; float v[D];
            y->r1[t] = rinv(x); y->s1[t] = q8t(x, q, y->r1[t]);
            mon_fwd(&y->a, q, y->s1[t], y->av + t * D, t);
            mon_fwd(&y->v, q, y->s1[t], v, t);
            for (int ch = 0; ch < D; ch++) {
                float r = rintf(v[ch] / c->s); mx = fmaxf(mx, fabsf(v[ch]));
                vq[ch] = r > 127 ? 127 : r < -127 ? -127 : (int8_t)r;
            }
        }
        if (train) c->alpha = c->alpha > 0 ? 0.99f * c->alpha + 0.01f * mx : mx;   // updated after use: causal
        #pragma omp parallel for
        for (int t = 0; t < n; t++) {           // phase 2 (per token): conv, gate, Mon_o, +res, rms, MLP, +res
            int tt = t % T; int16_t acc[D] = {0}; float g[D], o[D];   // |sum| <= T*127 < 2^15
            for (int j = 0; j <= tt; j++) {
                const int8_t *kj = c->kq + j * D, *vj = y->vq + (t - j) * D;
                for (int ch = 0; ch < D; ch++) acc[ch] += kj[ch] * vj[ch];
            }
            float *cc = y->cc + t * D, *av = y->av + t * D, *x = X[l] + t * D, *x1 = y->x1 + t * D, *h = y->h + t * D;
            for (int ch = 0; ch < D; ch++) { cc[ch] = acc[ch] * c->ks[ch] * c->s; g[ch] = av[ch] * cc[ch]; }
            y->so[t] = q8t(g, y->qo + t * D, 1);
            mon_fwd(&y->o, y->qo + t * D, y->so[t], o, t);
            for (int i = 0; i < D; i++) x1[i] = x[i] + o[i];
            y->r2[t] = rinv(x1); y->s2[t] = q8t(x1, y->q2 + t * D, y->r2[t]);
            mon_fwd(&y->u, y->q2 + t * D, y->s2[t], h, t);
            for (int i = 0; i < D; i++) g[i] = h[i] > 0 ? h[i] * h[i] : 0;
            y->sd[t] = q8t(g, y->qd + t * D, 1);
            mon_fwd(&y->d, y->qd + t * D, y->sd[t], o, t);
            for (int i = 0; i < D; i++) X[l + 1][t * D + i] = x1[i] + o[i];
        }
    }
    #pragma omp parallel for
    for (int t = 0; t < n; t++) {
        float r = RF[t] = rinv(X[L] + t * D);
        for (int i = 0; i < D; i++) NF[t * D + i] = X[L][t * D + i] * r;
        for (int c = 0; c < V; c++) {
            float s = 0; for (int i = 0; i < D; i++) s += NF[t * D + i] * E->w[c * D + i];
            LG[t * V + c] = s;
        }
    }
}

static float xent(const int *tgt, int n, int grad) {   // mean CE; if grad, LG <- dLoss/dlogits
    double loss = 0;
    for (int t = 0; t < n; t++) {
        float *z = LG + t * V, mx = z[0], s = 0;
        for (int c = 1; c < V; c++) mx = fmaxf(mx, z[c]);
        for (int c = 0; c < V; c++) s += expf(z[c] - mx);
        loss += logf(s) + mx - z[tgt[t]];
        if (grad) { for (int c = 0; c < V; c++) z[c] = expf(z[c] - mx) / s / n; z[tgt[t]] -= 1.f / n; }
    }
    return loss / n;
}

static void backward(const int *tok, int nb) {
    int n = nb * T;
    #pragma omp parallel for
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < D; i++) {
            float s = 0; for (int c = 0; c < V; c++) s += LG[t * V + c] * E->w[c * D + i];
            S1[t * D + i] = s; DX[t * D + i] = 0;
        }
        rms_bwd(X[L] + t * D, RF[t], S1 + t * D, DX + t * D);
    }
    #pragma omp parallel for
    for (int c = 0; c < V; c++) for (int t = 0; t < n; t++) for (int i = 0; i < D; i++)
        E->g[c * D + i] += LG[t * V + c] * NF[t * D + i];
    for (int l = L - 1; l >= 0; l--) {                  // DX: grad wrt X[l+1] -> grad wrt X[l]
        Layer *y = &ly[l]; Conv *c = &y->cv; float gc = gscale(&c->ga);
        #pragma omp parallel for
        for (int t = 0; t < n; t++) {                   // phase A (per token): MLP, rms2, Mon_o, gate
            int tid = TID; float a[D], b[D], am = 0, *dx = DX + t * D, *h = y->h + t * D, *av = y->av + t * D, *cc = y->cc + t * D;
            mon_bwd(&y->d, y->qd + t * D, y->sd[t], dx, a, t, tid);
            for (int i = 0; i < D; i++) a[i] *= h[i] > 0 ? 2 * h[i] : 0;
            mon_bwd(&y->u, y->q2 + t * D, y->s2[t], a, b, t, tid);
            rms_bwd(y->x1 + t * D, y->r2[t], b, dx);
            mon_bwd(&y->o, y->qo + t * D, y->so[t], dx, a, t, tid);
            for (int i = 0; i < D; i++) {               // av <- d av (in place); conv grad -> int8 codes
                float dc = a[i] * av[i]; av[i] = a[i] * cc[i]; am = fmaxf(am, fabsf(dc));
                y->gc[t * D + i] = sr8(dc / gc, (t * D + i) ^ (l * 0x51ED27u));
            }
            c->ga.cur[tid] = fmaxf(c->ga.cur[tid], am);
        }
        #pragma omp parallel for
        for (int j = 0; j < T; j++) {                   // dk[j] = sum_t g[t] v[t-j]
            int32_t acc[D] = {0};
            for (int bb = 0; bb < nb; bb++) for (int t = j; t < T; t++) {
                const int8_t *g = y->gc + (bb * T + t) * D, *vj = y->vq + (bb * T + t - j) * D;
                for (int ch = 0; ch < D; ch++) acc[ch] += g[ch] * vj[ch];
            }
            for (int ch = 0; ch < D; ch++) c->k->g[j * D + ch] += acc[ch] * gc * c->s;
        }
        #pragma omp parallel for
        for (int t = 0; t < n; t++) {                   // phase B (per token): conv^T, Mon_v, Mon_a, rms1
            int tid = TID, tt = t % T; int16_t acc[D] = {0}; float dv[D], a[D], b[D];
            for (int u = tt; u < T; u++) {
                const int8_t *kj = c->kq + (u - tt) * D, *g = y->gc + (t + u - tt) * D;
                for (int ch = 0; ch < D; ch++) acc[ch] += kj[ch] * g[ch];
            }
            for (int ch = 0; ch < D; ch++) dv[ch] = acc[ch] * c->ks[ch] * gc;
            mon_bwd(&y->v, y->q1 + t * D, y->s1[t], dv, a, t, tid);
            mon_bwd(&y->a, y->q1 + t * D, y->s1[t], y->av + t * D, b, t, tid);
            for (int i = 0; i < D; i++) a[i] += b[i];
            rms_bwd(X[l] + t * D, y->r1[t], a, DX + t * D);
        }
        mon_reduce(&y->a); mon_reduce(&y->v); mon_reduce(&y->o); mon_reduce(&y->u); mon_reduce(&y->d); roll(&c->ga);
    }
    for (int t = 0; t < n; t++) for (int i = 0; i < D; i++) E->g[tok[t] * D + i] += DX[t * D + i];
}

static void adam(int step) {
    float lr = step < 100 ? LR * step / 100 : LR * (0.1f + 0.45f * (1 + cosf(3.14159265f * step / STEPS)));
    float c1 = 1 - powf(0.9f, step), c2 = 1 - powf(0.95f, step);
    for (int k = 0; k < np; k++) {
        P *p = &ps[k];
        for (int i = 0; i < p->n; i++) {
            float g = p->g[i]; p->g[i] = 0;
            if (step <= 0) continue;
            p->m[i] = 0.9f * p->m[i] + 0.1f * g; p->v[i] = 0.95f * p->v[i] + 0.05f * g * g;
            p->w[i] -= lr * (p->m[i] / c1) / (sqrtf(p->v[i] / c2) + 1e-8f);
        }
    }
}

static int *data, ndata, ntrain;
static void batch(int *tok, int *tgt, int val) {
    for (int b = 0; b < B; b++) {
        int lo = val ? ntrain : 0, hi = val ? ndata : ntrain;
        int off = lo + (int)(urand() * (hi - lo - T - 1));
        for (int t = 0; t < T; t++) { tok[b * T + t] = data[off + t]; tgt[b * T + t] = data[off + t + 1]; }
    }
}
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    FILE *f = fopen(argc > 1 ? argv[1] : "input.txt", "rb");
    if (!f) f = fopen(__FILE__, "rb");                  // no data? learn to write moth.c
    if (!f) { fprintf(stderr, "no input\n"); return 1; }
    fseek(f, 0, SEEK_END); ndata = ftell(f); rewind(f);
    unsigned char *raw = malloc(ndata); if (fread(raw, 1, ndata, f) != (size_t)ndata) return 1; fclose(f);
    int stoi[256], itos[256]; memset(stoi, -1, sizeof stoi);
    for (int i = 0; i < ndata; i++) if (stoi[raw[i]] < 0) stoi[raw[i]] = 0;
    for (int c = 0; c < 256; c++) if (!stoi[c]) { itos[V] = c; stoi[c] = V++; }
    data = malloc(ndata * sizeof(int)); for (int i = 0; i < ndata; i++) data[i] = stoi[raw[i]];
    ntrain = ndata * 9 / 10;

    build();
    long nparam = 0; for (int k = 0; k < np; k++) nparam += ps[k].n;
    printf("moth: vocab %d, d %d, layers %d, T %d, %.2fM params (%.2fM ternary), %d threads\n",
           V, D, L, T, nparam / 1e6, (nparam - V * D) / 1e6, NT);

    int *tok = malloc(N * sizeof(int)), *tgt = malloc(N * sizeof(int));
    double t0 = now();
    for (int step = -1; step <= STEPS; step++) {        // steps -1, 0 calibrate the delayed scales
        seed = hash(step + 2);
        batch(tok, tgt, 0);
        forward(tok, B, 1);
        float loss = xent(tgt, N, 1);
        backward(tok, B);
        adam(step);
        if (step % 50 == 0 && step > 0) { printf("step %4d | loss %.4f | %.0f ms/step\n", step, loss, (now() - t0) * 1e3 / 50); fflush(stdout); t0 = now(); }
        if (step % 500 == 0 && step > 0) {
            float vl = 0; for (int k = 0; k < 8; k++) { batch(tok, tgt, 1); forward(tok, B, 0); vl += xent(tgt, N, 0) / 8; }
            printf("step %4d | val loss %.4f\n", step, vl); t0 = now();
        }
    }

    int ctx[T] = {0}, len = 1; ctx[0] = data[0];           // sample (recomputes the window per token)
    putchar(itos[ctx[0]]);
    for (int k = 0; k < 600; k++) {
        forward(ctx, 1, 0);
        float *z = LG + (len - 1) * V, mx = -1e30f, s = 0, r;
        for (int c = 0; c < V; c++) mx = fmaxf(mx, z[c]);
        for (int c = 0; c < V; c++) s += (z[c] = expf((z[c] - mx) / 0.8f));
        int c = 0; r = urand() * s; while (c < V - 1 && (r -= z[c]) > 0) c++;
        putchar(itos[c]); fflush(stdout);
        if (len == T) { memmove(ctx, ctx + 1, (T - 1) * sizeof(int)); len--; }
        ctx[len++] = c;
    }
    putchar('\n');
    return 0;
}
