// moth.c: Moth, a tiny ternary Monarch Mixer char-level LM in one file of pure C (nanoGPT style).
//
//   block(x):  u = rms(x);  x += Mon_o( Mon_a(u) * conv(Mon_v(u)) )      sequence mixer (gated causal long conv)
//              u = rms(x);  x += Mon_d( relu(Mon_u(u))^2 )                channel mixer
//   Mon(x)  =  P L P^T R x,  L and R block-diagonal (M blocks of MxM), d = M^2, 2 d^1.5 params per matrix
//
// forward : BitNet b1.58. weights ternary {-1,0,1} (absmean), activations int8 (per-token absmax),
//           int32 accumulate. The long conv uses a ternary per-channel kernel on int8 activations with
//           a static (EMA-calibrated) scale, so it stays strictly causal. Float master weights + STE.
// backward: gradients stochastically rounded to int8 (per tensor); dx = W_tern^T g8, dW = g8^T x8 are
//           integer GEMMs. Embedding / tied head / RMSNorm stay fp32, as in BitNet.
//
// cc -O3 -march=native -fopenmp moth.c -o moth -lm && ./moth input.txt
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define M 16            // monarch block size
#define D (M * M)       // model width
#define L 4             // layers
#define T 128           // context = long-conv length
#define B 16            // batch size
#define N (B * T)       // tokens per batch
#define STEPS 3000
#define LR 3e-3f

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static float urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 40) / 16777216.f; }
static float randn(void) { return sqrtf(-2 * logf(urand() + 1e-9f)) * cosf(6.2831853f * urand()); }
static float *fa(size_t n) { return calloc(n, sizeof(float)); }

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
static uint32_t hash(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16); }
static float q8(const float *x, int8_t *q, int n, int sr) {          // absmax int8 (sr: stochastic rounding)
    float m = 1e-8f; uint32_t seed = sr ? (uint32_t)(urand() * 4294967040.f) : 0;
    #pragma omp parallel for reduction(max:m) if (n > 4096)
    for (int i = 0; i < n; i++) m = fmaxf(m, fabsf(x[i]));
    float s = m / 127, is = 1 / s;
    #pragma omp parallel for if (n > 4096)
    for (int i = 0; i < n; i++) q[i] = (int8_t)(sr ? floorf(x[i] * is + (hash(i ^ seed) >> 8) / 16777216.f) : rintf(x[i] * is));
    return s;
}
static int8_t *G8, *X8;   // scratch int8 codes for the backward pass

// ---- BitBlockDiag: one factor of a Monarch matrix. tr=1 reads/writes through the transpose P --
#define IX(tr, b, j) ((tr) ? (j) * M + (b) : (b) * M + (j))
typedef struct { P *w; int tr; int8_t wq[M * M * M], wt[M * M * M], *xq; float ws; const float *x; } BD;  // wt: [b][i][o]
static void bd_fwd(BD *l, const float *x, float *y, int n) {
    l->x = x; l->ws = tern(l->w->w, l->wq, M * M * M, 1);
    for (int b = 0; b < M; b++) for (int o = 0; o < M; o++) for (int i = 0; i < M; i++)
        l->wt[(b * M + i) * M + o] = l->wq[(b * M + o) * M + i];
    #pragma omp parallel for
    for (int t = 0; t < n; t++) {
        int8_t *q = l->xq + t * D; float s = q8(x + t * D, q, D, 0) * l->ws;
        for (int b = 0; b < M; b++) {
            int32_t acc[M] = {0};                                   // y_b = W_b x_b, adds/subs only
            for (int i = 0; i < M; i++) {
                int xi = q[IX(l->tr, b, i)]; const int8_t *w = l->wt + (b * M + i) * M;
                for (int o = 0; o < M; o++) acc[o] += xi * w[o];
            }
            for (int o = 0; o < M; o++) y[t * D + IX(l->tr, b, o)] = acc[o] * s;
        }
    }
}
static void bd_bwd(BD *l, const float *dy, float *dx, int n) {
    float gs = q8(dy, G8, n * D, 1), xs = q8(l->x, X8, n * D, 0);   // per-tensor codes, so scales factor out
    #pragma omp parallel for
    for (int t = 0; t < n; t++) for (int b = 0; b < M; b++) {       // dx = W^T g   (ternary x int8)
        int32_t acc[M] = {0};
        for (int o = 0; o < M; o++) {
            int g = G8[t * D + IX(l->tr, b, o)];
            const int8_t *w = l->wq + (b * M + o) * M;
            for (int i = 0; i < M; i++) acc[i] += g * w[i];
        }
        for (int i = 0; i < M; i++) dx[t * D + IX(l->tr, b, i)] = acc[i] * l->ws * gs;
    }
    #pragma omp parallel for
    for (int b = 0; b < M; b++) {                                   // dW = g^T x   (int8 x int8), STE
        int32_t acc[M * M] = {0};
        for (int t = 0; t < n; t++) {
            int8_t g[M], xb[M];
            for (int j = 0; j < M; j++) { g[j] = G8[t * D + IX(l->tr, b, j)]; xb[j] = X8[t * D + IX(l->tr, b, j)]; }
            for (int o = 0; o < M; o++) for (int i = 0; i < M; i++) acc[o * M + i] += g[o] * xb[i];
        }
        for (int k = 0; k < M * M; k++) l->w->g[b * M * M + k] += acc[k] * gs * xs;
    }
}

// ---- Monarch = R, transpose, L, transpose -----------------------------------------------------
typedef struct { BD r, l; float *mid; } Mon;
static float *SM;
static void mon_init(Mon *m, float sd) {
    BD *f[2] = {&m->r, &m->l};
    for (int k = 0; k < 2; k++) { f[k]->w = param(M * M * M, sd / sqrtf(M)); f[k]->tr = k; f[k]->xq = malloc(N * D); }
    m->mid = fa(N * D);
}
static void mon_fwd(Mon *m, const float *x, float *y, int n) { bd_fwd(&m->r, x, m->mid, n); bd_fwd(&m->l, m->mid, y, n); }
static void mon_bwd(Mon *m, const float *dy, float *dx, int n) { bd_bwd(&m->l, dy, SM, n); bd_bwd(&m->r, SM, dx, n); }

// ---- causal depthwise long conv: ternary kernel [T][D], int8 input with a static EMA scale -----
typedef struct { P *k; int8_t kq[T * D], *vq; float ks[D], alpha, s; } Conv;
static void conv_fwd(Conv *c, const float *v, float *y, int nb, int train) {
    int n = nb * T; float mx = 1e-8f;
    for (int ch = 0; ch < D; ch++) c->ks[ch] = tern(c->k->w + ch, c->kq + ch, T, D);
    for (int i = 0; i < n * D; i++) mx = fmaxf(mx, fabsf(v[i]));
    if (!c->alpha) c->alpha = mx;                                   // first-batch calibration
    c->s = c->alpha / 127;
    for (int i = 0; i < n * D; i++) { float r = rintf(v[i] / c->s); c->vq[i] = r > 127 ? 127 : r < -127 ? -127 : r; }
    if (train) c->alpha = 0.99f * c->alpha + 0.01f * mx;            // update after use: no peeking ahead
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < nb; b++) for (int t = 0; t < T; t++) {
        int32_t acc[D] = {0};
        for (int j = 0; j <= t; j++) {
            const int8_t *kj = c->kq + j * D, *vj = c->vq + (b * T + t - j) * D;
            for (int ch = 0; ch < D; ch++) acc[ch] += kj[ch] * vj[ch];
        }
        for (int ch = 0; ch < D; ch++) y[(b * T + t) * D + ch] = acc[ch] * c->ks[ch] * c->s;
    }
}
static void conv_bwd(Conv *c, const float *dy, float *dv, int nb) {
    float gs = q8(dy, G8, nb * T * D, 1);
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < nb; b++) for (int u = 0; u < T; u++) {      // dv[u] = sum_t k[t-u] g[t]
        int32_t acc[D] = {0};
        for (int t = u; t < T; t++) {
            const int8_t *kj = c->kq + (t - u) * D, *g = G8 + (b * T + t) * D;
            for (int ch = 0; ch < D; ch++) acc[ch] += kj[ch] * g[ch];
        }
        for (int ch = 0; ch < D; ch++) dv[(b * T + u) * D + ch] = acc[ch] * c->ks[ch] * gs;
    }
    #pragma omp parallel for
    for (int j = 0; j < T; j++) {                                   // dk[j] = sum_t g[t] v[t-j]
        int32_t acc[D] = {0};
        for (int b = 0; b < nb; b++) for (int t = j; t < T; t++) {
            const int8_t *g = G8 + (b * T + t) * D, *vj = c->vq + (b * T + t - j) * D;
            for (int ch = 0; ch < D; ch++) acc[ch] += g[ch] * vj[ch];
        }
        for (int ch = 0; ch < D; ch++) c->k->g[j * D + ch] += acc[ch] * gs * c->s;
    }
}

// ---- RMSNorm (no gain) -------------------------------------------------------------------------
static void rms_fwd(const float *x, float *y, float *r, int n) {
    for (int t = 0; t < n; t++) {
        float ss = 0; for (int i = 0; i < D; i++) ss += x[t * D + i] * x[t * D + i];
        r[t] = 1 / sqrtf(ss / D + 1e-5f);
        for (int i = 0; i < D; i++) y[t * D + i] = x[t * D + i] * r[t];
    }
}
static void rms_bwd(const float *y, const float *r, const float *dy, float *dx, int n) {
    for (int t = 0; t < n; t++) {
        float dot = 0; for (int i = 0; i < D; i++) dot += dy[t * D + i] * y[t * D + i]; dot /= D;
        for (int i = 0; i < D; i++) dx[t * D + i] = (dy[t * D + i] - y[t * D + i] * dot) * r[t];
    }
}

// ---- model -------------------------------------------------------------------------------------
typedef struct { Mon a, v, o, u, d; Conv cv; float *n1, *r1, *av, *vv, *cc, *g, *x1, *n2, *r2, *h, *sq; } Layer;
static Layer ly[L];
static P *E;                                     // token embedding, tied with the output head (fp32)
static int V;
static float *X[L + 1], *NF, *RF, *LG, *DX, *S1, *S2, *S3, *S4;

static void build(void) {
    E = param(V * D, 0.02f);
    for (int l = 0; l < L; l++) {
        Layer *y = &ly[l]; float ro = 1 / sqrtf(2 * L);
        mon_init(&y->a, 1); mon_init(&y->v, 1); mon_init(&y->o, ro); mon_init(&y->u, 1); mon_init(&y->d, ro);
        y->cv.k = param(T * D, 1); y->cv.vq = malloc(N * D);
        for (int ch = 0; ch < D; ch++) {         // decaying init: far taps quantize to 0 -> local prior
            float tau = 1 + urand() * T / 4;
            for (int j = 0; j < T; j++) y->cv.k->w[j * D + ch] *= expf(-j / tau);
        }
        float **b[] = {&y->n1, &y->r1, &y->av, &y->vv, &y->cc, &y->g, &y->x1, &y->n2, &y->r2, &y->h, &y->sq};
        for (int i = 0; i < 11; i++) *b[i] = fa(N * D);
    }
    for (int l = 0; l <= L; l++) X[l] = fa(N * D);
    float **b[] = {&NF, &RF, &DX, &S1, &S2, &S3, &S4, &SM};
    for (int i = 0; i < 8; i++) *b[i] = fa(N * D);
    LG = fa(N * V); G8 = malloc(N * D); X8 = malloc(N * D);
}

static void forward(const int *tok, int nb, int train) {
    int n = nb * T;
    for (int t = 0; t < n; t++) memcpy(X[0] + t * D, E->w + tok[t] * D, D * sizeof(float));
    for (int l = 0; l < L; l++) {
        Layer *y = &ly[l]; float *x = X[l], *out = X[l + 1];
        rms_fwd(x, y->n1, y->r1, n);
        mon_fwd(&y->a, y->n1, y->av, n); mon_fwd(&y->v, y->n1, y->vv, n);
        conv_fwd(&y->cv, y->vv, y->cc, nb, train);
        for (int i = 0; i < n * D; i++) y->g[i] = y->av[i] * y->cc[i];
        mon_fwd(&y->o, y->g, S1, n);
        for (int i = 0; i < n * D; i++) y->x1[i] = x[i] + S1[i];
        rms_fwd(y->x1, y->n2, y->r2, n);
        mon_fwd(&y->u, y->n2, y->h, n);
        for (int i = 0; i < n * D; i++) y->sq[i] = y->h[i] > 0 ? y->h[i] * y->h[i] : 0;
        mon_fwd(&y->d, y->sq, S1, n);
        for (int i = 0; i < n * D; i++) out[i] = y->x1[i] + S1[i];
    }
    rms_fwd(X[L], NF, RF, n);
    #pragma omp parallel for
    for (int t = 0; t < n; t++) for (int c = 0; c < V; c++) {
        float s = 0; for (int i = 0; i < D; i++) s += NF[t * D + i] * E->w[c * D + i];
        LG[t * V + c] = s;
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
    for (int t = 0; t < n; t++) for (int i = 0; i < D; i++) {
        float s = 0; for (int c = 0; c < V; c++) s += LG[t * V + c] * E->w[c * D + i];
        S1[t * D + i] = s;
    }
    #pragma omp parallel for
    for (int c = 0; c < V; c++) for (int t = 0; t < n; t++) for (int i = 0; i < D; i++)
        E->g[c * D + i] += LG[t * V + c] * NF[t * D + i];
    rms_bwd(NF, RF, S1, DX, n);
    for (int l = L - 1; l >= 0; l--) {                  // DX: grad wrt X[l+1] -> grad wrt X[l]
        Layer *y = &ly[l];
        mon_bwd(&y->d, DX, S1, n);
        for (int i = 0; i < n * D; i++) S1[i] *= y->h[i] > 0 ? 2 * y->h[i] : 0;
        mon_bwd(&y->u, S1, S2, n);
        rms_bwd(y->n2, y->r2, S2, S3, n);
        for (int i = 0; i < n * D; i++) DX[i] += S3[i];
        mon_bwd(&y->o, DX, S1, n);
        for (int i = 0; i < n * D; i++) { S2[i] = S1[i] * y->cc[i]; S1[i] *= y->av[i]; }
        conv_bwd(&y->cv, S1, S4, nb);
        mon_bwd(&y->a, S2, S3, n); mon_bwd(&y->v, S4, S1, n);
        for (int i = 0; i < n * D; i++) S3[i] += S1[i];
        rms_bwd(y->n1, y->r1, S3, S2, n);
        for (int i = 0; i < n * D; i++) DX[i] += S2[i];
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
    printf("moth: vocab %d, d %d, layers %d, T %d, %.2fM params (%.2fM ternary)\n",
           V, D, L, T, nparam / 1e6, (nparam - V * D) / 1e6);

    int *tok = malloc(N * sizeof(int)), *tgt = malloc(N * sizeof(int));
    double t0 = now();
    for (int step = 1; step <= STEPS; step++) {
        batch(tok, tgt, 0);
        forward(tok, B, 1);
        float loss = xent(tgt, N, 1);
        backward(tok, B);
        adam(step);
        if (step % 50 == 0 || step == 1) { printf("step %4d | loss %.4f | %.0f ms/step\n", step, loss, (now() - t0) * 1e3 / (step == 1 ? 1 : 50)); fflush(stdout); t0 = now(); }
        if (step % 500 == 0) {
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
