// walshnet tests: kernels vs scalar references, finite-difference gradients, causality.
#include "wn.h"

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void test_sgemm(void) {
    int shapes[][3] = {{13, 45, 7}, {64, 65, 100}, {100, 256, 300}, {8, 32, 1}, {5, 3, 600}};
    Rng r = rng_seed(1);
    for (int s = 0; s < 5; s++) {
        const int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2];
        float *A = xmalloc(4 * M * K), *B = xmalloc(4 * K * N), *C = xmalloc(4 * M * N);
        for (int i = 0; i < M * K; i++) A[i] = rng_normal(&r);
        for (int i = 0; i < K * N; i++) B[i] = rng_normal(&r);
        for (int i = 0; i < M * N; i++) C[i] = 1.0f;
        sgemm(M, N, K, A, K, B, N, C, N, 1);
        double err = 0;
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
            double ref = 1; for (int k = 0; k < K; k++) ref += (double)A[i * K + k] * B[k * N + j];
            err = fmax(err, fabs(ref - C[i * N + j]) / (1 + fabs(ref)));
        }
        CHECK(err < 1e-4, "sgemm %dx%dx%d err %g", M, N, K, err);
        free(A); free(B); free(C);
    }
    for (int s = 0; s < 5; s++) {                                         // sgemm_tn and transpose
        const int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2];
        float *A = xmalloc(4 * K * M), *B = xmalloc(4 * K * N), *C = xmalloc(4 * M * N), *At = xmalloc(4 * K * M);
        for (int i = 0; i < K * M; i++) A[i] = rng_normal(&r);
        for (int i = 0; i < K * N; i++) B[i] = rng_normal(&r);
        sgemm_tn(M, N, K, A, M, B, N, C, N, 0);
        transpose(A, K, M, At);
        double err = 0, terr = 0;
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
            double ref = 0; for (int k = 0; k < K; k++) ref += (double)A[k * M + i] * B[k * N + j];
            err = fmax(err, fabs(ref - C[i * N + j]) / (1 + fabs(ref)));
        }
        for (int k = 0; k < K; k++) for (int i = 0; i < M; i++) terr = fmax(terr, fabs(At[i * K + k] - A[k * M + i]));
        CHECK(err < 1e-4, "sgemm_tn %dx%dx%d err %g", M, N, K, err);
        CHECK(terr == 0, "transpose %dx%d err %g", K, M, terr);
        free(A); free(B); free(C); free(At);
    }
    printf("sgemm / sgemm_tn / transpose: match naive references\n");
}

static void test_bitlinear(void) {
    Rng r = rng_seed(2); BitLinear l; const int K = 64, M = 48, N = 37;
    bl_init(&l, K, M, N, &r); bl_prepare(&l);
    float *x = xmalloc(4 * N * K), *y = xmalloc(4 * N * M);
    for (int i = 0; i < N * K; i++) x[i] = rng_normal(&r) * 3;
    bl_forward(&l, x, N, y);
    double err = 0;
    for (int n = 0; n < N; n++) for (int m = 0; m < M; m++) {
        double ref = 0; for (int k = 0; k < K; k++) ref += (double)l.xq[n * K + k] * l.WqT[m * K + k];
        err = fmax(err, fabs(ref - y[n * M + m]));
    }
    CHECK(err < 1e-3, "bitlinear VNNI vs dequantized fp32: err %g", err);
    int ternary = 1;
    for (int i = 0; i < K * M; i++) { float q = l.WqT[i] / l.ws; ternary &= fabsf(q) < 1e-6f || fabsf(fabsf(q) - 1) < 1e-6f; }
    CHECK(ternary, "weights not ternary");
    printf("bitlinear: VNNI exact vs fp32 reference (max err %.2e)\n", err);
}

static void test_longconv(int mode) {
    Rng r = rng_seed(3); WalshMix m; const int d = 32, T = 64, N = 3 * T;
    g_noquant = 1; wm_init(&m, d, T, mode, N, &r);
    for (int i = 0; i < T * d; i++) m.h.w[i] = rng_normal(&r);
    wm_prepare(&m);
    float *x = xmalloc(4 * N * d), *y = xmalloc(4 * N * d);
    for (int i = 0; i < N * d; i++) x[i] = rng_normal(&r);
    wm_forward(&m, x, N, y);
    double err = 0;
    for (int b = 0; b < N / T; b++) for (int t = 0; t < T; t++) for (int c = 0; c < d; c++) {
        double ref = 0;
        for (int s = 0; s < T; s++) if (mode == MODE_BIDIR || s <= t) ref += (double)m.h.w[(t ^ s) * d + c] * m.us[(b * T + s) * d + c];
        err = fmax(err, fabs(ref - m.v[(b * T + t) * d + c]) / (1 + fabs(ref)));
    }
    CHECK(err < 1e-4, "long conv (%s) vs brute force err %g", mode ? "bidir" : "causal", err);
    printf("walsh long conv %-6s: matches brute-force dyadic conv (max rel err %.2e)\n", mode ? "bidir" : "causal", err);
    g_noquant = 0;
}

static void test_gradients(int mode, int ffn) {
    g_noquant = 1;
    Config c = {.vocab = 11, .d = 16, .n_layers = 2, .T = 32, .mode = mode, .ffn = ffn, .hidden = 32,
                .dl_width = 64, .dl_depth = 3, .dl_nth = 4, .dl_temp = 0.7f, .dl_z = 1.0f, .dl_lr_mul = 1.0f, .seed = 7};
    const int N = 2 * c.T; Model m; model_init(&m, &c, N);
    Rng r = rng_seed(4);
    for (int l = 0; l < c.n_layers; l++) for (size_t i = 0; i < m.mix[l].h.n; i++) m.mix[l].h.w[i] += 0.3f * rng_normal(&r);
    model_prepare(&m);
    int *tok = malloc(4 * N), *tgt = malloc(4 * N);
    for (int n = 0; n < N; n++) { tok[n] = rng_u64(&r) % m.V_in; tgt[n] = (mode == MODE_BIDIR && n % 3) ? -1 : (int)(rng_u64(&r) % c.vocab); }
    zero_grads(&m); model_step(&m, tok, tgt, N, 1, 0);
    double worst = 0; int checked = 0;
    for (int p = 0; p < m.n_params; p++) {
        Param *P = m.params[p];
        for (int trial = 0; trial < 6; trial++) {
            size_t i = rng_u64(&r) % P->n; const float e = 1e-2f, w0 = P->w[i];     // fp32 loss: larger step beats rounding noise
            P->w[i] = w0 + e; model_prepare(&m); double lp = model_step(&m, tok, tgt, N, 0, 0);
            P->w[i] = w0 - e; model_prepare(&m); double lm = model_step(&m, tok, tgt, N, 0, 0);
            P->w[i] = w0; model_prepare(&m);
            const double num = (lp - lm) / (2 * e), ana = P->g[i];
            if (fabs(num) + fabs(ana) < 3e-4) continue;
            worst = fmax(worst, fabs(num - ana) / (fabs(num) + fabs(ana))); checked++;
        }
    }
    CHECK(worst < 2e-2, "gradcheck mode=%d ffn=%d worst rel err %g", mode, ffn, worst);
    printf("gradcheck %-6s + %-5s: worst rel err %.2e over %d entries\n", mode ? "bidir" : "causal", ffn == FFN_MLP ? "MLP" : "DDLGN", worst, checked);
    g_noquant = 0;
}

static void test_causality(void) {
    Config c = {.vocab = 11, .d = 32, .n_layers = 2, .T = 64, .mode = MODE_CAUSAL, .ffn = FFN_MLP, .hidden = 64, .seed = 9};
    const int N = c.T; Model m; model_init(&m, &c, N);
    Rng r = rng_seed(5);
    for (int l = 0; l < 2; l++) for (size_t i = 0; i < m.mix[l].h.n; i++) m.mix[l].h.w[i] += 0.3f * rng_normal(&r);
    model_prepare(&m);
    int tok[64], tgt[64]; for (int n = 0; n < N; n++) { tok[n] = rng_u64(&r) % 11; tgt[n] = 0; }
    float *a = xmalloc(4 * N * 11); model_step(&m, tok, tgt, N, 0, 0); memcpy(a, m.logits, 4 * N * 11);
    tok[40] = (tok[40] + 1) % 11; model_step(&m, tok, tgt, N, 0, 0);
    double before = 0, after = 0;
    for (int n = 0; n < N; n++) for (int v = 0; v < 11; v++) {
        const double dlt = fabs(a[n * 11 + v] - m.logits[n * 11 + v]);
        if (n < 40) before = fmax(before, dlt); else after = fmax(after, dlt);
    }
    CHECK(before == 0 && after > 0, "causality: change at t=40 moved earlier logits by %g (later by %g)", before, after);
    printf("causality: editing token 40 changes logits at t<40 by %g, at t>=40 by %.3f\n", before, after);
}

static void test_clopen(int fanin, int ste) {
    Rng r = rng_seed(11 + fanin + 7 * ste); Clopen f; const int d = 16, nth = 4, W = 64, depth = 3, N = 48, k = W / d;
    cl_init(&f, d, nth, 0.7f, W, depth, fanin, ste, 1.0f, N, &r);
    for (int i = 0; i < depth; i++) {
        for (size_t e = 0; e < f.w[i].n; e++) f.w[i].w[e] = 2.4f * rng_unif(&r) - 1.2f;
        for (int g = 0; g < W; g++) f.theta[i].w[g] = 4.0f * rng_unif(&r) - 2.0f;
    }
    for (int c = 0; c < d; c++) f.gain.w[c] = 0.5f + rng_unif(&r);
    float *x = xmalloc(4 * N * d), *y = xmalloc(4 * N * d), *dy = xmalloc(4 * N * d), *dx = xmalloc(4 * N * d);
    for (int i = 0; i < N * d; i++) { x[i] = rng_normal(&r); dy[i] = rng_normal(&r); }
    cl_forward(&f, x, N, y); cl_backward(&f, dy, dx);
    // ---- naive reference
    const int n_in0 = d * nth, maxr = W > n_in0 ? W : n_in0;
    double *act = calloc((size_t)(depth + 1) * maxr * N, 8), *pre = calloc((size_t)depth * W * N, 8), *xn = calloc(N * d, 8), *rr = calloc(N, 8);
    #define ACT(l, g, n) act[((size_t)(l) * maxr + (g)) * N + (n)]
    #define PRE(l, g, n) pre[((size_t)(l) * W + (g)) * N + (n)]
    for (int n = 0; n < N; n++) {
        double ss = 0; for (int c = 0; c < d; c++) ss += (double)x[n * d + c] * x[n * d + c];
        rr[n] = 1.0 / sqrt(ss / d + 1e-6);
        for (int c = 0; c < d; c++) { xn[n * d + c] = x[n * d + c] * rr[n]; for (int j = 0; j < nth; j++) ACT(0, c * nth + j, n) = xn[n * d + c] > f.th[j] ? 1 : -1; }
    }
    #define Q3(v) (fmax(-1.0, fmin(1.0, rint(v))))
    for (int l = 0; l < depth; l++) for (int g = 0; g < W; g++) for (int n = 0; n < N; n++) {
        double s = -f.theta[l].w[g];
        for (int j = 0; j < fanin; j++) s += Q3(f.w[l].w[g * fanin + j]) * ACT(l, f.idx[l][g * fanin + j], n);
        PRE(l, g, n) = s; ACT(l + 1, g, n) = s >= 0 ? 1 : -1;
    }
    double err_y = 0;
    for (int n = 0; n < N; n++) for (int c = 0; c < d; c++) {
        double s = 0; for (int q = 0; q < k; q++) s += ACT(depth, c * k + q, n);
        err_y = fmax(err_y, fabs(0.5 * s / k * f.gain.w[c] - y[n * d + c]));
    }
    double *gout = calloc((size_t)maxr * N, 8), *gin = calloc((size_t)maxr * N, 8), err_w = 0, err_t = 0, err_x = 0;
    for (int n = 0; n < N; n++) for (int c = 0; c < d; c++) for (int q = 0; q < k; q++) gout[(size_t)(c * k + q) * N + n] = dy[n * d + c] * f.gain.w[c] * 0.5 / k;
    for (int l = depth - 1; l >= 0; l--) {
        memset(gin, 0, 8 * (size_t)maxr * N);
        for (int g = 0; g < W; g++) {
            double dth = 0, dw[16] = {0};
            for (int n = 0; n < N; n++) {
                const double gy = gout[(size_t)g * N + n] * (ste ? fabs(PRE(l, g, n)) <= 1.0 : 1.0);
                dth -= gy;
                for (int j = 0; j < fanin; j++) {
                    const int s = f.idx[l][g * fanin + j];
                    dw[j] += gy * ACT(l, s, n); gin[(size_t)s * N + n] += gy * Q3(f.w[l].w[g * fanin + j]);
                }
            }
            err_t = fmax(err_t, fabs(dth - f.theta[l].g[g]) / (1 + fabs(dth)));
            for (int j = 0; j < fanin; j++) err_w = fmax(err_w, fabs(dw[j] - f.w[l].g[g * fanin + j]) / (1 + fabs(dw[j])));
        }
        double *tmp = gout; gout = gin; gin = tmp;
    }
    for (int n = 0; n < N; n++) {
        double dxn[64], dot = 0;
        for (int c = 0; c < d; c++) {
            double a = 0;
            for (int j = 0; j < nth; j++) { const double u = (xn[n * d + c] - f.th[j]) / 0.7; a += gout[(size_t)(c * nth + j) * N + n] * (ste ? fabs(u) <= 1.0 : 1.0); }
            dxn[c] = a / 0.7; dot += dxn[c] * xn[n * d + c];
        }
        dot /= d;
        for (int c = 0; c < d; c++) err_x = fmax(err_x, fabs(rr[n] * (dxn[c] - xn[n * d + c] * dot) - dx[n * d + c]) / (1 + fabs(dx[n * d + c])));
    }
    CHECK(err_y < 1e-5 && err_w < 1e-4 && err_t < 1e-4 && err_x < 1e-4, "clopen fanin=%d ste=%d: y %g w %g theta %g x %g", fanin, ste, err_y, err_w, err_t, err_x);
    printf("clopen fanin=%d ste=%-4s: forward + STE grads match naive reference (max err y %.1e, w %.1e, theta %.1e, x %.1e)\n",
           fanin, ste ? "clip" : "id", err_y, err_w, err_t, err_x);
}

int main(void) {
    printf("threads: %d\n", omp_get_max_threads());
    test_sgemm(); test_bitlinear();
    test_longconv(MODE_CAUSAL); test_longconv(MODE_BIDIR);
    for (int mode = 0; mode < 2; mode++) for (int ffn = 0; ffn < 2; ffn++) test_gradients(mode, ffn);
    test_causality();
    for (int ste = 0; ste < 2; ste++) { test_clopen(3, ste); test_clopen(8, ste); }
    printf(fails ? "\n%d FAILURES\n" : "\nall tests passed\n", fails);
    return fails != 0;
}
