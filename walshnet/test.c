// walshnet tests: kernels vs references, finite-difference gradients, causality/receptive field, streaming == batch.
#include "wn.h"

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void test_gemm(void) {
    int shapes[][3] = {{13, 45, 7}, {64, 65, 100}, {100, 256, 300}, {8, 32, 1}, {5, 3, 600}};
    Rng r = rng_seed(1); double worst = 0;
    for (int s = 0; s < 5; s++) {
        const int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2];
        float *A = xmalloc(4 * M * K), *At = xmalloc(4 * M * K), *B = xmalloc(4 * K * N), *C = xmalloc(4 * M * N), *C2 = xmalloc(4 * M * N);
        for (int i = 0; i < M * K; i++) A[i] = rng_normal(&r);
        for (int i = 0; i < K * N; i++) B[i] = rng_normal(&r);
        for (int i = 0; i < M * N; i++) C[i] = 1.0f;
        sgemm(M, N, K, A, K, B, N, C, N, 1);
        transpose(A, M, K, At);                                            // At [K][M]
        sgemm_tn(M, N, K, At, M, B, N, C2, N, 0);
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
            double ref = 0; for (int k = 0; k < K; k++) ref += (double)A[i * K + k] * B[k * N + j];
            worst = fmax(worst, fabs(ref + 1 - C[i * N + j]) / (1 + fabs(ref)));
            worst = fmax(worst, fabs(ref - C2[i * N + j]) / (1 + fabs(ref)));
        }
        free(A); free(At); free(B); free(C); free(C2);
    }
    CHECK(worst < 1e-4, "gemm err %g", worst);
    printf("sgemm / sgemm_tn / transpose : match naive references (max rel err %.1e)\n", worst);
}

static void test_bitlinear(void) {
    Rng r = rng_seed(2); BitLinear l; const int K = 64, M = 48, N = 37;
    bl_init(&l, K, M, N, &r); bl_prepare(&l); l.store_xq = 1;
    float *x = xmalloc(4 * N * K), *y = xmalloc(4 * N * M);
    for (int i = 0; i < N * K; i++) x[i] = rng_normal(&r) * 3;
    bl_forward(&l, x, N, y);
    double err = 0;
    for (int n = 0; n < N; n++) for (int m = 0; m < M; m++) {
        double ref = 0; for (int k = 0; k < K; k++) ref += (double)l.xq[n * K + k] * l.WqT[m * K + k];
        err = fmax(err, fabs(ref - y[n * M + m]));
    }
    CHECK(err < 1e-3, "bitlinear err %g", err);
    printf("bitlinear                    : int8 x ternary VNNI exact vs fp32 reference (max err %.1e)\n", err);
}

static void test_bf16_backward(void) {
    Rng r = rng_seed(8); const int K = 64, M = 128, N = 96; BitLinear l;
    bl_init(&l, K, M, N, &r); bl_prepare(&l);
    float *x = xmalloc(4 * N * K), *dy = xmalloc(4 * N * M), *dx32 = xmalloc(4 * N * K), *dx16 = xmalloc(4 * N * K), *g32 = xmalloc(4 * K * M);
    for (int i = 0; i < N * K; i++) x[i] = rng_normal(&r);
    for (int i = 0; i < N * M; i++) dy[i] = rng_normal(&r);
    float *y = xmalloc(4 * N * M);
    g_bf16_bwd = 0; bl_forward(&l, x, N, y); memset(l.W.g, 0, 4 * K * M); bl_backward(&l, dy, dx32); memcpy(g32, l.W.g, 4 * K * M);
    g_bf16_bwd = 1; bl_forward(&l, x, N, y); memset(l.W.g, 0, 4 * K * M); bl_backward(&l, dy, dx16);
    double ex = 0, nx = 0, ew = 0, nw = 0;
    for (int i = 0; i < N * K; i++) { ex += (dx16[i] - dx32[i]) * (dx16[i] - dx32[i]); nx += dx32[i] * dx32[i]; }
    for (int i = 0; i < K * M; i++) { ew += (l.W.g[i] - g32[i]) * (l.W.g[i] - g32[i]); nw += g32[i] * g32[i]; }
    ex = sqrt(ex / nx); ew = sqrt(ew / nw);
    CHECK(ex < 1e-2 && ew < 1e-2, "bf16 backward rel err dx %g dW %g", ex, ew);
    printf("bf16 backward                : vs fp32 backward, relative L2 error dx %.1e, dW %.1e\n", ex, ew);
}

static void test_amx(void) {
    if (!amx_init()) { printf("amx                          : not available, skipped\n"); return; }
    Rng r = rng_seed(12); BitLinear l; const int K = 128, M = 96, N = 64;
    bl_init(&l, K, M, N, &r); bl_prepare(&l);
    float *x = xmalloc(4 * N * K), *y1 = xmalloc(4 * N * M), *y2 = xmalloc(4 * N * M);
    for (int i = 0; i < N * K; i++) x[i] = rng_normal(&r);
    g_amx = 0; bl_forward(&l, x, N, y1); g_amx = 1; bl_forward(&l, x, N, y2);
    double ef = 0; for (int i = 0; i < N * M; i++) ef = fmax(ef, fabs(y1[i] - y2[i]));
    const int Mg = 64, Ng = 96, Kg = 160;
    bf16 *A = xmalloc(2 * Mg * Kg), *Bp = xmalloc(2 * Kg * Ng); float *Af = xmalloc(4 * Mg * Kg), *Bf = xmalloc(4 * Kg * Ng), *C1 = xmalloc(4 * Mg * Ng), *C2 = xmalloc(4 * Mg * Ng);
    for (int i = 0; i < Mg * Kg; i++) Af[i] = rng_normal(&r);
    for (int i = 0; i < Kg * Ng; i++) Bf[i] = rng_normal(&r);
    to_bf16(Af, A, Mg * Kg); pack_pairs_bf16(Bf, Kg, Ng, Ng, NULL, Bp);
    for (int i = 0; i < Mg * Ng; i++) C1[i] = C2[i] = 1.0f;
    g_amx = 0; gemm_bf16(Mg, Ng, Kg, A, Kg, Bp, C1, Ng, 0.5f, 1); g_amx = 1; gemm_bf16(Mg, Ng, Kg, A, Kg, Bp, C2, Ng, 0.5f, 1);
    double eg = 0; for (int i = 0; i < Mg * Ng; i++) eg = fmax(eg, fabs(C1[i] - C2[i]) / (1 + fabs(C1[i])));
    CHECK(ef < 1e-4 && eg < 1e-5, "amx fwd err %g, bf16 gemm err %g", ef, eg);
    printf("amx                          : int8 BitLinear fwd == VNNI (max err %.1e); bf16 tiles == AVX-512 bf16 (rel err %.1e)\n", ef, eg);
}

static void test_gradients(int mode, int K, int dc) {
    g_noquant = 1;
    Config c = {.vocab = 11, .d = 16, .n_layers = 3, .T = 16, .K = K, .mode = mode, .dil_cycle = dc, .seed = 7};
    const int N = 3 * c.T; Model m; model_init(&m, &c, N);
    Rng r = rng_seed(4);
    int *tok = malloc(4 * N), *tgt = malloc(4 * N);
    for (int n = 0; n < N; n++) { tok[n] = rng_u64(&r) % m.V_in; tgt[n] = (mode == MODE_BIDIR && n % 3) ? -1 : (int)(rng_u64(&r) % c.vocab); }
    zero_grads(&m); model_step(&m, tok, tgt, N, 1);
    double worst = 0; int checked = 0;
    for (int p = 0; p < m.n_params; p++) {
        Param *P = m.params[p];
        for (int trial = 0; trial < 8; trial++) {
            const size_t i = rng_u64(&r) % P->n; const float e = 1e-2f, w0 = P->w[i];
            P->w[i] = w0 + e; model_prepare(&m); const double lp = model_step(&m, tok, tgt, N, 0);
            P->w[i] = w0 - e; model_prepare(&m); const double lm = model_step(&m, tok, tgt, N, 0);
            P->w[i] = w0; model_prepare(&m);
            const double num = (lp - lm) / (2 * e), ana = P->g[i];
            if (fabs(num) + fabs(ana) < 3e-4) continue;
            worst = fmax(worst, fabs(num - ana) / (fabs(num) + fabs(ana))); checked++;
        }
    }
    CHECK(worst < 2e-2, "gradcheck mode=%d K=%d dil=%d worst %g", mode, K, dc, worst);
    printf("gradcheck %-6s K=%d dil=%d    : worst rel err %.1e over %d entries\n", mode ? "bidir" : "causal", K, dc, worst, checked);
    g_noquant = 0;
}

// changing token p may only move logits inside the receptive field [p - L*right, p + L*left]
static void test_receptive_field(int mode, int K, int dc) {
    Config c = {.vocab = 11, .d = 32, .n_layers = 4, .T = 64, .K = K, .mode = mode, .dil_cycle = dc, .seed = 9};
    const int N = c.T, p = 30; Model m; model_init(&m, &c, N);
    int tok[64]; Rng r = rng_seed(5);
    for (int n = 0; n < N; n++) tok[n] = rng_u64(&r) % 11;
    float *a = xmalloc(4 * N * 11); model_logits(&m, tok, N); memcpy(a, m.logits, 4 * N * 11);
    tok[p] = (tok[p] + 1) % 11; model_logits(&m, tok, N);
    int sd = 0; for (int l = 0; l < c.n_layers; l++) sd += model_dil(dc, l);
    const int back = mode == MODE_CAUSAL ? 0 : sd * (K / 2), fwd = mode == MODE_CAUSAL ? sd * (K - 1) : sd * (K / 2);
    double outside = 0, inside = 0;
    for (int n = 0; n < N; n++) for (int v = 0; v < 11; v++) {
        const double dl = fabs(a[n * 11 + v] - m.logits[n * 11 + v]);
        if (n < p - back || n > p + fwd) outside = fmax(outside, dl); else inside = fmax(inside, dl);
    }
    CHECK(outside == 0 && inside > 0, "receptive field mode=%d: outside %g inside %g", mode, outside, inside);
    printf("receptive field %-6s K=%d dil=%d: edit at t=%d moves logits only in [%d, %d] (outside %g, inside %.3f)\n",
           mode ? "bidir" : "causal", K, dc, p, p - back, p + fwd, outside, inside);
}

static void test_stream(int K, int dc) {
    Config c = {.vocab = 11, .d = 32, .n_layers = 4, .T = 64, .K = K, .mode = MODE_CAUSAL, .dil_cycle = dc, .seed = 3};
    const int T = c.T, S = 3; Model m; model_init(&m, &c, S * T);
    Rng r = rng_seed(6);
    for (int l = 0; l < c.n_layers; l++) for (size_t i = 0; i < m.blk[l].cw.n; i++) m.blk[l].cw.w[i] += 0.3f * rng_normal(&r);
    int *tok = malloc(sizeof(int) * S * T);
    for (int i = 0; i < S * T; i++) tok[i] = rng_u64(&r) % 11;
    model_logits(&m, tok, S * T);                                          // batch: S sequences of T
    float *ref = xmalloc(4 * S * T * 11); memcpy(ref, m.logits, 4 * S * T * 11);
    Stream s; stream_init(&s, &m, S); double err = 0;
    for (int t = 0; t < T; t++) {
        int step_tok[3]; for (int i = 0; i < S; i++) step_tok[i] = tok[i * T + t];
        stream_step(&s, step_tok);
        for (int i = 0; i < S; i++) for (int v = 0; v < 11; v++) err = fmax(err, fabs(m.logits[i * 11 + v] - ref[(i * T + t) * 11 + v]));
    }
    CHECK(err < 1e-4, "stream vs batch K=%d err %g", K, err);
    printf("streaming decode K=%d dil=%d   : %d streams x %d steps == batch forward (max |logit diff| %.1e)\n", K, dc, S, T, err);
}

static void test_data_parallel(void) {
    Config c = {.vocab = 13, .d = 64, .n_layers = 3, .T = 32, .K = 3, .mode = MODE_CAUSAL, .seed = 21};
    const int N = 8 * c.T, R = 4;
    Model a, m; model_init(&a, &c, N); model_init(&m, &c, N);             // identical weights (same seed)
    Model *reps = calloc(R, sizeof(Model)); for (int i = 0; i < R; i++) model_init_replica(&reps[i], &m, N / R);
    Rng r = rng_seed(22); int *tok = malloc(4 * N), *tgt = malloc(4 * N);
    for (int n = 0; n < N; n++) { tok[n] = rng_u64(&r) % 13; tgt[n] = (n % 5 == 0) ? -1 : (int)(rng_u64(&r) % 13); }
    zero_grads(&a); zero_grads(&m);
    const double la = model_step(&a, tok, tgt, N, 1), lm = train_step_dp(&m, reps, R, tok, tgt, N, c.T);
    double num = 0, den = 0;
    for (int i = 0; i < a.n_params; i++) for (size_t j = 0; j < a.params[i]->n; j++) {
        const double e = a.params[i]->g[j] - m.params[i]->g[j]; num += e * e; den += (double)a.params[i]->g[j] * a.params[i]->g[j];
    }
    const double rel = sqrt(num / den);
    CHECK(fabs(la - lm) < 1e-5 && rel < 1e-5, "data parallel: loss %g vs %g, grad rel err %g", la, lm, rel);
    printf("data-parallel step (%d thr)   : loss %.6f == %.6f, gradients match (rel L2 err %.1e)\n", R, lm, la, rel);
}

int main(void) {
    printf("threads: %d\n", omp_get_max_threads());
    test_gemm(); test_bitlinear(); test_bf16_backward(); test_amx();
    for (int mode = 0; mode < 2; mode++) for (int K = 3; K <= 5; K += 2) test_gradients(mode, K, 0);
    for (int mode = 0; mode < 2; mode++) test_gradients(mode, 3, 3);
    for (int mode = 0; mode < 2; mode++) { test_receptive_field(mode, 3, 0); test_receptive_field(mode, 3, 3); }
    test_receptive_field(MODE_CAUSAL, 4, 0);
    test_stream(3, 0); test_stream(4, 0); test_stream(3, 3); test_stream(4, 4);
    test_data_parallel();
    printf(fails ? "\n%d FAILURES\n" : "\nall tests passed\n", fails);
    return fails != 0;
}
