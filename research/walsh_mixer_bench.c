// Walsh-mixer blocks vs attention, combined with ternary MLP / DDLGN FFNs. Reuses kernels from kern.c.
// Build: gcc -O3 -march=native -ffast-math walsh.c -lm
#define main kern_main
#include "kern_bench.c"
#undef main

/* All token-mixing kernels use layout [N][d] (token-major), sequences of T contiguous tokens, d % 16 == 0.
   Every inner loop runs over channels, 16 per zmm. */

// in-place unnormalized FWHT along the sequence axis of rows x[0..n), each row d floats
static void fwht_rows(float *x, int n, int d) {
    for (int h = 1; h < n; h <<= 1)
        for (int i = 0; i < n; i += 2 * h)
            for (int j = i; j < i + h; j++) {
                float *a = x + (size_t)j * d, *b = x + (size_t)(j + h) * d;
                for (int c = 0; c < d; c += 16) {
                    __m512 va = _mm512_load_ps(a + c), vb = _mm512_load_ps(b + c);
                    _mm512_store_ps(a + c, _mm512_add_ps(va, vb)); _mm512_store_ps(b + c, _mm512_sub_ps(va, vb));
                }
            }
}

static void mul_rows(float *x, const float *s, int n, int d) {          // x[t][c] *= s[t][c]
    for (size_t i = 0; i < (size_t)n * d; i += 16)
        _mm512_store_ps(x + i, _mm512_mul_ps(_mm512_load_ps(x + i), _mm512_load_ps(s + i)));
}

// bidirectional dyadic conv: y[t] = sum_s h[t^s] u[s]  ==  WHT^-1( WHT(h) * WHT(u) );  Hs = WHT(h) / T precomputed
static void walsh_bidir(const float *u, float *y, int N, int T, int d, const float *Hs) {
    memcpy(y, u, sizeof(float) * (size_t)N * d);
    for (int b = 0; b < N / T; b++) {
        float *yb = y + (size_t)b * T * d;
        fwht_rows(yb, T, d); mul_rows(yb, Hs, T, d); fwht_rows(yb, T, d);
    }
}

// causal dyadic conv: y[t] = sum_{s<=t} h[t^s] u[s], divide and conquer O(T log^2 T).
// Block of size n: y_R += fullconv(u_L, h[n/2 .. n)); recurse into both halves. Base case n <= 8 is direct.
typedef struct { int T, d; float *h; float *Hhi; int *off; } WalshCausal;   // Hhi: WHT(h[n/2..n))/(n/2) for n = 16..T

static void wc_init(WalshCausal *w, const float *h, int T, int d) {       // h: [T][d]
    w->T = T; w->d = d; w->h = amalloc(4 * (size_t)T * d); memcpy(w->h, h, 4 * (size_t)T * d);
    w->Hhi = amalloc(4 * (size_t)T * d); w->off = malloc(sizeof(int) * 32);
    int o = 0;
    for (int n = 16, lv = 0; n <= T; n <<= 1, lv++) {
        int m = n / 2; w->off[lv] = o;
        memcpy(w->Hhi + (size_t)o * d, h + (size_t)m * d, 4 * (size_t)m * d);
        fwht_rows(w->Hhi + (size_t)o * d, m, d);
        for (size_t i = 0; i < (size_t)m * d; i++) w->Hhi[(size_t)o * d + i] /= m;
        o += m;
    }
}

static void wc_rec(const WalshCausal *w, const float *u, float *y, int n, int lv_of_n, float *scratch) {
    const int d = w->d;
    if (n <= 8) {                                                          // direct: y[t] = sum_{s<=t} h[t^s] u[s]
        for (int t = 0; t < n; t++) {
            float *yt = y + (size_t)t * d;
            for (int s = 0; s <= t; s++) {
                const float *hk = w->h + (size_t)(t ^ s) * d, *us = u + (size_t)s * d;
                for (int c = 0; c < d; c += 16)
                    _mm512_store_ps(yt + c, _mm512_fmadd_ps(_mm512_load_ps(hk + c), _mm512_load_ps(us + c), _mm512_load_ps(yt + c)));
            }
        }
        return;
    }
    const int m = n / 2;
    wc_rec(w, u, y, m, lv_of_n - 1, scratch);
    wc_rec(w, u + (size_t)m * d, y + (size_t)m * d, m, lv_of_n - 1, scratch);
    memcpy(scratch, u, 4 * (size_t)m * d);                                // cross term: first half -> second half
    fwht_rows(scratch, m, d);
    mul_rows(scratch, w->Hhi + (size_t)w->off[lv_of_n] * d, m, d);
    fwht_rows(scratch, m, d);
    float *yr = y + (size_t)m * d;
    for (size_t i = 0; i < (size_t)m * d; i += 16)
        _mm512_store_ps(yr + i, _mm512_add_ps(_mm512_load_ps(yr + i), _mm512_load_ps(scratch + i)));
}

static void walsh_causal(const WalshCausal *w, const float *u, float *y, int N, float *scratch) {
    const int T = w->T, d = w->d; int lv = 0; for (int n = 16; n < T; n <<= 1) lv++;
    memset(y, 0, sizeof(float) * (size_t)N * d);
    for (int b = 0; b < N / T; b++) wc_rec(w, u + (size_t)b * T * d, y + (size_t)b * T * d, T, lv, scratch);
}

// ordinary causal long conv (Hyena-style control), direct: y[t] = sum_{s<=t} h[t-s] u[s]
static void toeplitz_causal(const float *u, float *y, int N, int T, int d, const float *h) {
    for (int b = 0; b < N / T; b++)
        for (int t = 0; t < T; t++) {
            float *yt = y + ((size_t)b * T + t) * d;
            for (int c = 0; c < d; c += 16) {
                __m512 acc = _mm512_setzero_ps();
                for (int s = 0; s <= t; s++)
                    acc = _mm512_fmadd_ps(_mm512_load_ps(h + (size_t)(t - s) * d + c), _mm512_load_ps(u + ((size_t)b * T + s) * d + c), acc);
                _mm512_store_ps(yt + c, acc);
            }
        }
}

// Walsh block: [u|g] = BitLinear(x); u <- causal 3-tap depthwise conv; v = longconv(u); out = BitLinear(v * g)
enum { MIX_WALSH_BIDIR, MIX_WALSH_CAUSAL, MIX_SHORT_ONLY, MIX_TOEPLITZ };
typedef struct {
    int d, T, kind; BitLinear inp, out; float *sw /*[3][d]*/, *h /*[T][d]*/, *Hs; WalshCausal wc;
    float *ug, *us, *v, *scratch; uint8_t *xu; float *sx;
} WalshBlock;

static void wb_init(WalshBlock *w, int d, int T, int N, int kind) {
    w->d = d; w->T = T; w->kind = kind;
    bl_init(&w->inp, d, 2 * d); bl_init(&w->out, d, d);
    w->sw = amalloc(4 * 3 * (size_t)d); w->h = amalloc(4 * (size_t)T * d); w->Hs = amalloc(4 * (size_t)T * d);
    for (int i = 0; i < 3 * d; i++) w->sw[i] = frnd() * 0.5f;
    for (int i = 0; i < T * d; i++) w->h[i] = frnd() * 0.2f;
    memcpy(w->Hs, w->h, 4 * (size_t)T * d); fwht_rows(w->Hs, T, d);
    for (int i = 0; i < T * d; i++) w->Hs[i] /= T;
    wc_init(&w->wc, w->h, T, d);
    w->ug = amalloc(4 * (size_t)N * 2 * d); w->us = amalloc(4 * (size_t)N * d); w->v = amalloc(4 * (size_t)N * d);
    w->scratch = amalloc(4 * (size_t)T * d); w->xu = amalloc((size_t)N * 2 * d); w->sx = amalloc(4 * (size_t)N);
}

static void wb_forward(WalshBlock *w, const float *x, float *y, int N) {
    const int d = w->d, T = w->T;
    bl_forward(&w->inp, x, N, w->ug, w->xu, w->sx, 1);
    for (int n = 0; n < N; n++) {                                           // short conv, zero-padded per sequence
        const int t = n % T; float *o = w->us + (size_t)n * d;
        for (int c = 0; c < d; c += 16) {
            __m512 acc = _mm512_mul_ps(_mm512_load_ps(w->sw + c), _mm512_loadu_ps(w->ug + (size_t)n * 2 * d + c));
            if (t >= 1) acc = _mm512_fmadd_ps(_mm512_load_ps(w->sw + d + c), _mm512_loadu_ps(w->ug + (size_t)(n - 1) * 2 * d + c), acc);
            if (t >= 2) acc = _mm512_fmadd_ps(_mm512_load_ps(w->sw + 2 * d + c), _mm512_loadu_ps(w->ug + (size_t)(n - 2) * 2 * d + c), acc);
            _mm512_store_ps(o + c, acc);
        }
    }
    const float *v = w->v;
    switch (w->kind) {
        case MIX_WALSH_BIDIR:  walsh_bidir(w->us, w->v, N, T, d, w->Hs); break;
        case MIX_WALSH_CAUSAL: walsh_causal(&w->wc, w->us, w->v, N, w->scratch); break;
        case MIX_TOEPLITZ:     toeplitz_causal(w->us, w->v, N, T, d, w->h); break;
        case MIX_SHORT_ONLY:   v = w->us; break;
    }
    for (int n = 0; n < N; n++)                                             // gate (reuse us as z buffer)
        for (int c = 0; c < d; c += 16)
            _mm512_store_ps(w->us + (size_t)n * d + c, _mm512_mul_ps(_mm512_load_ps(v + (size_t)n * d + c), _mm512_loadu_ps(w->ug + (size_t)n * 2 * d + d + c)));
    bl_forward(&w->out, w->us, N, y, w->xu, w->sx, 1);
}

/* ---------------- checks vs scalar ---------------- */
static int check_mixers(void) {
    const int T = 64, d = 32, N = 128; int bad = 0;
    float *u = amalloc(4 * N * d), *y = amalloc(4 * N * d), *h = amalloc(4 * T * d), *Hs = amalloc(4 * T * d), *scr = amalloc(4 * T * d);
    for (int i = 0; i < N * d; i++) u[i] = frnd();
    for (int i = 0; i < T * d; i++) h[i] = frnd();
    memcpy(Hs, h, 4 * T * d); fwht_rows(Hs, T, d); for (int i = 0; i < T * d; i++) Hs[i] /= T;
    WalshCausal wc; wc_init(&wc, h, T, d);
    const char *names[] = {"walsh bidir", "walsh causal", "toeplitz causal"};
    for (int kind = 0; kind < 3; kind++) {
        if (kind == 0) walsh_bidir(u, y, N, T, d, Hs);
        if (kind == 1) walsh_causal(&wc, u, y, N, scr);
        if (kind == 2) toeplitz_causal(u, y, N, T, d, h);
        int b = 0;
        for (int s0 = 0; s0 < N; s0 += T) for (int t = 0; t < T; t++) for (int c = 0; c < d; c++) {
            double ref = 0;
            for (int s = 0; s < T; s++) {
                if (kind == 0) ref += h[(t ^ s) * d + c] * u[(s0 + s) * d + c];
                if (kind == 1 && s <= t) ref += h[(t ^ s) * d + c] * u[(s0 + s) * d + c];
                if (kind == 2 && s <= t) ref += h[(t - s) * d + c] * u[(s0 + s) * d + c];
            }
            if (fabs(ref - y[(s0 + t) * d + c]) > 1e-3) b++;
        }
        printf("check %s: %s\n", names[kind], b ? "FAIL" : "ok"); bad += b;
    }
    return bad;
}

int main(void) {
    if (check() || check_mixers()) return 1;
    const int N = 512;
    int configs[][3] = {{64, 64, 4}, {128, 128, 2}, {128, 512, 2}};     // {d, T, heads}
    for (int ci = 0; ci < 3; ci++) {
        const int d = configs[ci][0], T = configs[ci][1], H = configs[ci][2], dh = d / H, hid = 4 * d;
        printf("\n== d=%d  seq=%d  heads=%d  ffn_hidden=%d  | 512 tokens/batch, 1 thread, us per 512 tokens ==\n", d, T, H, hid);
        BitLinear qkv, o, up, down; bl_init(&qkv, d, 3 * d); bl_init(&o, d, d); bl_init(&up, d, hid); bl_init(&down, hid, d);
        float *x = amalloc(4 * (size_t)N * d), *y3 = amalloc(4 * (size_t)N * 3 * d), *att = amalloc(4 * (size_t)N * d),
              *hbuf = amalloc(4 * (size_t)N * hid), *y = amalloc(4 * (size_t)N * d), *sx = amalloc(4 * N),
              *Kt = amalloc(4 * (size_t)dh * T), *s = amalloc(4 * (size_t)QB * T);
        uint8_t *xu = amalloc((size_t)N * hid);
        for (int i = 0; i < N * d; i++) x[i] = frnd();
        for (int i = 0; i < N * 3 * d; i++) y3[i] = frnd();
        double mix[5];
        mix[0] = BENCH("mixer: attention (proj + core)", 30,
                       bl_forward(&qkv, x, N, y3, xu, sx, 1); attention(y3, att, N, T, H, dh, Kt, s); bl_forward(&o, att, N, y, xu, sx, 1));
        const char *mnames[] = {"", "mixer: WALSH bidirectional", "mixer: WALSH causal (D&C)", "mixer: short conv only", "mixer: toeplitz causal (direct)"};
        const int kinds[] = {0, MIX_WALSH_BIDIR, MIX_WALSH_CAUSAL, MIX_SHORT_ONLY, MIX_TOEPLITZ};
        for (int k = 1; k < 5; k++) { WalshBlock wb; wb_init(&wb, d, T, N, kinds[k]); mix[k] = BENCH(mnames[k], 100, wb_forward(&wb, x, y, N)); }
        double ffn[3];
        ffn[0] = BENCH("ffn: ternary MLP", 100,
                       bl_forward(&up, x, N, hbuf, xu, sx, 1);
                       for (int i = 0; i < N * hid; i++) { float v = hbuf[i] > 0 ? hbuf[i] : 0; hbuf[i] = v * v; }
                       bl_forward(&down, hbuf, N, y, xu, sx, 1));
        int widths[] = {8192 / 4, 32768 / 4};                    // 8K and 32K gates (4 layers) at every d
        for (int wi = 0; wi < 2; wi++) {
            DLGN m; dlgn_init(&m, d, 8, widths[wi], 4);
            int maxw = widths[wi] > d * 8 ? widths[wi] : d * 8;
            __m512i *A = amalloc(64 * (size_t)maxw), *B = amalloc(64 * (size_t)maxw);
            float *xt = amalloc(4 * (size_t)N * d), *ot = amalloc(4 * (size_t)N * d);
            char label[64]; snprintf(label, 64, "ffn: DDLGN %dK gates", widths[wi] * 4 / 1024);
            ffn[1 + wi] = BENCH(label, 200, dlgn_forward(&m, x, xt, A, B, ot));
        }
        const char *fn[] = {"MLP", "DDLGN-8K", "DDLGN-32K"}, *mn[] = {"attention", "walsh-bidir", "walsh-causal", "short-only", "toeplitz"};
        const double base = mix[0] + ffn[0];
        printf("  layer throughput vs BitNet (attention + MLP) = %.0f us:\n  %-14s", base * 1e6, "");
        for (int f = 0; f < 3; f++) printf("%12s", fn[f]);
        printf("\n");
        for (int k = 0; k < 5; k++) {
            printf("  %-14s", mn[k]);
            for (int f = 0; f < 3; f++) printf("%11.2fx", base / (mix[k] + ffn[f]));
            printf("\n");
        }
        printf("  tokens/s/core (2-layer model): attention+MLP %.0fk | walsh-causal+DDLGN-8K %.0fk | walsh-bidir+DDLGN-8K %.0fk\n",
               512 / (2 * base) / 1e3, 512 / (2 * (mix[2] + ffn[1])) / 1e3, 512 / (2 * (mix[1] + ffn[1])) / 1e3);
    }
    return 0;
}
