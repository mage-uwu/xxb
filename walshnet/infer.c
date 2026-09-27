// walshnet inference: checkpoint -> packed ternary BitLinears (VNNI), Walsh FWHT mixers, hardened bit-sliced DDLGN.
//   infer bench    --model M [--tokens 4096] [--iters 20]
//   infer generate --model M --prompt TEXT [--n 400] [--temp 0.8] [--seed 1]      (causal models)
//   infer fill     --model M --text "Th_ qu_ck br_wn f_x"                          (bidirectional models, '_' = mask)
#include "wn.h"

/* ------------------------------------------------------------ hardened bit-sliced DDLGN */
typedef struct { int n_in, n_out, bounds[17]; int32_t *ia, *ib; } HLayer;       // gates sorted by type
typedef struct { int d, nth, width, depth, k; float th[16]; const float *gain; HLayer *L; int32_t *group_pos; } HDL;

static void hdl_build(HDL *h, const DDLGN *f) {
    h->d = f->d; h->nth = f->nth; h->width = f->width; h->depth = f->depth; h->k = f->k; h->gain = f->gain.w;
    memcpy(h->th, f->th, sizeof(h->th));
    h->L = calloc(f->depth, sizeof(HLayer));
    int32_t *inv_prev = NULL;
    for (int i = 0; i < f->depth; i++) {
        HLayer *l = &h->L[i]; const int W = f->width; const float *th = f->theta[i].w;
        l->n_in = i ? W : f->d * f->nth; l->n_out = W;
        l->ia = xmalloc(4 * (size_t)W); l->ib = xmalloc(4 * (size_t)W);
        int *type = malloc(sizeof(int) * W), cnt[16] = {0}, pos[16];
        for (int g = 0; g < W; g++) {                                     // hard truth table -> 4-bit gate type
            int t = 0; for (int e = 0; e < 4; e++) t |= (th[4 * g + e] > 0) << e;
            type[g] = t; cnt[t]++;
        }
        l->bounds[0] = 0; for (int t = 0; t < 16; t++) l->bounds[t + 1] = l->bounds[t] + cnt[t];
        for (int t = 0; t < 16; t++) pos[t] = l->bounds[t];
        int32_t *inv = malloc(4 * (size_t)W);
        for (int g = 0; g < W; g++) {
            const int p = pos[type[g]]++; inv[g] = p;
            l->ia[p] = inv_prev ? inv_prev[f->ia[i][g]] : f->ia[i][g];
            l->ib[p] = inv_prev ? inv_prev[f->ib[i][g]] : f->ib[i][g];
        }
        free(type); free(inv_prev); inv_prev = inv;
    }
    h->group_pos = inv_prev;                                              // original gate c*k+q -> sorted position
}

#define IMM(t) ((((t) >> 0 & 1) * 0x03) | (((t) >> 1 & 1) * 0x0C) | (((t) >> 2 & 1) * 0x30) | (((t) >> 3 & 1) * 0xC0))
#define GCASE(T) case T: for (int g = lo; g < hi; g++) { \
        const __m512i a = _mm512_load_si512(in + ia[g]), b = _mm512_load_si512(in + ib[g]); \
        _mm512_store_si512(out + g, _mm512_ternarylogic_epi32(a, b, a, IMM(T))); } break;

static void hl_forward(const HLayer *l, const __m512i *in, __m512i *out) {
    const int32_t *ia = l->ia, *ib = l->ib;
    for (int t = 0; t < 16; t++) {
        const int lo = l->bounds[t], hi = l->bounds[t + 1];
        switch (t) { GCASE(0) GCASE(1) GCASE(2) GCASE(3) GCASE(4) GCASE(5) GCASE(6) GCASE(7)
                     GCASE(8) GCASE(9) GCASE(10) GCASE(11) GCASE(12) GCASE(13) GCASE(14) GCASE(15) }
    }
}

static int csa_count(__m512i *buf0, __m512i *buf1, int k, __m512i *cnt) {  // vertical popcount of k planes (in buf0)
    __m512i *bufs[2] = {buf0, buf1}; int n = k, P = 0, cur = 0;
    while (n > 0) {
        __m512i *src = bufs[cur], *car = bufs[cur ^ 1]; int nc = 0, ns = n;
        while (ns >= 3) {
            const __m512i a = src[--ns], b = src[--ns], c = src[--ns];
            src[ns++] = _mm512_ternarylogic_epi32(a, b, c, 0x96);        // sum
            car[nc++] = _mm512_ternarylogic_epi32(a, b, c, 0xE8);        // carry (majority)
        }
        if (ns == 2) { const __m512i a = src[0], b = src[1]; src[0] = _mm512_xor_si512(a, b); car[nc++] = _mm512_and_si512(a, b); ns = 1; }
        cnt[P++] = ns ? src[0] : _mm512_setzero_si512();
        cur ^= 1; n = nc;
    }
    return P;
}

static inline void tr16(__m512 r[16]) {
    for (int b = 8; b >= 1; b >>= 1) {
        int lo[16], hi[16];
        for (int l = 0; l < 16; l++) { lo[l] = (l & b) ? 16 + l - b : l; hi[l] = (l & b) ? 16 + l : l + b; }
        const __m512i il = _mm512_loadu_si512(lo), ih = _mm512_loadu_si512(hi);
        for (int i = 0; i < 16; i++) if (!(i & b)) {
            const __m512 a = r[i], c = r[i + b];
            r[i] = _mm512_permutex2var_ps(a, il, c); r[i + b] = _mm512_permutex2var_ps(a, ih, c);
        }
    }
}

typedef struct { __m512i *A, *B, *g0, *g1; float *ot; } HScratch;

// y[512][d] = hardened DDLGN(x[512][d]) for one 512-token block (512 tokens = 1 bit lane each)
static void hdl_forward512(const HDL *h, const float *x, float *y, HScratch *s) {
    const int d = h->d, nth = h->nth, N = 512;
    float rr[512];
    for (int n = 0; n < N; n++) {
        __m512 ss = _mm512_setzero_ps();
        for (int c = 0; c < d; c += 16) { const __m512 v = _mm512_loadu_ps(x + n * d + c); ss = _mm512_fmadd_ps(v, v, ss); }
        rr[n] = 1.0f / sqrtf(_mm512_reduce_add_ps(ss) / d + 1e-6f);
    }
    for (int q = 0; q < N / 16; q++) {                                    // RMSNorm + thermometer -> bit planes
        const __m512 rv = _mm512_loadu_ps(rr + 16 * q);
        for (int c0 = 0; c0 < d; c0 += 16) {
            __m512 t[16];
            for (int i = 0; i < 16; i++) t[i] = _mm512_loadu_ps(x + (size_t)(16 * q + i) * d + c0);
            tr16(t);
            for (int c = 0; c < 16; c++) {
                const __m512 v = _mm512_mul_ps(t[c], rv);
                for (int j = 0; j < nth; j++)
                    ((uint16_t *)(s->A + (c0 + c) * nth + j))[q] = _mm512_cmp_ps_mask(v, _mm512_set1_ps(h->th[j]), _CMP_GT_OQ);
            }
        }
    }
    __m512i *in = s->A, *out = s->B;
    for (int i = 0; i < h->depth; i++) { hl_forward(&h->L[i], in, out); __m512i *t = in; in = out; out = t; }
    for (int c = 0; c < d; c++) {                                         // GroupSum: vertical popcount per channel
        for (int q = 0; q < h->k; q++) s->g0[q] = in[h->group_pos[c * h->k + q]];
        __m512i cnt[16]; const int P = csa_count(s->g0, s->g1, h->k, cnt);
        const float sc = h->gain[c] / h->k, off = -0.5f * h->gain[c];
        for (int q = 0; q < 8; q++) {
            __m512i bytes = _mm512_setzero_si512();
            for (int p = 0; p < P; p++)
                bytes = _mm512_or_si512(bytes, _mm512_maskz_mov_epi8(((const uint64_t *)&cnt[p])[q], _mm512_set1_epi8((char)(1 << p))));
            for (int u = 0; u < 4; u++) {
                const __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bytes, u)));
                _mm512_store_ps(s->ot + (size_t)c * N + q * 64 + u * 16, _mm512_fmadd_ps(v, _mm512_set1_ps(sc), _mm512_set1_ps(off)));
            }
        }
    }
    for (int q = 0; q < N / 16; q++)                                      // back to token-major
        for (int c0 = 0; c0 < d; c0 += 16) {
            __m512 t[16];
            for (int c = 0; c < 16; c++) t[c] = _mm512_load_ps(s->ot + (size_t)(c0 + c) * N + 16 * q);
            tr16(t);
            for (int i = 0; i < 16; i++) _mm512_storeu_ps(y + (size_t)(16 * q + i) * d + c0, t[i]);
        }
}

static void hdl_forward(const HDL *h, const float *x, int N, float *y, HScratch *sc) {
    if (N % 512) { fprintf(stderr, "bit-sliced DDLGN needs N %% 512 == 0\n"); exit(1); }
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < N / 512; b++) hdl_forward512(h, x + (size_t)b * 512 * h->d, y + (size_t)b * 512 * h->d, &sc[omp_get_thread_num()]);
}

static HScratch *hscratch_alloc(const HDL *h) {
    const int nt = omp_get_max_threads(), rows = h->width > h->d * h->nth ? h->width : h->d * h->nth;
    HScratch *s = calloc(nt, sizeof(HScratch));
    for (int t = 0; t < nt; t++) {
        s[t].A = xmalloc(64 * (size_t)rows); s[t].B = xmalloc(64 * (size_t)rows);
        s[t].g0 = xmalloc(64 * (size_t)h->k); s[t].g1 = xmalloc(64 * (size_t)h->k); s[t].ot = xmalloc(4 * 512 * (size_t)h->d);
    }
    return s;
}

/* ------------------------------------------------------------ inference engine */
typedef struct { Model m; HDL *hdl; HScratch *hs; unsigned char vocab[256]; int stoi[256]; } Engine;

static void engine_load(Engine *e, const char *path, int cap) {
    const int rc = model_load(&e->m, path, e->vocab, cap);
    if (rc) { fprintf(stderr, "cannot load %s (%d)\n", path, rc); exit(1); }
    for (int i = 0; i < 256; i++) e->stoi[i] = -1;
    for (int i = 0; i < e->m.c.vocab; i++) e->stoi[e->vocab[i]] = i;
    e->hdl = NULL;
    if (e->m.c.ffn == FFN_DDLGN) {
        e->hdl = calloc(e->m.c.n_layers, sizeof(HDL));
        for (int l = 0; l < e->m.c.n_layers; l++) hdl_build(&e->hdl[l], &e->m.dl[l]);
        e->hs = hscratch_alloc(&e->hdl[0]);
    }
}

// logits for N tokens (N multiple of T). bitsliced=1 uses the hardened bit-sliced DDLGN (N % 512 == 0).
static void engine_forward(Engine *e, const int *tok, int N, int bitsliced, double *t_mix, double *t_ffn) {
    Model *m = &e->m; const int d = m->c.d, V = m->c.vocab;
    float *x = m->xs[0];
    for (int n = 0; n < N; n++) memcpy(x + (size_t)n * d, m->emb.w + (size_t)tok[n] * d, 4 * (size_t)d);
    for (int l = 0; l < m->c.n_layers; l++) {
        double t0 = now_sec();
        wm_forward(&m->mix[l], x, N, m->tmp);
        for (size_t i = 0; i < (size_t)N * d; i++) x[i] += m->tmp[i];
        double t1 = now_sec();
        if (m->mlp) mlp_forward(&m->mlp[l], x, N, m->tmp);
        else if (m->dl && bitsliced) hdl_forward(&e->hdl[l], x, N, m->tmp, e->hs);
        else if (m->dl) dl_forward(&m->dl[l], x, N, m->tmp, 1);
        if (m->mlp || m->dl) for (size_t i = 0; i < (size_t)N * d; i++) x[i] += m->tmp[i];
        double t2 = now_sec();
        if (t_mix) { *t_mix += t1 - t0; *t_ffn += t2 - t1; }
    }
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        float ss = 0; for (int c = 0; c < d; c++) ss += x[(size_t)n * d + c] * x[(size_t)n * d + c];
        const float r = 1.0f / sqrtf(ss / d + 1e-6f);
        for (int c = 0; c < d; c++) m->xn[(size_t)n * d + c] = x[(size_t)n * d + c] * r;
    }
    sgemm(N, V, d, m->xn, d, m->head.w, V, m->logits, V, 0);
}

static const char *arg(int argc, char **argv, const char *name, const char *def) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: infer bench|generate|fill --model M ...\n"); return 1; }
    const char *cmd = argv[1], *path = arg(argc, argv, "--model", "model.bin");
    if (arg(argc, argv, "--threads", NULL)) omp_set_num_threads(atoi(arg(argc, argv, "--threads", "4")));
    Engine e;
    if (!strcmp(cmd, "bench")) {
        const int N = atoi(arg(argc, argv, "--tokens", "4096")), iters = atoi(arg(argc, argv, "--iters", "20"));
        engine_load(&e, path, N);
        const Config *c = &e.m.c;
        printf("model: walsh-%s + %s | d=%d layers=%d T=%d | %d tokens/batch | %d threads\n", c->mode ? "bidir" : "causal",
               c->ffn == FFN_MLP ? "ternary MLP" : c->ffn == FFN_DDLGN ? "DDLGN" : "none", c->d, c->n_layers, c->T, N, omp_get_max_threads());
        int *tok = malloc(sizeof(int) * N); Rng r = rng_seed(3);
        for (int n = 0; n < N; n++) tok[n] = rng_u64(&r) % c->vocab;
        if (c->ffn == FFN_DDLGN) {                                        // bit-sliced engine must match the float hard path
            const size_t nl = (size_t)N * c->vocab; float *ref = xmalloc(4 * nl);
            engine_forward(&e, tok, N, 0, NULL, NULL); memcpy(ref, e.m.logits, 4 * nl);
            engine_forward(&e, tok, N, 1, NULL, NULL);
            double md = 0; for (size_t i = 0; i < nl; i++) md = fmax(md, fabs(ref[i] - e.m.logits[i]));
            printf("  check: bit-sliced vs float hard path, max |logit diff| = %.2e\n", md);
            free(ref);
        }
        for (int mode = 0; mode < (c->ffn == FFN_DDLGN ? 2 : 1); mode++) {
            const int bs = c->ffn == FFN_DDLGN && mode == 1;
            engine_forward(&e, tok, N, bs, NULL, NULL);
            double best = 1e9, bm = 0, bf = 0;
            for (int it = 0; it < iters; it++) {
                double tm = 0, tf = 0, t0 = now_sec(); engine_forward(&e, tok, N, bs, &tm, &tf);
                const double t = now_sec() - t0; if (t < best) { best = t; bm = tm; bf = tf; }
            }
            printf("  %-34s %10.0f tok/s   (%.3f ms/batch: mixers %.3f, ffn %.3f)\n",
                   c->ffn == FFN_DDLGN ? (bs ? "DDLGN hard, bit-sliced (vpternlog)" : "DDLGN hard, float path") : "forward",
                   N / best, best * 1e3, bm * 1e3, bf * 1e3);
        }
        return 0;
    }
    engine_load(&e, path, 0 + 512);                                       // capacity: one 512-token block
    const Config *c = &e.m.c; const int T = c->T, V = c->vocab;
    if (!strcmp(cmd, "generate")) {
        if (c->mode != MODE_CAUSAL) { fprintf(stderr, "generate needs a causal model\n"); return 1; }
        const char *prompt = arg(argc, argv, "--prompt", "\n"); const int n_new = atoi(arg(argc, argv, "--n", "400"));
        const float temp = atof(arg(argc, argv, "--temp", "0.8")); Rng r = rng_seed(atoi(arg(argc, argv, "--seed", "1")));
        int len = 0, *seq = malloc(sizeof(int) * (strlen(prompt) + n_new + 1));
        for (const char *p = prompt; *p; p++) if (e.stoi[(unsigned char)*p] >= 0) seq[len++] = e.stoi[(unsigned char)*p];
        if (!len) seq[len++] = 0;
        fputs(prompt, stdout);
        int *win = malloc(sizeof(int) * 512); double t0 = now_sec();
        for (int i = 0; i < n_new; i++) {                                 // sliding window of T tokens, left-aligned
            const int start = len > T ? len - T : 0, w = len - start;
            for (int t = 0; t < T; t++) win[t] = t < w ? seq[start + t] : 0;
            for (int t = T; t < 512; t++) win[t] = 0;
            engine_forward(&e, win, c->ffn == FFN_DDLGN ? 512 : T, 1, NULL, NULL);
            const float *lg = e.m.logits + (size_t)(w - 1) * V;
            float mx = lg[0]; for (int v = 1; v < V; v++) mx = fmaxf(mx, lg[v]);
            double sum = 0; float pr[256]; for (int v = 0; v < V; v++) { pr[v] = expf((lg[v] - mx) / temp); sum += pr[v]; }
            double u = rng_unif(&r) * sum; int pick = V - 1;
            for (int v = 0; v < V; v++) { u -= pr[v]; if (u <= 0) { pick = v; break; } }
            seq[len++] = pick; putchar(e.vocab[pick]); fflush(stdout);
        }
        fprintf(stderr, "\n[%d tokens in %.2fs]\n", n_new, now_sec() - t0);
        return 0;
    }
    if (!strcmp(cmd, "fill")) {
        if (c->mode != MODE_BIDIR) { fprintf(stderr, "fill needs a bidirectional model\n"); return 1; }
        const char *text = arg(argc, argv, "--text", "Th_ qu_ck br_wn f_x"); const int L = strlen(text);
        if (L > T) { fprintf(stderr, "text longer than T=%d\n", T); return 1; }
        int *win = calloc(512, sizeof(int));
        for (int t = 0; t < T; t++) win[t] = t < L ? (text[t] == '_' ? V : (e.stoi[(unsigned char)text[t]] >= 0 ? e.stoi[(unsigned char)text[t]] : 0)) : e.stoi[' '] >= 0 ? e.stoi[' '] : 0;
        engine_forward(&e, win, c->ffn == FFN_DDLGN ? 512 : T, 1, NULL, NULL);
        printf("input : %s\noutput: ", text);
        for (int t = 0; t < L; t++) {
            if (text[t] != '_') { putchar(text[t]); continue; }
            const float *lg = e.m.logits + (size_t)t * V; int best = 0;
            for (int v = 1; v < V; v++) if (lg[v] > lg[best]) best = v;
            putchar(e.vocab[best]);
        }
        putchar('\n');
        return 0;
    }
    fprintf(stderr, "unknown command %s\n", cmd); return 1;
}
