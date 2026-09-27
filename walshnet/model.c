#include "wn.h"

void model_init(Model *m, const Config *c, int cap) {
    memset(m, 0, sizeof(*m));
    static int amx_checked = 0; if (!amx_checked) { amx_init(); amx_checked = 1; }
    m->c = *c; m->cap = cap;
    const int d = c->d, L = c->n_layers, V = c->vocab;
    m->V_in = V + (c->mode == MODE_BIDIR);                                 // + [MASK]
    Rng rng = rng_seed(c->seed);
    param_init(&m->emb, (size_t)m->V_in * d, 1.0f);
    for (size_t i = 0; i < m->emb.n; i++) m->emb.w[i] = rng_normal(&rng);
    param_init(&m->head, (size_t)d * V, 1.0f);
    for (size_t i = 0; i < m->head.n; i++) m->head.w[i] = rng_normal(&rng) / sqrtf((float)d);
    m->blk = calloc(L, sizeof(Block));
    m->params = calloc(2 + 3 * L, sizeof(Param *));
    m->params[m->n_params++] = &m->emb; m->params[m->n_params++] = &m->head;
    for (int i = 0; i < L; i++) {
        blk_init(&m->blk[i], d, c->T, c->K, c->mode, model_dil(c->dil_cycle, i), cap, &rng);
        m->params[m->n_params++] = &m->blk[i].inp.W; m->params[m->n_params++] = &m->blk[i].out.W; m->params[m->n_params++] = &m->blk[i].cw;
    }
    m->xs = malloc(sizeof(float *) * (L + 1));
    for (int i = 0; i <= L; i++) m->xs[i] = xmalloc(4 * (size_t)cap * d);
    m->tmp = xmalloc(4 * (size_t)cap * d); m->dx = xmalloc(4 * (size_t)cap * d); m->dtmp = xmalloc(4 * (size_t)cap * d);
    m->xn = xmalloc(4 * (size_t)cap * d); m->r = xmalloc(4 * (size_t)cap); m->logits = xmalloc(4 * (size_t)cap * V);
    m->headT = xmalloc(4 * (size_t)V * d);
    model_prepare(m);
}

void model_prepare(Model *m) { for (int i = 0; i < m->c.n_layers; i++) blk_prepare(&m->blk[i]); }

size_t model_n_params(const Model *m) { size_t s = 0; for (int i = 0; i < m->n_params; i++) s += m->params[i]->n; return s; }

void zero_grads(Model *m) { for (int i = 0; i < m->n_params; i++) memset(m->params[i]->g, 0, 4 * m->params[i]->n); }

static void forward(Model *m, const int *tok, int N) {
    const int d = m->c.d, L = m->c.n_layers, V = m->c.vocab;
    if (N > m->cap) { fprintf(stderr, "model: N=%d > cap=%d\n", N, m->cap); exit(1); }
    m->N = N;
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) memcpy(m->xs[0] + (size_t)n * d, m->emb.w + (size_t)tok[n] * d, 4 * (size_t)d);
    for (int i = 0; i < L; i++) {
        blk_forward(&m->blk[i], m->xs[i], N, m->tmp);
        float *xo = m->xs[i + 1]; const float *xi = m->xs[i], *t = m->tmp;
        #pragma omp parallel for schedule(static)
        for (size_t e = 0; e < (size_t)N * d; e += 16) _mm512_storeu_ps(xo + e, _mm512_add_ps(_mm512_loadu_ps(xi + e), _mm512_loadu_ps(t + e)));
    }
    const float *xf = m->xs[L];
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {                                          // final RMSNorm
        __m512 ss = _mm512_setzero_ps();
        for (int c = 0; c < d; c += 16) { const __m512 v = _mm512_loadu_ps(xf + (size_t)n * d + c); ss = _mm512_fmadd_ps(v, v, ss); }
        const float r = 1.0f / sqrtf(_mm512_reduce_add_ps(ss) / d + 1e-6f); m->r[n] = r;
        for (int c = 0; c < d; c += 16) _mm512_storeu_ps(m->xn + (size_t)n * d + c, _mm512_mul_ps(_mm512_loadu_ps(xf + (size_t)n * d + c), _mm512_set1_ps(r)));
    }
    PROF(P_HEAD, sgemm(N, V, d, m->xn, d, m->head.w, V, m->logits, V, 0));
}

void model_logits(Model *m, const int *tok, int N) { forward(m, tok, N); }

double model_step(Model *m, const int *tok, const int *tgt, int N, int grad) {
    const int d = m->c.d, L = m->c.n_layers, V = m->c.vocab;
    forward(m, tok, N);
    int count = 0; for (int n = 0; n < N; n++) count += tgt[n] >= 0;
    if (!count) return 0;
    double loss = 0;
    #pragma omp parallel for schedule(static) reduction(+:loss)
    for (int n = 0; n < N; n++) {                                          // softmax cross-entropy
        float *lg = m->logits + (size_t)n * V;
        if (tgt[n] < 0) { if (grad) memset(lg, 0, 4 * (size_t)V); continue; }
        float mx = lg[0]; for (int v = 1; v < V; v++) mx = fmaxf(mx, lg[v]);
        double s = 0; for (int v = 0; v < V; v++) s += expf(lg[v] - mx);
        loss += log(s) - (lg[tgt[n]] - mx);
        if (grad) {
            const float nrm = m->loss_norm > 0 ? m->loss_norm : (float)count;
            const float inv = (float)(1.0 / s) / nrm;
            for (int v = 0; v < V; v++) lg[v] = expf(lg[v] - mx) * inv;
            lg[tgt[n]] -= 1.0f / nrm;
        }
    }
    loss /= count;
    if (!grad) return loss;
    PROF(P_HEAD, sgemm_tn(d, V, N, m->xn, d, m->logits, V, m->head.g, V, 1));
    transpose(m->head.w, d, V, m->headT);
    PROF(P_HEAD, sgemm(N, d, V, m->logits, V, m->headT, d, m->tmp, d, 0));
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {                                          // final RMSNorm backward
        const float r = m->r[n], *xn = m->xn + (size_t)n * d, *g = m->tmp + (size_t)n * d;
        float dot = 0; for (int c = 0; c < d; c++) dot += g[c] * xn[c];
        dot /= d;
        for (int c = 0; c < d; c++) m->dx[(size_t)n * d + c] = r * (g[c] - xn[c] * dot);
    }
    for (int i = L - 1; i >= 0; i--) {                                     // residual: dx += block^T(dx)
        blk_backward(&m->blk[i], m->dx, m->dtmp);
        float *dx = m->dx; const float *t = m->dtmp;
        #pragma omp parallel for schedule(static)
        for (size_t e = 0; e < (size_t)N * d; e += 16) _mm512_storeu_ps(dx + e, _mm512_add_ps(_mm512_loadu_ps(dx + e), _mm512_loadu_ps(t + e)));
    }
    for (int n = 0; n < N; n++) {                                          // embedding grad
        float *g = m->emb.g + (size_t)tok[n] * d; const float *s = m->dx + (size_t)n * d;
        for (int c = 0; c < d; c++) g[c] += s[c];
    }
    return loss;
}

void adam_step(Model *m, float lr, int t, float b1, float b2, float wd) {
#ifdef WN_PROF
    const double t0 = now_sec();
#endif
    const float c1 = 1.0f / (1.0f - powf(b1, t)), c2 = 1.0f / (1.0f - powf(b2, t));
    for (int i = 0; i < m->n_params; i++) {
        Param *p = m->params[i]; const float plr = lr * p->lr_mul;
        #pragma omp parallel for schedule(static) if (p->n > 65536)
        for (size_t j = 0; j < p->n; j++) {
            const float g = p->g[j];
            p->m[j] = b1 * p->m[j] + (1 - b1) * g;
            p->v[j] = b2 * p->v[j] + (1 - b2) * g * g;
            p->w[j] -= plr * ((p->m[j] * c1) / (sqrtf(p->v[j] * c2) + 1e-8f) + wd * p->w[j]);
            p->g[j] = 0;
        }
    }
#ifdef WN_PROF
    g_prof[P_ADAM] += now_sec() - t0;
#endif
    model_prepare(m);
}

/* checkpoint: magic "WSC2", Config, vocab chars[256], params (weights only) */
int model_save(const Model *m, const char *path, const unsigned char *vocab_chars) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    fwrite("WSC2", 1, 4, f); fwrite(&m->c, sizeof(Config), 1, f); fwrite(vocab_chars, 1, 256, f);
    for (int i = 0; i < m->n_params; i++) fwrite(m->params[i]->w, 4, m->params[i]->n, f);
    fclose(f); return 0;
}

int model_load(Model *m, const char *path, unsigned char *vocab_chars, int cap) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    char magic[4]; Config c;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "WSC2", 4) || fread(&c, sizeof(Config), 1, f) != 1 || fread(vocab_chars, 1, 256, f) != 256) { fclose(f); return -2; }
    model_init(m, &c, cap);
    for (int i = 0; i < m->n_params; i++) if (fread(m->params[i]->w, 4, m->params[i]->n, f) != m->params[i]->n) { fclose(f); return -3; }
    fclose(f); model_prepare(m); return 0;
}

/* ---- data-parallel replicas ---- */
static void share_bl(BitLinear *r, const BitLinear *m) {
    free(r->Wt); free(r->colsum); free(r->WqT); free(r->Wb); free(r->tmpT); free(r->tT);
    r->Wt = m->Wt; r->colsum = m->colsum; r->WqT = m->WqT; r->Wb = m->Wb; r->tmpT = r->tT = NULL; r->ws = m->ws;
}

void model_init_replica(Model *r, Model *master, int cap) {
    model_init(r, &master->c, cap);
    for (int i = 0; i < r->n_params; i++) { free(r->params[i]->w); free(r->params[i]->m); free(r->params[i]->v); r->params[i]->w = master->params[i]->w; r->params[i]->m = r->params[i]->v = NULL; }
    for (int l = 0; l < r->c.n_layers; l++) { share_bl(&r->blk[l].inp, &master->blk[l].inp); share_bl(&r->blk[l].out, &master->blk[l].out); }
}

void replicas_sync(Model *reps, int R, const Model *master) {
    for (int i = 0; i < R; i++) for (int l = 0; l < master->c.n_layers; l++) {
        reps[i].blk[l].inp.ws = master->blk[l].inp.ws; reps[i].blk[l].out.ws = master->blk[l].out.ws;
    }
}

double train_step_dp(Model *master, Model *reps, int R, const int *tok, const int *tgt, int N, int T) {
    const int nseq = N / T, per = (nseq + R - 1) / R;
    int total = 0; for (int n = 0; n < N; n++) total += tgt[n] >= 0;
    double loss = 0;
    const int saved = omp_get_max_active_levels(); omp_set_max_active_levels(1);   // inner regions run on 1 thread
    #pragma omp parallel num_threads(R) reduction(+:loss)
    {
        const int t = omp_get_thread_num(), s0 = t * per, s1 = s0 + per < nseq ? s0 + per : nseq;
        if (s0 < s1) {
            Model *r = &reps[t]; const int n0 = s0 * T, nn = (s1 - s0) * T;
            int cnt = 0; for (int n = n0; n < n0 + nn; n++) cnt += tgt[n] >= 0;
            r->loss_norm = (float)total;
            loss += model_step(r, tok + n0, tgt + n0, nn, 1) * cnt;
        }
    }
    omp_set_max_active_levels(saved);
    for (int i = 0; i < master->n_params; i++) {                           // reduce grads into the master
        Param *p = master->params[i];
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < p->n; j++) {
            float s = 0; for (int t = 0; t < R; t++) { s += reps[t].params[i]->g[j]; reps[t].params[i]->g[j] = 0; }
            p->g[j] += s;
        }
    }
    return total ? loss / total : 0;
}
