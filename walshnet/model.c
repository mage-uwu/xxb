#include "wn.h"

static void add_param(Model *m, Param *p) { m->params[m->n_params++] = p; }

void model_init(Model *m, const Config *c, int cap) {
    memset(m, 0, sizeof(*m));
    m->c = *c; m->cap = cap;
    const int d = c->d, L = c->n_layers, V = c->vocab;
    m->V_in = V + (c->mode == MODE_BIDIR);                                 // + [MASK]
    Rng rng = rng_seed(c->seed);
    param_init(&m->emb, (size_t)m->V_in * d, 1.0f);
    for (size_t i = 0; i < m->emb.n; i++) m->emb.w[i] = rng_normal(&rng);
    param_init(&m->head, (size_t)d * V, 1.0f);
    for (size_t i = 0; i < m->head.n; i++) m->head.w[i] = rng_normal(&rng) / sqrtf((float)d);
    m->mix = calloc(L, sizeof(WalshMix));
    if (c->ffn == FFN_MLP) m->mlp = calloc(L, sizeof(MLP));
    if (c->ffn == FFN_DDLGN) m->dl = calloc(L, sizeof(DDLGN));
    if (c->ffn == FFN_CLOPEN) m->cl = calloc(L, sizeof(Clopen));
    m->params = calloc(2 + L * (4 + 2 + (c->ffn == FFN_DDLGN ? c->dl_depth + 1 : 0) + (c->ffn == FFN_CLOPEN ? 2 * c->cl_depth + 1 : 0)), sizeof(Param *));
    add_param(m, &m->emb); add_param(m, &m->head);
    for (int i = 0; i < L; i++) {
        wm_init(&m->mix[i], d, c->T, c->mode, cap, &rng);
        add_param(m, &m->mix[i].inp.W); add_param(m, &m->mix[i].out.W); add_param(m, &m->mix[i].sw); add_param(m, &m->mix[i].h);
        if (c->ffn == FFN_MLP) {
            mlp_init(&m->mlp[i], d, c->hidden, cap, &rng);
            add_param(m, &m->mlp[i].up.W); add_param(m, &m->mlp[i].down.W);
        } else if (c->ffn == FFN_DDLGN) {
            dl_init(&m->dl[i], d, c->dl_nth, c->dl_temp, c->dl_width, c->dl_depth, c->dl_z, c->dl_lr_mul, cap, &rng);
            for (int j = 0; j < c->dl_depth; j++) add_param(m, &m->dl[i].theta[j]);
            add_param(m, &m->dl[i].gain);
        } else if (c->ffn == FFN_CLOPEN) {
            cl_init(&m->cl[i], d, c->cl_nth, c->cl_temp, c->cl_width, c->cl_depth, c->cl_fanin, c->cl_ste, c->cl_lr_mul, cap, &rng);
            for (int j = 0; j < c->cl_depth; j++) { add_param(m, &m->cl[i].w[j]); add_param(m, &m->cl[i].theta[j]); }
            add_param(m, &m->cl[i].gain);
        }
    }
    m->xs = malloc(sizeof(float *) * (2 * L + 1));
    for (int i = 0; i <= 2 * L; i++) m->xs[i] = xmalloc(4 * (size_t)cap * d);
    m->tmp = xmalloc(4 * (size_t)cap * d); m->dx = xmalloc(4 * (size_t)cap * d); m->dtmp = xmalloc(4 * (size_t)cap * d);
    m->xn = xmalloc(4 * (size_t)cap * d); m->r = xmalloc(4 * (size_t)cap); m->logits = xmalloc(4 * (size_t)cap * V);
    model_prepare(m);
}

void model_prepare(Model *m) {
    for (int i = 0; i < m->c.n_layers; i++) {
        wm_prepare(&m->mix[i]);
        if (m->mlp) mlp_prepare(&m->mlp[i]);
    }
}

size_t model_n_params(const Model *m) { size_t s = 0; for (int i = 0; i < m->n_params; i++) s += m->params[i]->n; return s; }

void zero_grads(Model *m) { for (int i = 0; i < m->n_params; i++) memset(m->params[i]->g, 0, 4 * m->params[i]->n); }

static inline void add_into(float *o, const float *a, size_t n) {
    #pragma omp parallel for schedule(static) if (n > 65536)
    for (size_t i = 0; i < n; i++) o[i] += a[i];
}

double model_step(Model *m, const int *tok, const int *tgt, int N, int grad, int hard) {
    const int d = m->c.d, L = m->c.n_layers, V = m->c.vocab;
    if (N > m->cap) { fprintf(stderr, "model_step: N > cap\n"); exit(1); }
    m->N = N; m->tok = tok;
    for (int n = 0; n < N; n++) memcpy(m->xs[0] + (size_t)n * d, m->emb.w + (size_t)tok[n] * d, 4 * (size_t)d);
    for (int i = 0; i < L; i++) {
        const size_t nd = (size_t)N * d;
        wm_forward(&m->mix[i], m->xs[2 * i], N, m->tmp);
        memcpy(m->xs[2 * i + 1], m->xs[2 * i], 4 * nd); add_into(m->xs[2 * i + 1], m->tmp, nd);
        memcpy(m->xs[2 * i + 2], m->xs[2 * i + 1], 4 * nd);
        if (m->mlp) { mlp_forward(&m->mlp[i], m->xs[2 * i + 1], N, m->tmp); add_into(m->xs[2 * i + 2], m->tmp, nd); }
        if (m->dl)  { PROF(P_DLF, dl_forward(&m->dl[i], m->xs[2 * i + 1], N, m->tmp, hard)); add_into(m->xs[2 * i + 2], m->tmp, nd); }
        if (m->cl)  { PROF(P_DLF, cl_forward(&m->cl[i], m->xs[2 * i + 1], N, m->tmp)); add_into(m->xs[2 * i + 2], m->tmp, nd); }
    }
    const float *xf = m->xs[2 * L];
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {                                          // final RMSNorm
        float ss = 0; for (int c = 0; c < d; c++) ss += xf[(size_t)n * d + c] * xf[(size_t)n * d + c];
        const float r = 1.0f / sqrtf(ss / d + 1e-6f); m->r[n] = r;
        for (int c = 0; c < d; c++) m->xn[(size_t)n * d + c] = xf[(size_t)n * d + c] * r;
    }
    PROF(P_HEAD, sgemm(N, V, d, m->xn, d, m->head.w, V, m->logits, V, 0));
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
            const float inv = (float)(1.0 / s) / count;
            for (int v = 0; v < V; v++) lg[v] = expf(lg[v] - mx) * inv;
            lg[tgt[n]] -= 1.0f / count;
        }
    }
    loss /= count;
    if (!grad) return loss;
    // ---- backward
    PROF(P_HEAD, sgemm_tn(d, V, N, m->xn, d, m->logits, V, m->head.g, V, 1));          // head grad
    float *headT = xmalloc(4 * (size_t)V * d); transpose(m->head.w, d, V, headT);
    sgemm(N, d, V, m->logits, V, headT, d, m->tmp, d, 0);                 // d xn
    free(headT);
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float r = m->r[n], *xn = m->xn + (size_t)n * d, *g = m->tmp + (size_t)n * d;
        float dot = 0; for (int c = 0; c < d; c++) dot += g[c] * xn[c];
        dot /= d;
        for (int c = 0; c < d; c++) m->dx[(size_t)n * d + c] = r * (g[c] - xn[c] * dot);
    }
    const size_t nd = (size_t)N * d;
    for (int i = L - 1; i >= 0; i--) {
        if (m->mlp) { mlp_backward(&m->mlp[i], m->dx, m->dtmp); add_into(m->dx, m->dtmp, nd); }
        if (m->dl)  { PROF(P_DLB, dl_backward(&m->dl[i], m->dx, m->dtmp)); add_into(m->dx, m->dtmp, nd); }
        if (m->cl)  { PROF(P_DLB, cl_backward(&m->cl[i], m->dx, m->dtmp)); add_into(m->dx, m->dtmp, nd); }
        wm_backward(&m->mix[i], m->dx, m->dtmp); add_into(m->dx, m->dtmp, nd);
    }
    for (int n = 0; n < N; n++) {                                          // embedding grad
        float *g = m->emb.g + (size_t)tok[n] * d; const float *s = m->dx + (size_t)n * d;
        for (int c = 0; c < d; c++) g[c] += s[c];
    }
    return loss;
}

void adam_step(Model *m, float lr, int t, float b1, float b2, float wd) {
#ifdef WN_PROF
    double _ta = now_sec();
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
    g_prof[P_ADAM] += now_sec() - _ta;
#endif
    model_prepare(m);
}

/* ---- checkpoints: magic, Config, vocab chars[256], params (w only), DDLGN wiring */
int model_save(const Model *m, const char *path, const unsigned char *vocab_chars) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    fwrite("WNT2", 1, 4, f); fwrite(&m->c, sizeof(Config), 1, f); fwrite(vocab_chars, 1, 256, f);
    for (int i = 0; i < m->n_params; i++) fwrite(m->params[i]->w, 4, m->params[i]->n, f);
    if (m->dl) for (int l = 0; l < m->c.n_layers; l++) for (int j = 0; j < m->c.dl_depth; j++) {
        fwrite(m->dl[l].ia[j], 4, m->c.dl_width, f); fwrite(m->dl[l].ib[j], 4, m->c.dl_width, f);
    }
    if (m->cl) for (int l = 0; l < m->c.n_layers; l++) for (int j = 0; j < m->c.cl_depth; j++)
        fwrite(m->cl[l].idx[j], 4, (size_t)m->c.cl_width * m->c.cl_fanin, f);
    fclose(f); return 0;
}

int model_load(Model *m, const char *path, unsigned char *vocab_chars, int cap) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    char magic[4]; Config c;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "WNT2", 4) || fread(&c, sizeof(Config), 1, f) != 1) { fclose(f); return -2; }
    if (fread(vocab_chars, 1, 256, f) != 256) { fclose(f); return -2; }
    model_init(m, &c, cap);
    for (int i = 0; i < m->n_params; i++) if (fread(m->params[i]->w, 4, m->params[i]->n, f) != m->params[i]->n) { fclose(f); return -3; }
    if (m->dl) for (int l = 0; l < c.n_layers; l++) for (int j = 0; j < c.dl_depth; j++) {
        if (fread(m->dl[l].ia[j], 4, c.dl_width, f) != (size_t)c.dl_width || fread(m->dl[l].ib[j], 4, c.dl_width, f) != (size_t)c.dl_width) { fclose(f); return -3; }
    }
    if (m->cl) for (int l = 0; l < c.n_layers; l++) for (int j = 0; j < c.cl_depth; j++) {
        const size_t n = (size_t)c.cl_width * c.cl_fanin;
        if (fread(m->cl[l].idx[j], 4, n, f) != n) { fclose(f); return -3; }
    }
    fclose(f); model_prepare(m); return 0;
}
