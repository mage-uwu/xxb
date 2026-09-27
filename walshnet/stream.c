#include "wn.h"

void stream_init(Stream *s, Model *m, int S) {
    if (m->c.mode != MODE_CAUSAL) { fprintf(stderr, "stream: needs a causal model\n"); exit(1); }
    if (S > m->cap) { fprintf(stderr, "stream: S > model capacity\n"); exit(1); }
    const int d = m->c.d, L = m->c.n_layers, K = m->c.K;
    s->m = m; s->S = S;
    s->x = xmalloc(4 * (size_t)S * d); s->y = xmalloc(4 * (size_t)S * d); s->ug = xmalloc(8 * (size_t)S * d); s->z = xmalloc(4 * (size_t)S * d);
    s->hist = xmalloc(4 * (size_t)L * (K > 1 ? K - 1 : 1) * S * d);
    stream_reset(s);
}

void stream_reset(Stream *s) {
    const Config *c = &s->m->c;
    memset(s->hist, 0, 4 * (size_t)c->n_layers * (c->K > 1 ? c->K - 1 : 1) * s->S * c->d);
    s->pos = 0;
}

void stream_step(Stream *s, const int *tok) {
    Model *m = s->m; const int d = m->c.d, L = m->c.n_layers, K = m->c.K, S = s->S, V = m->c.vocab, d2 = 2 * d;
    for (int i = 0; i < S; i++) memcpy(s->x + (size_t)i * d, m->emb.w + (size_t)tok[i] * d, 4 * (size_t)d);
    for (int l = 0; l < L; l++) {
        Block *b = &m->blk[l]; float *H = s->hist + (size_t)l * (K - 1) * S * d;     // H[j-1] = u at t - j
        bl_forward(&b->inp, s->x, S, s->ug);
        const float *cw = b->cw.w;
        for (int i = 0; i < S; i++)
            for (int c = 0; c < d; c += 16) {
                __m512 a = _mm512_mul_ps(_mm512_loadu_ps(cw + c), _mm512_loadu_ps(s->ug + (size_t)i * d2 + c));
                for (int j = 1; j < K; j++)
                    a = _mm512_fmadd_ps(_mm512_loadu_ps(cw + j * d + c), _mm512_loadu_ps(H + ((size_t)(j - 1) * S + i) * d + c), a);
                _mm512_storeu_ps(s->z + (size_t)i * d + c, _mm512_mul_ps(a, _mm512_loadu_ps(s->ug + (size_t)i * d2 + d + c)));
            }
        for (int j = K - 1; j >= 2; j--) memcpy(H + (size_t)(j - 1) * S * d, H + (size_t)(j - 2) * S * d, 4 * (size_t)S * d);
        if (K > 1) for (int i = 0; i < S; i++) memcpy(H + (size_t)i * d, s->ug + (size_t)i * d2, 4 * (size_t)d);
        bl_forward(&b->out, s->z, S, s->y);
        for (size_t e = 0; e < (size_t)S * d; e += 16) _mm512_storeu_ps(s->x + e, _mm512_add_ps(_mm512_loadu_ps(s->x + e), _mm512_loadu_ps(s->y + e)));
    }
    for (int i = 0; i < S; i++) {                                          // final RMSNorm
        __m512 ss = _mm512_setzero_ps();
        for (int c = 0; c < d; c += 16) { const __m512 v = _mm512_loadu_ps(s->x + (size_t)i * d + c); ss = _mm512_fmadd_ps(v, v, ss); }
        const __m512 r = _mm512_set1_ps(1.0f / sqrtf(_mm512_reduce_add_ps(ss) / d + 1e-6f));
        for (int c = 0; c < d; c += 16) _mm512_storeu_ps(m->xn + (size_t)i * d + c, _mm512_mul_ps(_mm512_loadu_ps(s->x + (size_t)i * d + c), r));
    }
    for (int i = 0; i < S; i++) {                                         // head GEMV (vocab-wide vectors, masked tail)
        const float *xn = m->xn + (size_t)i * d; float *lg = m->logits + (size_t)i * V;
        for (int v0 = 0; v0 < V; v0 += 64) {
            __m512 a[4]; __mmask16 mk[4];
            for (int q = 0; q < 4; q++) { const int rem = V - v0 - 16 * q; mk[q] = rem >= 16 ? 0xFFFF : rem > 0 ? (__mmask16)((1u << rem) - 1) : 0; a[q] = _mm512_setzero_ps(); }
            for (int c = 0; c < d; c++) {
                const __m512 xc = _mm512_set1_ps(xn[c]); const float *h = m->head.w + (size_t)c * V + v0;
                for (int q = 0; q < 4; q++) if (mk[q]) a[q] = _mm512_fmadd_ps(xc, _mm512_maskz_loadu_ps(mk[q], h + 16 * q), a[q]);
            }
            for (int q = 0; q < 4; q++) if (mk[q]) _mm512_mask_storeu_ps(lg + v0 + 16 * q, mk[q], a[q]);
        }
    }
    s->pos++;
}
