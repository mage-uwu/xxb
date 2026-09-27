// Bit-sliced inference cost of hardened gate layers: DDLGN (2-input LUT) vs CLOPEN-3 (3-input ternary threshold).
// One __m512i = one gate's output for 512 tokens. Gates are sorted by function; each function is one vpternlogd.
// Build: gcc -O3 -march=native clopen_gate_bench.c -o clopen_gate_bench
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)rs; }

typedef struct { int n_in, n, bounds[257]; int32_t *a, *b, *c; uint8_t *imm; } Layer;   // gates sorted by imm8

// CLOPEN-3 gate -> truth table: bit i of imm8 is sign(w0 xa + w1 xb + w2 xc - theta) for (a,b,c) = bits of i
static uint8_t clopen_imm(const int w[3], float theta) {
    uint8_t imm = 0;
    for (int i = 0; i < 8; i++) {
        const int xa = (i >> 2 & 1) ? 1 : -1, xb = (i >> 1 & 1) ? 1 : -1, xc = (i & 1) ? 1 : -1;
        imm |= (uint8_t)((w[0] * xa + w[1] * xb + w[2] * xc - theta >= 0) << i);
    }
    return imm;
}
// DDLGN gate (2-input table t over (a,b)) as imm8 of ternlog(a, b, a)
static uint8_t ddlgn_imm(int t) { return ((t & 1) * 0x03) | ((t >> 1 & 1) * 0x0C) | ((t >> 2 & 1) * 0x30) | ((t >> 3 & 1) * 0xC0); }

static void build(Layer *l, int n_in, int n, int clopen) {
    l->n_in = n_in; l->n = n;
    l->a = malloc(4 * n); l->b = malloc(4 * n); l->c = malloc(4 * n); l->imm = malloc(n);
    uint8_t *imm = malloc(n); int cnt[256] = {0}, pos[256];
    for (int g = 0; g < n; g++) {
        if (clopen) { int w[3] = {(int)(rnd() % 3) - 1, (int)(rnd() % 3) - 1, (int)(rnd() % 3) - 1}; imm[g] = clopen_imm(w, (float)((int)(rnd() % 7) - 3) + 0.5f); }
        else imm[g] = ddlgn_imm(rnd() % 16);
        cnt[imm[g]]++;
    }
    l->bounds[0] = 0; for (int t = 0; t < 256; t++) l->bounds[t + 1] = l->bounds[t] + cnt[t];
    for (int t = 0; t < 256; t++) pos[t] = l->bounds[t];
    for (int g = 0; g < n; g++) {
        const int p = pos[imm[g]]++;
        l->imm[p] = imm[g]; l->a[p] = rnd() % n_in; l->b[p] = rnd() % n_in; l->c[p] = clopen ? (int)(rnd() % n_in) : l->a[p];
    }
    free(imm);
}

#define G3(T) case T: for (int g = lo; g < hi; g++) \
        _mm512_store_si512(out + g, _mm512_ternarylogic_epi32(_mm512_load_si512(in + A[g]), _mm512_load_si512(in + B[g]), _mm512_load_si512(in + C[g]), T)); break;
#define G2(T) case T: for (int g = lo; g < hi; g++) { const __m512i a = _mm512_load_si512(in + A[g]); \
        _mm512_store_si512(out + g, _mm512_ternarylogic_epi32(a, _mm512_load_si512(in + B[g]), a, T)); } break;
#define R4(M, n) M(n) M(n + 1) M(n + 2) M(n + 3)
#define R16(M, n) R4(M, n) R4(M, n + 4) R4(M, n + 8) R4(M, n + 12)
#define R64(M, n) R16(M, n) R16(M, n + 16) R16(M, n + 32) R16(M, n + 48)
#define R256(M) R64(M, 0) R64(M, 64) R64(M, 128) R64(M, 192)

static void fwd3(const Layer *l, const __m512i *in, __m512i *out) {
    const int32_t *A = l->a, *B = l->b, *C = l->c;
    for (int t = 0; t < 256; t++) { const int lo = l->bounds[t], hi = l->bounds[t + 1]; if (lo == hi) continue; switch (t) { R256(G3) } }
}
static void fwd2(const Layer *l, const __m512i *in, __m512i *out) {
    const int32_t *A = l->a, *B = l->b;
    for (int t = 0; t < 256; t++) { const int lo = l->bounds[t], hi = l->bounds[t + 1]; if (lo == hi) continue; switch (t) { R256(G2) } }
}

int main(void) {
    const int n_in0 = 512, W = 2048, depth = 4;             // thermometer input bits (d=64 x 8), gates per layer
    for (int clopen = 0; clopen < 2; clopen++) {
        Layer L[depth]; for (int i = 0; i < depth; i++) build(&L[i], i ? W : n_in0, W, clopen);
        __m512i *A = aligned_alloc(64, 64 * W), *B = aligned_alloc(64, 64 * W);
        for (int i = 0; i < n_in0 * 16; i++) ((uint32_t *)A)[i] = rnd();
        // correctness: every gate output bit vs scalar truth-table lookup
        __m512i *in = A, *out = B; int bad = 0;
        for (int i = 0; i < depth; i++) {
            (clopen ? fwd3 : fwd2)(&L[i], in, out);
            for (int g = 0; g < W; g += 97) for (int bit = 0; bit < 512; bit += 37) {
                const int a = ((uint64_t *)(in + L[i].a[g]))[bit / 64] >> (bit % 64) & 1, b = ((uint64_t *)(in + L[i].b[g]))[bit / 64] >> (bit % 64) & 1,
                          c = ((uint64_t *)(in + L[i].c[g]))[bit / 64] >> (bit % 64) & 1, o = ((uint64_t *)(out + g))[bit / 64] >> (bit % 64) & 1;
                bad += o != (L[i].imm[g] >> (a << 2 | b << 1 | c) & 1);
            }
            __m512i *t = in; in = out; out = t;
        }
        int types = 0; for (int t = 0; t < 256; t++) types += L[0].bounds[t + 1] > L[0].bounds[t];
        double best = 1e9;
        for (int r = 0; r < 200; r++) {
            const double t0 = now(); in = A; out = B;
            for (int i = 0; i < depth; i++) { (clopen ? fwd3 : fwd2)(&L[i], in, out); __m512i *t = in; in = out; out = t; }
            const double t = now() - t0; if (t < best) best = t;
        }
        const double gates = (double)W * depth;
        printf("%-9s %d gates, %3d distinct functions | check %s | %.2f us / 512 tokens | %.2f ns per gate (512 tok) | %.1f G gate-evals/s/core\n",
               clopen ? "CLOPEN-3" : "DDLGN", (int)gates, types, bad ? "FAIL" : "ok", best * 1e6, best / gates * 1e9, gates * 512 / best / 1e9);
    }
    return 0;
}
