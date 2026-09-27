#include "wn.h"

/* BitNet b1.58 FFN: BitLinear(d -> hid) -> ReLU^2 -> BitLinear(hid -> d) */
void mlp_init(MLP *f, int d, int hid, int cap, Rng *rng) {
    f->d = d; f->hid = hid; f->cap = cap;
    bl_init(&f->up, d, hid, cap, rng); bl_init(&f->down, hid, d, cap, rng);
    f->u = xmalloc(4 * (size_t)cap * hid); f->a = xmalloc(4 * (size_t)cap * hid); f->da = xmalloc(4 * (size_t)cap * hid);
}

void mlp_prepare(MLP *f) { bl_prepare(&f->up); bl_prepare(&f->down); }

void mlp_forward(MLP *f, const float *x, int N, float *y) {
    bl_forward(&f->up, x, N, f->u);
    const size_t n = (size_t)N * f->hid;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { const float v = f->u[i] > 0 ? f->u[i] : 0; f->a[i] = v * v; }
    bl_forward(&f->down, f->a, N, y);
}

void mlp_backward(MLP *f, const float *dy, float *dx) {
    bl_backward(&f->down, dy, f->da);
    const size_t n = (size_t)f->up.N * f->hid;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) f->da[i] *= f->u[i] > 0 ? 2 * f->u[i] : 0;
    bl_backward(&f->up, f->da, dx);
}
