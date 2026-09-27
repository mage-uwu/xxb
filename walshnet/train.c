// walshnet pretraining: causal LM (causal short conv) or masked LM (centred short conv) on a byte/char corpus.
#include "wn.h"

typedef struct { const char *data, *out; int steps, batch, eval_every, eval_windows, warmup, save_best; float lr, wd, mask_p; } Opts;

static void usage(void) {
    fprintf(stderr,
        "usage: train --data FILE [options]\n"
        "  --mode causal|bidir  --d 64 --layers 7 --T 64 --K 3 --dilate 0 (cycle c: layer l spacing 2^(l mod c))\n"
        "  --batch 16 --steps 3000 --lr 3e-2 --wd 0 --warmup 100 --seed 0 --mask 0.15\n"
        "  --eval-every 500 --eval-windows 256 --out model.bin --save-best --threads N\n");
    exit(1);
}

static int *load_corpus(const char *path, size_t *n_out, unsigned char *vocab, int *V) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *raw = malloc(n); if (fread(raw, 1, n, f) != n) { perror("read"); exit(1); } fclose(f);
    int used[256] = {0}, map[256]; for (size_t i = 0; i < n; i++) used[raw[i]] = 1;
    *V = 0; for (int c = 0; c < 256; c++) if (used[c]) { map[c] = *V; vocab[(*V)++] = (unsigned char)c; }
    int *tok = malloc(sizeof(int) * n); for (size_t i = 0; i < n; i++) tok[i] = map[raw[i]];
    free(raw); *n_out = n; return tok;
}

// fill one batch: B windows of length T. causal: next-token targets. bidir: BERT-style masking (80/10/10).
static void make_batch(const int *data, size_t len, const Config *c, int B, float mask_p, Rng *r, int *tok, int *tgt) {
    const int T = c->T;
    for (int b = 0; b < B; b++) {
        const size_t s = rng_u64(r) % (len - T - 1);
        for (int t = 0; t < T; t++) {
            const int n = b * T + t, x = data[s + t];
            if (c->mode == MODE_CAUSAL) { tok[n] = x; tgt[n] = data[s + t + 1]; continue; }
            tok[n] = x; tgt[n] = -1;
            if (rng_unif(r) < mask_p) {
                tgt[n] = x; const float u = rng_unif(r);
                tok[n] = u < 0.8f ? c->vocab : u < 0.9f ? (int)(rng_u64(r) % c->vocab) : x;
            }
        }
    }
}

static double evaluate(Model *m, const int *val, size_t len, const Opts *o, int B) {
    Rng r = rng_seed(12345); int *tok = malloc(sizeof(int) * B * m->c.T), *tgt = malloc(sizeof(int) * B * m->c.T);
    double s = 0; int nb = 0;
    for (int w = 0; w < o->eval_windows; w += B, nb++) {
        make_batch(val, len, &m->c, B, o->mask_p, &r, tok, tgt);
        s += model_step(m, tok, tgt, B * m->c.T, 0);
    }
    free(tok); free(tgt); return s / nb;
}

int main(int argc, char **argv) {
    Config c = {.d = 64, .n_layers = 7, .T = 64, .K = 3, .mode = MODE_CAUSAL, .seed = 0};
    Opts o = {.data = NULL, .out = "model.bin", .steps = 3000, .batch = 16, .eval_every = 500, .eval_windows = 256,
              .warmup = 100, .lr = 3e-2f, .wd = 0, .mask_p = 0.15f};
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        #define ARG(name) (!strcmp(a, name) && v && ++i)
        if      ARG("--data") o.data = v;
        else if ARG("--out") o.out = v;
        else if ARG("--mode") c.mode = !strcmp(v, "bidir") ? MODE_BIDIR : MODE_CAUSAL;
        else if ARG("--d") c.d = atoi(v);
        else if ARG("--layers") c.n_layers = atoi(v);
        else if ARG("--T") c.T = atoi(v);
        else if ARG("--K") c.K = atoi(v);
        else if ARG("--dilate") c.dil_cycle = atoi(v);
        else if (!strcmp(a, "--save-best")) o.save_best = 1;
        else if ARG("--seed") c.seed = strtoull(v, 0, 10);
        else if ARG("--batch") o.batch = atoi(v);
        else if ARG("--steps") o.steps = atoi(v);
        else if ARG("--lr") o.lr = atof(v);
        else if ARG("--wd") o.wd = atof(v);
        else if ARG("--warmup") o.warmup = atoi(v);
        else if ARG("--mask") o.mask_p = atof(v);
        else if ARG("--eval-every") o.eval_every = atoi(v);
        else if ARG("--eval-windows") o.eval_windows = atoi(v);
        else if ARG("--threads") omp_set_num_threads(atoi(v));
        else usage();
    }
    if (!o.data) usage();
    unsigned char vocab[256] = {0}; size_t len;
    int *data = load_corpus(o.data, &len, vocab, &c.vocab);
    const size_t n_train = (size_t)(0.9 * len);
    const int *train = data, *val = data + n_train; const size_t n_val = len - n_train;
    const int B = o.batch, N = B * c.T;
    Model m; model_init(&m, &c, N);
    printf("walshnet | %s LM | %d x gated %s short-conv (K=%d) BitNet blocks | d=%d T=%d | vocab %d | %zu params | %d threads | AMX %s\n",
           c.mode ? "masked" : "causal", c.n_layers, c.mode ? "centred" : "causal", c.K, c.d, c.T, c.vocab, model_n_params(&m), omp_get_max_threads(), g_amx ? "on" : "off");
    int rf = 1; for (int l = 0; l < c.n_layers; l++) rf += (c.mode ? c.K / 2 : c.K - 1) * model_dil(c.dil_cycle, l);
    printf("receptive field: %d tokens%s | training window T=%d\n", rf, c.mode ? " each side" : "", c.T);
    int R = omp_get_max_threads(); if (R > B) R = B;                       // data-parallel replicas (one per thread)
    Model *reps = NULL;
    if (R > 1) { reps = calloc(R, sizeof(Model)); for (int i = 0; i < R; i++) model_init_replica(&reps[i], &m, ((B + R - 1) / R) * c.T); }
    int *tok = malloc(sizeof(int) * N), *tgt = malloc(sizeof(int) * N);
    Rng r = rng_seed(1000 + c.seed);
    double t_train = 0, run_loss = 0, best_val = 1e9; int run_n = 0;
    for (int it = 1; it <= o.steps; it++) {
        make_batch(train, n_train, &c, B, o.mask_p, &r, tok, tgt);
        const double t0 = now_sec();
        run_loss += R > 1 ? train_step_dp(&m, reps, R, tok, tgt, N, c.T) : model_step(&m, tok, tgt, N, 1); run_n++;
        const float warm = it < o.warmup ? (float)it / o.warmup : 1.0f;
        adam_step(&m, o.lr * warm * (1.0f - (float)it / (o.steps + 1)), it, 0.9f, 0.95f, o.wd);
        if (R > 1) replicas_sync(reps, R, &m);
        t_train += now_sec() - t0;
        if (it % o.eval_every == 0 || it == o.steps) {
            const double vl = evaluate(&m, val, n_val, &o, B);
            const int improved = vl < best_val; if (improved) best_val = vl;
            printf("step %5d | train %.4f | val %.4f%s | %.0f tok/s | %.1fs\n", it, run_loss / run_n, vl, improved && o.save_best ? " *" : "", (double)it * N / t_train, t_train);
            if (improved && o.save_best) model_save(&m, o.out, vocab);
            fflush(stdout); run_loss = 0; run_n = 0;
        }
    }
    const double fv = evaluate(&m, val, n_val, &o, B);
    printf("final val %.4f | best val %.4f | train throughput %.0f tok/s | %.1fs\n", fv, fv < best_val ? fv : best_val, (double)o.steps * N / t_train, t_train);
    if (o.save_best && fv >= best_val) printf("kept best checkpoint in %s (val %.4f)\n", o.out, best_val);
    else if (model_save(&m, o.out, vocab)) fprintf(stderr, "could not save %s\n", o.out); else printf("saved %s\n", o.out);
    return 0;
}
