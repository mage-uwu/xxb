// walshnet inference
//   infer bench    --model M [--tokens 8192]               batched forward + streaming decode throughput
//   infer generate --model M --prompt TEXT [--n 400] [--temp 0.8] [--seed 1]    constant-state streaming (causal)
//   infer fill     --model M --text "Wh_t is th_ m_tter"                        masked infill (bidirectional)
#include "wn.h"

static const char *arg(int argc, char **argv, const char *name, const char *def) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}

static int sample(const float *lg, int V, float temp, Rng *r) {
    float mx = lg[0]; for (int v = 1; v < V; v++) mx = fmaxf(mx, lg[v]);
    double sum = 0; float p[256];
    for (int v = 0; v < V; v++) { p[v] = expf((lg[v] - mx) / temp); sum += p[v]; }
    double u = rng_unif(r) * sum;
    for (int v = 0; v < V; v++) { u -= p[v]; if (u <= 0) return v; }
    return V - 1;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: infer bench|generate|fill --model M ...\n"); return 1; }
    const char *cmd = argv[1], *path = arg(argc, argv, "--model", "model.bin");
    if (arg(argc, argv, "--threads", NULL)) omp_set_num_threads(atoi(arg(argc, argv, "--threads", "4")));
    const int cap = atoi(arg(argc, argv, "--tokens", "8192"));
    Model m; unsigned char vocab[256]; int stoi[256];
    const int rc = model_load(&m, path, vocab, cap);
    if (rc) { fprintf(stderr, "cannot load %s (%d)\n", path, rc); return 1; }
    for (int i = 0; i < 256; i++) stoi[i] = -1;
    for (int i = 0; i < m.c.vocab; i++) stoi[vocab[i]] = i;
    const Config *c = &m.c; const int V = c->vocab, T = c->T;

    if (!strcmp(cmd, "bench")) {
        printf("model: %d x gated %s short-conv (K=%d) BitNet blocks | d=%d T=%d | %zu params | %d threads | AMX %s\n",
               c->n_layers, c->mode ? "centred" : "causal", c->K, c->d, T, model_n_params(&m), omp_get_max_threads(), g_amx ? "on" : "off");
        const int N = cap / T * T; int *tok = malloc(sizeof(int) * N); Rng r = rng_seed(3);
        for (int n = 0; n < N; n++) tok[n] = rng_u64(&r) % V;
        for (int w = 0; w < 3; w++) model_logits(&m, tok, N);
        double best = 1e9;
        for (int it = 0; it < 20; it++) { const double t0 = now_sec(); model_logits(&m, tok, N); best = fmin(best, now_sec() - t0); }
        printf("  batched forward (%d tokens)       : %10.0f tok/s\n", N, N / best);
        if (c->mode == MODE_CAUSAL) {
            const int streams[] = {1, 16, 64, 256, 1024};
            const int R = omp_get_max_threads();
            for (int si = 0; si < 5; si++) {
                const int S = streams[si]; if (S > cap) break;
                if (S >= 64 && R > 1) {                                   // independent streams: one replica per core, no sync
                    Model *reps = calloc(R, sizeof(Model)); Stream *sts = calloc(R, sizeof(Stream));
                    for (int i = 0; i < R; i++) { model_init_replica(&reps[i], &m, S / R); stream_init(&sts[i], &reps[i], S / R); }
                    const int steps = S <= 256 ? 2000 : 500;
                    const int saved = omp_get_max_active_levels(); omp_set_max_active_levels(1);
                    double t = 0;
                    #pragma omp parallel num_threads(R)
                    {
                        Stream *st = &sts[omp_get_thread_num()]; int *tk = calloc(st->S, sizeof(int));
                        for (int i = 0; i < 50; i++) stream_step(st, tk);
                        #pragma omp barrier
                        const double t0 = now_sec();
                        for (int i = 0; i < steps; i++) stream_step(st, tk);
                        #pragma omp barrier
                        #pragma omp master
                        t = now_sec() - t0;
                    }
                    omp_set_max_active_levels(saved);
                    printf("  streaming decode, %4d streams   : %10.0f tok/s  (%d cores x %d streams, %.2f us per step)\n", S, (double)S * steps / t, R, S / R, t / steps * 1e6);
                    continue;
                }
                Stream st; stream_init(&st, &m, S); int *st_tok = malloc(sizeof(int) * S);
                for (int i = 0; i < S; i++) st_tok[i] = rng_u64(&r) % V;
                const int steps = S == 1 ? 20000 : S <= 16 ? 4000 : 1000;
                for (int i = 0; i < 100; i++) stream_step(&st, st_tok);
                const double t0 = now_sec();
                for (int i = 0; i < steps; i++) stream_step(&st, st_tok);
                const double t = now_sec() - t0;
                printf("  streaming decode, %4d stream%s   : %10.0f tok/s  (1 core, %.2f us per step)\n", S, S > 1 ? "s" : " ", (double)S * steps / t, t / steps * 1e6);
            }
        }
        return 0;
    }
    if (!strcmp(cmd, "generate")) {
        if (c->mode != MODE_CAUSAL) { fprintf(stderr, "generate needs a causal model\n"); return 1; }
        const char *prompt = arg(argc, argv, "--prompt", "\n"); const int n_new = atoi(arg(argc, argv, "--n", "400"));
        const float temp = atof(arg(argc, argv, "--temp", "0.8")); Rng r = rng_seed(atoi(arg(argc, argv, "--seed", "1")));
        Stream st; stream_init(&st, &m, 1);
        int cur = 0; fputs(prompt, stdout);
        for (const char *p = prompt; *p; p++) {                            // feed the prompt through the stream state
            const int id = stoi[(unsigned char)*p]; if (id < 0) continue;
            cur = id; if (p[1]) stream_step(&st, &cur);
        }
        const double t0 = now_sec();
        for (int i = 0; i < n_new; i++) {                                 // constant work per token, unbounded length
            stream_step(&st, &cur);
            cur = sample(m.logits, V, temp, &r);
            putchar(vocab[cur]);
        }
        const double t = now_sec() - t0;
        fprintf(stderr, "\n[%d tokens in %.3f s: %.0f tok/s, single stream]\n", n_new, t, n_new / t);
        return 0;
    }
    if (!strcmp(cmd, "fill")) {
        if (c->mode != MODE_BIDIR) { fprintf(stderr, "fill needs a bidirectional model\n"); return 1; }
        const char *text = arg(argc, argv, "--text", "Wh_t is th_ m_tter"); const int L = strlen(text);
        if (L > T) { fprintf(stderr, "text longer than T=%d\n", T); return 1; }
        int *win = malloc(sizeof(int) * T); const int sp = stoi[' '] >= 0 ? stoi[' '] : 0;
        for (int t = 0; t < T; t++) win[t] = t >= L ? sp : text[t] == '_' ? V : (stoi[(unsigned char)text[t]] >= 0 ? stoi[(unsigned char)text[t]] : sp);
        model_logits(&m, win, T);
        printf("input : %s\noutput: ", text);
        for (int t = 0; t < L; t++) {
            if (text[t] != '_') { putchar(text[t]); continue; }
            const float *lg = m.logits + (size_t)t * V; int best = 0;
            for (int v = 1; v < V; v++) if (lg[v] > lg[best]) best = v;
            putchar(vocab[best]);
        }
        putchar('\n');
        return 0;
    }
    fprintf(stderr, "unknown command %s\n", cmd); return 1;
}
