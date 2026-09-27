#include "wn.h"
#include <sys/syscall.h>
#include <unistd.h>

/* Intel AMX paths (Sapphire Rapids+):
     int8:  TDPBUSD   C[16x16 i32] += A[16 x 64 u8] . B[16 x (16 cols x 4 k) s8]    (BitLinear forward)
     bf16:  TDPBF16PS C[16x16 f32] += A[16 x 32 bf16] . B[16 x (16 cols x 2 k) bf16] (BitLinear backward)
   B operands are exactly the layouts already used by the VNNI / bf16 code: [K/4][N][4] bytes and [K/2][N][2] bf16.
   Each task computes a 32 x 32 output block from four accumulator tiles (tmm0-3), A tiles tmm4-5, B tiles tmm6-7. */

int g_amx = 0;

typedef struct { uint8_t palette, start_row, rsv[14]; uint16_t colsb[16]; uint8_t rows[16]; } __attribute__((packed)) TileCfg;

static void tile_config(void) {
    static __thread int configured = 0;
    if (configured) return;
    TileCfg cfg; memset(&cfg, 0, sizeof cfg); cfg.palette = 1;
    for (int i = 0; i < 8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
    _tile_loadconfig(&cfg); configured = 1;
}

int amx_init(void) {
    if (getenv("WN_NO_AMX")) return g_amx = 0;
    #ifndef ARCH_REQ_XCOMP_PERM
    #define ARCH_REQ_XCOMP_PERM 0x1023
    #endif
    if (!__builtin_cpu_supports("amx-int8") || !__builtin_cpu_supports("amx-bf16")) return g_amx = 0;
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, 18 /* XFEATURE_XTILEDATA */)) return g_amx = 0;
    return g_amx = 1;
}

/* y[N][M] = ((xu[N][K] . Wt) - 128 colsum) * sx[n] * ws     (N % 32, M % 32, K % 64 == 0) */
void amx_bitlinear_fwd(int N, int M, int K, const uint8_t *xu, const int8_t *Wt, const int32_t *colsum, const float *sx, float ws, float *y) {
    const int nb = N / 32, mb = M / 32;
    #pragma omp parallel if (N >= 128)
    {
        tile_config();
        int32_t Cb[4][16 * 16] __attribute__((aligned(64)));
        #pragma omp for collapse(2) schedule(static)
        for (int ib = 0; ib < nb; ib++)
            for (int jb = 0; jb < mb; jb++) {
                const int n0 = ib * 32, m0 = jb * 32;
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                for (int k0 = 0; k0 < K; k0 += 64) {
                    _tile_loadd(4, xu + (size_t)n0 * K + k0, K);
                    _tile_loadd(5, xu + (size_t)(n0 + 16) * K + k0, K);
                    _tile_loadd(6, Wt + ((size_t)(k0 / 4) * M + m0) * 4, M * 4);
                    _tile_loadd(7, Wt + ((size_t)(k0 / 4) * M + m0 + 16) * 4, M * 4);
                    _tile_dpbusd(0, 4, 6); _tile_dpbusd(1, 4, 7); _tile_dpbusd(2, 5, 6); _tile_dpbusd(3, 5, 7);
                }
                _tile_stored(0, Cb[0], 64); _tile_stored(1, Cb[1], 64); _tile_stored(2, Cb[2], 64); _tile_stored(3, Cb[3], 64);
                for (int t = 0; t < 4; t++) {
                    const int rn = n0 + (t >> 1) * 16, cm = m0 + (t & 1) * 16;
                    const __m512i cs = _mm512_slli_epi32(_mm512_loadu_si512(colsum + cm), 7);
                    for (int r = 0; r < 16; r++) {
                        const __m512 v = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_load_si512(Cb[t] + r * 16), cs));
                        _mm512_storeu_ps(y + (size_t)(rn + r) * M + cm, _mm512_mul_ps(v, _mm512_set1_ps(sx[rn + r] * ws)));
                    }
                }
            }
    }
}

/* C[M][N] (+)= alpha * A[M][K] . Bp,  A bf16 row-major (lda), Bp pairs [K/2][N][2]   (M % 32, N % 32, K % 32 == 0) */
void amx_gemm_bf16(int M, int N, int K, const bf16 *A, int lda, const bf16 *Bp, float *C, int ldc, float alpha, int accumulate) {
    const int ib_n = M / 32, jb_n = N / 32;
    #pragma omp parallel if ((double)M * N * K > 1e6)
    {
        tile_config();
        float Cb[4][16 * 16] __attribute__((aligned(64)));
        #pragma omp for collapse(2) schedule(static)
        for (int ib = 0; ib < ib_n; ib++)
            for (int jb = 0; jb < jb_n; jb++) {
                const int i0 = ib * 32, j0 = jb * 32;
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                for (int k0 = 0; k0 < K; k0 += 32) {
                    _tile_loadd(4, A + (size_t)i0 * lda + k0, lda * 2);
                    _tile_loadd(5, A + (size_t)(i0 + 16) * lda + k0, lda * 2);
                    _tile_loadd(6, Bp + ((size_t)(k0 / 2) * N + j0) * 2, N * 4);
                    _tile_loadd(7, Bp + ((size_t)(k0 / 2) * N + j0 + 16) * 2, N * 4);
                    _tile_dpbf16ps(0, 4, 6); _tile_dpbf16ps(1, 4, 7); _tile_dpbf16ps(2, 5, 6); _tile_dpbf16ps(3, 5, 7);
                }
                _tile_stored(0, Cb[0], 64); _tile_stored(1, Cb[1], 64); _tile_stored(2, Cb[2], 64); _tile_stored(3, Cb[3], 64);
                const __m512 al = _mm512_set1_ps(alpha);
                for (int t = 0; t < 4; t++) {
                    const int ri = i0 + (t >> 1) * 16, cj = j0 + (t & 1) * 16;
                    for (int r = 0; r < 16; r++) {
                        float *c = C + (size_t)(ri + r) * ldc + cj;
                        __m512 v = _mm512_mul_ps(_mm512_load_ps(Cb[t] + r * 16), al);
                        if (accumulate) v = _mm512_add_ps(v, _mm512_loadu_ps(c));
                        _mm512_storeu_ps(c, v);
                    }
                }
            }
    }
}
