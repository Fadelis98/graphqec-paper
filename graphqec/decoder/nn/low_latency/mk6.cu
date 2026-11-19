// Persistent BF16 megakernel v2 for GraphLinearAttnDecoderV2A full-history inference (batch 1).
// One cooperative block per SM interprets a host-built op program; phases are separated by grid barriers.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define NT 256
#define NWARP 8
#define SMEM_BYTES 96256
// 256 phases / 448 ops keep static + dynamic smem (100,354 B) under the sm_120 opt-in limit (101,376 B)
#define MAXPH 256
#define MAXOPS 448

typedef __nv_bfloat16 bf16;

enum { OP_GEMM = 1, OP_ATTN = 2, OP_EMBED = 3, OP_SCAT_MID = 4, OP_GDN = 5, OP_SCAT_LOG = 6, OP_FINAL = 7, OP_ZERO = 8, OP_NORM = 9 };
enum { EPI_STORE = 0, EPI_RESID = 1, EPI_SWIGLU = 2, EPI_RESID_TANH = 3, EPI_MIX = 4 };
enum { PRO_NONE = 0, PRO_RMSNORM = 1, PRO_RESADD = 2 };

struct Op { long long f[32]; };

struct Args {
    const Op* ops;
    const int* phase_begin;   // [n_phases+1] op index ranges
    const int* phase_total;   // [n_phases] work items per phase
    const int* op_end;        // [n_ops] cumulative item end within its phase
    int n_phases;
    const uint8_t* syn;
    int n_det;
    int n_inputs;
    int n_samples;
    float* logits;
    int nlog;
    unsigned long long* times;
    unsigned int* bar;
    unsigned long long* phase_t;
    int barrier_mode;
    int prefetch;
    unsigned long long* dbg;
};

__device__ unsigned long long* g_dbg;
#ifdef MK_DBG
#define DBG(i) do { if (g_dbg && blockIdx.x == 0 && threadIdx.x == 0) { g_dbg[2 * (i)] = clock64(); g_dbg[2 * (i) + 1] = globaltimer(); } } while (0)
#else
#define DBG(i) do {} while (0)
#endif

#define FLD(op, i) ((op)->f[(i)])
#define PTR(T, op, i) (reinterpret_cast<T>(FLD(op, i)))

__device__ __forceinline__ float bf2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2bf(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ldcg_bf(const bf16* p) {
    unsigned short v = __ldcg(reinterpret_cast<const unsigned short*>(p));
    return __bfloat162float(__ushort_as_bfloat16(v));
}
__device__ __forceinline__ float ldg_bf(const bf16* p) {
    unsigned short v = __ldg(reinterpret_cast<const unsigned short*>(p));
    return __bfloat162float(__ushort_as_bfloat16(v));
}
__device__ __forceinline__ float i2f(long long bits) { return __int_as_float((int)bits); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void bf8_to_f(const uint4& u, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) { float2 t = __bfloat1622float2(h[i]); f[2 * i] = t.x; f[2 * i + 1] = t.y; }
}

// ---------------------------------------------------------------- grid barriers
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release(unsigned* p, unsigned v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void red_release_add(unsigned* p, unsigned v) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// mode 0: counter + generation flag; mode 1: monotonic counter, everyone polls the counter.
__device__ __forceinline__ void grid_sync(unsigned* bar, unsigned nblk, unsigned& gen, int mode) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (mode == 1) {
            gen += nblk;
            red_release_add(bar, 1u);
            while (ld_acquire(bar) < gen) {}
        } else {
            unsigned g = gen;
            __threadfence();
            unsigned prev = atomicAdd(&bar[0], 1u);
            if (prev == nblk - 1) {
                atomicExch(&bar[0], 0u);
                __threadfence();
                st_release(&bar[1], g + 1);
            } else {
                while (ld_acquire(&bar[1]) == g) {}
            }
            gen = g + 1;
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------- async copy / mma helpers
__device__ __forceinline__ void cp_async16(void* s, const void* g, bool pred) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(unsigned& r0, unsigned& r1, const void* p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void mma16816(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------- GEMM tile
// out[M,N] = epi(pro(A)[M,K] @ W[N,K]^T). W is nn.Linear layout (rows = output features).
// fields: 4 M, 5 N, 6 K, 7 A, 8 lda, 9 W, 10 bias, 11 out, 12 ldo, 13 aux1(resid), 14 ldr,
//         15 norm_w, 16 norm_b, 17 eps, 18 epi, 19 pro, 20 aux2, 21 aux3, 22 tiles_n
#define BK 64
#define SPAD 8
#define SROW (BK + SPAD)

template <int MI, int NI, int TM, int TN>
__device__ __forceinline__ void gemm_epi(float (&acc)[MI][NI][4], const Op* op, int m0, int n0, int wm, int wn, int lane,
                                         int M, int N) {
    const int epi = (int)FLD(op, 18);
    const bf16* bias = PTR(const bf16*, op, 10);
    bf16* out = PTR(bf16*, op, 11);
    const int ldo = (int)FLD(op, 12);
    const bf16* aux1 = PTR(const bf16*, op, 13);
    const int ldr = (int)FLD(op, 14);
    const bf16* aux2 = PTR(const bf16*, op, 20);
    const bf16* aux3 = PTR(const bf16*, op, 21);
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = m0 + wm * TM + i * 16 + (lane >> 2) + h * 8;
                const int c = n0 + wn * TN + j * 8 + 2 * (lane & 3);
                if (r >= M || c >= N) continue;
                float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                if (epi == EPI_SWIGLU) {
                    float a = rbf(v0), b = rbf(v1);
                    out[(size_t)r * ldo + (c >> 1)] = f2bf(rbf(silu_f(a)) * b);
                    continue;
                }
                if (bias) {
                    float2 bb = __bfloat1622float2(__ldg(reinterpret_cast<const __nv_bfloat162*>(bias + c)));
                    v0 += bb.x; v1 += bb.y;
                }
                v0 = rbf(v0); v1 = rbf(v1);
                if (epi != EPI_STORE) {
                    size_t o = (size_t)r * ldr + c;
                    float2 r1 = __bfloat1622float2(__ldcg(reinterpret_cast<const __nv_bfloat162*>(aux1 + o)));
                    if (epi == EPI_RESID) {
                        v0 += r1.x; v1 += r1.y;
                    } else if (epi == EPI_RESID_TANH) {
                        v0 = tanhf(rbf(v0 + r1.x)); v1 = tanhf(rbf(v1 + r1.y));
                    } else if (epi == EPI_MIX) {
                        float2 d = __bfloat1622float2(__ldcg(reinterpret_cast<const __nv_bfloat162*>(aux2 + o)));
                        float2 rr = __bfloat1622float2(__ldcg(reinterpret_cast<const __nv_bfloat162*>(aux3 + o)));
                        float m0_ = rbf(tanhf(rbf(v0 + r1.x))), m1_ = rbf(tanhf(rbf(v1 + r1.y)));
                        v0 = rbf(m0_ * d.x) + rbf(rbf(1.f - m0_) * rr.x);
                        v1 = rbf(m1_ * d.y) + rbf(rbf(1.f - m1_) * rr.y);
                    }
                }
                *reinterpret_cast<__nv_bfloat162*>(out + (size_t)r * ldo + c) = __floats2bfloat162_rn(v0, v1);
            }
}

template <int BM, int BN, int WM, int WN, int STAGES>
__device__ void gemm_tile(const Op* op, int tile, char* smem) {
    constexpr int TM = BM / WM, TN = BN / WN, MI = TM / 16, NI = TN / 8;
    constexpr int RPW = (BM + NWARP - 1) / NWARP;
    static_assert(WM * WN == NWARP, "8 warps");
    static_assert((BM + BN) * SROW * 2 * STAGES + BM * 4 <= SMEM_BYTES, "smem");
    const int M = (int)FLD(op, 4), N = (int)FLD(op, 5), K = (int)FLD(op, 6);
    const bf16* A = PTR(const bf16*, op, 7);
    const int lda = (int)FLD(op, 8);
    const bf16* W = PTR(const bf16*, op, 9);
    const int tiles_n = (int)FLD(op, 22);
    const int pro = (int)FLD(op, 19);
    const int m0 = (tile / tiles_n) * BM, n0 = (tile % tiles_n) * BN;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;

    bf16* As = reinterpret_cast<bf16*>(smem);
    bf16* Ws = As + STAGES * BM * SROW;
    float* s_rstd = reinterpret_cast<float*>(Ws + STAGES * BN * SROW);
    const int nk = K / BK;

    auto load_stage = [&](int kc, int st) {
        const int k0 = kc * BK;
        bf16* as = As + st * BM * SROW;
        bf16* ws = Ws + st * BN * SROW;
        for (int s = tid; s < BM * 8; s += NT) {
            int r = s >> 3, cs = s & 7;
            int gr = m0 + r;
            bool ok = gr < M;
            cp_async16(as + r * SROW + cs * 8, A + (size_t)(ok ? gr : 0) * lda + k0 + cs * 8, ok);
        }
        for (int s = tid; s < BN * 8; s += NT) {
            int r = s >> 3, cs = s & 7;
            int gr = n0 + r;
            bool ok = gr < N;
            cp_async16(ws + r * SROW + cs * 8, W + (size_t)(ok ? gr : 0) * K + k0 + cs * 8, ok);
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nk) load_stage(s, s);
        cp_commit();
    }

    if (pro == PRO_RMSNORM) {
        // K <= 256: one 16-byte load per lane per row, all rows of this warp issued before reducing
        const float eps = i2f(FLD(op, 17));
        const bool act = lane * 8 < K;
#pragma unroll
        for (int qb = 0; qb < RPW; qb += 4) {
            uint4 v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                int r = warp * RPW + qb + q, gr = m0 + r;
                v[q] = make_uint4(0, 0, 0, 0);
                if (qb + q < RPW && r < BM && gr < M && act)
                    v[q] = __ldcg(reinterpret_cast<const uint4*>(A + (size_t)gr * lda + lane * 8));
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float f[8];
                bf8_to_f(v[q], f);
                float ss = 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) ss += f[i] * f[i];
                ss = warp_sum(ss);
                int r = warp * RPW + qb + q;
                if (lane == 0 && qb + q < RPW && r < BM) s_rstd[r] = rsqrtf(ss / (float)K + eps);
            }
        }
    }

    float acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

    const bf16* nw = PTR(const bf16*, op, 15);
    const bf16* nb = PTR(const bf16*, op, 16);

    for (int kc = 0; kc < nk; ++kc) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        {
            int nxt = kc + STAGES - 1;
            if (nxt < nk) load_stage(nxt, nxt % STAGES);
            cp_commit();
        }
        const int st = kc % STAGES;
        bf16* as = As + st * BM * SROW;
        const bf16* ws = Ws + st * BN * SROW;
        if (pro == PRO_RMSNORM) {
            const int k0 = kc * BK;
            for (int e = tid; e < BM * (BK / 2); e += NT) {
                int r = e / (BK / 2), c = 2 * (e % (BK / 2));
                __nv_bfloat162* p2 = reinterpret_cast<__nv_bfloat162*>(as + r * SROW + c);
                float2 x = __bfloat1622float2(*p2);
                float rs = s_rstd[r];
                float y0 = rbf(rbf(rbf(x.x * rs) * ldg_bf(nw + k0 + c)) + ldg_bf(nb + k0 + c));
                float y1 = rbf(rbf(rbf(x.y * rs) * ldg_bf(nw + k0 + c + 1)) + ldg_bf(nb + k0 + c + 1));
                *p2 = __floats2bfloat162_rn(y0, y1);
            }
            __syncthreads();
        }
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            unsigned af[MI][4];
#pragma unroll
            for (int i = 0; i < MI; ++i) {
                const bf16* p = as + (wm * TM + i * 16 + (lane & 15)) * SROW + kk * 16 + (lane >> 4) * 8;
                ldsm_x4(af[i][0], af[i][1], af[i][2], af[i][3], p);
            }
            unsigned bfr[NI][2];
#pragma unroll
            for (int j = 0; j < NI; j += 2) {
                if (j + 1 < NI) {
                    const bf16* p = ws + (wn * TN + j * 8 + (lane & 7) + (lane >> 4) * 8) * SROW + kk * 16 + ((lane >> 3) & 1) * 8;
                    ldsm_x4(bfr[j][0], bfr[j][1], bfr[j + 1][0], bfr[j + 1][1], p);
                } else {
                    const bf16* p = ws + (wn * TN + j * 8 + (lane & 7)) * SROW + kk * 16 + ((lane >> 3) & 1) * 8;
                    ldsm_x2(bfr[j][0], bfr[j][1], p);
                }
            }
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NI; ++j)
                    mma16816(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bfr[j][0], bfr[j][1]);
        }
    }
    cp_wait<0>();
    gemm_epi<MI, NI, TM, TN>(acc, op, m0, n0, wm, wn, lane, M, N);
}

// Resident GEMM: whole K panel of A and W in smem, plus norm vectors / bias / residual tiles, all in one round trip.
// W may be prefetched before the grid barrier.
// extra fields: 23 resident, 24 acc(f32), 25 ld_acc, 26 resadd bias, 27 x1 out, 28 ld_x1
template <int BM, int BN>
__device__ __forceinline__ void gemm_res_prefetch(const Op* op, int tile, char* smem) {
    const int N = (int)FLD(op, 5), K = (int)FLD(op, 6), tiles_n = (int)FLD(op, 22);
    const bf16* W = PTR(const bf16*, op, 9);
    const int n0 = (tile % tiles_n) * BN, KP = K + 8, segs = K / 8;
    bf16* Ws = reinterpret_cast<bf16*>(smem) + BM * KP;
    for (int s = threadIdx.x; s < BN * segs; s += NT) {
        int r = s / segs, c = s % segs, gr = n0 + r;
        bool ok = gr < N;
        cp_async16(Ws + r * KP + c * 8, W + (size_t)(ok ? gr : 0) * K + c * 8, ok);
    }
    cp_commit();
}

__device__ __forceinline__ void cp_vec(bf16* dst, const bf16* src, int n) {
    if (!src) return;
    for (int s = threadIdx.x; s < n / 8; s += NT) cp_async16(dst + s * 8, src + s * 8, true);
}

template <int BM, int BN>
__device__ __forceinline__ void cp_tile(bf16* dst, const bf16* src, int ld, int m0, int n0, int M) {
    if (!src) return;
    for (int s = threadIdx.x; s < BM * (BN / 8); s += NT) {
        int r = s / (BN / 8), c = s % (BN / 8);
        bool ok = m0 + r < M;
        cp_async16(dst + r * BN + c * 8, src + (size_t)(ok ? m0 + r : 0) * ld + n0 + c * 8, ok);
    }
}

template <int BM, int BN, int WM, int WN>
__device__ void gemm_res(const Op* op, int tile, char* smem, int w_pref) {
    constexpr int TM = BM / WM, TN = BN / WN, MI = TM / 16, NI = TN / 8;
    static_assert(WM * WN == NWARP, "8 warps");
    const int M = (int)FLD(op, 4), N = (int)FLD(op, 5), K = (int)FLD(op, 6);
    const bf16* A = PTR(const bf16*, op, 7);
    const int lda = (int)FLD(op, 8);
    const int tiles_n = (int)FLD(op, 22);
    const int pro = (int)FLD(op, 19), epi = (int)FLD(op, 18);
    const int tm = tile / tiles_n, tn = tile % tiles_n;
    const int m0 = tm * BM, n0 = tn * BN;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KP = K + 8, segs = K / 8;
    const bool resadd = pro == PRO_RESADD;
    const int naux = epi == EPI_MIX ? 3 : (epi == EPI_STORE || epi == EPI_SWIGLU ? 0 : 1);
    bf16* As = reinterpret_cast<bf16*>(smem);
    bf16* Ws = As + BM * KP;
    float* Acc = reinterpret_cast<float*>(Ws + BN * KP);
    bf16* gam = reinterpret_cast<bf16*>(Acc + (resadd ? BM * K : 0));
    bf16* bet = gam + K;
    bf16* sbo = bet + K;
    bf16* sbias = sbo + K;
    bf16* saux = sbias + BN;
    float* s_rstd = reinterpret_cast<float*>(saux + naux * BM * BN);
    const bf16* bias = PTR(const bf16*, op, 10);
    const int ldr = (int)FLD(op, 14);

    if (!w_pref) gemm_res_prefetch<BM, BN>(op, tile, smem);
    DBG(0);
    for (int s = tid; s < BM * segs; s += NT) {
        int r = s / segs, c = s % segs, gr = m0 + r;
        bool ok = gr < M;
        cp_async16(As + r * KP + c * 8, A + (size_t)(ok ? gr : 0) * lda + c * 8, ok);
    }
    if (resadd) {
        const float* accg = PTR(const float*, op, 24);
        const int lacc = (int)FLD(op, 25), s4 = K / 4;
        for (int s = tid; s < BM * s4; s += NT) {
            int r = s / s4, c = s % s4, gr = m0 + r;
            bool ok = gr < M;
            cp_async16(Acc + r * K + c * 4, accg + (size_t)(ok ? gr : 0) * lacc + c * 4, ok);
        }
        cp_vec(sbo, PTR(const bf16*, op, 26), K);
    }
    if (pro != PRO_NONE) {
        cp_vec(gam, PTR(const bf16*, op, 15), K);
        cp_vec(bet, PTR(const bf16*, op, 16), K);
    }
    if (bias) cp_vec(sbias, bias + n0, min(BN, N - n0));
    if (naux >= 1) cp_tile<BM, BN>(saux, PTR(const bf16*, op, 13), ldr, m0, n0, M);
    if (naux == 3) {
        cp_tile<BM, BN>(saux + BM * BN, PTR(const bf16*, op, 20), ldr, m0, n0, M);
        cp_tile<BM, BN>(saux + 2 * BM * BN, PTR(const bf16*, op, 21), ldr, m0, n0, M);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    DBG(1);
    if (pro != PRO_NONE) {
        // one warp per row: residual-add, RMSNorm statistics and normalization in registers
        const float eps = i2f(FLD(op, 17));
        bf16* x1 = PTR(bf16*, op, 27);
        const int ldx1 = (int)FLD(op, 28);
        const int np2 = K / 64;
        for (int r = warp; r < BM; r += NWARP) {
            float2 v[4];
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (i < np2) {
                    const int c = 2 * lane + 64 * i;
                    float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(As + r * KP + c));
                    if (resadd) {
                        float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(sbo + c));
                        x.x = rbf(x.x + rbf(Acc[r * K + c] + b.x));
                        x.y = rbf(x.y + rbf(Acc[r * K + c + 1] + b.y));
                        if (tn == 0 && m0 + r < M)
                            *reinterpret_cast<__nv_bfloat162*>(x1 + (size_t)(m0 + r) * ldx1 + c) = __floats2bfloat162_rn(x.x, x.y);
                    }
                    v[i] = x;
                    ss += x.x * x.x + x.y * x.y;
                }
            }
            ss = warp_sum(ss);
            const float rs = rsqrtf(ss / (float)K + eps);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (i < np2) {
                    const int c = 2 * lane + 64 * i;
                    float2 w = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gam + c));
                    float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(bet + c));
                    *reinterpret_cast<__nv_bfloat162*>(As + r * KP + c) = __floats2bfloat162_rn(
                        rbf(rbf(rbf(v[i].x * rs) * w.x) + b.x), rbf(rbf(rbf(v[i].y * rs) * w.y) + b.y));
                }
            }
        }
        __syncthreads();
    }
    float acc[MI][NI][4];
    DBG(3);
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    for (int kk = 0; kk < K / 16; ++kk) {
        unsigned af[MI][4];
#pragma unroll
        for (int i = 0; i < MI; ++i)
            ldsm_x4(af[i][0], af[i][1], af[i][2], af[i][3],
                    As + (wm * TM + i * 16 + (lane & 15)) * KP + kk * 16 + (lane >> 4) * 8);
        unsigned bfr[NI][2];
#pragma unroll
        for (int j = 0; j < NI; j += 2) {
            if (j + 1 < NI)
                ldsm_x4(bfr[j][0], bfr[j][1], bfr[j + 1][0], bfr[j + 1][1],
                        Ws + (wn * TN + j * 8 + (lane & 7) + (lane >> 4) * 8) * KP + kk * 16 + ((lane >> 3) & 1) * 8);
            else
                ldsm_x2(bfr[j][0], bfr[j][1], Ws + (wn * TN + j * 8 + (lane & 7)) * KP + kk * 16 + ((lane >> 3) & 1) * 8);
        }
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
            for (int j = 0; j < NI; ++j) mma16816(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bfr[j][0], bfr[j][1]);
    }
    bf16* out = PTR(bf16*, op, 11);
    const int ldo = (int)FLD(op, 12);
    DBG(4);
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = wm * TM + i * 16 + (lane >> 2) + h * 8;
                const int lc = wn * TN + j * 8 + 2 * (lane & 3);
                const int r = m0 + lr, c = n0 + lc;
                if (r >= M || c >= N) continue;
                float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                if (epi == EPI_SWIGLU) {
                    out[(size_t)r * ldo + (c >> 1)] = f2bf(rbf(silu_f(rbf(v0))) * rbf(v1));
                    continue;
                }
                if (bias) {
                    float2 bb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(sbias + lc));
                    v0 += bb.x; v1 += bb.y;
                }
                v0 = rbf(v0); v1 = rbf(v1);
                if (naux) {
                    const int o = lr * BN + lc;
                    float2 r1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(saux + o));
                    if (epi == EPI_RESID) {
                        v0 += r1.x; v1 += r1.y;
                    } else if (epi == EPI_RESID_TANH) {
                        v0 = tanhf(rbf(v0 + r1.x)); v1 = tanhf(rbf(v1 + r1.y));
                    } else if (epi == EPI_MIX) {
                        float2 d = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(saux + BM * BN + o));
                        float2 rr = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(saux + 2 * BM * BN + o));
                        float q0 = rbf(tanhf(rbf(v0 + r1.x))), q1 = rbf(tanhf(rbf(v1 + r1.y)));
                        v0 = rbf(q0 * d.x) + rbf(rbf(1.f - q0) * rr.x);
                        v1 = rbf(q1 * d.y) + rbf(rbf(1.f - q1) * rr.y);
                    }
                }
                *reinterpret_cast<__nv_bfloat162*>(out + (size_t)r * ldo + c) = __floats2bfloat162_rn(v0, v1);
            }
    DBG(5);
}

__device__ void run_gemm(const Op* op, int tile, char* smem, int pref) {
    if (FLD(op, 23)) {
        switch ((int)FLD(op, 3)) {
            case 0: gemm_res<32, 32, 2, 4>(op, tile, smem, pref); break;
            case 1: gemm_res<64, 64, 2, 4>(op, tile, smem, pref); break;
            case 3: gemm_res<16, 64, 1, 8>(op, tile, smem, pref); break;
            case 4: gemm_res<32, 128, 2, 4>(op, tile, smem, pref); break;
            case 6: gemm_res<16, 128, 1, 8>(op, tile, smem, pref); break;
            default: break;
        }
        return;
    }
    switch ((int)FLD(op, 3)) {
        case 0: gemm_tile<32, 32, 2, 4, 8>(op, tile, smem); break;
        case 1: gemm_tile<64, 64, 2, 4, 5>(op, tile, smem); break;
        case 2: gemm_tile<128, 64, 4, 2, 3>(op, tile, smem); break;
        case 3: gemm_tile<16, 64, 1, 8, 8>(op, tile, smem); break;
        case 4: gemm_tile<32, 128, 2, 4, 4>(op, tile, smem); break;
        case 5: gemm_tile<64, 128, 2, 4, 3>(op, tile, smem); break;
        case 6: gemm_tile<16, 128, 1, 8, 4>(op, tile, smem); break;
        case 7: gemm_tile<128, 128, 2, 4, 2>(op, tile, smem); break;
        default: break;
    }
}

__device__ int gemm_prefetch(const Op* op, int tile, char* smem) {
    if (!FLD(op, 23)) return 0;
    switch ((int)FLD(op, 3)) {
        case 0: gemm_res_prefetch<32, 32>(op, tile, smem); return 1;
        case 1: gemm_res_prefetch<64, 64>(op, tile, smem); return 1;
        case 3: gemm_res_prefetch<16, 64>(op, tile, smem); return 1;
        case 4: gemm_res_prefetch<32, 128>(op, tile, smem); return 1;
        case 6: gemm_res_prefetch<16, 128>(op, tile, smem); return 1;
        default: return 0;
    }
}

// ---------------------------------------------------------------- attention on tensor cores (per group, head, query chunk)
// fields: 4 qkv, 5 ld_qkv, 6 E, 7 hd, 8 heads, 9 out (bf16 ATT, or f32 acc when fused), 10 ldo, 11 row0, 12 gsz,
//         13 ngroups, 14 nqc, 15 scale, 16 qc, 17 fuse_wo, 18 Wo ([E][E] nn.Linear layout)
#define MAXG 160
#define HP 40
#define SPS (MAXG + 4)
#define PPS (MAXG + 8)
__device__ __forceinline__ void ldsm_x2_t(unsigned& r0, unsigned& r1, const void* p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
struct AttnSmem {
    bf16 Wo[256 * HP];
    bf16 Q[32 * HP];
    bf16 K[MAXG * HP];
    bf16 V[MAXG * HP];
    bf16 O[32 * HP];
    bf16 Pm[32 * PPS];
    float S[32 * SPS];
    float rs[32];
};

__device__ __forceinline__ void attn_prefetch_wo(const Op* op, int item, char* smem) {
    AttnSmem* sm = reinterpret_cast<AttnSmem*>(smem);
    const int E = (int)FLD(op, 6), hd = (int)FLD(op, 7), H = (int)FLD(op, 8), nqc = (int)FLD(op, 14);
    const bf16* Wo = PTR(const bf16*, op, 18);
    const int h = (item / nqc) % H, segs_valid = hd / 8;
    for (int s = threadIdx.x; s < E * 4; s += NT) {
        int n = s >> 2, sg = s & 3;
        bool ok = sg < segs_valid;
        cp_async16(sm->Wo + n * HP + sg * 8, Wo + (size_t)n * E + h * hd + (ok ? sg * 8 : 0), ok);
    }
    cp_commit();
}

// ---------------------------------------------------------------- flash-style attention (registers, online softmax)
// fields: 4 qkv, 5 ld_qkv, 6 E, 7 hd, 8 heads, 9 out (bf16 ATT, or f32 acc when fused), 10 ldo, 11 row0, 12 gsz,
//         13 ngroups, 15 scale, 17 fuse_wo, 18 Wo ([E][E]), 20 heads per item (HB)
#define FA_MAXG 144
#define FA_HP 40
struct FaSmem {
    bf16 K[2][FA_MAXG * FA_HP];
    bf16 V[2][FA_MAXG * FA_HP];
    bf16 Q[2][FA_MAXG * FA_HP];
    bf16 Wo[256 * FA_HP];
};

__device__ __forceinline__ unsigned pack_bf2(float a, float b) {
    __nv_bfloat162 t = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<unsigned*>(&t);
}

__device__ __forceinline__ void fa_prefetch_wo(const Op* op, int item, char* smem) {
    FaSmem* sm = reinterpret_cast<FaSmem*>(smem);
    const int E = (int)FLD(op, 6), hd = (int)FLD(op, 7), H = (int)FLD(op, 8);
    const bf16* Wo = PTR(const bf16*, op, 18);
    const int gsz = (int)FLD(op, 12), MT = ((gsz + 15) & ~15) / 16;
    const int tpi = min(MT, max(1, (int)FLD(op, 21)));
    const int h = (item / ((MT + tpi - 1) / tpi)) % H, segs_valid = hd / 8;
    for (int s = threadIdx.x; s < E * 4; s += NT) {
        int n = s >> 2, sg = s & 3;
        bool ok = sg < segs_valid;
        cp_async16(sm->Wo + n * FA_HP + sg * 8, Wo + (size_t)n * E + h * hd + (ok ? sg * 8 : 0), ok);
    }
    cp_commit();
}

template <int HD, bool FUSE>
__device__ void attn_fa(const Op* op, int item, char* smem, int pref) {
    FaSmem* sm = reinterpret_cast<FaSmem*>(smem);
    const bf16* qkv = PTR(const bf16*, op, 4);
    const int ld = (int)FLD(op, 5), E = (int)FLD(op, 6), H = (int)FLD(op, 8);
    const int ldo = (int)FLD(op, 10), row0 = (int)FLD(op, 11), gsz = (int)FLD(op, 12);
    const int HB = FUSE ? 1 : max(1, (int)FLD(op, 20));
    const float scale = i2f(FLD(op, 15));
    const int nhb = (H + HB - 1) / HB;
    const int GP = (gsz + 15) & ~15, MT = GP / 16;
    const int tpi = min(MT, max(1, (int)FLD(op, 21)));
    const int nmc = (MT + tpi - 1) / tpi;
    const int mc = item % nmc, hb = (item / nmc) % nhb, g = item / (nmc * nhb);
    const int h0 = hb * HB, nh = min(HB, H - h0);
    const int mt0 = mc * tpi, nmt = min(tpi, MT - mt0);
    const int base = row0 + g * gsz;
    constexpr int SV = HD / 8;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (FUSE && !pref) fa_prefetch_wo(op, item, smem);
    for (int s = tid; s < nh * 2 * GP * 4; s += NT) {
        const int sg = s & 3, r = (s >> 2) % GP, k = 1 + ((s >> 2) / GP) % 2, hh = (s >> 2) / (2 * GP);
        const bool ok = sg < SV && r < gsz;
        const bf16* src = qkv + (size_t)(base + (ok ? r : 0)) * ld + k * E + (h0 + hh) * HD + (ok ? sg * 8 : 0);
        bf16* dst = (k == 1 ? sm->K[hh] : sm->V[hh]) + r * FA_HP + sg * 8;
        cp_async16(dst, src, ok);
    }
    for (int s = tid; s < nh * nmt * 16 * 4; s += NT) {
        const int sg = s & 3, rr = (s >> 2) % (nmt * 16), hh = (s >> 2) / (nmt * 16), r = mt0 * 16 + rr;
        const bool ok = sg < SV && r < gsz;
        cp_async16(sm->Q[hh] + r * FA_HP + sg * 8, qkv + (size_t)(base + (ok ? r : 0)) * ld + (h0 + hh) * HD + (ok ? sg * 8 : 0), ok);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    const float sl2 = scale * 1.4426950408889634f;
    for (int task = warp; task < nh * nmt; task += NWARP) {
        const int hh = task / nmt, mt = mt0 + task % nmt;
        const bf16* Qs = sm->Q[hh];
        const bf16* Ks = sm->K[hh];
        const bf16* Vs = sm->V[hh];
        unsigned qa[2][4];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
            ldsm_x4(qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], Qs + (mt * 16 + (lane & 15)) * FA_HP + ks * 16 + (lane >> 4) * 8);
        float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
        float o[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) o[j][q] = 0.f;
        for (int kc = 0; kc < GP; kc += 32) {
            float s[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int q = 0; q < 4; ++q) s[nt][q] = 0.f;
                if (kc + nt * 8 < GP) {
#pragma unroll
                    for (int ks = 0; ks < 2; ++ks) {
                        unsigned b0, b1;
                        ldsm_x2(b0, b1, Ks + (kc + nt * 8 + (lane & 7)) * FA_HP + ks * 16 + ((lane >> 3) & 1) * 8);
                        mma16816(s[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b0, b1);
                    }
                }
            }
            float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int c = kc + nt * 8 + 2 * (lane & 3);
                s[nt][0] = c < gsz ? s[nt][0] * sl2 : -1e30f;
                s[nt][1] = c + 1 < gsz ? s[nt][1] * sl2 : -1e30f;
                s[nt][2] = c < gsz ? s[nt][2] * sl2 : -1e30f;
                s[nt][3] = c + 1 < gsz ? s[nt][3] * sl2 : -1e30f;
                mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
                mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
            const float c0 = exp2f(m0 - mn0), c1 = exp2f(m1 - mn1);
            m0 = mn0; m1 = mn1;
            l0 *= c0; l1 *= c1;
#pragma unroll
            for (int j = 0; j < 4; ++j) { o[j][0] *= c0; o[j][1] *= c0; o[j][2] *= c1; o[j][3] *= c1; }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                s[nt][0] = exp2f(s[nt][0] - mn0); s[nt][1] = exp2f(s[nt][1] - mn0);
                s[nt][2] = exp2f(s[nt][2] - mn1); s[nt][3] = exp2f(s[nt][3] - mn1);
                l0 += s[nt][0] + s[nt][1];
                l1 += s[nt][2] + s[nt][3];
            }
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                if (kc + kk * 16 >= GP) break;
                const unsigned a0 = pack_bf2(s[2 * kk][0], s[2 * kk][1]);
                const unsigned a1 = pack_bf2(s[2 * kk][2], s[2 * kk][3]);
                const unsigned a2 = pack_bf2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
                const unsigned a3 = pack_bf2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    unsigned b0, b1;
                    ldsm_x2_t(b0, b1, Vs + (kc + kk * 16 + (lane & 15)) * FA_HP + j * 8);
                    mma16816(o[j], a0, a1, a2, a3, b0, b1);
                }
            }
        }
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
        const int r0 = mt * 16 + (lane >> 2), r1 = r0 + 8;
        const int h = h0 + hh;
        if (!FUSE) {
            bf16* out = PTR(bf16*, op, 9);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = j * 8 + 2 * (lane & 3);
                if (d < HD) {
                    if (r0 < gsz)
                        *reinterpret_cast<__nv_bfloat162*>(out + (size_t)(base + r0) * ldo + h * HD + d) = __floats2bfloat162_rn(o[j][0] * i0, o[j][1] * i0);
                    if (r1 < gsz)
                        *reinterpret_cast<__nv_bfloat162*>(out + (size_t)(base + r1) * ldo + h * HD + d) = __floats2bfloat162_rn(o[j][2] * i1, o[j][3] * i1);
                }
            }
        } else {
            // bf16-rounded attention output tile staged in smem (Q[1] is free when one head per item)
            bf16* Ot = sm->Q[1] + (task * 16) * FA_HP;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = j * 8 + 2 * (lane & 3);
                *reinterpret_cast<__nv_bfloat162*>(Ot + (lane >> 2) * FA_HP + d) = __floats2bfloat162_rn(o[j][0] * i0, o[j][1] * i0);
                *reinterpret_cast<__nv_bfloat162*>(Ot + ((lane >> 2) + 8) * FA_HP + d) = __floats2bfloat162_rn(o[j][2] * i1, o[j][3] * i1);
            }
        }
    }
    if (FUSE) {
        // out-projection of all staged tiles shared by every warp, f32 atomics into acc
        __syncthreads();
        float* acc = PTR(float*, op, 9);
        const int NE = E / 8;
        for (int t2 = warp; t2 < nmt * NE; t2 += NWARP) {
            const int task = t2 / NE, nt = t2 % NE;
            const bf16* Ot = sm->Q[1] + (task * 16) * FA_HP;
            float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned a0, a1, a2, a3, b0, b1;
                ldsm_x4(a0, a1, a2, a3, Ot + (lane & 15) * FA_HP + kk * 16 + (lane >> 4) * 8);
                ldsm_x2(b0, b1, sm->Wo + (nt * 8 + (lane & 7)) * FA_HP + kk * 16 + ((lane >> 3) & 1) * 8);
                mma16816(c, a0, a1, a2, a3, b0, b1);
            }
            const int r0 = (mt0 + task) * 16 + (lane >> 2), r1 = r0 + 8, n = nt * 8 + 2 * (lane & 3);
            if (r0 < gsz) { float* p = acc + (size_t)(base + r0) * ldo + n; atomicAdd(p, c[0]); atomicAdd(p + 1, c[1]); }
            if (r1 < gsz) { float* p = acc + (size_t)(base + r1) * ldo + n; atomicAdd(p, c[2]); atomicAdd(p + 1, c[3]); }
        }
    }
}

__device__ void run_attn(const Op* op, int item, char* smem, int pref) {
    const int hd = (int)FLD(op, 7), fuse = (int)FLD(op, 17);
    if (fuse) { if (hd == 24) attn_fa<24, true>(op, item, smem, pref); else attn_fa<32, true>(op, item, smem, pref); }
    else { if (hd == 24) attn_fa<24, false>(op, item, smem, 0); else attn_fa<32, false>(op, item, smem, 0); }
}

// NORM fields: 4 X, 5 ldx, 6 Xn, 7 ldn, 8 M, 9 K, 10 gamma, 11 beta, 12 eps. 32 rows per item, one warp per row.
__device__ void norm_item(const Op* op, int item) {
    const bf16* X = PTR(const bf16*, op, 4);
    bf16* Xn = PTR(bf16*, op, 6);
    const int ldx = (int)FLD(op, 5), ldn = (int)FLD(op, 7), M = (int)FLD(op, 8), K = (int)FLD(op, 9);
    const bf16* gm = PTR(const bf16*, op, 10);
    const bf16* bt = PTR(const bf16*, op, 11);
    const float eps = i2f(FLD(op, 12));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const bool act = lane * 8 < K;
    float gw[8], gb[8];
    if (act) { bf8_to_f(__ldg(reinterpret_cast<const uint4*>(gm + lane * 8)), gw); bf8_to_f(__ldg(reinterpret_cast<const uint4*>(bt + lane * 8)), gb); }
    uint4 v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        int r = item * 32 + warp * 4 + q;
        v[q] = make_uint4(0, 0, 0, 0);
        if (r < M && act) v[q] = __ldcg(reinterpret_cast<const uint4*>(X + (size_t)r * ldx + lane * 8));
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        int r = item * 32 + warp * 4 + q;
        float f[8];
        bf8_to_f(v[q], f);
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) ss += f[i] * f[i];
        ss = warp_sum(ss);
        const float rs = rsqrtf(ss / (float)K + eps);
        if (r < M && act) {
            __align__(16) __nv_bfloat162 o[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                o[i] = __floats2bfloat162_rn(rbf(rbf(rbf(f[2 * i] * rs) * gw[2 * i]) + gb[2 * i]),
                                             rbf(rbf(rbf(f[2 * i + 1] * rs) * gw[2 * i + 1]) + gb[2 * i + 1]));
            *reinterpret_cast<uint4*>(Xn + (size_t)r * ldn + lane * 8) = *reinterpret_cast<uint4*>(o);
        }
    }
}

// ZERO fields: 4 ptr (f32), 5 n (multiple of 4)
__device__ void zero_item(const Op* op, int item) {
    float4* p = PTR(float4*, op, 4);
    const int n4 = (int)FLD(op, 5) / 4;
    for (int i = item * NT * 4 + threadIdx.x; i < min(n4, (item + 1) * NT * 4); i += NT) p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}


// ---------------------------------------------------------------- encoder embedding + first scatter
// fields: 4 syn_off, 5 syn_stride, 6 ngroups, 7 nchk, 8 ndata, 9 emb, 10 cpe, 11 dpe, 12 csr_ptr, 13 csr_idx,
//         14 outC, 15 outD, 16 ld, 17 chunks, 18 dim
#define MAXDEG 16
__device__ void embed_item(const Op* op, int item, const uint8_t* syn_all) {
    const int syn_off = (int)FLD(op, 4), syn_stride = (int)FLD(op, 5);
    const int nchk = (int)FLD(op, 7), ndata = (int)FLD(op, 8);
    const bf16* emb = PTR(const bf16*, op, 9);
    const bf16* cpe = PTR(const bf16*, op, 10);
    const bf16* dpe = PTR(const bf16*, op, 11);
    const int* cp = PTR(const int*, op, 12);
    const int* ci = PTR(const int*, op, 13);
    bf16* outC = PTR(bf16*, op, 14);
    bf16* outD = PTR(bf16*, op, 15);
    const int ld = (int)FLD(op, 16), chunks = (int)FLD(op, 17), D = (int)FLD(op, 18);
    const int g = item / chunks, ch = item % chunks;
    const uint8_t* syn = syn_all + syn_off + g * syn_stride;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int rr = warp; rr < 16; rr += NWARP) {
        int r = ch * 16 + rr;
        if (r >= nchk + ndata) break;
        if (r < nchk) {
            int s = __ldg(syn + r);
            for (int d = lane; d < D; d += 32) {
                float xa = rbf(tanhf(ldg_bf(emb + s * D + d)));
                outC[(size_t)(g * nchk + r) * ld + d] = f2bf(xa + ldg_bf(cpe + r * D + d));
            }
        } else {
            int dn = r - nchk;
            int e0 = __ldg(cp + dn), deg = __ldg(cp + dn + 1) - e0;
            int bit = 0;
            if (lane < deg) bit = __ldg(syn + __ldg(ci + e0 + lane));
            unsigned mask = __ballot_sync(0xffffffffu, bit);
            for (int d = lane; d < D; d += 32) {
                float t0 = rbf(tanhf(ldg_bf(emb + d))), t1 = rbf(tanhf(ldg_bf(emb + D + d)));
                float pr = 1.f;
                for (int e = 0; e < deg; ++e) pr = rbf(pr * (((mask >> e) & 1u) ? t1 : t0));
                outD[(size_t)(g * ndata + dn) * ld + d] = f2bf(pr + ldg_bf(dpe + dn * D + d));
            }
        }
    }
}

// fields: 4 XC, 5 chk_stride, 6 XD, 7 ngroups, 8 ndata, 9 csr_ptr, 10 csr_idx, 11 ld, 12 dim, 13 chunks
__device__ void scat_mid_item(const Op* op, int item) {
    const bf16* XC = PTR(const bf16*, op, 4);
    const int cs = (int)FLD(op, 5);
    bf16* XD = PTR(bf16*, op, 6);
    const int ndata = (int)FLD(op, 8);
    const int* cp = PTR(const int*, op, 9);
    const int* ci = PTR(const int*, op, 10);
    const int ld = (int)FLD(op, 11), D = (int)FLD(op, 12), chunks = (int)FLD(op, 13);
    const int g = item / chunks, ch = item % chunks;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int rr = warp; rr < 16; rr += NWARP) {
        int dn = ch * 16 + rr;
        if (dn >= ndata) break;
        int e0 = __ldg(cp + dn), deg = __ldg(cp + dn + 1) - e0;
        int myc = lane < deg ? __ldg(ci + e0 + lane) : 0;
        bf16* orow = XD + (size_t)(g * ndata + dn) * ld;
        for (int d = lane; d < D; d += 32) {
            float vals[MAXDEG];
#pragma unroll
            for (int e = 0; e < MAXDEG; ++e) {
                int c = __shfl_sync(0xffffffffu, myc, e);
                vals[e] = e < deg ? ldcg_bf(XC + (size_t)(g * cs + c) * ld + d) : 1.f;
            }
            float pr = 1.f;
#pragma unroll
            for (int e = 0; e < MAXDEG; ++e) if (e < deg) pr = rbf(pr * vals[e]);
            orow[d] = f2bf(ldcg_bf(orow + d) + pr);
        }
    }
}

// ---------------------------------------------------------------- gated delta rule (one warp per (data node, head))
// fields: 4 P, 5 ldp, 6 q_off, 7 k_off, 8 v_off, 9 g_off, 10 a_off, 11 b_off, 12 cwq, 13 cwk, 14 cwv,
//         15 A_log, 16 dt_bias, 17 onorm_w, 18 O, 19 ldo, 20 ndata, 21 L, 22 heads, 23 t_out0, 24 eps
struct GdnIn { float q, k, v0, v1, a, b; };
__device__ void gdn_item(const Op* op, int item, char* smem) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ndata = (int)FLD(op, 20), L = (int)FLD(op, 21), H = (int)FLD(op, 22), t0 = (int)FLD(op, 23);
    const int pidx = item * NWARP + warp;
    if (pidx >= ndata * H) return;
    const int n = pidx / H, h = pidx % H;
    const bf16* P = PTR(const bf16*, op, 4);
    const int ldp = (int)FLD(op, 5);
    const int cq = (int)FLD(op, 6) + h * 32 + lane;
    const int ck = (int)FLD(op, 7) + h * 32 + lane;
    const int cv = (int)FLD(op, 8) + h * 64 + 2 * lane;
    const int cg = (int)FLD(op, 9) + h * 64 + 2 * lane;
    const int ca = (int)FLD(op, 10) + h, cb = (int)FLD(op, 11) + h;
    const bf16* cwq = PTR(const bf16*, op, 12) + (h * 32 + lane) * 4;
    const bf16* cwk = PTR(const bf16*, op, 13) + (h * 32 + lane) * 4;
    const bf16* cwv = PTR(const bf16*, op, 14) + (h * 64 + 2 * lane) * 4;
    const float negA = -__expf(ldg_bf(PTR(const bf16*, op, 15) + h));
    const float dtb = ldg_bf(PTR(const bf16*, op, 16) + h);
    const bf16* onw = PTR(const bf16*, op, 17);
    bf16* O = PTR(bf16*, op, 18);
    const int ldo = (int)FLD(op, 19);
    const float eps = i2f(FLD(op, 24));
    const float wn0 = ldg_bf(onw + 2 * lane), wn1 = ldg_bf(onw + 2 * lane + 1);
    float wq[4], wk[4], wv0[4], wv1[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        wq[i] = ldg_bf(cwq + i); wk[i] = ldg_bf(cwk + i);
        wv0[i] = ldg_bf(cwv + i); wv1[i] = ldg_bf(cwv + 4 + i);
    }
    float hq[3] = {0.f, 0.f, 0.f}, hk[3] = {0.f, 0.f, 0.f}, hv0[3] = {0.f, 0.f, 0.f}, hv1[3] = {0.f, 0.f, 0.f};
    float S0[32], S1[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) { S0[i] = 0.f; S1[i] = 0.f; }
    float* qsh = reinterpret_cast<float*>(smem) + warp * 64;
    float* ksh = qsh + 32;
    const float scale = rsqrtf(32.f);
    auto fetch = [&](int t) {
        GdnIn x;
        const bf16* pr = P + (size_t)(t * ndata + n) * ldp;
        x.q = ldcg_bf(pr + cq); x.k = ldcg_bf(pr + ck);
        float2 vv = __bfloat1622float2(__ldcg(reinterpret_cast<const __nv_bfloat162*>(pr + cv)));
        x.v0 = vv.x; x.v1 = vv.y;
        x.a = ldcg_bf(pr + ca); x.b = ldcg_bf(pr + cb);
        return x;
    };
    GdnIn n1 = fetch(0), n2 = n1, n3 = n1;
    if (L > 1) n2 = fetch(1);
    if (L > 2) n3 = fetch(2);
    for (int t = 0; t < L; ++t) {
        GdnIn x = n1;
        n1 = n2;
        n2 = n3;
        if (t + 3 < L) n3 = fetch(t + 3);
        float yq = 0.f, yk = 0.f, yv0 = 0.f, yv1 = 0.f;
        yq += hq[0] * wq[0]; yq += hq[1] * wq[1]; yq += hq[2] * wq[2]; yq += x.q * wq[3];
        yk += hk[0] * wk[0]; yk += hk[1] * wk[1]; yk += hk[2] * wk[2]; yk += x.k * wk[3];
        yv0 += hv0[0] * wv0[0]; yv0 += hv0[1] * wv0[1]; yv0 += hv0[2] * wv0[2]; yv0 += x.v0 * wv0[3];
        yv1 += hv1[0] * wv1[0]; yv1 += hv1[1] * wv1[1]; yv1 += hv1[2] * wv1[2]; yv1 += x.v1 * wv1[3];
        hq[0] = hq[1]; hq[1] = hq[2]; hq[2] = x.q;
        hk[0] = hk[1]; hk[1] = hk[2]; hk[2] = x.k;
        hv0[0] = hv0[1]; hv0[1] = hv0[2]; hv0[2] = x.v0;
        hv1[0] = hv1[1]; hv1[1] = hv1[2]; hv1[2] = x.v1;
        yq = rbf(silu_f(yq)); yk = rbf(silu_f(yk)); yv0 = rbf(silu_f(yv0)); yv1 = rbf(silu_f(yv1));
        float sq = warp_sum(yq * yq), sk = warp_sum(yk * yk);
        qsh[lane] = yq / sqrtf(sq + 1e-6f) * scale;
        ksh[lane] = yk / sqrtf(sk + 1e-6f);
        __syncwarp();
        float beta = rbf(sigmoid_f(x.b));
        float a = x.a + dtb;
        float sp = a > 20.f ? a : log1pf(__expf(a));
        float eg = __expf(negA * sp);
        float kv0 = 0.f, kv1 = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            S0[i] *= eg; S1[i] *= eg;
            float kk = ksh[i];
            kv0 += S0[i] * kk; kv1 += S1[i] * kk;
        }
        float u0 = beta * (yv0 - kv0), u1 = beta * (yv1 - kv1);
        float o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            float kk = ksh[i], qq = qsh[i];
            S0[i] += kk * u0; S1[i] += kk * u1;
            o0 += S0[i] * qq; o1 += S1[i] * qq;
        }
        __syncwarp();
        if (t >= t0) {
            const bf16* pr = P + (size_t)(t * ndata + n) * ldp;
            o0 = rbf(o0); o1 = rbf(o1);
            float ss = warp_sum(o0 * o0 + o1 * o1);
            float rstd = 1.f / sqrtf(ss / 64.f + eps);
            float2 gg = __bfloat1622float2(__ldcg(reinterpret_cast<const __nv_bfloat162*>(pr + cg)));
            float y0 = o0 * rstd * wn0, y1 = o1 * rstd * wn1;
            y0 = y0 * gg.x * sigmoid_f(gg.x);
            y1 = y1 * gg.y * sigmoid_f(gg.y);
            *reinterpret_cast<__nv_bfloat162*>(O + (size_t)(t * ndata + n) * ldo + h * 64 + 2 * lane) =
                __floats2bfloat162_rn(y0, y1);
        }
    }
}

// ---------------------------------------------------------------- readout pooling and final projection
// SCAT_LOG fields: 4 V, 5 ld, 6 nlog, 7 csr_ptr, 8 csr_idx, 9 out, 10 dim, 11 nrows
__device__ void scat_log_item(const Op* op, int item, char* smem) {
    const bf16* V = PTR(const bf16*, op, 4);
    const int ld = (int)FLD(op, 5), nlog = (int)FLD(op, 6), D = (int)FLD(op, 10), nrows = (int)FLD(op, 11);
    const int* cp = PTR(const int*, op, 7);
    const int* ci = PTR(const int*, op, 8);
    bf16* out = PTR(bf16*, op, 9);
    bf16* Vs = reinterpret_cast<bf16*>(smem);
    const int segs = D / 8;
    for (int s = threadIdx.x; s < nrows * segs; s += NT) {
        int r = s / segs, sg = s % segs;
        cp_async16(Vs + r * D + sg * 8, V + (size_t)r * ld + sg * 8, true);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    for (int e = threadIdx.x; e < nlog * D; e += NT) {
        int j = e / D, d = e % D;
        float pr = 1.f;
        for (int q = __ldg(cp + j); q < __ldg(cp + j + 1); ++q) pr = rbf(pr * bf2f(Vs[__ldg(ci + q) * D + d]));
        out[(size_t)j * ld + d] = f2bf(pr);
    }
}

// FINAL fields: 4 Lm, 5 ld, 6 nlog, 7 w, 8 b, 9 dim
__device__ void final_item(const Op* op, float* logits) {
    const bf16* Lm = PTR(const bf16*, op, 4);
    const int ld = (int)FLD(op, 5), nlog = (int)FLD(op, 6), D = (int)FLD(op, 9);
    const bf16* w = PTR(const bf16*, op, 7);
    const bf16* b = PTR(const bf16*, op, 8);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int j = warp; j < nlog; j += NWARP) {
        float s = 0.f;
        for (int d = lane; d < D; d += 32) s += ldcg_bf(Lm + (size_t)j * ld + d) * ldg_bf(w + d);
        s = warp_sum(s);
        if (lane == 0) logits[j] = rbf(s + ldg_bf(b));
    }
}


// ---------------------------------------------------------------- interpreter
__device__ void run_op(const Op* op, int item, char* smem, const uint8_t* syn, float* logits, int pref) {
    switch ((int)FLD(op, 0)) {
        case OP_GEMM: run_gemm(op, item, smem, pref); break;
        case OP_ATTN: run_attn(op, item, smem, pref); break;
        case OP_EMBED: embed_item(op, item, syn); break;
        case OP_SCAT_MID: scat_mid_item(op, item); break;
        case OP_GDN: gdn_item(op, item, smem); break;
        case OP_SCAT_LOG: scat_log_item(op, item, smem); break;
        case OP_FINAL: final_item(op, logits); break;
        case OP_ZERO: zero_item(op, item); break;
        case OP_NORM: norm_item(op, item); break;
        default: break;
    }
}

__device__ int prefetch_op(const Op* op, int item, char* smem) {
    switch ((int)FLD(op, 0)) {
        case OP_GEMM: return gemm_prefetch(op, item, smem);
        case OP_ATTN:
            if (FLD(op, 17)) { fa_prefetch_wo(op, item, smem); return 1; }
            return 0;
        default: return 0;
    }
}

__shared__ unsigned short s_pb[MAXPH + 1];
__shared__ unsigned int s_pt[MAXPH], s_oe[MAXOPS];
__device__ __forceinline__ int find_op(const Args& a, int p, int w) {
    int oi = s_pb[p];
    while (w >= (int)s_oe[oi]) ++oi;
    return oi;
}

__device__ __forceinline__ void load_op(Op* dst, const Op* src) {
    if (threadIdx.x < 16) reinterpret_cast<uint4*>(dst)[threadIdx.x] = __ldg(reinterpret_cast<const uint4*>(src) + threadIdx.x);
}

extern "C" __global__ void __launch_bounds__(NT, 1) mk_kernel(Args a) {
    extern __shared__ __align__(16) char smem[];
    __shared__ __align__(16) Op s_op[3];
    unsigned gen = 0;
    if (threadIdx.x == 0) gen = a.barrier_mode == 1 ? 0u : ld_acquire(&a.bar[1]);
    for (int i = threadIdx.x; i <= a.n_phases; i += NT) s_pb[i] = (unsigned short)__ldg(a.phase_begin + i);
    for (int i = threadIdx.x; i < a.n_phases; i += NT) s_pt[i] = (unsigned int)__ldg(a.phase_total + i);
    for (int i = threadIdx.x; i < __ldg(a.phase_begin + a.n_phases); i += NT) s_oe[i] = (unsigned int)__ldg(a.op_end + i);
    __syncthreads();
    int cur = 0, pref = 0;
    if ((int)blockIdx.x < (int)s_pt[0]) {
        load_op(&s_op[0], a.ops + find_op(a, 0, blockIdx.x));
        __syncthreads();
        pref = a.prefetch ? prefetch_op(&s_op[0], blockIdx.x - (int)s_op[0].f[1], smem) : 0;
    }
    for (int s = 0; s < a.n_samples; ++s) {
        if (blockIdx.x == 0 && threadIdx.x == 0) a.times[2 * s] = globaltimer();
        const uint8_t* syn = a.syn + (size_t)(s % a.n_inputs) * a.n_det;
        float* logits = a.logits + (size_t)s * a.nlog;
        for (int p = 0; p < a.n_phases; ++p) {
            const int total = s_pt[p];
            int np = p + 1, ns = s;
            if (np == a.n_phases) { np = 0; ns = s + 1; }
            const bool has_next = ns < a.n_samples && (int)blockIdx.x < (int)s_pt[np];
            uint4 nxt = make_uint4(0, 0, 0, 0);
            if (has_next && threadIdx.x < 16)
                nxt = __ldg(reinterpret_cast<const uint4*>(a.ops + find_op(a, np, blockIdx.x)) + threadIdx.x);
            int w = blockIdx.x;
            if (w < total) {
                __syncthreads();
                DBG(6);
                run_op(&s_op[cur], w - (int)s_op[cur].f[1], smem, syn, logits, pref);
                DBG(7);
                for (w += gridDim.x; w < total; w += gridDim.x) {
                    const int oi = find_op(a, p, w);
                    __syncthreads();
                    load_op(&s_op[2], a.ops + oi);
                    __syncthreads();
                    run_op(&s_op[2], w - (int)s_op[2].f[1], smem, syn, logits, 0);
                }
            }
            cp_wait<0>();
            __syncthreads();
            if (has_next && threadIdx.x < 16) reinterpret_cast<uint4*>(&s_op[cur ^ 1])[threadIdx.x] = nxt;
            __syncthreads();
            pref = (has_next && a.prefetch) ? prefetch_op(&s_op[cur ^ 1], blockIdx.x - (int)s_op[cur ^ 1].f[1], smem) : 0;
            DBG(8);
            grid_sync(a.bar, gridDim.x, gen, a.barrier_mode);
            DBG(9);
            cur ^= 1;
            if (a.phase_t && blockIdx.x == 0 && threadIdx.x == 0) a.phase_t[p] = globaltimer();
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) a.times[2 * s + 1] = globaltimer();
    }
}

// ---------------------------------------------------------------- barrier-only probe
extern "C" __global__ void __launch_bounds__(NT, 1) barrier_probe(unsigned* bar, int iters, unsigned long long* t, int mode) {
    unsigned gen = 0;
    if (threadIdx.x == 0) gen = mode == 1 ? 0u : ld_acquire(&bar[1]);
    grid_sync(bar, gridDim.x, gen, mode);
    unsigned long long t0 = globaltimer();
    for (int i = 0; i < iters; ++i) grid_sync(bar, gridDim.x, gen, mode);
    if (blockIdx.x == 0 && threadIdx.x == 0) { t[0] = t0; t[1] = globaltimer(); }
}

extern "C" int mk_launch(const void* ops, const int* phase_begin, const int* phase_total, const int* op_end, int n_phases,
                         const uint8_t* syn, int n_det, int n_inputs, int n_samples, float* logits, int nlog,
                         unsigned long long* times, unsigned* bar, int grid, void* stream, unsigned long long* phase_t,
                         int barrier_mode, int prefetch) {
    if (n_phases > MAXPH) return -2;
    static bool init = false;
    if (!init) {
        cudaError_t e = cudaFuncSetAttribute(mk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        if (e != cudaSuccess) return (int)e;
        init = true;
    }
    Args a;
    a.ops = reinterpret_cast<const Op*>(ops);
    a.phase_begin = phase_begin;
    a.phase_total = phase_total;
    a.op_end = op_end;
    a.n_phases = n_phases;
    a.syn = syn;
    a.n_det = n_det;
    a.n_inputs = n_inputs;
    a.n_samples = n_samples;
    a.logits = logits;
    a.nlog = nlog;
    a.times = times;
    a.bar = bar;
    a.phase_t = phase_t;
    a.barrier_mode = barrier_mode;
    a.prefetch = prefetch;
    a.dbg = nullptr;
    void* params[] = {&a};
    cudaError_t e = cudaLaunchCooperativeKernel((void*)mk_kernel, dim3(grid), dim3(NT), params, SMEM_BYTES, (cudaStream_t)stream);
    return (int)e;
}

// Single-sample host round trip: pinned row i -> d_syn, one launch, logits -> pinned row i, stream sync.
struct E2EState {
    const void* ops; const int* pb; const int* pt; const int* oe; int n_phases;
    const uint8_t* h_syn; long long h_stride; uint8_t* d_syn; int n_det;
    float* d_logits; float* h_logits; int nlog; unsigned long long* times; unsigned* bars; int bar_stride;
    int grid; void* stream; int barrier_mode; int prefetch; int ncalls;
};
static E2EState g_e2e;

extern "C" int mk_e2e_prepare(const void* ops, const int* pb, const int* pt, const int* oe, int n_phases,
                              const uint8_t* h_syn, long long h_stride, uint8_t* d_syn, int n_det, float* d_logits,
                              float* h_logits, int nlog, unsigned long long* times, unsigned* bars, int bar_stride,
                              int grid, void* stream, int barrier_mode, int prefetch, int ncalls) {
    if (n_phases > MAXPH) return -2;
    g_e2e = E2EState{ops, pb, pt, oe, n_phases, h_syn, h_stride, d_syn, n_det, d_logits, h_logits, nlog, times, bars,
                     bar_stride, grid, stream, barrier_mode, prefetch, ncalls};
    return 0;
}

extern "C" int mk_e2e_call(int i) {
    const E2EState& g = g_e2e;
    if (i < 0 || i >= g.ncalls) return -1;
    cudaStream_t st = (cudaStream_t)g.stream;
    cudaError_t e = cudaMemcpyAsync(g.d_syn, g.h_syn + (size_t)i * g.h_stride, (size_t)g.n_det, cudaMemcpyHostToDevice, st);
    if (e != cudaSuccess) return (int)e;
    float* dl = g.d_logits + (size_t)i * g.nlog;
    int r = mk_launch(g.ops, g.pb, g.pt, g.oe, g.n_phases, g.d_syn, g.n_det, 1, 1, dl, g.nlog, g.times + 2 * (size_t)i,
                      g.bars + (size_t)i * g.bar_stride, g.grid, g.stream, nullptr, g.barrier_mode, g.prefetch);
    if (r) return r;
    e = cudaMemcpyAsync(g.h_logits + (size_t)i * g.nlog, dl, sizeof(float) * g.nlog, cudaMemcpyDeviceToHost, st);
    if (e != cudaSuccess) return (int)e;
    return (int)cudaStreamSynchronize(st);
}

extern "C" int mk_barrier_probe(unsigned* bar, int iters, unsigned long long* t, int grid, void* stream, int mode) {
    void* params[] = {&bar, &iters, &t, &mode};
    return (int)cudaLaunchCooperativeKernel((void*)barrier_probe, dim3(grid), dim3(NT), params, 0, (cudaStream_t)stream);
}

extern "C" int mk_occupancy(int* blocks_per_sm) {
    cudaFuncSetAttribute(mk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    return (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, mk_kernel, NT, SMEM_BYTES);
}

extern "C" int mk_set_dbg(unsigned long long* p) { return (int)cudaMemcpyToSymbol(g_dbg, &p, sizeof(p)); }

static_assert(sizeof(FaSmem) <= SMEM_BYTES, "fa smem");

// Shared-memory need of mk_kernel versus the device opt-in limit (build/runtime capability check).
extern "C" int mk_resource_info(int* static_smem, int* dynamic_smem, int* optin_limit, int* coop_supported) {
    cudaFuncAttributes attr;
    cudaError_t e = cudaFuncGetAttributes(&attr, mk_kernel);
    if (e != cudaSuccess) return (int)e;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(coop_supported, cudaDevAttrCooperativeLaunch, dev);
    *static_smem = (int)attr.sharedSizeBytes;
    *dynamic_smem = SMEM_BYTES;
    return 0;
}
