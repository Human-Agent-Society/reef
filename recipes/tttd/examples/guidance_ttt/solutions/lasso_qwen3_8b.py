# EVOLVE-BLOCK-START

CPP_CODE = r'''
#define EIGEN_NO_DEBUG
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstdint>
#include <unistd.h>
#include <algorithm>
#include <limits>
#include <cstring>
#include <utility>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;

static void readbin(void* ptr, size_t sz) {
    char* buf = (char*)ptr;
    size_t got = 0;
    while (got < sz) {
        ssize_t r = ::read(0, buf+got, sz-got);
        if (r <= 0) std::_Exit(1);
        got += (size_t)r;
    }
}
static void writebin(const void* ptr, size_t sz) {
    const char* buf = (const char*)ptr;
    size_t put = 0;
    while (put < sz) {
        ssize_t r = ::write(1, buf+put, sz-put);
        if (r <= 0) std::_Exit(1);
        put += (size_t)r;
    }
}

// ========== SIMD AXPY: y -= alpha * x ==========
#ifdef __AVX512F__
static inline void axpy_neg(double* __restrict__ y, const double* __restrict__ x, double alpha, int n) {
    if (n <= 0) return;
    __m512d va = _mm512_set1_pd(alpha);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512d x0 = _mm512_loadu_pd(x + i);
        __m512d x1 = _mm512_loadu_pd(x + i + 8);
        __m512d y0 = _mm512_loadu_pd(y + i);
        __m512d y1 = _mm512_loadu_pd(y + i + 8);
        _mm512_storeu_pd(y + i, _mm512_fnmadd_pd(va, x0, y0));
        _mm512_storeu_pd(y + i + 8, _mm512_fnmadd_pd(va, x1, y1));
    }
    for (; i + 7 < n; i += 8) {
        _mm512_storeu_pd(y + i, _mm512_fnmadd_pd(va, _mm512_loadu_pd(x + i), _mm512_loadu_pd(y + i)));
    }
    for (; i < n; i++) y[i] -= alpha * x[i];
}

static inline double dot_simd(const double* __restrict__ a, const double* __restrict__ b, int n) {
    if (n <= 0) return 0.0;
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i), _mm512_loadu_pd(b + i), acc0);
        acc1 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i + 8), _mm512_loadu_pd(b + i + 8), acc1);
    }
    for (; i + 7 < n; i += 8) {
        acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i), _mm512_loadu_pd(b + i), acc0);
    }
    double s = _mm512_reduce_add_pd(_mm512_add_pd(acc0, acc1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline void dot2_simd(const double* __restrict__ x, const double* __restrict__ r,
                             const double* __restrict__ u, int n, double& cj, double& aj) {
    if (n <= 0) { cj = 0; aj = 0; return; }
    __m512d vc0 = _mm512_setzero_pd(), vc1 = _mm512_setzero_pd();
    __m512d va0 = _mm512_setzero_pd(), va1 = _mm512_setzero_pd();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512d x0 = _mm512_loadu_pd(x + i), x1 = _mm512_loadu_pd(x + i + 8);
        vc0 = _mm512_fmadd_pd(x0, _mm512_loadu_pd(r + i), vc0);
        vc1 = _mm512_fmadd_pd(x1, _mm512_loadu_pd(r + i + 8), vc1);
        va0 = _mm512_fmadd_pd(x0, _mm512_loadu_pd(u + i), va0);
        va1 = _mm512_fmadd_pd(x1, _mm512_loadu_pd(u + i + 8), va1);
    }
    for (; i + 7 < n; i += 8) {
        __m512d x0 = _mm512_loadu_pd(x + i);
        vc0 = _mm512_fmadd_pd(x0, _mm512_loadu_pd(r + i), vc0);
        va0 = _mm512_fmadd_pd(x0, _mm512_loadu_pd(u + i), va0);
    }
    __m512d vc = _mm512_add_pd(vc0, vc1);
    __m512d va = _mm512_add_pd(va0, va1);
    cj = _mm512_reduce_add_pd(vc);
    aj = _mm512_reduce_add_pd(va);
    for (; i < n; i++) { cj += x[i] * r[i]; aj += x[i] * u[i]; }
}
#elif defined(__AVX2__)
static inline void axpy_neg(double* __restrict__ y, const double* __restrict__ x, double alpha, int n) {
    if (n <= 0) return;
    __m256d va = _mm256_set1_pd(alpha);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256d x0 = _mm256_loadu_pd(x + i);
        __m256d x1 = _mm256_loadu_pd(x + i + 4);
        __m256d y0 = _mm256_loadu_pd(y + i);
        __m256d y1 = _mm256_loadu_pd(y + i + 4);
        _mm256_storeu_pd(y + i, _mm256_sub_pd(y0, _mm256_mul_pd(va, x0)));
        _mm256_storeu_pd(y + i + 4, _mm256_sub_pd(y1, _mm256_mul_pd(va, x1)));
    }
    for (; i + 3 < n; i += 4) {
        _mm256_storeu_pd(y + i, _mm256_sub_pd(_mm256_loadu_pd(y + i), _mm256_mul_pd(va, _mm256_loadu_pd(x + i))));
    }
    for (; i < n; i++) y[i] -= alpha * x[i];
}

static inline double dot_simd(const double* __restrict__ a, const double* __restrict__ b, int n) {
    if (n <= 0) return 0.0;
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4), acc1);
    }
    for (; i + 3 < n; i += 4) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), acc0);
    }
    __m256d acc = _mm256_add_pd(acc0, acc1);
    double tmp[4] __attribute__((aligned(32)));
    _mm256_store_pd(tmp, acc);
    double s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline void dot2_simd(const double* __restrict__ x, const double* __restrict__ r,
                             const double* __restrict__ u, int n, double& cj, double& aj) {
    if (n <= 0) { cj = 0; aj = 0; return; }
    __m256d vc0 = _mm256_setzero_pd(), vc1 = _mm256_setzero_pd();
    __m256d va0 = _mm256_setzero_pd(), va1 = _mm256_setzero_pd();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256d x0 = _mm256_loadu_pd(x + i), x1 = _mm256_loadu_pd(x + i + 4);
        vc0 = _mm256_fmadd_pd(x0, _mm256_loadu_pd(r + i), vc0);
        vc1 = _mm256_fmadd_pd(x1, _mm256_loadu_pd(r + i + 4), vc1);
        va0 = _mm256_fmadd_pd(x0, _mm256_loadu_pd(u + i), va0);
        va1 = _mm256_fmadd_pd(x1, _mm256_loadu_pd(u + i + 4), va1);
    }
    for (; i + 3 < n; i += 4) {
        __m256d x0 = _mm256_loadu_pd(x + i);
        vc0 = _mm256_fmadd_pd(x0, _mm256_loadu_pd(r + i), vc0);
        va0 = _mm256_fmadd_pd(x0, _mm256_loadu_pd(u + i), va0);
    }
    __m256d vc = _mm256_add_pd(vc0, vc1);
    __m256d va = _mm256_add_pd(va0, va1);
    double tc[4] __attribute__((aligned(32)));
    double ta[4] __attribute__((aligned(32)));
    _mm256_store_pd(tc, vc);
    _mm256_store_pd(ta, va);
    cj = tc[0] + tc[1] + tc[2] + tc[3];
    aj = ta[0] + ta[1] + ta[2] + ta[3];
    for (; i < n; i++) { cj += x[i] * r[i]; aj += x[i] * u[i]; }
}
#else
static inline void axpy_neg(double* y, const double* x, double alpha, int n) {
    for (int i = 0; i < n; i++) y[i] -= alpha * x[i];
}
static inline double dot_simd(const double* a, const double* b, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
static inline void dot2_simd(const double* x, const double* r, const double* u, int n, double& cj, double& aj) {
    cj = 0; aj = 0;
    for (int i = 0; i < n; i++) { cj += x[i] * r[i]; aj += x[i] * u[i]; }
}
#endif

// ========== Branchless max-select for segment tree ==========
// Selects (lv, li) if lv >= rv, else (rv, ri), using bitwise masking (no branch)
static inline void max_select_branchless(double lv, int li, double rv, int ri,
                                         double& ov, int& oi) {
    int ge = (lv >= rv);
    int mask_i = -ge;               // 0xFFFFFFFF if ge, 0x00000000 if not
    uint64_t mask_u = (uint64_t)0 - (uint64_t)(unsigned)ge;
    uint64_t lu, ru;
    std::memcpy(&lu, &lv, sizeof(double));
    std::memcpy(&ru, &rv, sizeof(double));
    uint64_t ou = (lu & mask_u) | (ru & ~mask_u);
    std::memcpy(&ov, &ou, sizeof(double));
    oi = (li & mask_i) | (ri & ~mask_i);
}

// Propagate one leaf change up the tree using branchless selects
static inline void seg_propagate_up(double* segVal, int* segIdx, int segN, int pos) {
    pos >>= 1;
    while (pos >= 1) {
        int c2 = 2 * pos;
        double ov;
        int oi;
        max_select_branchless(segVal[c2], segIdx[c2], segVal[c2+1], segIdx[c2+1], ov, oi);
        segVal[pos] = ov;
        segIdx[pos] = oi;
        pos >>= 1;
    }
}

static inline void update_residual_parallel(double* r, const double* uA, double gamma, int n) {
    if (n <= 0) return;
#ifdef _OPENMP
    if (n > 4096) {
        #pragma omp parallel for schedule(static)
        for (int blk = 0; blk < n; blk += 4096) {
            int blk_end = blk + 4096;
            if (blk_end > n) blk_end = n;
            axpy_neg(r + blk, uA + blk, gamma, blk_end - blk);
        }
    } else {
        axpy_neg(r, uA, gamma, n);
    }
#else
    axpy_neg(r, uA, gamma, n);
#endif
}

int main() {
    int32_t hdr[3];
    readbin(hdr, 12);
    int n = hdr[0], p = hdr[1], n_lambda = hdr[2];

    const double dn = (double)n;
    const double eps = 1e-12;
    const double tolC = 1e-10;
    const double INF = std::numeric_limits<double>::infinity();

    bool useGram = ((size_t)p*(size_t)p*sizeof(double) <= (size_t)64*1024*1024)
                   && ((double)n * (double)p * (double)p <= 5e8)
                   && (p <= 3000)
                   && (n > 5 * p);

    RowMatrix Xrm(n, p);
    readbin(Xrm.data(), (size_t)n*p*sizeof(double));

    VectorXd y(n);
    readbin(y.data(), n*sizeof(double));
    VectorXd lambda_path(n_lambda);
    readbin(lambda_path.data(), n_lambda*sizeof(double));

#ifdef _OPENMP
    Eigen::setNbThreads(omp_get_max_threads());
#endif

    VectorXd Xty(p);
    MatrixXd Gram;
    MatrixXd X;
    VectorXd Gdiag(p);

    if (useGram) {
        Xty.noalias() = Xrm.transpose() * y / dn;
        Gram.noalias() = Xrm.transpose() * Xrm / dn;
        Gdiag = Gram.diagonal();
    } else {
        X = Xrm;
        #pragma omp parallel for schedule(static) if(p > 64)
        for (int j = 0; j < p; j++) {
            const double* col = &X.data()[(size_t)j * n];
            Xty(j) = dot_simd(col, y.data(), n) / dn;
            Gdiag(j) = dot_simd(col, col, n) / dn;
        }
    }
    RowMatrix().swap(Xrm);

    VectorXd r;
    if (!useGram) r = y;

    VectorXd beta = VectorXd::Zero(p);

    std::vector<double> activeMaskD(p, 0.0);
    int maxK = std::min(n, p) + 1;
    std::vector<int> activeIdx(maxK);
    std::vector<double> signs(maxK);
    std::vector<const double*> active_cols(maxK, nullptr);
    int nActive = 0;

    std::vector<double> Lbuf(maxK * maxK, 0.0);
    int Lk = 0;

    std::vector<double> G_active;
    if (useGram) G_active.assign((size_t)maxK * p, 0.0);

    std::vector<double> sA_buf(maxK), zbuf(maxK), avec_buf(maxK), wA_buf(maxK), vbuf(maxK);
    VectorXd uA;
    if (!useGram) uA = VectorXd::Zero(n);

    double delta_accum = 0.0;

    // Segment tree for max |c_j| queries over inactive features
    // Split into separate arrays for branchless bitwise max-selection
    int segN = 1;
    while (segN < p) segN <<= 1;
    if (segN < 1) segN = 1;
    std::vector<double> segVal(2 * segN, -1.0);
    std::vector<int> segIdx(2 * segN, -1);

    auto segUpdate = [&](int j, double val) {
        int pos = segN + j;
        segVal[pos] = val;
        segIdx[pos] = j;
        seg_propagate_up(segVal.data(), segIdx.data(), segN, pos);
    };

    auto segDeactivate = [&](int j) {
        int pos = segN + j;
        segVal[pos] = -1.0;
        segIdx[pos] = -1;
        seg_propagate_up(segVal.data(), segIdx.data(), segN, pos);
    };

    std::vector<std::pair<double,int>> checked;
    checked.reserve(256);

    std::vector<double> knot_lambda;
    std::vector<std::vector<std::pair<int,double>>> knot_sparse;

    auto recordKnot = [&](double lam) {
        std::vector<std::pair<int,double>> snap;
        for (int i=0;i<nActive;i++) {
            int j = activeIdx[i];
            double v = beta(j);
            if (v != 0.0) snap.emplace_back(j, v);
        }
        knot_lambda.push_back(lam);
        knot_sparse.push_back(std::move(snap));
    };

    auto compute_cj_aj = [&](int j, double& cj, double& aj) {
        if (useGram) {
            cj = Xty(j);
            aj = 0.0;
            const double* gram_col = &Gram.data()[(size_t)j * p];
            for (int i = 0; i < nActive; i++) {
                double g = gram_col[activeIdx[i]];
                cj -= beta(activeIdx[i]) * g;
                aj += wA_buf[i] * g;
            }
        } else {
            const double* xj = &X.data()[(size_t)j * n];
            dot2_simd(xj, r.data(), uA.data(), n, cj, aj);
            cj /= dn;
            aj /= dn;
        }
    };

    auto rebuild_tree = [&]() {
        if (useGram) {
            for (int j = 0; j < p; j++) {
                if (activeMaskD[j] < 0.5) {
                    double cj = Xty(j);
                    const double* gram_col = &Gram.data()[(size_t)j * p];
                    for (int i = 0; i < nActive; i++)
                        cj -= beta(activeIdx[i]) * gram_col[activeIdx[i]];
                    segVal[segN + j] = std::fabs(cj);
                    segIdx[segN + j] = j;
                } else {
                    segVal[segN + j] = -1.0;
                    segIdx[segN + j] = -1;
                }
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < p; j++) {
                if (activeMaskD[j] < 0.5) {
                    double cj = dot_simd(&X.data()[(size_t)j * n], r.data(), n) / dn;
                    segVal[segN + j] = std::fabs(cj);
                    segIdx[segN + j] = j;
                } else {
                    segVal[segN + j] = -1.0;
                    segIdx[segN + j] = -1;
                }
            }
        }
        for (int j = p; j < segN; j++) {
            segVal[segN + j] = -1.0;
            segIdx[segN + j] = -1;
        }
        // Bottom-up build with branchless propagation
        for (int i = segN - 1; i >= 1; i--) {
            int c2 = 2 * i;
            double ov;
            int oi;
            max_select_branchless(segVal[c2], segIdx[c2], segVal[c2+1], segIdx[c2+1], ov, oi);
            segVal[i] = ov;
            segIdx[i] = oi;
        }
        delta_accum = 0.0;
    };

    auto lsolve = [&](double* x, const double* b) {
        for (int i = 0; i < Lk; i++) {
            const double* Li = &Lbuf[i * maxK];
            double s = b[i] - dot_simd(Li, x, i);
            x[i] = s / Li[i];
        }
    };

    auto usolve = [&](double* x, double* b) {
        for (int j = Lk - 1; j >= 0; j--) {
            x[j] = b[j] / Lbuf[j * maxK + j];
            double xj = x[j];
            if (j > 0) axpy_neg(b, &Lbuf[j * maxK], xj, j);
        }
    };

    auto chol_update = [&](int j_new) -> bool {
        if (useGram) {
            for (int i = 0; i < nActive; i++)
                vbuf[i] = Gram(activeIdx[i], j_new);
        } else {
            const double* xj_new = &X.data()[(size_t)j_new * n];
            #pragma omp parallel for schedule(static) if(nActive > 32)
            for (int i = 0; i < nActive; i++)
                vbuf[i] = dot_simd(active_cols[i], xj_new, n) / dn;
        }
        lsolve(zbuf.data(), vbuf.data());
        double dd = Gdiag(j_new) - dot_simd(zbuf.data(), zbuf.data(), Lk);
        if (dd <= 1e-20) return false;
        std::memcpy(&Lbuf[Lk * maxK], zbuf.data(), Lk * sizeof(double));
        Lbuf[Lk * maxK + Lk] = std::sqrt(dd);
        Lk++;
        if (useGram) {
            std::memcpy(&G_active[(size_t)nActive * p],
                        &Gram.data()[(size_t)j_new * p],
                        (size_t)p * sizeof(double));
        }
        return true;
    };

    auto chol_downdate = [&](int drop_idx) {
        if (Lk <= 1) { Lk = 0; return; }
        int k = Lk;
        int newk = k - 1;
        for (int i = drop_idx; i < newk; i++) {
            int ncols = i + 2;
            std::memcpy(&Lbuf[i * maxK], &Lbuf[(i+1) * maxK], (size_t)ncols * sizeof(double));
        }
        for (int rr = drop_idx; rr < newk; rr++) {
            double av = Lbuf[rr * maxK + rr];
            double bv = Lbuf[rr * maxK + rr + 1];
            if (std::fabs(bv) < 1e-300) continue;
            double rroot = std::sqrt(av * av + bv * bv);
            double cc = av / rroot;
            double ss = bv / rroot;
            for (int row = rr; row < newk; row++) {
                double t1 = cc * Lbuf[row * maxK + rr] + ss * Lbuf[row * maxK + rr + 1];
                double t2 = -ss * Lbuf[row * maxK + rr] + cc * Lbuf[row * maxK + rr + 1];
                Lbuf[row * maxK + rr] = t1;
                Lbuf[row * maxK + rr + 1] = t2;
            }
        }
        Lk = newk;
        if (useGram && drop_idx < nActive) {
            std::memmove(&G_active[(size_t)drop_idx * p],
                         &G_active[(size_t)(drop_idx + 1) * p],
                         (size_t)(nActive - drop_idx) * p * sizeof(double));
        }
    };

    // Find lambda_max
    int j_first = 0;
    double lambda_max = 0;
    for (int j=0;j<p;j++) {
        double av = std::fabs(Xty(j));
        if (av > lambda_max) { lambda_max = av; j_first = j; }
    }
    recordKnot(lambda_max);

    double smallest_lam = lambda_path(n_lambda-1);

    if (lambda_max > tolC) {
        activeMaskD[j_first] = 1.0;
        activeIdx[0] = j_first;
        signs[0] = Xty(j_first) >= 0 ? 1.0 : -1.0;
        if (!useGram) active_cols[0] = &X.data()[(size_t)j_first * n];
        nActive = 1;

        Lbuf[0] = std::sqrt(Gdiag(j_first));
        Lk = 1;
        if (useGram) {
            std::memcpy(&G_active[0], &Gram.data()[(size_t)j_first * p],
                        (size_t)p * sizeof(double));
        }

        // Initialize segment tree leaves with |Xty| for inactive features
        for (int j = 0; j < p; j++) {
            if (activeMaskD[j] < 0.5) {
                segVal[segN + j] = std::fabs(Xty(j));
                segIdx[segN + j] = j;
            } else {
                segVal[segN + j] = -1.0;
                segIdx[segN + j] = -1;
            }
        }
        for (int j = p; j < segN; j++) {
            segVal[segN + j] = -1.0;
            segIdx[segN + j] = -1;
        }
        // Bottom-up build with branchless propagation
        for (int i = segN - 1; i >= 1; i--) {
            int c2 = 2 * i;
            double ov;
            int oi;
            max_select_branchless(segVal[c2], segIdx[c2], segVal[c2+1], segIdx[c2+1], ov, oi);
            segVal[i] = ov;
            segIdx[i] = oi;
        }

        double C = lambda_max;
        int maxIter = 4 * std::min(n, p) + 200;

        for (int iter=0; iter<maxIter; iter++) {
            int k = nActive;
            if (k == 0) break;

            for (int i=0;i<k;i++) sA_buf[i] = signs[i];

            lsolve(zbuf.data(), sA_buf.data());
            usolve(avec_buf.data(), zbuf.data());

            double dot_sa = dot_simd(sA_buf.data(), avec_buf.data(), k);
            double A_A = 1.0 / std::sqrt(dot_sa);

            for (int i=0;i<k;i++) wA_buf[i] = A_A * avec_buf[i];

            if (!useGram) {
                std::memset(uA.data(), 0, (size_t)n * sizeof(double));
                #pragma omp parallel for schedule(static) if(n > 2048)
                for (int blk = 0; blk < n; blk += 512) {
                    int blk_end = blk + 512;
                    if (blk_end > n) blk_end = n;
                    int blk_len = blk_end - blk;
                    for (int i = 0; i < k; i++) {
                        double w = wA_buf[i];
                        if (w != 0.0) {
                            axpy_neg(&uA.data()[blk], &active_cols[i][blk], -w, blk_len);
                        }
                    }
                }
            }

            double gamma_lasso = INF;
            int drop_idx = -1;
            for (int ii=0;ii<k;ii++) {
                double w=wA_buf[ii], bv=beta(activeIdx[ii]);
                if (std::fabs(w)>1e-15) {
                    double d = -bv/w;
                    if (d>1e-12 && d<gamma_lasso) { gamma_lasso=d; drop_idx=ii; }
                }
            }

            // Segment tree-based gamma_hat screening
            double gamma_best = std::isfinite(gamma_lasso) ? gamma_lasso : INF;
            int j_new = -1; double s_new = 0.0;

            checked.clear();

            while (true) {
                // O(1) max query: read root (branchless tree)
                double stale_cj = segVal[1];
                int j = segIdx[1];

                if (j < 0) break; // no inactive features

                if (activeMaskD[j] > 0.5) {
                    segDeactivate(j);
                    continue;
                }

                // Safe screening: O(1) check, no push-back needed
                double safe_thresh = C - 2.0 * A_A * gamma_best - delta_accum;
                if (std::isfinite(gamma_best) && safe_thresh > 0 && stale_cj < safe_thresh) {
                    break; // all remaining features have smaller stale|c_j|
                }

                // Temporarily remove from tree for consideration
                segDeactivate(j);

                // Compute actual c_j and a_j (lazy evaluation)
                double cj, aj;
                compute_cj_aj(j, cj, aj);

                double abs_cj = std::fabs(cj);

                // Exact screening
                double exact_thresh = C - 2.0 * A_A * gamma_best;
                if (std::isfinite(gamma_best) && exact_thresh > 0 && abs_cj < exact_thresh) {
                    checked.emplace_back(abs_cj, j);
                    continue;
                }

                // Compute gamma_j
                double d1 = A_A - aj, d2 = A_A + aj;
                double gp = (d1 > eps) ? (C - cj) / d1 : INF;
                double gm = (d2 > eps) ? (C + cj) / d2 : INF;
                if (gp < 0) gp = INF;
                if (gm < 0) gm = INF;

                if (gp < gamma_best) { gamma_best = gp; j_new = j; s_new = 1.0; }
                if (gm < gamma_best) { gamma_best = gm; j_new = j; s_new = -1.0; }

                checked.emplace_back(abs_cj, j);
            }

            // Restore checked features to tree with updated |c_j|
            for (auto& pr : checked) {
                segUpdate(pr.second, pr.first);
            }

            double gamma_hat = (j_new >= 0) ? gamma_best : INF;
            double gamma = std::min(gamma_hat, gamma_lasso);

            if (!std::isfinite(gamma)) {
                gamma = C / A_A;
                for (int ii=0;ii<k;ii++) beta(activeIdx[ii]) += gamma*wA_buf[ii];
                if (!useGram) update_residual_parallel(r.data(), uA.data(), gamma, n);
                double lambda_new = C - gamma*A_A;
                recordKnot(lambda_new);
                break;
            }

            if (gamma <= 1e-15) {
                if (gamma_lasso < gamma_hat && drop_idx>=0) {
                    int dj = activeIdx[drop_idx];
                    beta(dj)=0.0;
                    activeMaskD[dj]=0.0;
                    for (int i=drop_idx;i<nActive-1;i++) {
                        activeIdx[i] = activeIdx[i+1];
                        signs[i] = signs[i+1];
                        active_cols[i] = active_cols[i+1];
                    }
                    nActive--;
                    chol_downdate(drop_idx);
                    double cj_d, aj_d;
                    compute_cj_aj(dj, cj_d, aj_d);
                    segUpdate(dj, std::fabs(cj_d));
                } else if (j_new>=0) {
                    if (!chol_update(j_new)) { recordKnot(C); continue; }
                    activeMaskD[j_new]=1.0;
                    segDeactivate(j_new);
                    activeIdx[nActive] = j_new;
                    signs[nActive] = s_new;
                    if (!useGram) active_cols[nActive] = &X.data()[(size_t)j_new * n];
                    nActive++;
                }
                recordKnot(C);
                continue;
            }

            for (int ii=0;ii<k;ii++) beta(activeIdx[ii]) += gamma*wA_buf[ii];

            if (!useGram) update_residual_parallel(r.data(), uA.data(), gamma, n);

            double lambda_new = C - gamma*A_A;

            if (gamma_lasso < gamma_hat) {
                int dj = activeIdx[drop_idx];
                beta(dj)=0.0;
                activeMaskD[dj]=0.0;
                for (int i=drop_idx;i<nActive-1;i++) {
                    activeIdx[i] = activeIdx[i+1];
                    signs[i] = signs[i+1];
                    active_cols[i] = active_cols[i+1];
                }
                nActive--;
                chol_downdate(drop_idx);
                double cj_d, aj_d;
                compute_cj_aj(dj, cj_d, aj_d);
                segUpdate(dj, std::fabs(cj_d));
            } else {
                if (!chol_update(j_new)) {
                    recordKnot(lambda_new);
                    C = lambda_new;
                    delta_accum += gamma * A_A;
                    if (delta_accum > std::max(C * 0.1, 1e-4))
                        rebuild_tree();
                    if (C < smallest_lam) break;
                    continue;
                }
                activeMaskD[j_new]=1.0;
                segDeactivate(j_new);
                activeIdx[nActive] = j_new;
                signs[nActive] = s_new;
                if (!useGram) active_cols[nActive] = &X.data()[(size_t)j_new * n];
                nActive++;
            }

            recordKnot(lambda_new);
            C = lambda_new;

            delta_accum += gamma * A_A;
            if (delta_accum > std::max(C * 0.1, 1e-4))
                rebuild_tree();

            if (C < smallest_lam - 1e-15) break;
        }
    }

    int nk = (int)knot_lambda.size();
    VectorXd coef_path = VectorXd::Zero(p * n_lambda);

    if (nk >= 1) {
        VectorXd b(p);
        int seg = 0;
        for (int kk = 0; kk < n_lambda; kk++) {
            double lam = lambda_path(kk);
            b.setZero();

            if (lam >= knot_lambda[0] - 1e-15) {
                continue;
            }

            if (lam <= knot_lambda[nk-1] + 1e-15) {
                for (auto& kv : knot_sparse[nk-1])
                    b(kv.first) = kv.second;
                coef_path.segment(kk * p, p) = b;
                continue;
            }

            int lo = 0, hi = nk - 1;
            while (lo < hi - 1) {
                int mid = (lo + hi) / 2;
                if (knot_lambda[mid] > lam) lo = mid;
                else hi = mid;
            }
            seg = lo;

            double la = knot_lambda[seg], lb = knot_lambda[seg+1];
            if (std::fabs(la - lb) < 1e-15) {
                for (auto& kv : knot_sparse[seg])
                    b(kv.first) = kv.second;
            } else {
                double t = (la - lam) / (la - lb);
                for (auto& kv : knot_sparse[seg])
                    b(kv.first) = (1.0 - t) * kv.second;
                for (auto& kv : knot_sparse[seg+1])
                    b(kv.first) += t * kv.second;
            }

            for (auto& kv : knot_sparse[seg])
                if (std::fabs(b(kv.first)) < 1e-12) b(kv.first) = 0.0;
            for (auto& kv : knot_sparse[seg+1])
                if (std::fabs(b(kv.first)) < 1e-12) b(kv.first) = 0.0;

            coef_path.segment(kk * p, p) = b;
        }
    }

    writebin(coef_path.data(), (size_t)p*n_lambda*sizeof(double));
    return 0;
}
'''

COMPILE_FLAGS = ["-fopenmp"]

# EVOLVE-BLOCK-END