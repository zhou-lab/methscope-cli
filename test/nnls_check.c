/* SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
 * Use of this software is available to academic and non-profit institutions
 * for research purposes under the 2-Clause BSD License; for use or transfers
 * to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
 * See the LICENSE file at the root of the repository for the full terms.
 *
 * nnls_check -- exercises the vendored Lawson-Hanson solver (src/nnls.c)
 * directly, which `deconv` alone cannot: its references are well
 * conditioned, so the active set only ever grows and the solver's
 * backtrack (a coefficient turning negative and moving from set P back to
 * set Z) never runs. Built and run by `make test`; no input, no network.
 *
 * Three checks. (1) Hand cases with known answers, including the empty
 * problem (mode 2). (2) An oracle: for n <= 4, every support is enumerated,
 * solved unconstrained, and the feasible one with the smallest residual is
 * the NNLS answer; the solver must match its residual. (3) A seeded sweep
 * of random problems checked against the KKT conditions -- x >= 0, every
 * dual w_j = a_j . (b - A x) <= 0, and w_j = 0 wherever x_j > 0 -- which is
 * what makes a point THE constrained minimiser. Random, near-collinear
 * columns are what force the backtrack.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nnls.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; \
  fprintf(stderr, "nnls_check: "); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

/* Solve min ||A x - b|| s.t. x >= 0 with A given ROW-major (m x n). Returns
 * mode; fills x and rnorm. A and b are copied: nnls_c overwrites its input. */
static int solve(int m, int n, const double *A, const double *b,
                 double *x, double *rnorm) {
  double *a = calloc((size_t)(m ? m : 1) * (n ? n : 1), sizeof *a);
  double *bb = calloc(m ? m : 1, sizeof *bb);
  double *w = calloc(n ? n : 1, sizeof *w), *zz = calloc(m ? m : 1, sizeof *zz);
  int *idx = calloc(n ? n : 1, sizeof *idx), mode = 0, mda = m ? m : 1;
  for (int i = 0; i < m; ++i) {
    bb[i] = b[i];
    for (int j = 0; j < n; ++j) a[(size_t)j * m + i] = A[(size_t)i * n + j];
  }
  nnls_c(a, &mda, &m, &n, bb, x, rnorm, w, zz, idx, &mode);
  free(a); free(bb); free(w); free(zz); free(idx);
  return mode;
}

static double resid(int m, int n, const double *A, const double *b, const double *x) {
  double s = 0;
  for (int i = 0; i < m; ++i) {
    double r = b[i];
    for (int j = 0; j < n; ++j) r -= A[(size_t)i * n + j] * x[j];
    s += r * r;
  }
  return sqrt(s);
}

/* Unconstrained least squares on the columns in `on` (a bitmask), by the
 * normal equations with partial pivoting. Returns 0 on a singular Gram
 * matrix, else fills x (zeros off the support). Fine for tiny problems. */
static int ls_support(int m, int n, const double *A, const double *b,
                      unsigned on, double *x) {
  int cols[8], k = 0;
  for (int j = 0; j < n; ++j) if (on & (1u << j)) cols[k++] = j;
  double G[8][9];
  for (int p = 0; p < k; ++p) {
    for (int q = 0; q <= k; ++q) {
      double s = 0;
      for (int i = 0; i < m; ++i)
        s += A[(size_t)i * n + cols[p]] * (q < k ? A[(size_t)i * n + cols[q]] : b[i]);
      G[p][q] = s;
    }
  }
  for (int p = 0; p < k; ++p) {
    int piv = p;
    for (int r = p + 1; r < k; ++r) if (fabs(G[r][p]) > fabs(G[piv][p])) piv = r;
    if (fabs(G[piv][p]) < 1e-10) return 0;
    if (piv != p) for (int q = 0; q <= k; ++q) { double t = G[p][q]; G[p][q] = G[piv][q]; G[piv][q] = t; }
    for (int r = 0; r < k; ++r) {
      if (r == p) continue;
      double f = G[r][p] / G[p][p];
      for (int q = p; q <= k; ++q) G[r][q] -= f * G[p][q];
    }
  }
  for (int j = 0; j < n; ++j) x[j] = 0;
  for (int p = 0; p < k; ++p) x[cols[p]] = G[p][k] / G[p][p];
  return 1;
}

/* The NNLS optimum by enumeration: the feasible support solution with the
 * smallest residual (the empty support, x = 0, is always feasible). */
static double oracle(int m, int n, const double *A, const double *b) {
  double best = resid(m, n, A, b, (double[8]){0}), x[8];
  for (unsigned on = 1; on < (1u << n); ++on) {
    if (!ls_support(m, n, A, b, on, x)) continue;
    int ok = 1;
    for (int j = 0; j < n; ++j) if (x[j] < -1e-12) ok = 0;
    if (!ok) continue;
    double r = resid(m, n, A, b, x);
    if (r < best) best = r;
  }
  return best;
}

/* KKT: x >= 0; w = A^T (b - A x) <= 0; w_j = 0 where x_j > 0. */
static int kkt(int m, int n, const double *A, const double *b, const double *x, double tol) {
  for (int j = 0; j < n; ++j) {
    if (x[j] < 0) return 0;
    double w = 0;
    for (int i = 0; i < m; ++i) {
      double r = b[i];
      for (int q = 0; q < n; ++q) r -= A[(size_t)i * n + q] * x[q];
      w += A[(size_t)i * n + j] * r;
    }
    if (w > tol) return 0;
    if (x[j] > tol && fabs(w) > tol) return 0;
  }
  return 1;
}

static unsigned long long rng = 20260924ULL;   /* fixed seed: the sweep is a test, not a search */
static double urand(void) {                     /* uniform in [-1, 1) */
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)(rng >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}

int main(void) {
  double x[8], rn;

  /* ---- 1. hand cases ---------------------------------------------------- */
  { double A[] = {1, 0,  0, 1,  1, 1}, b[] = {1, 2, 3};      /* exact fit */
    int mode = solve(3, 2, A, b, x, &rn);
    CHECK(mode == 1, "exact fit: mode %d", mode);
    CHECK(fabs(x[0] - 1) < 1e-12 && fabs(x[1] - 2) < 1e-12, "exact fit: x = (%g, %g)", x[0], x[1]);
    CHECK(rn < 1e-12, "exact fit: rnorm %g", rn); }
  { double A[] = {1, 0,  0, 1}, b[] = {1, -1};               /* one bound active */
    int mode = solve(2, 2, A, b, x, &rn);
    CHECK(mode == 1 && fabs(x[0] - 1) < 1e-12 && x[1] == 0, "bound: x = (%g, %g)", x[0], x[1]);
    CHECK(fabs(rn - 1) < 1e-12, "bound: rnorm %g", rn); }
  { double A[] = {1, 1, 1}, b[] = {3};                       /* wide: n > m */
    int mode = solve(1, 3, A, b, x, &rn);
    CHECK(mode == 1 && rn < 1e-12, "wide: mode %d rnorm %g", mode, rn);
    CHECK(fabs(x[0] + x[1] + x[2] - 3) < 1e-12, "wide: sum %g", x[0] + x[1] + x[2]);
    CHECK(kkt(1, 3, A, b, x, 1e-9), "wide: KKT"); }
  { double A[] = {1, 0}, b[] = {0};                          /* zero right-hand side */
    int mode = solve(1, 2, A, b, x, &rn);
    CHECK(mode == 1 && x[0] == 0 && x[1] == 0 && rn == 0, "zero b: x = (%g, %g) rnorm %g", x[0], x[1], rn); }
  { double A[] = {1}, b[] = {1};                             /* empty problem */
    int mode = solve(0, 1, A, b, x, &rn);
    CHECK(mode == 2, "m = 0: mode %d, want 2", mode);
    mode = solve(1, 0, A, b, x, &rn);
    CHECK(mode == 2, "n = 0: mode %d, want 2", mode); }
  /* a backtrack, verified under gcov (2026-09-24): columns 1 and 2 are
   * near-collinear, the dual test admits one, and the joint solve after the
   * next column enters drives it negative, so it is moved back to set Z.
   * Found as sweep problem 0 below; kept on its own so the branch is covered
   * even if the sweep changes. Solution: x = (0, 0, 0.0219, 0, 0.4717). */
  { double A[] = {-0.64806977511529329, -0.6808121922230107, 0.33494438767515056,
                  -0.56586892128256094, 0.071664372090188833,
                  -0.66264564897901668, -0.63959009000083455, 0.64587351576099672,
                  0.00047999303382639802, -0.44327436206174409};
    double b[] = {0.041150037489303681, -0.19495559721614053};
    int mode = solve(2, 5, A, b, x, &rn);
    CHECK(mode == 1, "backtrack: mode %d", mode);
    CHECK(kkt(2, 5, A, b, x, 1e-9), "backtrack: KKT fails, x = (%g, %g, %g, %g, %g)", x[0], x[1], x[2], x[3], x[4]);
    CHECK(x[0] == 0 && x[1] == 0 && x[3] == 0 && fabs(x[2] - 0.0219215) < 1e-6 && fabs(x[4] - 0.471749) < 1e-6,
          "backtrack: x = (%g, %g, %g, %g, %g)", x[0], x[1], x[2], x[3], x[4]); }

  /* ---- 2 + 3. seeded sweep: oracle where enumerable, KKT everywhere ------ */
  int n_prob = 0, n_oracle = 0;
  for (int t = 0; t < 400; ++t) {
    int m = 1 + (int)((urand() + 1) * 3), n = 1 + (int)((urand() + 1) * 3);   /* 1..6 */
    if (m > 6) m = 6;
    if (n > 6) n = 6;
    double A[36], b[6];
    for (int i = 0; i < m * n; ++i) A[i] = urand();
    /* every third problem: make two columns nearly collinear, which is what
     * drives a coefficient negative after its partner enters */
    if (n >= 2 && t % 3 == 0)
      for (int i = 0; i < m; ++i) A[i * n + 1] = A[i * n] + 0.05 * urand();
    for (int i = 0; i < m; ++i) b[i] = urand();
    int mode = solve(m, n, A, b, x, &rn);
    CHECK(mode == 1, "sweep %d (%dx%d): mode %d", t, m, n, mode);
    if (mode != 1) continue;
    ++n_prob;
    CHECK(fabs(rn - resid(m, n, A, b, x)) < 1e-9, "sweep %d: rnorm %g vs residual %g", t, rn, resid(m, n, A, b, x));
    CHECK(kkt(m, n, A, b, x, 1e-8), "sweep %d (%dx%d): KKT fails", t, m, n);
    if (n <= 4) {
      ++n_oracle;
      double o = oracle(m, n, A, b);
      CHECK(fabs(rn - o) < 1e-8, "sweep %d (%dx%d): rnorm %g vs oracle %g", t, m, n, rn, o);
    }
  }
  if (failures) { fprintf(stderr, "nnls_check: %d failure(s)\n", failures); return 1; }
  printf("nnls_check: hand cases, %d random problems KKT-checked, %d against the oracle\n", n_prob, n_oracle);
  return 0;
}
