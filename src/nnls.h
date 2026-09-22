// SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
// Use of this software is available to academic and non-profit institutions
// for research purposes under the 2-Clause BSD License; for use or transfers
// to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
// See the LICENSE file at the root of the repository for the full terms.
/**
 * Lawson-Hanson NONNEGATIVE LEAST SQUARES (NNLS).
 *
 * Declares the f2c-translated reference implementation vendored verbatim in
 * src/nnls.c (Charles L. Lawson & Richard J. Hanson, JPL 1973; SIAM 1995
 * reprint; netlib.org/lawson-hanson). This is the same algorithm the R `nnls`
 * package wraps, so methscope's `deconv` matches the R nnls_deconv() numerics.
 *
 * Solves  min ||A x - b||_2  subject to  x >= 0.
 *
 * NOTE: `a` is COLUMN-MAJOR (Fortran order), dimensioned mda x n, and is
 * OVERWRITTEN (becomes Q*A). `b` is overwritten with Q*b. Caller supplies all
 * scratch (w: n, zz: m, index: n). mode: 1 = ok, 2 = bad dims, 3 = no converge.
 */
#ifndef METHSCOPE_NNLS_H
#define METHSCOPE_NNLS_H

int nnls_c(double *a, const int *mda, const int *m, const int *n,
           double *b, double *x, double *rnorm, double *w, double *zz,
           int *index, int *mode);

#endif /* METHSCOPE_NNLS_H */
