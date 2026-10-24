/* SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
 * Use of this software is available to academic and non-profit institutions
 * for research purposes under the 2-Clause BSD License; for use or transfers
 * to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
 * See the LICENSE file at the root of the repository for the full terms.
 *
 * rf_check -- exercises the random forest (src/rforest.c) on its own.
 *
 * With no argument it runs the built-in checks, for `make test`: a seeded
 * problem where 3 of 40 features carry the class, which must top both
 * importances and give a low OOB error; the same fit at 1 and 4 threads,
 * which must agree exactly; and a fit on separable data, whose training
 * rows the forest must call right.
 *
 * With a matrix it fits and prints, for comparison against R randomForest:
 *   rf_check <in.tsv> <ntree> <seed> <threads> [test.tsv]
 * in.tsv: a header line, then one row per sample: an integer class label
 * (0-based), then the feature values, tab-separated. Prints "oob_err\t<e>",
 * "test_acc\t<a>" when a test set is given (same layout; the call is the
 * class with the most votes), then one "<feature>\t<mda>\t<gini>" line per
 * feature.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rforest.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; \
  fprintf(stderr, "rf_check: "); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static uint64_t lcg = 88172645463325252ULL;
static double unif(void) {
  lcg ^= lcg << 13; lcg ^= lcg >> 7; lcg ^= lcg << 17;
  return (double)(lcg >> 11) * (1.0 / 9007199254740992.0);
}

static void builtin(void) {
  enum { N = 120, P = 40 };
  static float X[N * P];
  static int y[N];
  for (int i = 0; i < N; ++i) {
    y[i] = i % 2;
    for (int j = 0; j < P; ++j) X[i * P + j] = (float)unif();
    for (int j = 0; j < 3; ++j)                /* features 0-2 carry y */
      X[i * P + j] = (float)(0.6 * y[i] + 0.7 * unif());
  }
  static double mda1[P], gini1[P], mda4[P], gini4[P];
  rf_param_t par = { 300, 0, 7, 1 };
  rf_forest_t *f1 = rf_fit(X, N, P, y, 2, &par, mda1, gini1);
  par.threads = 4;
  rf_forest_t *f4 = rf_fit(X, N, P, y, 2, &par, mda4, gini4);
  int same = f1->oob_err == f4->oob_err;
  for (int j = 0; j < P; ++j)
    same = same && mda1[j] == mda4[j] && gini1[j] == gini4[j];
  CHECK(same, "1 and 4 threads differ");
  CHECK(f1->oob_err < 0.25, "OOB error %.3f on an easy problem", f1->oob_err);
  for (int j = 3; j < P; ++j)
    for (int k = 0; k < 3; ++k) {
      CHECK(mda1[k] > mda1[j], "noise feature %d outranks %d by MDA", j, k);
      CHECK(gini1[k] > gini1[j], "noise feature %d outranks %d by Gini", j, k);
    }
  double *prob = malloc(N * 2 * sizeof(double));
  rf_predict(f1, X, N, prob);
  int right = 0;
  for (int i = 0; i < N; ++i) right += ((prob[i * 2 + 1] >= 0.5) == y[i]);
  CHECK(right == N, "training rows called right: %d of %d", right, N);
  free(prob);
  rf_free(f1); rf_free(f4);
}

typedef struct { float *X; int *y; char **names; uint32_t n, p; } tsv_t;

/* Header: a label column then feature names; rows: label, then values. */
static tsv_t read_tsv(const char *path) {
  FILE *fh = fopen(path, "r");
  if (!fh) { perror(path); exit(1); }
  tsv_t t = { 0 };
  size_t cap = 1 << 20, len = 0;
  char *line = NULL; size_t lcap = 0;
  if (getline(&line, &lcap, fh) < 0) { fprintf(stderr, "empty\n"); exit(1); }
  for (char *tok = strtok(line, "\t\n"), *first = tok; tok;
       tok = strtok(NULL, "\t\n")) {
    if (tok == first) continue;
    t.names = realloc(t.names, (t.p + 1) * sizeof(char *));
    t.names[t.p++] = strdup(tok);
  }
  t.X = malloc(cap * sizeof(float));
  while (getline(&line, &lcap, fh) > 0) {
    char *s = line, *e;
    t.y = realloc(t.y, (t.n + 1) * sizeof(int));
    t.y[t.n] = (int)strtol(s, &e, 10); s = e;
    for (uint32_t j = 0; j < t.p; ++j) {
      if (len == cap) t.X = realloc(t.X, (cap *= 2) * sizeof(float));
      t.X[len++] = strtof(s, &e); s = e;
    }
    ++t.n;
  }
  fclose(fh); free(line);
  return t;
}

static void from_file(const char *path, uint32_t ntree, uint64_t seed,
                      int threads, const char *test_path) {
  tsv_t t = read_tsv(path);
  int nclass = 0;
  for (uint32_t i = 0; i < t.n; ++i) if (t.y[i] + 1 > nclass) nclass = t.y[i] + 1;
  double *mda = malloc(t.p * sizeof(double)), *gini = malloc(t.p * sizeof(double));
  rf_param_t par = { ntree, 0, seed, threads };
  rf_forest_t *f = rf_fit(t.X, t.n, t.p, t.y, (uint32_t)nclass, &par, mda, gini);
  printf("oob_err\t%.6f\n", f->oob_err);
  { double nn = 0;
    for (uint32_t k = 0; k < f->ntree; ++k) nn += f->tree[k].n_node;
    fprintf(stderr, "mean_leaves\t%.2f\n", (nn / f->ntree + 1) / 2); }
  if (test_path) {
    tsv_t q = read_tsv(test_path);
    if (q.p != t.p) { fprintf(stderr, "test has %u features, train %u\n", q.p, t.p); exit(1); }
    double *prob = malloc((size_t)q.n * nclass * sizeof(double));
    rf_predict(f, q.X, q.n, prob);
    uint32_t right = 0;
    for (uint32_t i = 0; i < q.n; ++i) {
      int c = 0, nties = 1;                 /* ties at random, as R does */
      for (int k = 1; k < nclass; ++k) {
        double a = prob[(size_t)i * nclass + k], b = prob[(size_t)i * nclass + c];
        if (a > b) { c = k; nties = 1; }
        else if (a == b && unif() * ++nties < 1.0) c = k;
      }
      right += (c == q.y[i]);
    }
    printf("test_acc\t%.6f\n", (double)right / q.n);
    free(prob);
  }
  for (uint32_t j = 0; j < t.p; ++j)
    printf("%s\t%.6g\t%.6g\n", t.names[j], mda[j], gini[j]);
  rf_free(f);
}

int main(int argc, char **argv) {
  if (argc == 5 || argc == 6) {
    from_file(argv[1], (uint32_t)atoi(argv[2]),
              strtoull(argv[3], NULL, 10), atoi(argv[4]),
              argc == 6 ? argv[5] : NULL);
    return 0;
  }
  builtin();
  if (failures) { fprintf(stderr, "rf_check: %d failure(s)\n", failures); return 1; }
  fprintf(stderr, "rf_check: ok\n");
  return 0;
}
