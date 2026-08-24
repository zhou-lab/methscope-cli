// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Rename a CLASS LABEL inside a trained model, without retraining it.
 *
 * WHY THIS EXISTS
 *
 * A class name is a claim about what the reference contains, and it can be
 * wrong while every number in the model is right. The 33-label human model's
 * `Macrophage` was 20 of 26 donors of BLUEPRINT cord/venous-blood
 * `inflammatory_` and `alternatively_activated_macrophage` -- monocytes
 * differentiated and polarised IN VITRO. It represents no tissue-resident
 * macrophage: the shipped tree called 8 of 8 sorted Loyfer tissue macrophages
 * Monocyte, and 2,019 of 2,020 Zhou single cells something other than
 * Macrophage, while SELF-deconvolving its own reference row at 0.966. Nothing
 * was broken except the name.
 *
 * Retraining to fix a name would be absurd and would perturb every number, so
 * this edits the one place a label is stored: the XGBoost booster attribute
 * MS_ATTR_LABELS ("methscope_labels"), a comma-separated list in class-index
 * order. Predictions are bit-identical; only the emitted string changes.
 *
 * A TREE bundle keeps one booster per node and EVERY node carrying the class
 * has its own copy of the list, so the rename has to walk all of them -- doing
 * the root alone would leave the leaves disagreeing with it.
 *
 * The label is deliberately NOT rewritten anywhere else. The bundled MRMP
 * carries reference sample names from <ref>.cg.idx, which are a record of what
 * was built, not a claim about what it is; rewriting those would falsify the
 * provenance this command exists to correct.
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <xgboost/c_api.h>
#include "methscope.h"
#include "bundle.h"
#include "bmeta.h"

static void rdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] relabel: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] relabel: %s\n", msg);
  exit(1);
}

int main_relabel(int argc, char *argv[]) {
  const char *out_path = NULL, *from = NULL, *to = NULL;
  int force = 0, i = 1;
  for (; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-o") && i + 1 < argc) out_path = argv[++i];
    else if (!strcmp(a, "--force")) force = 1;
    else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      ms_help(stdout,
"Usage:\n"
"  methscope relabel [options] OLD=NEW <in.clfx> -o <out.clfx>\n"
"\n"
"Purpose:\n"
"  Rename a class label inside a trained model. The label lives in one XGBoost\n"
"  booster attribute (\"methscope_labels\", comma-separated in class-index\n"
"  order), so this rewrites that string and nothing else -- every prediction is\n"
"  bit-identical and only the emitted name changes.\n"
"\n"
"  A class name is a claim about what the reference CONTAINS, and it can be\n"
"  wrong while the model is right. Retraining to correct a name would perturb\n"
"  numbers that were never at fault.\n"
"\n"
"  On a TREE bundle every node carrying the class has its own copy of the label\n"
"  list, so all of them are walked; renaming the root alone would leave the\n"
"  leaves disagreeing with it.\n"
"\n"
"Arguments:\n"
"  OLD=NEW            The existing label and its replacement.\n"
"  <in.clfx>          The model to read. It is never modified.\n"
"\n"
"Options:\n"
"  -o <out.clfx>      Write the relabelled model here. Required.\n"
"  --force            Overwrite an existing output.\n"
"  -h                 Show this help message.\n"
"\n"
"Notes:\n"
"  The bundled MRMP's reference sample names are left ALONE on purpose: they\n"
"  record what was built, not what it is.\n"
"\n"
"Example:\n"
"  methscope relabel 'Macrophage=Macrophage.(Monocyte.Derived)' \\\n"
"      hg38_celltype.clfx -o hg38_celltype_v2.clfx\n");
      return 0;
    }
    else if (a[0] == '-' && a[1]) rdie("unrecognized option", a);
    else break;
  }
  if (argc - i != 2 || !out_path) {
    fprintf(stderr,
            "Usage: methscope relabel OLD=NEW IN.clfx -o OUT.clfx\n");
    return 1;
  }
  char *spec = strdup(argv[i]);
  const char *in_path = argv[i + 1];
  char *eq = strchr(spec, '=');
  if (!eq) rdie("expected OLD=NEW", argv[i]);
  *eq = '\0'; from = spec; to = eq + 1;
  if (!*from || !*to) rdie("OLD and NEW must both be non-empty", argv[i]);
  if (!ms_bundle_is(in_path)) rdie("not a model bundle", in_path);
  if (!force) {
    FILE *t = fopen(out_path, "rb");
    if (t) { fclose(t); rdie("output exists (use --force)", out_path); }
  }

  int n_sec = 0;
  ms_bundle_entry_t *sec = ms_bundle_list(in_path, &n_sec);
  if (!sec || n_sec < 1) rdie("bundle has no sections", in_path);

  /* Node sections are everything that is not the MRMP prefix or the kind
   * mark; on a flat bundle that is the single "model" section. */
  uint32_t n_node = 0;
  char **name = calloc(n_sec, sizeof(char *));
  void **blob = calloc(n_sec, sizeof(void *));
  uint64_t *blen = calloc(n_sec, sizeof(uint64_t));
  if (!name || !blob || !blen) rdie("out of memory", NULL);

  uint32_t n_hit = 0, n_walked = 0;
  for (int s = 0; s < n_sec; ++s) {
    if (!strcmp(sec[s].name, "mrmp") || !strcmp(sec[s].name, "kind")) continue;
    size_t len = 0;
    void *buf = ms_bundle_section(in_path, sec[s].name, &len);
    if (!buf) rdie("cannot read section", sec[s].name);

    BoosterHandle b = NULL;
    if (XGBoosterCreate(NULL, 0, &b) != 0)
      rdie("cannot create booster", sec[s].name);
    if (XGBoosterLoadModelFromBuffer(b, buf, (bst_ulong)len) != 0)
      rdie("section is not an XGBoost booster", sec[s].name);
    ++n_walked;

    int K = 0;
    char **lab = ms_booster_get_labels(b, &K);
    if (!lab || !K) rdie("booster carries no class labels", sec[s].name);
    int hit = 0;
    for (int c = 0; c < K; ++c)
      if (!strcmp(lab[c], from)) {
        free(lab[c]); lab[c] = strdup(to); hit = 1; ++n_hit;
      }
    if (hit) {
      ms_booster_set_meta(b, lab, K);
      const char *raw = NULL; bst_ulong rlen = 0;
      if (XGBoosterSaveModelToBuffer(b, "{\"format\":\"ubj\"}", &rlen, &raw) != 0)
        rdie("cannot serialize booster", sec[s].name);
      free(buf);
      buf = malloc(rlen);
      if (!buf) rdie("out of memory", NULL);
      memcpy(buf, raw, rlen);
      len = rlen;
      fprintf(stderr, "[methscope] relabel: %-14s %d classes, renamed\n",
              sec[s].name, K);
    } else {
      fprintf(stderr, "[methscope] relabel: %-14s %d classes, not present\n",
              sec[s].name, K);
    }
    for (int c = 0; c < K; ++c) free(lab[c]);
    free(lab);
    XGBoosterFree(b);

    name[n_node] = strdup(sec[s].name);
    blob[n_node] = buf;
    blen[n_node] = len;
    ++n_node;
  }
  if (!n_walked) rdie("no booster sections found", in_path);
  if (!n_hit) rdie("label not found in any node", from);

  /* The MRMP prefix is the file's leading bytes, so the input doubles as the
   * chain source -- pack_tree copies it across verbatim. */
  ms_bundle_pack_tree(out_path, in_path, n_node, name, blob, blen);
  fprintf(stderr,
    "[methscope] relabel: '%s' -> '%s' in %u of %u node(s) -> %s\n",
    from, to, n_hit, n_walked, out_path);

  for (uint32_t k = 0; k < n_node; ++k) { free(name[k]); free(blob[k]); }
  free(name); free(blob); free(blen); free(sec); free(spec);
  return 0;
}
