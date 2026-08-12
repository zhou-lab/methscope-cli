// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Generic model bundle (see bundle.h). Provides the section reader used by
 * `upscale` to unpack a `.updecx`, plus the `bundle` / `unbundle` subcommands
 * that wrap/unwrap a model with its MRMP definition.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/stat.h>
#include "bundle.h"
#include "mrmp.h"
#include "methscope.h"    /* ms_annotate_booster (for bundle -l) */
#include "index.h"        /* get_fname_index -- the resolve temp has a sibling .idx */

#define NAMELEN 16
#define FOOTER_BYTES 8         /* trailing uint64 container-header offset */
#define CONTAINER_HDR_BYTES 12 /* MSBNDL1 magic (8) + section count (4) */

typedef struct { char name[NAMELEN]; uint64_t offset, length; } entry_t;

static void bdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] bundle: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] bundle: %s\n", msg);
  exit(1);
}

/* Read only the footer, container header, and section directory. */
static entry_t *bundle_directory(FILE *fp, const char *path, uint64_t *total_out,
                                 uint64_t *container_out, uint32_t *n_out) {
  if (fseeko(fp, 0, SEEK_END)) bdie("cannot seek", path);
  off_t end = ftello(fp);
  if (end < FOOTER_BYTES + CONTAINER_HDR_BYTES) /* 20 = smallest possible */
    bdie("not a methscope bundle (too short)", path);
  uint64_t total = (uint64_t)end, container_off;
  if (fseeko(fp, (off_t)(total - FOOTER_BYTES), SEEK_SET) ||
      fread(&container_off, sizeof(container_off), 1, fp) != 1)
    bdie("cannot read bundle footer", path);
  if (container_off > total - FOOTER_BYTES ||
      total - FOOTER_BYTES - container_off < CONTAINER_HDR_BYTES)
    bdie("not a methscope bundle (bad footer)", path);
  char magic[8];
  uint32_t n;
  if (fseeko(fp, (off_t)container_off, SEEK_SET) || fread(magic, 1, 8, fp) != 8 ||
      fread(&n, sizeof(n), 1, fp) != 1 ||
      memcmp(magic, MS_BUNDLE_MAGIC, 8))
    bdie("not a methscope bundle (no MSBNDL1 footer)", path);
  uint64_t table_bytes = (uint64_t)n * sizeof(entry_t);
  if (table_bytes > total - FOOTER_BYTES - container_off - CONTAINER_HDR_BYTES ||
      table_bytes > SIZE_MAX)
    bdie("truncated bundle section table", path);
  entry_t *entries = malloc(n ? (size_t)table_bytes : 1);
  if (!entries) bdie("out of memory reading bundle directory", path);
  if (n && fread(entries, sizeof(*entries), n, fp) != n)
    bdie("truncated bundle section table", path);
  for (uint32_t i = 0; i < n; ++i) {
    entries[i].name[NAMELEN - 1] = '\0';
    if (entries[i].offset > total ||
        entries[i].length > total - entries[i].offset)
      bdie("section out of bounds", entries[i].name);
  }
  *total_out = total; *container_out = container_off; *n_out = n;
  return entries;
}

int ms_bundle_is(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return 0;
  int ok = 0;
  if (fseek(fp, 0, SEEK_END) == 0) {
    long total = ftell(fp);
    uint64_t container_off;
    if (total >= FOOTER_BYTES + CONTAINER_HDR_BYTES &&
        fseek(fp, total - FOOTER_BYTES, SEEK_SET) == 0 &&
        fread(&container_off, 8, 1, fp) == 1 &&
        container_off <= (uint64_t)total - FOOTER_BYTES &&
        (uint64_t)total - FOOTER_BYTES - container_off >= CONTAINER_HDR_BYTES) {
      char m[8];
      if (fseek(fp, (long)container_off, SEEK_SET) == 0 && fread(m, 1, 8, fp) == 8)
        ok = (memcmp(m, MS_BUNDLE_MAGIC, 8) == 0);
    }
  }
  fclose(fp);
  return ok;
}

void *ms_bundle_section_opt(const char *path, const char *name, size_t *len_out) {
  FILE *fp = fopen(path, "rb");
  if (!fp) bdie("cannot open", path);
  uint64_t total, container_off; uint32_t n;
  entry_t *entries = bundle_directory(fp, path, &total, &container_off, &n);
  (void)total; (void)container_off;
  for (uint32_t i = 0; i < n; ++i) {
    entry_t e = entries[i];
    if (strcmp(e.name, name) == 0) {
      if (e.length > SIZE_MAX) bdie("section is too large to materialize", name);
      void *buf = malloc(e.length ? e.length : 1);
      if (!buf) bdie("out of memory", name);
      if (fseeko(fp, (off_t)e.offset, SEEK_SET) ||
          (e.length && fread(buf, 1, (size_t)e.length, fp) != (size_t)e.length))
        bdie("short read", name);
      *len_out = (size_t)e.length;
      free(entries); fclose(fp);
      return buf;
    }
  }
  free(entries); fclose(fp);
  return NULL;                 /* absent: caller decides if that's fatal */
}

void *ms_bundle_section(const char *path, const char *name, size_t *len_out) {
  void *buf = ms_bundle_section_opt(path, name, len_out);
  if (!buf) bdie("section not found in bundle", name);
  return buf;
}

ms_bundle_entry_t *ms_bundle_list(const char *path, int *n_out) {
  FILE *fp = fopen(path, "rb");
  if (!fp) bdie("cannot open", path);
  uint64_t total, container_off; uint32_t n;
  entry_t *entries = bundle_directory(fp, path, &total, &container_off, &n);
  (void)total; (void)container_off;
  ms_bundle_entry_t *arr = malloc((n ? n : 1) * sizeof(*arr));
  if (!arr) bdie("out of memory", path);
  for (uint32_t i = 0; i < n; ++i) {
    entry_t e = entries[i];
    memset(arr[i].name, 0, sizeof(arr[i].name));
    memcpy(arr[i].name, e.name, NAMELEN);
    arr[i].name[NAMELEN - 1] = '\0';
    arr[i].offset = e.offset;
    arr[i].length = e.length;
  }
  free(entries); fclose(fp);
  *n_out = (int)n;
  return arr;
}

int ms_bundle_find(const char *path, const char *name, ms_bundle_entry_t *out) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return 0;
  uint64_t total, container_off; uint32_t n;
  entry_t *entries = bundle_directory(fp, path, &total, &container_off, &n);
  (void)total; (void)container_off;
  int found = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (!strcmp(entries[i].name, name)) {
      memset(out, 0, sizeof(*out));
      memcpy(out->name, entries[i].name, sizeof(out->name));
      out->offset = entries[i].offset;
      out->length = entries[i].length;
      found = 1;
      break;
    }
  }
  free(entries); fclose(fp);
  return found;
}

const char *ms_mrmp_resolve(const char *path, char **tmp_out) {
  *tmp_out = NULL;
  /* A loose .cm is already what open_cfile wants, and a .cm-bearing bundle is
     too: its .cm is the file prefix and yame stops at that .cm's BGZF EOF,
     ignoring the MSBNDL1 trailer. Either way the path goes straight through.

     A .mrmp does not: it is the artifact, not the runtime mask. Callers that
     want EVERY set of a chain expand it themselves (classify-featurize does).
     This is the single-set path -- deconv, build-reference -- so it materializes
     the FIRST block as a temp .cm and hands that back, which is what the
     (path, tmp_out) contract and ms_mrmp_cleanup() were always for. */
  FILE *f = fopen(path, "rb");
  if (!f) return path;
  char magic[8];
  int is_mrmp = fread(magic, 1, 8, f) == 8 && !memcmp(magic, MRMPIDX_MAGIC, 8);
  fclose(f);
  if (!is_mrmp) return path;

  char tpl[] = "/tmp/methscope_resolve_XXXXXX.cm";
  int fd = mkstemps(tpl, 3);
  if (fd < 0) bdie("cannot create a temporary mask", tpl);
  close(fd);
  ms_mrmp_write_mask(path, tpl, "Pna", UINT32_MAX);
  *tmp_out = strdup(tpl);
  return *tmp_out;
}

void ms_mrmp_cleanup(char *tmp) {
  if (!tmp) return;
  unlink(tmp);
  /* A chained artifact resolves to YAME's multi-record form, which drops a
   * sibling .idx next to the .cm. Removing only the .cm leaks one temp index
   * per resolve. */
  char *ipath = get_fname_index(tmp);
  if (ipath) { unlink(ipath); free(ipath); }
  free(tmp);
}

static uint64_t path_bytes(const char *path) {
  struct stat st;
  if (stat(path, &st) || st.st_size < 0) bdie("cannot stat", path);
  return (uint64_t)st.st_size;
}

static void stream_path(FILE *out, const char *path, uint64_t expected,
                        const char *out_path) {
  FILE *in = fopen(path, "rb");
  if (!in) bdie("cannot open", path);
  const size_t cap = 16u * 1024u * 1024u;
  unsigned char *buf = malloc(cap);
  if (!buf) bdie("out of memory streaming bundle", path);
  uint64_t done = 0;
  while (done < expected) {
    size_t want = (size_t)((expected - done) > cap ? cap : expected - done);
    size_t got = fread(buf, 1, want, in);
    if (got != want || fwrite(buf, 1, got, out) != got)
      bdie("streaming bundle failed", out_path);
    done += got;
  }
  if (fgetc(in) != EOF) bdie("input changed while bundling", path);
  free(buf); fclose(in);
}

/* ------------------------------------------------------------------ */
/* bundle                                                             */
/* ------------------------------------------------------------------ */
static int bundle_usage(void) {
  ms_help(stderr,
    "\n"
    "Usage:\n"
    "  methscope bundle -m <ref.mrmp> -o <out> <model>\n"
    "\n"
    "Purpose:\n"
    "  Bundle a model with the MRMP definition it needs into one self-contained\n"
    "  file, so a query .cg can be featurized + run without a separate .mrmp.\n"
    "  Works for any model; by convention the output extension names the ROLE:\n"
    "    a classifier          -> model.clfx    (classify)\n"
    "    an upscale decoder    -> model.updecx  (upscale)\n"
    "    a deconv reference    -> panel.refx    (deconv)\n"
    "  All three are the same MSBNDL1 container and are detected by magic, not by\n"
    "  name; '.ubjx' is the former classifier extension and is still accepted.\n"
    "\n"
    "Options:\n"
    "  -m <ref.mrmp>   the MRMP definition (a YAME .cm) to bundle (required).\n"
    "  -k <kind>       framework mark to record (xgboost/violation/threshold/logistic);\n"
    "                  required for a classifier .clfx that `classify` will run (it\n"
    "                  rejects an unmarked bundle). Also re-stamps an existing model.\n"
    "  -l <meta.tsv>   embed class labels (a 'labels<TAB>l1,l2,...' line) into the\n"
    "                  booster before bundling — for a raw (e.g. R-exported) .ubj.\n"
    "  -O <outcpg.cm>  output-CpG locations (upscale only): a YAME mask marking the\n"
    "                  CpGs the model imputes. With it, `upscale` writes a full-genome\n"
    "                  .cg; without it, a dense block .cg.\n"
    "  -o <out>        output bundle path (required).\n"
    "  --tree          bundle a routing TREE: -m is the mrmp CHAIN and the\n"
    "                  positional arguments are the per-node .clfx files, in\n"
    "                  any order. Each is matched to a chain set BY NAME, and\n"
    "                  only its booster is taken -- node->classes comes from\n"
    "                  the chain, so nothing is duplicated. Scored with\n"
    "                  `classify query.cg tree.clfx`.\n"
    "  -h              Show this help message.\n"
    "\n");
  return 1;
}

void ms_bundle_pack(const char *out, const char *kind, const char *model_path,
                    const char *mrmp_path, const char *outcpg_path) {
  uint64_t rlen = path_bytes(mrmp_path), mlen = path_bytes(model_path);
  uint64_t olen = outcpg_path ? path_bytes(outcpg_path) : 0;
  int nsec = 2 + !!kind + !!outcpg_path; /* mrmp + optional kind/outcpg + model */
  uint64_t container_off = rlen;
  uint64_t hdr = CONTAINER_HDR_BYTES + (uint64_t)nsec * sizeof(entry_t);
  FILE *fp = !strcmp(out, "-") ? stdout : fopen(out, "wb");
  if (!fp) bdie("cannot open output", out);
  stream_path(fp, mrmp_path, rlen, out);
  if (fwrite(MS_BUNDLE_MAGIC, 1, 8, fp) != 8) bdie("write error", out);
  uint32_t nn = (uint32_t)nsec;
  if (fwrite(&nn, sizeof(nn), 1, fp) != 1) bdie("write error", out);
  uint64_t off = container_off + hdr;
  entry_t e;
  memset(&e, 0, sizeof(e)); strncpy(e.name, "mrmp", NAMELEN - 1);
  e.offset = 0; e.length = rlen;
  if (fwrite(&e, sizeof(e), 1, fp) != 1) bdie("write error", out);
  if (kind) {
    memset(&e, 0, sizeof(e)); strncpy(e.name, "kind", NAMELEN - 1);
    e.offset = off; e.length = strlen(kind); off += e.length;
    if (fwrite(&e, sizeof(e), 1, fp) != 1) bdie("write error", out);
  }
  if (outcpg_path) {
    memset(&e, 0, sizeof(e)); strncpy(e.name, "outcpg", NAMELEN - 1);
    e.offset = off; e.length = olen; off += olen;
    if (fwrite(&e, sizeof(e), 1, fp) != 1) bdie("write error", out);
  }
  memset(&e, 0, sizeof(e)); strncpy(e.name, "model", NAMELEN - 1);
  e.offset = off; e.length = mlen;
  if (fwrite(&e, sizeof(e), 1, fp) != 1) bdie("write error", out);
  if (kind && fwrite(kind, 1, strlen(kind), fp) != strlen(kind))
    bdie("write error", out);
  if (outcpg_path) stream_path(fp, outcpg_path, olen, out);
  stream_path(fp, model_path, mlen, out);
  if (fwrite(&container_off, sizeof(container_off), 1, fp) != 1) bdie("write error", out);
  if (fp != stdout && fclose(fp)) bdie("write error", out);
}

/* A TREE bundle: the chain as the file prefix, then one booster section per
 * node, named for the node.
 *
 * The chain is the prefix rather than any one node's MRMP because that is what
 * a shared featurize wants, and because blocks stay individually addressable
 * inside it -- ms_mrmp_top_read_at() and the featurizer already take
 * (path, base, len) -- so scoring one path lazily is still possible. Storing
 * the seven per-node .clfx files whole would have duplicated the chain and
 * needed a bundle reader that works on memory rather than a path; the boosters
 * alone lose nothing, since node->classes comes from the chain's own blocks.
 *
 * Section names are the node names, which is what makes the sections
 * addressable without a table of contents. NAMELEN caps that at 15 characters
 * -- "root.5.13" fits, a tree deeper than about six levels would not, so it is
 * checked rather than truncated into a name collision. */
void ms_bundle_pack_tree(const char *out, const char *chain_path,
                         uint32_t n_nodes, char *const *node_name,
                         void *const *booster, const uint64_t *booster_len) {
  const char *kind = "tree";
  uint64_t rlen = path_bytes(chain_path);
  int nsec = 2 + (int)n_nodes;                 /* mrmp + kind + one per node */
  uint64_t container_off = rlen;
  uint64_t hdr = CONTAINER_HDR_BYTES + (uint64_t)nsec * sizeof(entry_t);
  for (uint32_t k = 0; k < n_nodes; ++k) {
    if (strlen(node_name[k]) >= NAMELEN)
      bdie("node name too long for a bundle section (max 15 chars)", node_name[k]);
    for (uint32_t j = 0; j < k; ++j)
      if (!strcmp(node_name[k], node_name[j]))
        bdie("duplicate node name", node_name[k]);
  }
  FILE *fp = fopen(out, "wb");
  if (!fp) bdie("cannot open output", out);
  stream_path(fp, chain_path, rlen, out);
  if (fwrite(MS_BUNDLE_MAGIC, 1, 8, fp) != 8) bdie("write error", out);
  uint32_t nn = (uint32_t)nsec;
  if (fwrite(&nn, sizeof(nn), 1, fp) != 1) bdie("write error", out);
  uint64_t off = container_off + hdr;
  entry_t e;
  memset(&e, 0, sizeof(e)); strncpy(e.name, "mrmp", NAMELEN - 1);
  e.offset = 0; e.length = rlen;
  if (fwrite(&e, sizeof(e), 1, fp) != 1) bdie("write error", out);
  memset(&e, 0, sizeof(e)); strncpy(e.name, "kind", NAMELEN - 1);
  e.offset = off; e.length = strlen(kind); off += e.length;
  if (fwrite(&e, sizeof(e), 1, fp) != 1) bdie("write error", out);
  for (uint32_t k = 0; k < n_nodes; ++k) {
    memset(&e, 0, sizeof(e)); strncpy(e.name, node_name[k], NAMELEN - 1);
    e.offset = off; e.length = booster_len[k]; off += booster_len[k];
    if (fwrite(&e, sizeof(e), 1, fp) != 1) bdie("write error", out);
  }
  if (fwrite(kind, 1, strlen(kind), fp) != strlen(kind)) bdie("write error", out);
  for (uint32_t k = 0; k < n_nodes; ++k)
    if (booster_len[k] &&
        fwrite(booster[k], 1, (size_t)booster_len[k], fp) != booster_len[k])
      bdie("write error", out);
  if (fwrite(&container_off, sizeof(container_off), 1, fp) != 1) bdie("write error", out);
  if (fclose(fp)) bdie("write error", out);
}

char *ms_bundle_kind(const char *path) {
  size_t len; void *buf = ms_bundle_section_opt(path, "kind", &len);
  if (!buf) return NULL;
  char *s = malloc(len + 1);
  if (!s) bdie("out of memory", "kind");
  memcpy(s, buf, len); s[len] = '\0';
  free(buf);
  return s;
}

/* Every bundle is the same MSBNDL1 container and is DETECTED by magic
 * (ms_bundle_is), so this list is only about what a writer will accept as an
 * output name. The extensions name the model's role: .clfx classifier, .updecx
 * upscale decoder, .refx deconvolution reference. ".ubjx" was the classifier's
 * former name -- it described the payload format (a UBJSON booster) rather than
 * the role, and stopped being true the moment the threshold/logistic
 * frameworks shipped a plain-text model section under it. Kept accepted
 * indefinitely: detection never depended on it, so nothing is gained by
 * breaking existing files or scripts. ".refx" was missing here entirely, which
 * is the drift that having three names for one container invites. */
int ms_path_is_bundle_ext(const char *path) {
  size_t n = strlen(path);
  return (n >= 5 && strcmp(path + n - 5, ".clfx")   == 0) ||
         (n >= 7 && strcmp(path + n - 7, ".updecx") == 0) ||
         (n >= 5 && strcmp(path + n - 5, ".refx")   == 0) ||
         (n >= 5 && strcmp(path + n - 5, ".ubjx")   == 0);  /* legacy */
}

int main_bundle(int argc, char *argv[]) {
  const char *mrmp = NULL, *out = NULL, *outcpg = NULL, *kind = NULL, *meta = NULL;
  int tree = 0, n_inner = 0;
  char **inner_list = NULL;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-m") == 0 && i+1 < argc) mrmp   = argv[++i];
    else if (strcmp(argv[i], "-k") == 0 && i+1 < argc) kind   = argv[++i];
    else if (strcmp(argv[i], "-l") == 0 && i+1 < argc) meta   = argv[++i];
    else if (strcmp(argv[i], "-O") == 0 && i+1 < argc) outcpg = argv[++i];
    else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) out    = argv[++i];
    else if (strcmp(argv[i], "--tree") == 0) tree = 1;
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      bundle_usage(); return 0;
    }
    else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0)
      bdie("unrecognized or incomplete option", argv[i]);
    else break;
  }
  if (!mrmp || !out) return bundle_usage();
  if (tree) { inner_list = argv + i; n_inner = argc - i; }
  if (!tree && argc - i != 1) return bundle_usage();
  const char *model = tree ? NULL : argv[i];

  /* -l: embed labels into the (raw) booster first, then bundle the annotated copy. */
  const char *inner = model;
  if (tree && (meta || outcpg || kind))
    bdie("--tree takes only -m and -o; the kind mark is 'tree' and per-node "
         "labels already travel inside each booster", out);
  char *tmp_ubj = NULL;
  if (meta) {
    char tmpl[] = "/tmp/methscope_ann_XXXXXX.ubj";
    int fd = mkstemps(tmpl, 4);          /* keep the .ubj suffix for XGBoost */
    if (fd < 0) bdie("cannot create temp booster file", NULL);
    close(fd);
    ms_annotate_booster(model, meta, tmpl);
    tmp_ubj = strdup(tmpl); inner = tmp_ubj;
  }

  if (tree) {
    /* Match each per-node .clfx to a chain block BY NAME rather than by
     * position: a mismatched order would otherwise pair a node's booster with
     * another node's columns and score confidently on the wrong features. */
    ms_mrmpset_t *ch = ms_mrmpset_open(mrmp);
    uint32_t n = ch->n_sets;
    char **nm = calloc(n, sizeof(char *));
    void **bl = calloc(n, sizeof(void *));
    uint64_t *bn = calloc(n, sizeof(uint64_t));
    if (!nm || !bl || !bn) bdie("out of memory", "tree bundle");
    for (uint32_t s2 = 0; s2 < n; ++s2) {
      nm[s2] = strdup(ch->name[s2]);
      for (int a = 0; a < n_inner; ++a) {
        ms_mrmpset_t *one = ms_mrmpset_open(inner_list[a]);
        int hit = one->n_sets == 1 && !strcmp(one->name[0], nm[s2]);
        ms_mrmpset_free(one);
        if (!hit) continue;
        size_t bl2;
        bl[s2] = ms_bundle_section(inner_list[a], "model", &bl2);
        bn[s2] = bl2;
        break;
      }
      if (!bl[s2]) bdie("no per-node model supplied for chain set", nm[s2]);
    }
    ms_bundle_pack_tree(out, mrmp, n, nm, bl, bn);
    fprintf(stderr, "[methscope] bundle: tree of %u node(s) -> %s\n", n, out);
    for (uint32_t s2 = 0; s2 < n; ++s2) { free(nm[s2]); free(bl[s2]); }
    free(nm); free(bl); free(bn);
    ms_mrmpset_free(ch);
    return 0;
  }
  ms_bundle_pack(out, kind, inner, mrmp, outcpg);   /* kind mark (NULL = omit) */
  if (tmp_ubj) { unlink(tmp_ubj); free(tmp_ubj); }
  if (outcpg)
    fprintf(stderr, "[methscope] bundled %s + %s + %s -> %s\n", model, mrmp, outcpg, out);
  else
    fprintf(stderr, "[methscope] bundled %s + %s -> %s\n", model, mrmp, out);
  return 0;
}

/* ------------------------------------------------------------------ */
/* unbundle                                                           */
/* ------------------------------------------------------------------ */
static int unbundle_usage(void) {
  ms_help(stderr,
    "\n"
    "Usage:\n"
    "  methscope unbundle [-o <model_out>] [--mrmp <mrmp_out>] <bundle>\n"
    "\n"
    "Purpose:\n"
    "  Unpack a bundle (.clfx / .updecx / .refx) into its inner model, its MRMP,\n"
    "  and (if present) the output-CpG mask.\n"
    "\n"
    "  With no -o/--mrmp, output names are derived from the bundle path (the\n"
    "  original .mrmp filename is not stored): drop the 'x' from the extension for\n"
    "  the model, and add sibling suffixes for the rest --\n"
    "    foo.clfx   -> foo.model + foo.mrmp  (+ foo.outcpg.cm if present)\n"
    "    foo.updecx -> foo.updec + foo.mrmp\n"
    "    foo.refx   -> foo.ref   + foo.mrmp\n"
    "\n"
    "Options:\n"
    "  -o <model_out>    write the inner model here (default: derived, see above).\n"
    "  --mrmp <mrmp_out> write the bundled MRMP (.cm) here (default: derived).\n"
    "  -h                Show this help message.\n"
    "\n");
  return 1;
}

static void write_out(const char *path, void *buf, size_t len) {
  FILE *fp = fopen(path, "wb");
  if (!fp) bdie("cannot open output", path);
  if (len && fwrite(buf, 1, len, fp) != len) bdie("write error", path);
  fclose(fp);
}

/* Default inner-model name: the bundle path with a trailing 'x' dropped
   (foo.ubjx -> foo.ubj); if it doesn't end in 'x', append ".model". */
static char *derive_model_out(const char *b) {
  size_t n = strlen(b);
  if (n && b[n-1] == 'x') { char *s = malloc(n); memcpy(s, b, n - 1); s[n - 1] = '\0'; return s; }
  char *s = malloc(n + 7); snprintf(s, n + 7, "%s.model", b); return s;
}

/* Default sibling name: the bundle path with its extension replaced by <suffix>
   (foo.ubjx, ".mrmp" -> foo.mrmp). */
static char *derive_sibling(const char *b, const char *suffix) {
  const char *dot = strrchr(b, '.'), *sl = strrchr(b, '/');
  size_t stem = (dot && (!sl || dot > sl)) ? (size_t)(dot - b) : strlen(b);
  size_t sn = strlen(suffix);
  char *s = malloc(stem + sn + 1);
  memcpy(s, b, stem); memcpy(s + stem, suffix, sn + 1);
  return s;
}

int main_unbundle(int argc, char *argv[]) {
  const char *model_out = NULL, *mrmp_out = NULL;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-o") == 0 && i+1 < argc) model_out = argv[++i];
    else if (strcmp(argv[i], "--mrmp") == 0 && i+1 < argc) mrmp_out = argv[++i];
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      unbundle_usage(); return 0;
    }
    else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0)
      bdie("unrecognized or incomplete option", argv[i]);
    else break;
  }
  if (argc - i != 1) return unbundle_usage();
  const char *bundle = argv[i];

  char *model_def = model_out ? NULL : derive_model_out(bundle);
  char *mrmp_def  = mrmp_out  ? NULL : derive_sibling(bundle, ".mrmp");
  const char *mo = model_out ? model_out : model_def;
  const char *ro = mrmp_out  ? mrmp_out  : mrmp_def;

  { size_t len; void *buf = ms_bundle_section(bundle, "model", &len);
    write_out(mo, buf, len); free(buf);
    fprintf(stderr, "[methscope] wrote inner model -> %s\n", mo); }
  { size_t len; void *buf = ms_bundle_section(bundle, "mrmp", &len);
    write_out(ro, buf, len); free(buf);
    fprintf(stderr, "[methscope] wrote bundled mrmp -> %s\n", ro); }
  { size_t len; void *buf = ms_bundle_section_opt(bundle, "outcpg", &len);
    if (buf) {
      char *oc = derive_sibling(bundle, ".outcpg.cm");
      write_out(oc, buf, len); free(buf);
      fprintf(stderr, "[methscope] wrote outcpg mask -> %s\n", oc);
      free(oc);
    } }
  free(model_def); free(mrmp_def);
  return 0;
}
