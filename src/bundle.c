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
#include "bundle.h"
#include "methscope.h"    /* ms_annotate_booster (for bundle -l) */

#define NAMELEN 16

typedef struct { char name[NAMELEN]; uint64_t offset, length; } entry_t;

static void bdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] %s\n", msg);
  exit(1);
}

static void *read_file(const char *path, size_t *len) {
  FILE *fp = fopen(path, "rb");
  if (!fp) bdie("cannot open", path);
  fseek(fp, 0, SEEK_END); long sz = ftell(fp); rewind(fp);
  if (sz < 0) bdie("cannot size", path);
  void *buf = malloc((size_t)sz ? (size_t)sz : 1);
  if (!buf) bdie("out of memory", path);
  if (sz > 0 && fread(buf, 1, (size_t)sz, fp) != (size_t)sz) bdie("short read", path);
  fclose(fp);
  *len = (size_t)sz;
  return buf;
}

/* Locate the MSBNDL1 container inside a file already read into raw[0..total).
   The last 8 bytes are a footer holding the container offset M (the magic sits at
   raw[M], right after the leading MRMP .cm). Returns M and *n_out, or -1 if the
   file has no valid footer/magic (i.e. it's a plain .cm, not a bundle). */
static long bundle_locate(const unsigned char *raw, size_t total, uint32_t *n_out) {
  if (total < 20) return -1;                       /* footer(8)+magic(8)+count(4) min */
  size_t footer = total - 8;
  uint64_t M; memcpy(&M, raw + footer, 8);
  if (M > footer || footer - M < 12) return -1;    /* room for magic(8)+count(4) */
  if (memcmp(raw + M, MS_BUNDLE_MAGIC, 8) != 0) return -1;
  uint32_t n; memcpy(&n, raw + M + 8, 4);
  if (M + 12 + (uint64_t)n * sizeof(entry_t) > footer) return -1;   /* table must fit */
  *n_out = n;
  return (long)M;
}

int ms_bundle_is(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return 0;
  int ok = 0;
  if (fseek(fp, 0, SEEK_END) == 0) {
    long total = ftell(fp);
    uint64_t M;
    if (total >= 20 && fseek(fp, total - 8, SEEK_SET) == 0 && fread(&M, 8, 1, fp) == 1
        && M <= (uint64_t)total - 8 && (uint64_t)total - 8 - M >= 12) {
      char m[8];
      if (fseek(fp, (long)M, SEEK_SET) == 0 && fread(m, 1, 8, fp) == 8)
        ok = (memcmp(m, MS_BUNDLE_MAGIC, 8) == 0);
    }
  }
  fclose(fp);
  return ok;
}

void *ms_bundle_section_opt(const char *path, const char *name, size_t *len_out) {
  size_t total; unsigned char *raw = read_file(path, &total);
  uint32_t n; long M = bundle_locate(raw, total, &n);
  if (M < 0) bdie("not a methscope bundle (no MSBNDL1 footer)", path);
  for (uint32_t i = 0; i < n; ++i) {
    entry_t e; memcpy(&e, raw + M + 12 + (size_t)i * sizeof(entry_t), sizeof(e));
    e.name[NAMELEN-1] = '\0';
    if (strcmp(e.name, name) == 0) {
      if (e.offset + e.length > total) bdie("section out of bounds", name);
      void *buf = malloc(e.length ? e.length : 1);
      if (!buf) bdie("out of memory", name);
      memcpy(buf, raw + e.offset, e.length);
      *len_out = e.length;
      free(raw);
      return buf;
    }
  }
  free(raw);
  return NULL;                 /* absent: caller decides if that's fatal */
}

void *ms_bundle_section(const char *path, const char *name, size_t *len_out) {
  void *buf = ms_bundle_section_opt(path, name, len_out);
  if (!buf) bdie("section not found in bundle", name);
  return buf;
}

ms_bundle_entry_t *ms_bundle_list(const char *path, int *n_out) {
  size_t total; unsigned char *raw = read_file(path, &total);
  uint32_t n; long M = bundle_locate(raw, total, &n);
  if (M < 0) bdie("not a methscope bundle (no MSBNDL1 footer)", path);
  ms_bundle_entry_t *arr = malloc((n ? n : 1) * sizeof(*arr));
  if (!arr) bdie("out of memory", path);
  for (uint32_t i = 0; i < n; ++i) {
    entry_t e; memcpy(&e, raw + M + 12 + (size_t)i * sizeof(entry_t), sizeof(e));
    memset(arr[i].name, 0, sizeof(arr[i].name));
    memcpy(arr[i].name, e.name, NAMELEN);
    arr[i].name[NAMELEN - 1] = '\0';
    arr[i].offset = e.offset;
    arr[i].length = e.length;
  }
  free(raw);
  *n_out = (int)n;
  return arr;
}

const char *ms_mrmp_resolve(const char *path, char **tmp_out) {
  *tmp_out = NULL;
  /* bundles begin with the raw MRMP .cm bytes (a valid YAME file whose BGZF EOF
     makes yame stop before the MSBNDL1 trailer), and a loose .cm is already what we
     want — so in both cases the path can be handed straight to open_cfile. */
  return path;
}

void ms_mrmp_cleanup(char *tmp) {
  if (tmp) { unlink(tmp); free(tmp); }
}

/* layout: [mrmp .cm bytes][MSBNDL1 magic][uint32 n][section table][blobs][footer].
   The mrmp is written first (offset 0) so `yame` can read it directly; it is also
   listed as section "mrmp" (offset 0). `names/blob/blen` are the NON-mrmp sections
   (kind, outcpg, model, ...). The 8-byte footer holds the container offset M. */
static void write_bundle(const char *out, void *mrmp, size_t mrmp_len,
                         const char *names[], void *blob[], size_t blen[], int n) {
  int nsec = n + 1;                                  /* + the mrmp prefix section */
  uint64_t M   = mrmp_len;                           /* container starts after mrmp */
  uint64_t hdr = 8 + 4 + (uint64_t)nsec * sizeof(entry_t);  /* magic+count+table */
  FILE *fp = (strcmp(out, "-") == 0) ? stdout : fopen(out, "wb");  /* "-" = stdout (pipe) */
  if (!fp) bdie("cannot open output", out);
  if (mrmp_len && fwrite(mrmp, 1, mrmp_len, fp) != mrmp_len) bdie("write error", out);
  if (fwrite(MS_BUNDLE_MAGIC, 1, 8, fp) != 8) bdie("write error", out);  /* 7+NUL */
  uint32_t nn = (uint32_t)nsec;
  if (fwrite(&nn, 4, 1, fp) != 1) bdie("write error", out);
  uint64_t off = M + hdr;                            /* first non-mrmp blob offset */
  { entry_t e; memset(&e, 0, sizeof(e)); strncpy(e.name, "mrmp", NAMELEN - 1);
    e.offset = 0; e.length = mrmp_len;
    if (fwrite(&e, sizeof(e), 1, fp) != 1) bdie("write error", out); }
  for (int i = 0; i < n; ++i) {
    entry_t e; memset(&e, 0, sizeof(e));
    strncpy(e.name, names[i], NAMELEN - 1);
    e.offset = off; e.length = blen[i];
    if (fwrite(&e, sizeof(e), 1, fp) != 1) bdie("write error", out);
    off += blen[i];
  }
  for (int i = 0; i < n; ++i)
    if (blen[i] && fwrite(blob[i], 1, blen[i], fp) != blen[i]) bdie("write error", out);
  if (fwrite(&M, 8, 1, fp) != 1) bdie("write error", out);   /* footer = container offset */
  if (fp != stdout) fclose(fp);
}

/* ------------------------------------------------------------------ */
/* bundle                                                             */
/* ------------------------------------------------------------------ */
static int bundle_usage(void) {
  fprintf(stderr,
    "\n"
    "Usage:\n"
    "  methscope bundle -m <ref.mrmp> -o <out> <model>\n"
    "\n"
    "Purpose:\n"
    "  Bundle a model with the MRMP definition it needs into one self-contained\n"
    "  file, so a query .cg can be featurized + run without a separate .mrmp.\n"
    "  Works for any model; by convention give the output an 'x' suffix:\n"
    "    booster.ubj  -> booster.ubjx   (predict)\n"
    "    model.updec  -> model.updecx   (upscale)\n"
    "\n"
    "Options:\n"
    "  -m <ref.mrmp>   the MRMP definition (a YAME .cm) to bundle (required).\n"
    "  -k <kind>       framework mark to record (xgboost/threshold/logistic); required\n"
    "                  for a classifier .ubjx that `predict` will run (predict rejects\n"
    "                  an unmarked bundle). Also used to re-stamp an existing model.\n"
    "  -l <meta.tsv>   embed class labels (a 'labels<TAB>l1,l2,...' line) into the\n"
    "                  booster before bundling — for a raw (e.g. R-exported) .ubj.\n"
    "  -O <outcpg.cm>  output-CpG locations (upscale only): a YAME mask marking the\n"
    "                  CpGs the model imputes. With it, `upscale` writes a full-genome\n"
    "                  .cg; without it, a dense block .cg.\n"
    "  -o <out>        output bundle path (required).\n"
    "  -h              Show this help message.\n"
    "\n");
  return 1;
}

void ms_bundle_pack(const char *out, const char *kind, const char *model_path,
                    const char *mrmp_path, const char *outcpg_path) {
  size_t mlen, rlen, olen = 0;
  void *mbuf = read_file(model_path, &mlen);
  void *rbuf = read_file(mrmp_path,  &rlen);       /* the MRMP .cm -> file prefix */
  void *obuf = outcpg_path ? read_file(outcpg_path, &olen) : NULL;
  const char *names[3]; void *blob[3]; size_t blen[3];
  int n = 0;                                        /* order: [kind] [outcpg] model */
  if (kind) { names[n] = "kind";   blob[n] = (void *)kind; blen[n] = strlen(kind); n++; }
  if (obuf) { names[n] = "outcpg"; blob[n] = obuf; blen[n] = olen; n++; }
  names[n] = "model"; blob[n] = mbuf; blen[n] = mlen; n++;
  write_bundle(out, rbuf, rlen, names, blob, blen, n);
  free(mbuf); free(rbuf); free(obuf);
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

int ms_path_is_bundle_ext(const char *path) {
  size_t n = strlen(path);
  return (n >= 5 && strcmp(path + n - 5, ".ubjx")   == 0) ||
         (n >= 7 && strcmp(path + n - 7, ".updecx") == 0);
}

int main_bundle(int argc, char *argv[]) {
  const char *mrmp = NULL, *out = NULL, *outcpg = NULL, *kind = NULL, *meta = NULL;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-m") == 0 && i+1 < argc) mrmp   = argv[++i];
    else if (strcmp(argv[i], "-k") == 0 && i+1 < argc) kind   = argv[++i];
    else if (strcmp(argv[i], "-l") == 0 && i+1 < argc) meta   = argv[++i];
    else if (strcmp(argv[i], "-O") == 0 && i+1 < argc) outcpg = argv[++i];
    else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) out    = argv[++i];
    else if (strcmp(argv[i], "-h") == 0) return bundle_usage();
    else break;
  }
  if (!mrmp || !out || argc - i != 1) return bundle_usage();
  const char *model = argv[i];

  /* -l: embed labels into the (raw) booster first, then bundle the annotated copy. */
  const char *inner = model;
  char *tmp_ubj = NULL;
  if (meta) {
    char tmpl[] = "/tmp/methscope_ann_XXXXXX.ubj";
    int fd = mkstemps(tmpl, 4);          /* keep the .ubj suffix for XGBoost */
    if (fd < 0) bdie("cannot create temp booster file", NULL);
    close(fd);
    ms_annotate_booster(model, meta, tmpl);
    tmp_ubj = strdup(tmpl); inner = tmp_ubj;
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
  fprintf(stderr,
    "\n"
    "Usage:\n"
    "  methscope unbundle [-o <model_out>] [--mrmp <mrmp_out>] <bundle>\n"
    "\n"
    "Purpose:\n"
    "  Unpack a bundle (.ubjx / .updecx / .refx) into its inner model, its MRMP,\n"
    "  and (if present) the output-CpG mask.\n"
    "\n"
    "  With no -o/--mrmp, output names are derived from the bundle path (the\n"
    "  original .mrmp filename is not stored): drop the 'x' from the extension for\n"
    "  the model, and add sibling suffixes for the rest --\n"
    "    foo.ubjx   -> foo.ubj   + foo.mrmp  (+ foo.outcpg.cm if present)\n"
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
    else if (strcmp(argv[i], "-h") == 0) return unbundle_usage();
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
