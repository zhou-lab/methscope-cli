// SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
// Use of this software is available to academic and non-profit institutions
// for research purposes under the 2-Clause BSD License; for use or transfers
// to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
// See the LICENSE file at the root of the repository for the full terms.
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
#include "methscope.h"
#include "msfm.h"
#include "mrmp.h"
#include "methscope.h"
#include "assets.h"   /* the registry, the store root, per-file state */
#include "index.h"        /* get_fname_index -- the resolve temp has a sibling .idx */

#define NAMELEN 16

/* Section names live in a fixed char[16], which capped node names at 15
 * characters -- and a routing tree only six levels deep breaks that
 * ("root.0.0.0.3.0.7.2" is 18). Long names are stored under a deterministic
 * ALIAS instead: '~' + 14 hex chars of the name's FNV-1a hash, 15 chars
 * exactly. Writer and reader both apply it, so no mapping section is needed;
 * a name that fits is stored verbatim, so every bundle written before this
 * reads identically. A collision needs two >=16-char node names sharing 56
 * hash bits -- and the writer's duplicate check runs on aliases, so even that
 * would refuse loudly rather than misroute. */
static void sec_alias(const char *name, char out[NAMELEN]) {
  size_t len = strlen(name);
  if (len < NAMELEN) { memcpy(out, name, len + 1); return; }
  uint64_t h = 0xcbf29ce484222325ull;
  for (const unsigned char *p2 = (const unsigned char *)name; *p2; ++p2)
    h = (h ^ *p2) * 0x100000001b3ull;
  snprintf(out, NAMELEN, "~%014llx",
           (unsigned long long)(h & 0xFFFFFFFFFFFFFFull));
}
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

/* M if `path` is a bundle, -1 otherwise. Never exits: callers use it to ASK
   whether a path is a bundle, so an unreadable or too-short file is an answer
   ("not a bundle"), not a failure. */
static int64_t bundle_container_off(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return -1;
  int64_t off = -1;
  if (fseeko(fp, 0, SEEK_END) == 0) {
    off_t total = ftello(fp);
    uint64_t container_off;
    if (total >= FOOTER_BYTES + CONTAINER_HDR_BYTES &&
        fseeko(fp, total - FOOTER_BYTES, SEEK_SET) == 0 &&
        fread(&container_off, 8, 1, fp) == 1 &&
        container_off <= (uint64_t)total - FOOTER_BYTES &&
        (uint64_t)total - FOOTER_BYTES - container_off >= CONTAINER_HDR_BYTES) {
      char m[8];
      if (fseeko(fp, (off_t)container_off, SEEK_SET) == 0 && fread(m, 1, 8, fp) == 8 &&
          memcmp(m, MS_BUNDLE_MAGIC, 8) == 0)
        off = (int64_t)container_off;
    }
  }
  fclose(fp);
  return off;
}

int ms_bundle_is(const char *path) {
  return bundle_container_off(path) >= 0;
}

int64_t ms_bundle_cx_limit(const char *path) {
  return bundle_container_off(path);   /* -1 == CX_NO_LIMIT for a non-bundle */
}

void *ms_bundle_section_opt(const char *path, const char *name, size_t *len_out) {
  FILE *fp = fopen(path, "rb");
  if (!fp) bdie("cannot open", path);
  uint64_t total, container_off; uint32_t n;
  entry_t *entries = bundle_directory(fp, path, &total, &container_off, &n);
  (void)total; (void)container_off;
  char want[NAMELEN]; sec_alias(name, want);
  for (uint32_t i = 0; i < n; ++i) {
    entry_t e = entries[i];
    if (strcmp(e.name, want) == 0) {
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

/* See bundle.h. Since YAME v1.52 the lookup itself is yame_store_resolve(),
 * shared with the other tools so three repos cannot each get the name rules
 * subtly different. What stays here is the POLICY, which the library
 * deliberately does not enforce: stop when the file is not there, warn and
 * go on when it is merely from an earlier release. */
const char *ms_model_resolve(const char *spec, char **owned) {
  if (owned) *owned = NULL;
  if (!spec || !*spec) return spec;

  const yame_fetch_cfg_t *cfg = yame_default_fetch_cfg();
  if (!cfg) return spec;                 /* no registry in this build */

  char path[8448], advice[512];
  const yame_asset_file_t *rec = NULL;
  yame_store_state_t st = yame_store_resolve(cfg, spec, NULL, path, sizeof path,
                                             &rec, advice, sizeof advice);
  switch (st) {
  case YAME_STORE_ABSENT:
    /* Nothing to open. Stop, with the line that fetches it, rather than an
     * open() failure naming a path the user never typed. */
    if (advice[0]) fprintf(stderr, "[methscope] %s\n", advice);
    return NULL;
  case YAME_STORE_STALE:
    /* A real model, just from an earlier release. Refusing would strand a run
     * that worked yesterday; the advice carries the repair, so the choice
     * stays the reader's. Same policy `fetch` itself applies to a stale
     * store. */
    if (advice[0]) fprintf(stderr, "[methscope] %s\n", advice);
    break;
  case YAME_STORE_NOT_CATALOGUED:
    /* Two cases, told apart by `path`: empty means the name is ambiguous and
     * the advice names the candidates; otherwise it is simply not ours, and
     * `path` is the spec unchanged so the opener reports it in its own
     * words -- a typo'd local file should say "cannot open foo.clfx", not
     * "not in the catalogue". */
    if (!path[0]) {
      if (advice[0]) fprintf(stderr, "[methscope] %s\n", advice);
      return NULL;
    }
    return spec;
  default:
    break;
  }

  if (!strcmp(path, spec)) return spec;  /* an existing path, used as given */
  char *p = strdup(path);
  if (!p) return NULL;
  if (owned) *owned = p;
  return p;
}

const char *ms_mrmp_resolve(const char *path, char **tmp_out) {
  *tmp_out = NULL;
  /* A loose .cm is already what open_cfile wants, and a .cm-bearing bundle is
     too: its .cm is the file prefix. Either way the path goes straight through
     -- but a bundle only reads correctly if the CALLER bounds the read at
     ms_bundle_cx_limit(). yame does NOT stop by itself at the prefix's BGZF EOF
     marker; it did until YAME v1.40, and assuming it still does is what broke
     `methscope upscale` in v0.7. See the layout note in bundle.h.

     A .mrmp does not: it is the artifact, not the runtime mask. Callers that
     want EVERY set of a chain expand it themselves (classify-featurize does).
     This is the single-set path -- deconv, deconv-build-ref -- so it materializes
     the FIRST block as a temp .cm and hands that back, which is what the
     (path, tmp_out) contract and ms_mrmp_cleanup() were always for. */
  FILE *f = fopen(path, "rb");
  if (!f) return path;
  char magic[8];
  int is_mrmp = fread(magic, 1, 8, f) == 8 && !memcmp(magic, MRMPIDX_MAGIC, 8);
  fclose(f);
  if (!is_mrmp) return path;

  char tpl[4096];
  const char *td = getenv("TMPDIR");
  snprintf(tpl, sizeof tpl, "%s/methscope_resolve_XXXXXX.cm",
           td && *td ? td : "/tmp");
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
static int bundle_usage(FILE *out) {
  ms_help(out,
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
    "  Both are the same MSBNDL1 container and are detected by magic, not by\n"
    "  name; '.ubjx' is the former classifier extension and is still accepted.\n"
    "\n"
    "Options:\n"
    "  -m <ref.mrmp>   the MRMP definition (a YAME .cm) to bundle (required).\n"
    "  -k <kind>       framework mark to record (xgboost/violation/logistic);\n"
    "                  required for a classifier .clfx that `classify` will run (it\n"
    "                  rejects an unmarked bundle). Also re-stamps an existing model.\n"
    "  -O <outcpg.cm>  output-CpG locations (upscale only): a YAME mask marking the\n"
    "                  CpGs the model imputes. With it, `upscale` writes a full-genome\n"
    "                  .cg; without it, a dense block .cg.\n"
    "  -o <out>        output bundle path (required).\n"
    "  -h              Show this help message.\n"
    "\n");
  return out == stdout ? 0 : 1;
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
  char (*alias)[NAMELEN] = malloc((size_t)(n_nodes ? n_nodes : 1) * NAMELEN);
  if (!alias) bdie("out of memory", out);
  for (uint32_t k = 0; k < n_nodes; ++k) {
    sec_alias(node_name[k], alias[k]);
    for (uint32_t j = 0; j < k; ++j)
      if (!strcmp(alias[k], alias[j]))
        bdie("duplicate node name (or alias collision)", node_name[k]);
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
    memset(&e, 0, sizeof(e)); memcpy(e.name, alias[k], NAMELEN);
    e.offset = off; e.length = booster_len[k]; off += booster_len[k];
    if (fwrite(&e, sizeof(e), 1, fp) != 1) bdie("write error", out);
  }
  free(alias);
  if (fwrite(kind, 1, strlen(kind), fp) != strlen(kind)) bdie("write error", out);
  for (uint32_t k = 0; k < n_nodes; ++k)
    if (booster_len[k] &&
        fwrite(booster[k], 1, (size_t)booster_len[k], fp) != booster_len[k])
      bdie("write error", out);
  if (fwrite(&container_off, sizeof(container_off), 1, fp) != 1) bdie("write error", out);
  if (fclose(fp)) bdie("write error", out);
}

void ms_bundle_repack(const char *out, const char *mrmp_path, const char *src) {
  FILE *in = fopen(src, "rb");
  if (!in) bdie("cannot open bundle", src);
  uint64_t total, coff; uint32_t n;
  entry_t *e = bundle_directory(in, src, &total, &coff, &n);
  uint64_t rlen = path_bytes(mrmp_path);
  /* every section but the prefix travels; count them and size the table */
  uint32_t nsec = 1;
  for (uint32_t i = 0; i < n; ++i) if (strcmp(e[i].name, "mrmp")) ++nsec;
  uint64_t hdr = CONTAINER_HDR_BYTES + (uint64_t)nsec * sizeof(entry_t);
  FILE *fp = fopen(out, "wb");
  if (!fp) bdie("cannot open output", out);
  stream_path(fp, mrmp_path, rlen, out);
  if (fwrite(MS_BUNDLE_MAGIC, 1, 8, fp) != 8) bdie("write error", out);
  if (fwrite(&nsec, sizeof(nsec), 1, fp) != 1) bdie("write error", out);
  uint64_t off = rlen + hdr;
  entry_t w;
  memset(&w, 0, sizeof(w)); strncpy(w.name, "mrmp", NAMELEN - 1);
  w.offset = 0; w.length = rlen;
  if (fwrite(&w, sizeof(w), 1, fp) != 1) bdie("write error", out);
  for (uint32_t i = 0; i < n; ++i) {
    if (!strcmp(e[i].name, "mrmp")) continue;
    memset(&w, 0, sizeof(w)); memcpy(w.name, e[i].name, NAMELEN);
    w.offset = off; w.length = e[i].length; off += e[i].length;
    if (fwrite(&w, sizeof(w), 1, fp) != 1) bdie("write error", out);
  }
  char *buf = malloc(1 << 20);
  if (!buf) bdie("out of memory", out);
  for (uint32_t i = 0; i < n; ++i) {
    if (!strcmp(e[i].name, "mrmp")) continue;
    if (fseeko(in, (off_t)e[i].offset, SEEK_SET)) bdie("cannot seek", src);
    uint64_t left = e[i].length;
    while (left) {
      size_t chunk = left < (1 << 20) ? (size_t)left : (1 << 20);
      if (fread(buf, 1, chunk, in) != chunk) bdie("truncated section", e[i].name);
      if (fwrite(buf, 1, chunk, fp) != chunk) bdie("write error", out);
      left -= chunk;
    }
  }
  free(buf); free(e); fclose(in);
  if (fwrite(&rlen, sizeof(rlen), 1, fp) != 1) bdie("write error", out);
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
 * upscale decoder. ".ubjx" was the classifier's
 * former name -- it described the payload format (a UBJSON booster) rather than
 * the role, and stopped being true the moment the threshold/logistic
 * frameworks shipped a plain-text model section under it. Kept accepted
 * indefinitely: detection never depended on it, so nothing is gained by
 * breaking existing files or scripts. ".refx" was the deconvolution reference
 * before .msdref replaced it; nothing produces or reads one now. */
int ms_path_is_bundle_ext(const char *path) {
  size_t n = strlen(path);
  return (n >= 5 && strcmp(path + n - 5, ".clfx")   == 0) ||
         (n >= 7 && strcmp(path + n - 7, ".updecx") == 0) ||
         (n >= 5 && strcmp(path + n - 5, ".ubjx")   == 0);  /* legacy */
}

int main_bundle(int argc, char *argv[]) {
  const char *pos[4]; int npos = 0;
  const char *mrmp = NULL, *out = NULL, *outcpg = NULL, *kind = NULL;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-m") == 0 && i+1 < argc) mrmp   = argv[++i];
    else if (strcmp(argv[i], "-k") == 0 && i+1 < argc) kind   = argv[++i];
    else if (strcmp(argv[i], "-O") == 0 && i+1 < argc) outcpg = argv[++i];
    else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) out    = argv[++i];
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      return bundle_usage(stdout);
    }
    else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0)
      bdie("unrecognized or incomplete option", argv[i]);
    else if (npos < (int)(sizeof pos / sizeof *pos)) pos[npos++] = argv[i];
    else break;   /* too many positionals: the tail's own check reports it */
  }
  if (!mrmp || !out) return bundle_usage(stderr);
  if (npos != 1) return bundle_usage(stderr);
  const char *model = pos[0];
  const char *inner = model;

  ms_bundle_pack(out, kind, inner, mrmp, outcpg);   /* kind mark (NULL = omit) */
  if (outcpg)
    fprintf(stderr, "[methscope] bundled %s + %s + %s -> %s\n", model, mrmp, outcpg, out);
  else
    fprintf(stderr, "[methscope] bundled %s + %s -> %s\n", model, mrmp, out);
  return 0;
}

/* ------------------------------------------------------------------ */
/* unbundle                                                           */
/* ------------------------------------------------------------------ */
static int unbundle_usage(FILE *out) {
  ms_help(out,
    "\n"
    "Usage:\n"
    "  methscope unbundle [-o <model_out>] [--mrmp <mrmp_out>] <bundle>\n"
    "\n"
    "Purpose:\n"
    "  Unpack a bundle (.clfx / .updecx) into its inner model, its MRMP,\n"
    "  and (if present) the output-CpG mask.\n"
    "\n"
    "  With no -o/--mrmp, output names are derived from the bundle path (the\n"
    "  original .mrmp filename is not stored): drop the 'x' from the extension for\n"
    "  the model, and add sibling suffixes for the rest --\n"
    "    foo.clfx   -> foo.model + foo.mrmp  (+ foo.outcpg.cm if present)\n"
    "    foo.updecx -> foo.updec + foo.mrmp\n"
    "\n"
    "Options:\n"
    "  -o <model_out>    write the inner model here (default: derived, see above).\n"
    "  --mrmp <mrmp_out> write the bundled MRMP (.cm) here (default: derived).\n"
    "  -h                Show this help message.\n"
    "\n");
  return out == stdout ? 0 : 1;
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
  const char *pos[4]; int npos = 0;
  const char *model_out = NULL, *mrmp_out = NULL;
  int i = 1;
  for (; i < argc; ++i) {
    if      (strcmp(argv[i], "-o") == 0 && i+1 < argc) model_out = argv[++i];
    else if (strcmp(argv[i], "--mrmp") == 0 && i+1 < argc) mrmp_out = argv[++i];
    else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      return unbundle_usage(stdout);
    }
    else if (argv[i][0] == '-' && strcmp(argv[i], "-") != 0)
      bdie("unrecognized or incomplete option", argv[i]);
    else if (npos < (int)(sizeof pos / sizeof *pos)) pos[npos++] = argv[i];
    else break;   /* too many positionals: the tail's own check reports it */
  }
  if (npos != 1) return unbundle_usage(stderr);
  const char *bundle = pos[0];

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
