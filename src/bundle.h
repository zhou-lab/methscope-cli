// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * Generic model bundle (MSBNDL1): wraps an inner model together with the MRMP
 * definition it needs, so one self-contained file lets a query .cg be featurized +
 * run without a separate .mrmp. The `x` extension suffix marks the bundled form:
 *   .ubj   (cell-type booster / linear model) -> .ubjx   (run by `predict`)
 *   .updec (upscale block decoder)            -> .updecx  (run by `upscale`)
 *   .ref   (celltype x pattern signature TSV) -> .refx    (run by `deconv`)
 *
 * layout — the MRMP .cm is the FIRST bytes of the file, so `yame` (and any tool
 * reading a .cm) can read a bundle directly: yame stops at the .cm's BGZF EOF marker
 * and ignores the MSBNDL1 trailer. A footer at the very end points back at the
 * container, so readers don't have to parse the .cm to find it. Native byte order:
 *
 *   [ MRMP .cm bytes ]                                  offset 0 (a complete YAME .cm)
 *   "MSBNDL1\0"                                         8 bytes   (container magic at offset M)
 *   uint32 n_sections
 *   n_sections x { char name[16]; uint64 offset; uint64 length }   (absolute file offsets)
 *   section blobs                                       (kind, outcpg, model, ...)
 *   uint64 msbndl_offset = M                            8-byte footer (last bytes of file)
 *
 * Magic "MSBNDL1\0" = "MethScope BuNDLe, version 1" (7 chars + NUL; the trailing
 * digit is a format-version stamp). A plain .cm has no footer -> not a bundle.
 *
 * Sections are a name->blob directory (order: mrmp, [kind], [outcpg], model). The
 * "mrmp" entry has offset 0 / length = the leading-.cm size. Others:
 *   "mrmp"       the bundled MRMP (a YAME .cm) = the file prefix; featurizes the
 *                query internally (ms_mrmp_resolve just hands back the bundle path).
 *   "model"      the inner-model bytes; meaning is format-specific:
 *                  .ubjx   -> an XGBoost .ubj booster OR a "methscope-linear" text
 *                             spec (threshold/logistic). Class labels live INSIDE
 *                             the .ubj as XGBoost attributes, not a separate section.
 *                  .updecx -> a .updec MLP decoder (little-endian float32 weights).
 *                  .refx   -> the celltype x pattern beta signature (a TSV).
 *   "outcpg"     (upscale only, optional) a genome-wide mask of the imputed CpG
 *                locations; if present, `upscale` writes a whole-genome .cg. The
 *                format allows repeated (outcpg, model) pairs for a multi-block
 *                .updecx (execution of multiple models is not yet implemented).
 *   "kind"       the framework mark string:
 *                  .ubjx   -> "xgboost" | "threshold" | "logistic"  (REQUIRED;
 *                             `predict` rejects an unmarked bundle).
 *                  .refx   -> "refx".
 *                  .updecx has no kind (dispatched by the `upscale` subcommand).
 *
 * A wrong pairing (e.g. a .updecx fed to `predict`) fails on the inner model's own
 * magic. Build/read via ms_bundle_pack() / ms_bundle_kind() / ms_bundle_section() /
 * ms_bundle_list() below; ms_mrmp_resolve() returns the path as-is (the front .cm is
 * read directly wherever a loose <ref.mrmp> is expected).
 */
#ifndef METHSCOPE_BUNDLE_H
#define METHSCOPE_BUNDLE_H

#include <stddef.h>
#include <stdint.h>

#define MS_BUNDLE_MAGIC "MSBNDL1"   /* 7 chars + NUL = 8 bytes; MRMP-first layout */

/* A bundle section-table entry (name is NUL-terminated within its 16 bytes). */
typedef struct { char name[16]; uint64_t offset, length; } ms_bundle_entry_t;

/* Enumerate a bundle's sections in file order into a malloc'd array (*n_out
 * entries; caller frees). Exits on a bad/truncated bundle. */
ms_bundle_entry_t *ms_bundle_list(const char *path, int *n_out);

/* 1 if `path` is a bundle (its trailing footer points at the MSBNDL1 magic), else 0. */
int ms_bundle_is(const char *path);

/* Extract a named section into a malloc'd buffer (caller frees); exits on a
 * missing section or read error. */
void *ms_bundle_section(const char *path, const char *name, size_t *len_out);

/* Like ms_bundle_section but returns NULL (not fatal) if the section is absent. */
void *ms_bundle_section_opt(const char *path, const char *name, size_t *len_out);

/* Resolve a path given where a loose .mrmp (a YAME .cm) is expected. A bundle
 * begins with its MRMP .cm, and a loose .cm is already what we want, so `path` is
 * returned as-is in both cases (*tmp_out = NULL). Kept for API symmetry with
 * ms_mrmp_cleanup(); no temp file is created. */
const char *ms_mrmp_resolve(const char *path, char **tmp_out);

/* Unlink + free a temp path returned via ms_mrmp_resolve()'s *tmp_out (NULL ok). */
void ms_mrmp_cleanup(char *tmp);

/* Pack an inner model file + its MRMP (+ optional outcpg mask) into an MSBNDL1
 * bundle at `out` (the on-disk form `bundle` writes): the MRMP .cm becomes the file
 * prefix, then the container with sections [kind] [outcpg] model. `kind` (e.g.
 * "xgboost"/"threshold"/"refx") is written as a "kind" section marking the model
 * framework; pass NULL to omit it. `outcpg_path` may be NULL. Exits on I/O error. */
void ms_bundle_pack(const char *out, const char *kind, const char *model_path,
                    const char *mrmp_path, const char *outcpg_path);

/* Read the framework mark (the "kind" section) as a malloc'd NUL-terminated
 * string, or NULL if the bundle has no kind section. (`predict` requires a kind
 * and rejects an unmarked bundle; upscale/deconv dispatch by subcommand.) */
char *ms_bundle_kind(const char *path);

/* 1 if `path` ends in ".ubjx" or ".updecx" (the bundled-model extensions). */
int ms_path_is_bundle_ext(const char *path);

/* subcommands */
int main_bundle(int argc, char *argv[]);
int main_unbundle(int argc, char *argv[]);

#endif /* METHSCOPE_BUNDLE_H */
