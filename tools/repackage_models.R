#!/usr/bin/env Rscript
# Repackage MethScope's pretrained models into self-contained methscope-cli .msm
# bundles (booster.ubj + reference.cm + meta.tsv).
#
# Usage:
#   Rscript repackage_models.R <methscope_repo> <methscope_bin> <out_dir>
#
# The .msm bundles the FULL reference .cm; predict uses the first `npattern`
# patterns (numeric order), matching the R GenerateInput()/PredictCellType path.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: repackage_models.R <methscope_repo> <methscope_bin> <out_dir>")
methscope_repo <- args[1]
ms2_bin        <- normalizePath(args[2])
out_dir        <- args[3]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

load(file.path(methscope_repo, "R/sysdata.rda"))   # MethScope_model_metadata

# model name -> bundled reference .cm and genome build
ref_map <- list(
  Liu2021_MouseBrain_P1000  = list(cm = "mm10_Liu2021.cm",  genome = "mm10"),
  Zhou2025_HumanAtlas_P1000 = list(cm = "hg38_Zhou2025.cm", genome = "hg38")
)

extdata <- file.path(methscope_repo, "inst/extdata")

for (name in names(MethScope_model_metadata)) {
  m   <- MethScope_model_metadata[[name]]
  ref <- ref_map[[name]]
  if (is.null(ref)) { cat("skip (no reference mapping):", name, "\n"); next }
  cm_path <- file.path(extdata, ref$cm)
  if (!file.exists(cm_path)) { cat("skip (missing reference):", cm_path, "\n"); next }

  work    <- tempfile("msm_"); dir.create(work)
  booster <- file.path(work, "booster.ubj")
  meta    <- file.path(work, "meta.tsv")

  # gunzip the .ubj.gz booster into raw ubj bytes
  gz  <- file.path(extdata, m$file)
  con <- gzfile(gz, "rb"); raw <- readBin(con, "raw", n = 1e9); close(con)
  writeBin(raw, booster)

  writeLines(c(
    paste0("num_class\t", m$num_class),
    paste0("npattern\t",  m$npattern),
    paste0("genome\t",    ref$genome),
    paste0("labels\t",    paste(m$cell_type, collapse = ","))
  ), meta)

  out <- file.path(out_dir, paste0(name, ".msm"))
  cmd <- sprintf("%s pack-model -o %s %s %s %s",
                 shQuote(ms2_bin), shQuote(out),
                 shQuote(booster), shQuote(cm_path), shQuote(meta))
  cat("packing", name, "->", out, "\n")
  if (system(cmd) != 0) stop("pack-model failed for ", name)
  unlink(work, recursive = TRUE)
}
cat("done\n")
