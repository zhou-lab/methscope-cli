#!/usr/bin/env Rscript
# Emit standalone meta.tsv files for the pretrained models (committed to the
# repo so `methscope pack-model` can rebuild a .msm from wget'd pieces without R).
#   Rscript dump_model_meta.R <methscope_repo> <out_dir>
args <- commandArgs(trailingOnly = TRUE)
methscope_repo <- args[1]
out_dir        <- args[2]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

load(file.path(methscope_repo, "R/sysdata.rda"))
genome <- list(Liu2021_MouseBrain_P1000 = "mm10",
               Zhou2025_HumanAtlas_P1000 = "hg38")

for (name in names(MethScope_model_metadata)) {
  m <- MethScope_model_metadata[[name]]
  writeLines(c(
    paste0("num_class\t", m$num_class),
    paste0("npattern\t",  m$npattern),
    paste0("genome\t",    genome[[name]]),
    paste0("labels\t",    paste(m$cell_type, collapse = ","))
  ), file.path(out_dir, paste0(name, ".meta.tsv")))
  cat("wrote", file.path(out_dir, paste0(name, ".meta.tsv")), "\n")
}
