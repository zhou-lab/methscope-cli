# Produce R reference outputs (matrix + predictions) for the parity test,
# bypassing the MethScope package namespace so only light deps are needed.
# Configured via env: MS_REPO (path to MethScope repo), OUT_DIR.
MS  <- Sys.getenv("MS_REPO",  "../MethScope")
OUT <- Sys.getenv("OUT_DIR",  "/tmp/ms2_parity")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

dyn.load(file.path(MS, "src/MethScope.so"))
suppressMessages({
  library(magrittr); library(dplyr); library(tidyr)
  library(stringr); library(data.table); library(xgboost)
})
source(file.path(MS, "R/GenerateInput.R"))
source(file.path(MS, "R/PredictCellType.R"))

load(file.path(MS, "R/sysdata.rda"))
meta <- MethScope_model_metadata[["Liu2021_MouseBrain_P1000"]]
con  <- gzfile(file.path(MS, "inst/extdata", meta$file), "rb")
braw <- readBin(con, "raw", 1e8); close(con)
model <- structure(list(
  booster   = xgboost::xgb.load.raw(braw),
  cell_type = meta$cell_type,
  npattern  = meta$npattern,
  num_class = meta$num_class), class = "MethScopeModel")

ip   <- GenerateInput(file.path(MS, "inst/extdata/example.cg"),
                      file.path(MS, "inst/extdata/mm10_Liu2021.cm"))
pred <- PredictCellType(model, ip)

write.table(data.frame(cell = rownames(pred),
                       prediction_label = as.character(pred$prediction_label),
                       confidence = pred$confidence_score),
            file.path(OUT, "r_pred.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(cbind(cell = rownames(ip), as.data.frame(ip)),
            file.path(OUT, "r_matrix.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("R reference written to", OUT, "\n")
