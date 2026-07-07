# Compare methscope (C) vs MethScope (R) outputs; exits non-zero on failure.
OUT <- Sys.getenv("OUT_DIR", "/tmp/ms2_parity")
fail <- 0

## ---- predictions ----
c <- read.table(file.path(OUT, "c_pred.tsv"), header = TRUE, sep = "\t", stringsAsFactors = FALSE)
r <- read.table(file.path(OUT, "r_pred.tsv"), header = TRUE, sep = "\t", stringsAsFactors = FALSE)
m <- merge(c[, c("cell","prediction_label","confidence")], r, by = "cell", suffixes = c(".c",".r"))
lab_ok  <- sum(m$prediction_label.c == m$prediction_label.r)
conf_d  <- max(abs(m$confidence.c - m$confidence.r))
cat(sprintf("predict: %d/%d labels agree, max|conf diff|=%.2e\n", lab_ok, nrow(m), conf_d))
if (lab_ok != nrow(m) || conf_d > 1e-4) { cat("  -> FAIL\n"); fail <- 1 }

## ---- feature matrix ----
cm <- as.matrix(read.table(file.path(OUT, "c_matrix.tsv"), header = TRUE, sep = "\t",
                           row.names = 1, check.names = FALSE))
rm_ <- as.matrix(read.table(file.path(OUT, "r_matrix.tsv"), header = TRUE, sep = "\t",
                            row.names = 1, check.names = FALSE))
cells <- intersect(rownames(cm), rownames(rm_)); pats <- intersect(colnames(cm), colnames(rm_))
order_ok <- identical(colnames(cm)[seq_len(min(1000, ncol(cm)))],
                      colnames(rm_)[seq_len(min(1000, ncol(rm_)))])
A <- cm[cells, pats]; B <- rm_[cells, pats]
na_ok <- mean(is.na(A) == is.na(B))
d <- abs(A - B); d <- max(d[!is.na(d)])
cat(sprintf("matrix: %d cells x %d patterns, colorder=%s, NA-agree=%.3f, max|beta diff|=%.2e\n",
            length(cells), length(pats), order_ok, na_ok, d))
if (!order_ok || na_ok < 1 || d > 2e-3) { cat("  -> FAIL\n"); fail <- 1 }

cat(if (fail) "PARITY: FAIL\n" else "PARITY: PASS\n")
quit(status = fail)
