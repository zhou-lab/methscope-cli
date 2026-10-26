#!/bin/bash
## title: Train a random forest on array probes
## norun: the recorded build of EPICv2_thyroid_malignancy.clfx; needs the 117 labelled EPICv2 tumors.
# beta.cg = sesame-cli betas (format 4) for the training tumors, one record each;
# labels.txt = one label per record, in record order (Benign / Malignant)
methscope classify-train --framework rf --seed 123 --threads 8 \
  -l labels.txt -o EPICv2_thyroid_malignancy.clfx beta.cg

# a query on the same platform: its betas are read probe by probe
methscope classify --probs EPICv2_thyroid_malignancy.clfx query_beta.cg
