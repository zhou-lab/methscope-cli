#!/bin/bash
## title: A random forest on array probes
# the shipped benign-vs-malignant thyroid classifier, in the EPICv2 row space
methscope fetch -c EPICv2/models/EPICv2_thyroid_malignancy.clfx
methscope inspect EPICv2_thyroid_malignancy.clfx | tail -6
