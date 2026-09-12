#!/bin/bash
## title: Build your own .msdref
methscope fetch -c hg38/data/human_hg38_celltypes.cg
methscope deconv-build-ref -o self.msdref human_hg38_celltypes.cg
methscope deconv self.msdref human_hg38_celltypes.cg
