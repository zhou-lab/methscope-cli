#!/bin/bash
## title: Train
## norun: the recorded command that built hg38_wg.updecx (models v10)
sbatch -q gpu --gres=gpu:1 --mem=64G -c 8 -t 16:00:00 --wrap '\
  methscope upscale-train -i $W/loyfer_100r_flat500_bank100.msur \
    --units $W/units_16k.msui --mrmp $W/loyfer_train_flat500_bank100.mrmp \
    -o $W/hg38_wg.updecx --work-dir $W/work --split $SPLIT \
    --features scalar --activation leaky \
    --pure-bottleneck 16 --mixed-bottleneck 32 --mixed-mode factor \
    --min-steps 2000 --max-steps 60000 --patience 400 --eval-every 200 \
    --eval-rows 24 --device 0'
