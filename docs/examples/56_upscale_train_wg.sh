#!/bin/bash
## title: Train
## norun: the recorded command that built hg38_wg.updecx
sbatch -q gpu --gres=gpu:1 --mem=64G -c 8 -t 8:00:00 --wrap '\
  methscope upscale-train -i $W/loyfer_100r.msur --units $W/units_16k.msui \
    -o $W/hg38_wg.updecx --work-dir $W/work --split $SPLIT \
    --patterns 500 --features scalar'
