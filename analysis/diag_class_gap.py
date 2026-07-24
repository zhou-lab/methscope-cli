#!/usr/bin/env python3
"""Locate the UPDEC2-vs-hybrid error gap by CpG class.

Reuses the exact paired sampler and model readers from
compare_upscale_models.py, then splits every scored target by whether it falls
in the top-1000 MRMP head (sidecar group > 0) or the residual set (group == 0),
and reports MAE / RMSE per class for the unified UPDEC2 and the hybrid. This
shows whether the unified model's heavy error tail is concentrated in the hard
residual CpGs or spread across the top head.
"""
import argparse
import struct

import numpy as np

from compare_upscale_models import HybridModel, sample_rows
from compare_updec2_models import Model as Updec2Model


def group_map(data_path, n_cpg):
    with open(data_path, "rb") as f:
        d = struct.unpack("<8s4IQ2I4Q", f.read(72))
    groups_off = d[8]
    return np.memmap(data_path, "<u2", "r", groups_off, (n_cpg,))


def stats(pred, y):
    d = pred - y
    return len(y), float(np.abs(d).mean()), float(np.sqrt((d @ d) / len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updec2", required=True)
    ap.add_argument("--hybrid-encoder", required=True)
    ap.add_argument("--hybrid-residual", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--targets-per-row", type=int, default=131072)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    up = Updec2Model(args.updec2)
    hy = HybridModel(args.hybrid_encoder, args.hybrid_residual)
    groups = group_map(args.data, up.n_cpg)
    rows = sample_rows(args.data, up, "all", args.rows, args.targets_per_row,
                       args.seed, include_observed=True)

    cls = {"top": [], "residual": [], "all": []}
    up_pred = {k: [] for k in cls}
    hy_pred = {k: [] for k in cls}
    ys = {k: [] for k in cls}
    for row in rows:
        g = row["beta"][:up.patterns]
        c = row["count"][:up.patterns]
        genomic, y = row["genomic"], row["truth"]
        pu = up.predict(g, c, genomic)
        ph = hy.predict(row["beta"][:hy.patterns], row["count"][:hy.patterns],
                        genomic)
        gr = np.asarray(groups[genomic])
        is_top = gr > 0
        for key, mask in (("top", is_top), ("residual", ~is_top),
                          ("all", np.ones(len(y), bool))):
            ys[key].append(y[mask]); up_pred[key].append(pu[mask])
            hy_pred[key].append(ph[mask])

    print("class\tn\tupdec2_mae\thybrid_mae\tgap\tupdec2_rmse\thybrid_rmse")
    for key in ("all", "top", "residual"):
        y = np.concatenate(ys[key])
        nu, umae, urmse = stats(np.concatenate(up_pred[key]), y)
        _, hmae, hrmse = stats(np.concatenate(hy_pred[key]), y)
        print(f"{key}\t{nu}\t{umae:.6f}\t{hmae:.6f}\t{umae - hmae:+.6f}"
              f"\t{urmse:.6f}\t{hrmse:.6f}")


if __name__ == "__main__":
    main()
