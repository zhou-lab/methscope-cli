#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Export a trained methscope-cli "upscale" block decoder to a portable .updec file
that the pure-C `methscope upscale` subcommand can read (no torch at inference).

Input: a block directory containing
  - bottleneck_weights_best.pth   (torch state_dict of BottleneckDecoder)
  - imputer_scaler.pkl            ({"imputer": SimpleImputer, "scaler": StandardScaler})
as produced by 20260202_mlp_wg_10k_para.py.

The decoder is:
  Linear(net.0) -> ReLU(net.1) -> BatchNorm1d(net.2) -> Linear(net.3)

This script needs torch + sklearn ONLY because .pth/.pkl are Python pickles; it
is a one-time, CPU-only conversion (batchable over all blocks). Run it in an
env that has them, e.g. Hao's `sturgeon` (`source activate_2023 sturgeon`).

.updec binary layout (little-endian; all arrays float32, written sequentially):
  magic   "UPDEC1\0\0"          (8 bytes)
  int32   n_in, n_hidden, n_out
  float32 bn_eps
  float32 imputer_mean[n_in]
  float32 scaler_mean[n_in]
  float32 scaler_scale[n_in]
  float32 W1[n_hidden*n_in]     (row-major [out,in]; PyTorch Linear weight layout)
  float32 b1[n_hidden]
  float32 bn_gamma[n_hidden], bn_beta[n_hidden], bn_mean[n_hidden], bn_var[n_hidden]
  float32 W2[n_out*n_hidden]    (row-major [out,in])
  float32 b2[n_out]
"""
import argparse
import pickle
import struct
import sys
from pathlib import Path

import numpy as np
import torch

BN_EPS = 1e-5  # torch.nn.BatchNorm1d default; BottleneckDecoder uses the default


def f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype="<f4"))


def main():
    ap = argparse.ArgumentParser(description="Export an upscale block decoder to .updec")
    ap.add_argument("block_dir", help="directory with bottleneck_weights_best.pth + imputer_scaler.pkl")
    ap.add_argument("-o", "--out", required=True, help="output .updec path")
    ap.add_argument("--weights", default="bottleneck_weights_best.pth")
    ap.add_argument("--scaler", default="imputer_scaler.pkl")
    args = ap.parse_args()

    bd = Path(args.block_dir)
    sd = torch.load(bd / args.weights, map_location="cpu")
    # state_dict may be wrapped; accept either a dict of tensors or {'state_dict':...}
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    W1 = sd["net.0.weight"].cpu().numpy(); b1 = sd["net.0.bias"].cpu().numpy()
    bn_g = sd["net.2.weight"].cpu().numpy(); bn_b = sd["net.2.bias"].cpu().numpy()
    bn_m = sd["net.2.running_mean"].cpu().numpy(); bn_v = sd["net.2.running_var"].cpu().numpy()
    W2 = sd["net.3.weight"].cpu().numpy(); b2 = sd["net.3.bias"].cpu().numpy()

    n_hidden, n_in = W1.shape
    n_out = W2.shape[0]
    assert W2.shape[1] == n_hidden, "Linear2 in-dim != hidden"
    assert b1.shape[0] == n_hidden and b2.shape[0] == n_out

    with open(bd / args.scaler, "rb") as fh:
        saved = pickle.load(fh)
    imp_mean = np.asarray(saved["imputer"].statistics_, dtype=np.float64)
    sc_mean = np.asarray(saved["scaler"].mean_, dtype=np.float64)
    sc_scale = np.asarray(saved["scaler"].scale_, dtype=np.float64)
    for name, arr in (("imputer", imp_mean), ("scaler.mean", sc_mean), ("scaler.scale", sc_scale)):
        if arr.shape[0] != n_in:
            sys.exit(f"{name} length {arr.shape[0]} != n_in {n_in}")

    with open(args.out, "wb") as f:
        f.write(b"UPDEC1\x00\x00")
        f.write(struct.pack("<iii", n_in, n_hidden, n_out))
        f.write(struct.pack("<f", BN_EPS))
        f.write(f32(imp_mean).tobytes())
        f.write(f32(sc_mean).tobytes())
        f.write(f32(sc_scale).tobytes())
        f.write(f32(W1).tobytes())   # [n_hidden, n_in] row-major
        f.write(f32(b1).tobytes())
        f.write(f32(bn_g).tobytes()); f.write(f32(bn_b).tobytes())
        f.write(f32(bn_m).tobytes()); f.write(f32(bn_v).tobytes())
        f.write(f32(W2).tobytes())   # [n_out, n_hidden] row-major
        f.write(f32(b2).tobytes())

    print(f"[export_upscale_model] wrote {args.out}  "
          f"(n_in={n_in}, n_hidden={n_hidden}, n_out={n_out})", file=sys.stderr)


if __name__ == "__main__":
    main()
