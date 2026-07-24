#!/usr/bin/env python3
"""Cross-check a C `upscale --probs` row by reading UPDEC1 with NumPy."""
import argparse
import struct

import numpy as np


def take(raw, off, count):
    arr = np.frombuffer(raw, dtype="<f4", count=count, offset=off)
    return arr.astype(np.float64), off + count * 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("features", help="TSV with a header and one feature row")
    ap.add_argument("c_probs", help="one row from methscope upscale --probs")
    args = ap.parse_args()

    raw = open(args.model, "rb").read()
    if raw[:8] != b"UPDEC1\0\0":
        raise SystemExit("bad UPDEC1 magic")
    I, H, O = struct.unpack_from("<iii", raw, 8)
    eps = struct.unpack_from("<f", raw, 20)[0]
    off = 24
    imp, off = take(raw, off, I)
    mean, off = take(raw, off, I)
    scale, off = take(raw, off, I)
    W1, off = take(raw, off, H * I); W1 = W1.reshape(H, I)
    b1, off = take(raw, off, H)
    gamma, off = take(raw, off, H)
    beta, off = take(raw, off, H)
    run_mean, off = take(raw, off, H)
    run_var, off = take(raw, off, H)
    W2, off = take(raw, off, O * H); W2 = W2.reshape(O, H)
    b2, off = take(raw, off, O)
    if off != len(raw):
        raise SystemExit("unexpected trailing model bytes")

    with open(args.features) as fh:
        next(fh)
        cells = fh.readline().rstrip("\r\n").split("\t")
    if len(cells) != I:
        raise SystemExit(f"feature count {len(cells)} != {I}")
    x = np.array([np.nan if v in ("", "NA", "nan", "NaN") else float(v)
                  for v in cells])
    x = np.where(np.isnan(x), imp, x)
    x = (x - mean) / scale
    hidden = np.maximum(W1 @ x + b1, 0.0)
    hidden = gamma * (hidden - run_mean) / np.sqrt(run_var + eps) + beta
    logits = W2 @ hidden + b2
    expected = np.where(logits >= 0, 1 / (1 + np.exp(-logits)),
                        np.exp(logits) / (1 + np.exp(logits)))
    got = np.loadtxt(args.c_probs, dtype=np.float64)
    diff = np.max(np.abs(got - expected))
    print(f"UPDEC1 forward max_abs_diff={diff:.3g}")
    if diff > 2e-6:
        raise SystemExit("forward parity failed")


if __name__ == "__main__":
    main()
