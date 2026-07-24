#!/usr/bin/env python3
"""Paired Zhou-cohort comparison of UPDEC2, the frozen hybrid, and Hao's MLP.

The three models receive the same simulated MRMP values and are scored on the
same (cell, replicate, genomic-CpG) tuples.  Hao's 2,941 checkpoints are
streamed one at a time and only the requested output rows are evaluated.
"""
import argparse
import math
import pickle
import struct
import time
from pathlib import Path

import numpy as np

from compare_updec2_models import Model as Updec2Model
from updec2_eval import source_split


def sigmoid(x):
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    z = np.exp(x[~pos])
    out[~pos] = z / (1.0 + z)
    return out


class HybridModel:
    """Selective CPU inference for the archived UPFAC3 + UPRES1 model."""

    def __init__(self, encoder_path, residual_path):
        self.encoder_path = Path(encoder_path)
        self.residual_path = Path(residual_path)
        with self.encoder_path.open("rb") as f:
            e = struct.unpack("<8s12I2Q4f5Q", f.read(128))
        with self.residual_path.open("rb") as f:
            r = struct.unpack("<8s12I3Q8f8Q", f.read(176))
        if e[0] != b"UPFAC3\0\0" or e[1] != 3:
            raise ValueError("encoder is not UPFAC3")
        if r[0] != b"UPRES1\0\0" or r[1] != 1:
            raise ValueError("residual model is not UPRES1")

        self.patterns, self.grank, self.hidden, self.input_dim = \
            int(e[2]), int(e[3]), int(e[4]), int(e[5])
        self.n_active, self.n_cpg = int(e[6]), int(e[13])
        self.n_bins, self.rrank, self.n_residual = \
            int(r[4]), int(r[2]), int(r[14])
        if self.hidden != r[3] or self.n_cpg != r[13] or \
                self.n_active + self.n_residual != self.n_cpg:
            raise ValueError("hybrid model dimensions disagree")

        prep_off, active_off, eparam_off = int(e[20]), int(e[21]), int(e[22])
        prep = np.memmap(self.encoder_path, "<f4", "r", prep_off,
                         (3 * self.input_dim,))
        self.imputer = prep[:self.input_dim]
        self.mean = prep[self.input_dim:2 * self.input_dim]
        self.scale = prep[2 * self.input_dim:]
        self.top_cpg = np.memmap(self.encoder_path, "<u4", "r", active_off,
                                 (self.n_active,))
        self.top_group = np.memmap(
            self.encoder_path, "<u2", "r",
            active_off + 4 * self.n_active, (self.n_active,))

        q = eparam_off
        self.w1 = np.memmap(self.encoder_path, "<f4", "r", q,
                            (self.hidden, self.input_dim))
        q += 4 * self.hidden * self.input_dim
        self.b1 = np.memmap(self.encoder_path, "<f4", "r", q, (self.hidden,))
        q += 4 * self.hidden
        self.w2 = np.memmap(self.encoder_path, "<f4", "r", q,
                            (self.hidden, self.hidden))
        q += 4 * self.hidden * self.hidden
        self.b2 = np.memmap(self.encoder_path, "<f4", "r", q, (self.hidden,))
        q += 4 * self.hidden
        self.gw = np.memmap(self.encoder_path, "<f4", "r", q,
                            (self.patterns, self.grank, self.hidden))
        q += 4 * self.patterns * self.grank * self.hidden
        self.gb = np.memmap(self.encoder_path, "<f4", "r", q,
                            (self.patterns, self.grank))
        q += 4 * self.patterns * self.grank
        self.ge = np.memmap(self.encoder_path, "<f4", "r", q,
                            (self.n_active, self.grank))
        q += 4 * self.n_active * self.grank
        self.geb = np.memmap(self.encoder_path, "<f4", "r", q,
                             (self.n_active,))

        bin_off_off, cpg_off, rparam_off = int(r[25]), int(r[26]), int(r[28])
        self.bin_offsets = np.memmap(self.residual_path, "<u8", "r",
                                     bin_off_off, (self.n_bins + 1,))
        self.res_cpg = np.memmap(self.residual_path, "<u4", "r", cpg_off,
                                 (self.n_residual,))
        q = rparam_off
        self.ra = np.memmap(self.residual_path, "<f4", "r", q,
                            (self.n_bins, self.rrank, self.hidden))
        q += 4 * self.n_bins * self.rrank * self.hidden
        self.rab = np.memmap(self.residual_path, "<f4", "r", q,
                             (self.n_bins, self.rrank))
        q += 4 * self.n_bins * self.rrank
        self.re = np.memmap(self.residual_path, "<f4", "r", q,
                            (self.n_residual, self.rrank))
        q += 4 * self.n_residual * self.rrank
        self.reb = np.memmap(self.residual_path, "<f4", "r", q,
                             (self.n_residual,))

        # A compact genome -> head/local-row map makes selective inference cheap.
        self.is_residual = np.zeros(self.n_cpg, np.bool_)
        self.local = np.empty(self.n_cpg, np.uint32)
        self.local[self.top_cpg] = np.arange(self.n_active, dtype=np.uint32)
        self.is_residual[self.res_cpg] = True
        self.local[self.res_cpg] = np.arange(self.n_residual, dtype=np.uint32)
        self.res_bin = np.empty(self.n_residual, np.uint16)
        for b in range(self.n_bins):
            self.res_bin[self.bin_offsets[b]:self.bin_offsets[b + 1]] = b

    def representation(self, beta, count):
        missing = ~np.isfinite(beta)
        raw = np.empty(self.input_dim, np.float32)
        raw[0::3] = beta
        raw[1::3] = np.log1p(count)
        raw[2::3] = missing
        raw = np.where(np.isfinite(raw), raw, self.imputer)
        x = (raw - self.mean) / self.scale
        h1 = self.w1 @ x + self.b1
        h1 = np.where(h1 >= 0, h1, .01 * h1)
        h2 = self.w2 @ h1 + self.b2
        return h1 + np.where(h2 >= 0, h2, .01 * h2)

    def predict(self, beta, count, genomic):
        h = self.representation(beta, count)
        out = np.empty(len(genomic), np.float32)
        residual = self.is_residual[genomic]
        if np.any(~residual):
            take = np.flatnonzero(~residual)
            local = self.local[genomic[take]]
            group = np.asarray(self.top_group[local])
            for g in np.unique(group):
                gt = take[group == g]
                gl = local[group == g]
                z = self.gw[g] @ h + self.gb[g]
                out[gt] = sigmoid(self.ge[gl] @ z + self.geb[gl])
        if np.any(residual):
            take = np.flatnonzero(residual)
            local = self.local[genomic[take]]
            bins = self.res_bin[local]
            for b in np.unique(bins):
                bt = take[bins == b]
                bl = local[bins == b]
                z = self.ra[b] @ h + self.rab[b]
                z = np.where(z >= 0, z, .01 * z)
                out[bt] = sigmoid(self.re[bl] @ z + self.reb[bl])
        return out


def sample_rows(data_path, model, split, rows, targets_per_row, seed,
                include_observed):
    path = Path(data_path)
    with path.open("rb") as f:
        d = struct.unpack("<8s4IQ2I4Q", f.read(72))
    magic, _, n_cells, n_reps, side_p, n_cpg, sampled, flags, _, truth_off, \
        records_off, record_bytes = d
    if magic != b"MSURAW2\0" or not flags & 1 or \
            n_cpg != model.n_cpg or side_p < model.patterns:
        raise ValueError("sidecar dimensions disagree")
    raw = np.memmap(path, "u1", "r")
    truth = np.memmap(path, "<u2", "r", truth_off, (n_cells, n_cpg))
    _, val, test = source_split(n_cells, seed)
    cells = val if split == "validation" else \
        test if split == "test" else list(range(n_cells))
    rng = np.random.default_rng(seed + 987654321)
    sampled_rows = []
    for ri in range(rows):
        cell = int(cells[rng.integers(len(cells))])
        rep = int(rng.integers(n_reps))
        roff = records_off + (rep * n_cells + cell) * record_bytes
        beta = np.array(np.ndarray((side_p,), "<f4", raw, roff), copy=True)
        count = np.array(np.ndarray((side_p,), "<u4", raw,
                                    roff + side_p * 4), copy=True)
        observed = np.ndarray((sampled,), "<u4", raw,
                              roff + side_p * 8)
        chosen = np.empty(0, np.uint32)
        while len(chosen) < targets_per_row:
            candidate = rng.choice(n_cpg, min(n_cpg, 2 * targets_per_row),
                                   replace=False).astype(np.uint32)
            valid = np.asarray(truth[cell, candidate]) != 65535
            if not include_observed:
                at = np.searchsorted(observed, candidate)
                in_range = at < sampled
                seen = np.zeros(len(candidate), bool)
                seen[in_range] = observed[at[in_range]] == candidate[in_range]
                valid &= ~seen
            chosen = np.unique(np.concatenate((chosen, candidate[valid])))
        genomic = rng.choice(chosen, targets_per_row, replace=False).astype(
            np.uint32)
        y = np.asarray(truth[cell, genomic], np.float32) / 65534.0
        sampled_rows.append({
            "row": ri, "cell": cell, "rep": rep, "beta": beta,
            "count": count, "selected": np.array(observed, copy=True),
            "selected_truth": np.asarray(
                truth[cell, observed], np.float32) / 65534.0,
            "genomic": genomic, "truth": y,
        })
    return sampled_rows


def attach_hao_features(rows, group_path, expected_n_cpg):
    groups = np.memmap(group_path, "<u2", "r", shape=(expected_n_cpg,))
    for row in rows:
        g = np.asarray(groups[row["selected"]], dtype=np.int64)
        keep = (g > 0) & (g <= 100)
        count = np.bincount(g[keep], minlength=101)
        sums = np.bincount(g[keep], weights=row["selected_truth"][keep],
                           minlength=101)
        beta = np.full(100, np.nan, np.float32)
        present = count[1:] != 0
        beta[present] = (sums[1:][present] / count[1:][present]).astype(
            np.float32)
        row["hao_beta"] = beta


def predict_hao(rows, model_root, torch, log_every):
    """Evaluate only requested checkpoint output rows, block by block."""
    root = Path(model_root)
    n_rows = len(rows)
    n_cpg = max(int(r["genomic"].max()) for r in rows) + 1
    n_blocks = (n_cpg + 9999) // 10000
    pred = [np.empty(len(r["genomic"]), np.float32) for r in rows]
    per_block = [[] for _ in range(n_blocks)]
    for ri, row in enumerate(rows):
        blocks = row["genomic"] // 10000
        for b in np.unique(blocks):
            positions = np.flatnonzero(blocks == b)
            per_block[int(b)].append((ri, positions))

    t0 = time.monotonic()
    done = 0
    for block, requests in enumerate(per_block):
        if not requests:
            continue
        block_dir = root / f"10k{block}"
        with (block_dir / "imputer_scaler.pkl").open("rb") as f:
            prep = pickle.load(f)
        x = np.stack([r["hao_beta"] for r in rows])
        x = prep["scaler"].transform(
            prep["imputer"].transform(x)).astype(np.float32, copy=False)
        state = torch.load(block_dir / "bottleneck_weights_best.pth",
                           map_location="cpu", weights_only=True)
        xt = torch.from_numpy(x)
        h = torch.nn.functional.linear(
            xt, state["net.0.weight"], state["net.0.bias"]).relu_()
        h = torch.nn.functional.batch_norm(
            h, state["net.2.running_mean"], state["net.2.running_var"],
            state["net.2.weight"], state["net.2.bias"], training=False,
            momentum=0.1, eps=1e-5)
        ow, ob = state["net.3.weight"], state["net.3.bias"]
        for ri, positions in requests:
            local = (rows[ri]["genomic"][positions] - block * 10000).astype(
                np.int64)
            idx = torch.from_numpy(local)
            logits = (ow.index_select(0, idx) * h[ri]).sum(1) + \
                ob.index_select(0, idx)
            pred[ri][positions] = torch.sigmoid(logits).numpy()
        done += 1
        if done % log_every == 0 or block == len(per_block) - 1:
            elapsed = time.monotonic() - t0
            print(f"[hao] blocks={done}/{sum(bool(x) for x in per_block)} "
                  f"elapsed={elapsed:.1f}s", flush=True)
    return pred


def metrics(y_rows, pred_rows):
    y, p = np.concatenate(y_rows), np.concatenate(pred_rows)
    d = p - y
    return {
        "n": len(y),
        "rmse": math.sqrt(float(d @ d) / len(y)),
        "mae": float(np.abs(d).mean()),
        "pearson": float(np.corrcoef(y, p)[0, 1]),
        "within_0.05": float((np.abs(d) <= .05).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updec2", required=True)
    ap.add_argument("--hybrid-encoder", required=True)
    ap.add_argument("--hybrid-residual", required=True)
    ap.add_argument("--hao-root", required=True)
    ap.add_argument(
        "--hao-groups", required=True,
        help="uint16 CpG-to-group map decoded from Hao's original P1-P100 "
             "mask (0=PNA)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=("validation", "test", "all"),
                    default="all")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--targets-per-row", type=int, default=131072)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--include-observed", action="store_true")
    ap.add_argument("--row-tsv")
    ap.add_argument("--log-every-blocks", type=int, default=100)
    args = ap.parse_args()

    current = Updec2Model(args.updec2)
    hybrid = HybridModel(args.hybrid_encoder, args.hybrid_residual)
    if current.n_cpg != hybrid.n_cpg or \
            current.patterns != hybrid.patterns:
        raise ValueError("current and hybrid dimensions disagree")
    rows = sample_rows(args.data, current, args.split, args.rows,
                       args.targets_per_row, args.seed, args.include_observed)

    current_pred, hybrid_pred = [], []
    for i, row in enumerate(rows):
        current_pred.append(current.predict(
            row["beta"][:current.patterns],
            row["count"][:current.patterns], row["genomic"]))
        hybrid_pred.append(hybrid.predict(
            row["beta"][:hybrid.patterns],
            row["count"][:hybrid.patterns], row["genomic"]))
        print(f"[native] rows={i + 1}/{len(rows)}", flush=True)

    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "Hao evaluation requires the sturgeon PyTorch environment") from e
    torch.set_num_threads(4)
    attach_hao_features(rows, args.hao_groups, current.n_cpg)
    print("[hao] exact P1-P100 summaries reconstructed from paired sampled "
          "CpGs", flush=True)
    hao_pred = predict_hao(rows, args.hao_root, torch,
                           args.log_every_blocks)

    truth = [r["truth"] for r in rows]
    all_pred = {
        "updec2": current_pred,
        "last_hybrid": hybrid_pred,
        "hao_original": hao_pred,
    }
    print("model\tn\trmse\tmae\tpearson\twithin_0.05")
    for name, predictions in all_pred.items():
        m = metrics(truth, predictions)
        print(f"{name}\t{m['n']}\t{m['rmse']:.9g}\t{m['mae']:.9g}\t"
              f"{m['pearson']:.9g}\t{m['within_0.05']:.9g}")

    if args.row_tsv:
        names = list(all_pred)
        row_metrics = []
        for ri, row in enumerate(rows):
            values = [ri, row["cell"], row["rep"], len(row["truth"])]
            values.extend(float(np.abs(all_pred[n][ri] -
                                       row["truth"]).mean()) for n in names)
            row_metrics.append(values)
        with open(args.row_tsv, "w") as f:
            f.write("row\tcell\treplicate\tn\t" +
                    "\t".join(f"mae_{n}" for n in names) + "\n")
            for values in row_metrics:
                f.write("\t".join(map(str, values)) + "\n")

        # Paired cell/replicate-row bootstrap for every model contrast.
        rng = np.random.default_rng(args.seed + 5)
        maes = np.asarray([r[4:] for r in row_metrics])
        boot_idx = rng.integers(len(rows), size=(10000, len(rows)))
        print("contrast\tmae_delta\tbootstrap_95pct_low\t"
              "bootstrap_95pct_high")
        for ai in range(len(names)):
            for bi in range(ai + 1, len(names)):
                delta = maes[:, bi] - maes[:, ai]
                boot = delta[boot_idx].mean(1)
                lo, hi = np.quantile(boot, [.025, .975])
                print(f"{names[bi]}_minus_{names[ai]}\t"
                      f"{delta.mean():.9g}\t{lo:.9g}\t{hi:.9g}")


if __name__ == "__main__":
    main()
