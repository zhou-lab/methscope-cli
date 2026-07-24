#!/usr/bin/env python3
"""Synthetic MSUIDX1 layout and UPDEC2 numerical/validation checks."""
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*args, input_text=None, ok=True):
    p = subprocess.run(args, input=input_text, text=True, capture_output=True)
    if ok and p.returncode:
        raise AssertionError(f"{args} failed:\n{p.stderr}")
    if not ok and not p.returncode:
        raise AssertionError(f"{args} unexpectedly succeeded")
    return p


def main():
    exe = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "./methscope")
    p0, p1, pna, psmall = "0" * 35, "1" * 35, "2" * 35, "0" * 34 + "1"
    counts = [(3, p0), (2, p1), (5, pna), (2, psmall)]
    order = [pna, p0, psmall, p1, pna, p0, pna, psmall, p1, pna, p0, pna]
    with tempfile.TemporaryDirectory(prefix="methscope-updec2-") as td:
        d = Path(td)
        (d / "counts").write_text("".join(f"{n} {p}\n" for n, p in counts))
        (d / "binstrings").write_text("".join(p + "\n" for p in order))
        idx = d / "units.msui"
        run(exe, "_upscale", "index", "--binstrings", str(d / "binstrings"),
            "--pattern-counts", str(d / "counts"), "--unit-cpgs", "4",
            "-o", str(idx))
        raw = idx.read_bytes()
        h = struct.unpack_from("<8s8I11Q", raw)
        assert h[0] == b"MSUIDX1\0" and h[1] == 1
        assert h[5:8] == (4, 3, 2)  # units, real memberships, PNA units
        assert h[9:12] == (12, 7, 5)
        unit_off, cpg_off, mem_off, file_bytes = h[12:16]
        assert file_bytes == len(raw)
        units = [struct.unpack_from("<Q4I", raw, unit_off + 24 * i)
                 for i in range(4)]
        assert [u[3] for u in units] == [3, 4, 4, 1]
        assert [u[4] for u in units] == [1, 0, 2, 2]
        cpg = struct.unpack_from("<12I", raw, cpg_off)
        assert sorted(cpg) == list(range(12))
        assert cpg[7:] == (0, 4, 6, 9, 11)  # PNA last, genomic order
        members = [struct.unpack_from("<2Q2I", raw, mem_off + 24 * i)
                   for i in range(3)]
        assert [m[2] for m in members] == [3, 2, 2]

        # Two public betas -> four numeric inputs. Unit 0 is factorized/rank 1;
        # unit 1 is direct. Output map deliberately exercises genomic scatter.
        H = struct.Struct("<8s8I11Q")
        U = struct.Struct("<3Q2I2HI")
        mean_off, scale_off, uoff, coff, moff, poff = 128, 136, 144, 224, 236, 236
        par0 = [1, 0, 0, 0, 0, 1, -1, 0, 0]
        par1 = [0, 0, 0, 1, 0]
        nbytes = poff + 4 * (len(par0) + len(par1))
        mh = H.pack(b"UPDEC2\0\0", 2, 1, 2, 4, 2, 0, 4, 0, 3,
                    mean_off, scale_off, uoff, coff, moff, poff, nbytes,
                    0, 0, 0)
        u0 = U.pack(0, poff, 4 * len(par0), 2, 1, 1, 1, 1)
        u1 = U.pack(2, poff + 4 * len(par0), 4 * len(par1), 1, 1, 0, 0, 2)
        model = d / "model.updec2"
        with model.open("wb") as f:
            f.write(mh)
            f.write(struct.pack("<2f", .5, .25))
            f.write(struct.pack("<2f", .25, .25))
            f.write(u0 + u1)
            f.write(struct.pack("<3I", 2, 0, 1))
            f.write(struct.pack(f"<{len(par0) + len(par1)}f", *(par0 + par1)))
        out = run(exe, "upscale", "--probs", str(model), "-",
                  input_text="0.75\tNaN\n").stdout
        got = [float(x) for x in out.split()]
        want = [1 / (1 + math.exp(1)), 1 / (1 + math.exp(-1)),
                1 / (1 + math.exp(-1))]
        assert max(abs(a - b) for a, b in zip(got, want)) < 1e-5

        # Duplicate genomic output coordinates must be rejected.
        bad = bytearray(model.read_bytes())
        struct.pack_into("<3I", bad, coff, 2, 0, 0)
        (d / "bad.updec2").write_bytes(bad)
        run(exe, "upscale", "--probs", str(d / "bad.updec2"), "-",
            input_text="0.75\tNaN\n", ok=False)

        # Version 3: beta+log1p(count), frozen two-layer residual trunk, then
        # a rank-1 unit head. Bare count-model input is P betas followed by
        # P integer counts. Here x=[1, log(5)-1], the identity trunk preserves
        # it, and the expected logit is log(5).
        mean_off, scale_off, uoff, coff, moff, poff = 128, 136, 144, 184, 188, 188
        trunk = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        head = [1, 1, 0, 1, 0]
        nbytes = poff + 4 * (len(trunk) + len(head))
        mh = H.pack(b"UPDEC2\0\0", 3, 7, 1, 2, 1, 0, 4, 1, 1,
                    mean_off, scale_off, uoff, coff, moff, poff, nbytes,
                    0, 0, 2)
        uu = U.pack(0, poff + 4 * len(trunk), 4 * len(head),
                    1, 1, 1, 1, 1)
        v3 = d / "count-trunk.updec2"
        with v3.open("wb") as f:
            f.write(mh)
            f.write(struct.pack("<2f", .5, 1))
            f.write(struct.pack("<2f", .5, 1))
            f.write(uu)
            f.write(struct.pack("<I", 0))
            f.write(struct.pack(f"<{len(trunk) + len(head)}f",
                                *(trunk + head)))
        got = float(run(exe, "upscale", "--probs", str(v3), "-",
                        input_text="1\t4\n").stdout)
        assert abs(got - 5 / 6) < 1e-5

        # Version 3 beta-only input and mean-imputation.
        mean_off, scale_off, uoff, coff, moff, poff = 128, 132, 136, 176, 180, 180
        head = [1, 0, 1, 0]
        nbytes = poff + 4 * len(head)
        mh = H.pack(b"UPDEC2\0\0", 3, 9, 1, 1, 1, 0, 4, 1, 1,
                    mean_off, scale_off, uoff, coff, moff, poff, nbytes,
                    0, 0, 0)
        uu = U.pack(0, poff, 4 * len(head), 1, 1, 1, 1, 1)
        beta_only = d / "beta-only.updec2"
        with beta_only.open("wb") as f:
            f.write(mh)
            f.write(struct.pack("<f", .5))
            f.write(struct.pack("<f", .25))
            f.write(uu)
            f.write(struct.pack("<I", 0))
            f.write(struct.pack("<4f", *head))
        got = float(run(exe, "upscale", "--probs", str(beta_only), "-",
                        input_text=".75\n").stdout)
        assert abs(got - 1 / (1 + math.exp(-1))) < 1e-5
    print("UPDEC2 checks passed")


if __name__ == "__main__":
    main()
