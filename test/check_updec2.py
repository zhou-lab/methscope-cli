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
    order = [pna, p0, psmall, p1, pna, p0, pna, psmall, p1, pna, p0, pna]
    yame = Path(exe).parent / "YAME" / "yame"
    with tempfile.TemporaryDirectory(prefix="methscope-updec2-") as td:
        d = Path(td)
        # upscale-set-units reads the reference STORE (the .mrmp and text modes
        # were retired 2026-08-22): build a 35-sample fmt3 .cg whose binstring
        # map is exactly `order`. Trit 1 -> M=1,U=0; 0 -> M=0,U=1; 2 -> M=U=0
        # (below --mincov, which is what makes the all-2 rows PNA).
        mu = {"1": "1\t0", "0": "0\t1", "2": "0\t0"}
        with (d / "ref.cg").open("wb") as ref:
            for k in range(35):
                txt = d / f"s{k:02d}.txt"
                txt.write_text("".join(mu[p[k]] + "\n" for p in order))
                one = d / f"s{k:02d}.cg"
                run(str(yame), "pack", "-f", "m", str(txt), str(one))
                ref.write(one.read_bytes())
        (d / "names.txt").write_text("".join(f"c{k:02d}\n" for k in range(35)))
        run(str(yame), "index", "-s", str(d / "names.txt"), str(d / "ref.cg"))
        idx = d / "units.msui"
        run(exe, "upscale-set-units", "--unit-cpgs", "4",
            str(d / "ref.cg"), str(idx))
        # the retired modes must FAIL, and name why
        run(exe, "upscale-set-units", "--binstrings", str(d / "names.txt"),
            "--pattern-counts", str(d / "names.txt"), str(idx), ok=False)
        run(exe, "upscale-set-units", str(d / "anything.mrmp"), str(idx),
            ok=False)
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
        # Version 3 PARTIAL (flag 32): one unit at output offset 1 of 3, so the
        # units do NOT tile the genome. The reader must accept it, and the CpGs
        # outside the unit must come back NA (-1), not stale memory -- this is
        # the check that catches a missing prob[] init. (A unit record is 40
        # bytes -- 3Q 2I 2H I -- hence coff = uoff + 40 per unit.)
        head = [1, 0, 1, 0]
        mean_off, scale_off, uoff = 128, 132, 136
        coff = uoff + 40
        moff = poff = coff + 4 * 3
        nbytes = poff + 4 * len(head)
        mh = H.pack(b"UPDEC2\0\0", 3, 8 | 1 | 32, 1, 1, 1, 0, 4, 1, 3,
                    mean_off, scale_off, uoff, coff, moff, poff, nbytes,
                    0, 0, 0)
        uu = U.pack(1, poff, 4 * len(head), 1, 1, 1, 1, 1)
        part = d / "partial.updec2"
        with part.open("wb") as f:
            f.write(mh)
            f.write(struct.pack("<f", .5))
            f.write(struct.pack("<f", .25))
            f.write(uu)
            f.write(struct.pack("<3I", 2, 0, 1))  # covered genomic CpG = cpg[1] = 0
            f.write(struct.pack("<4f", *head))
        got = [float(x) for x in run(exe, "upscale", "--probs", str(part), "-",
                                     input_text=".75\n").stdout.split()]
        assert len(got) == 3
        assert abs(got[0] - 1 / (1 + math.exp(-1))) < 1e-5
        assert got[1] == -1.0 and got[2] == -1.0  # NA outside the unit

        # Without the PARTIAL flag the same gap must still be rejected...
        nopart = bytearray(part.read_bytes())
        struct.pack_into("<I", nopart, 12, 8 | 1)
        (d / "nopart.updec2").write_bytes(bytes(nopart))
        run(exe, "upscale", "--probs", str(d / "nopart.updec2"), "-",
            input_text=".75\n", ok=False)
        # ...and PARTIAL units may not OVERLAP: [0,2) then [1,2).
        coff2 = uoff + 80
        moff2 = poff2 = coff2 + 4 * 3
        par_a = [1, 0, 1, 1, 0, 0]           # rank-1 factor over 2 CpGs
        nbytes2 = poff2 + 4 * (len(par_a) + len(head))
        mh2 = H.pack(b"UPDEC2\0\0", 3, 8 | 1 | 32, 1, 1, 2, 0, 4, 1, 3,
                     mean_off, scale_off, uoff, coff2, moff2, poff2, nbytes2,
                     0, 0, 0)
        u_a = U.pack(0, poff2, 4 * len(par_a), 2, 1, 1, 1, 1)
        u_b = U.pack(1, poff2 + 4 * len(par_a), 4 * len(head), 1, 1, 1, 1, 1)
        with (d / "overlap.updec2").open("wb") as f:
            f.write(mh2)
            f.write(struct.pack("<f", .5))
            f.write(struct.pack("<f", .25))
            f.write(u_a + u_b)
            f.write(struct.pack("<3I", 2, 0, 1))
            f.write(struct.pack("<6f", *par_a))
            f.write(struct.pack("<4f", *head))
        run(exe, "upscale", "--probs", str(d / "overlap.updec2"), "-",
            input_text=".75\n", ok=False)

    print("UPDEC2 checks passed")


if __name__ == "__main__":
    main()
