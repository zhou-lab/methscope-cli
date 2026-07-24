# methscope-cli

Pure-C command-line tool for ultra-fast analysis of sparse DNA methylomes via
Most Recurrent Methylation Pattern (MRMP) encoding. methscope-cli is the
command-line counterpart of the [MethScope](https://github.com/zhou-lab/MethScope)
R package: it performs the headline path — query `.cg` + MRMP reference
→ cell×pattern feature matrix → XGBoost cell-type prediction / NNLS
deconvolution — with no R runtime.

It builds [YAME](https://github.com/zhou-lab/YAME) as a static library
(`libyame.a`) for all `.cg/.cm` I/O and the `summary` computation, and links
`libxgboost` for inference.

## Models

**Pretrained models live in the [methscope_data](https://github.com/zhou-lab/methscope_data)
repo (`models/`) — see [its README](https://github.com/zhou-lab/methscope_data/blob/main/README.md)
for the catalog (files, genome, labels, framework, source MRMP).** This section
defines the bundle *formats*.

## Build

methscope-cli depends on YAME (vendored as a git submodule) and on `libxgboost`
(from conda-forge — the one external dependency).

```sh
# 1. clone with the YAME submodule
git clone --recurse-submodules git@github.com:zhou-lab/methscope-cli.git
cd methscope-cli

# 2. libxgboost (provides c_api.h + libxgboost.{so,dylib})
conda create -n methscope -c conda-forge libxgboost
conda activate methscope        # sets CONDA_PREFIX

# 3. build (links libyame.a + libxgboost)
make                             # or: make XGB_PREFIX=/path/to/env
```

The binary records an rpath to `$XGB_PREFIX/lib`, so at runtime the conda env
that provided `libxgboost` must be on the library path (activating it is enough).

## Testing

The **models and the query `.cg` fixtures are both downloaded from the
[methscope_data](https://github.com/zhou-lab/methscope_data) repo** (`models/` and
`test/`). Ensure the conda env that provided `libxgboost` is active (or the binary's
rpath points at it). Each test featurizes against the whole-genome human MRMP, so
allow ~1–2 min per test.

```sh
MS=~/repo/methscope-cli/methscope   # the built binary  (`yame` must also be on PATH)

mkdir -p ~/tmp/methscope-test && cd ~/tmp/methscope-test # scratch: everything below is fetched here

# fetch models (~80 MB) + query .cg fixtures (~12 MB) from methscope_data
MD=https://raw.githubusercontent.com/zhou-lab/methscope_data/main
wget -q $MD/models/hg38_celltype.ubjx     # 62-cell-type human classifier (xgboost)
wget -q $MD/models/hg38_65celltypes.refx  # 65-cell-type whole-body deconvolution reference
wget -q $MD/models/hg38_10k1.updecx       # block-10k1 upscale decoder
wget -q $MD/test/human_hg38_celltypes.cg $MD/test/human_hg38_celltypes.cg.idx  # 4 typed cells
wget -q $MD/test/human_hg38_immune_mixture.cg     # simulated 70% macrophage / 30% monocyte
wget -q $MD/test/human_hg38_test.cg $MD/test/human_hg38_test.truth.cg  # upscale input + truth
CT=$PWD/hg38_celltype.ubjx
DX=$PWD/hg38_65celltypes.refx
UP=$PWD/hg38_10k1.updecx
```

Each model bundle carries its own MRMP, and any subcommand that wants a
`<ref.mrmp>` also accepts a bundle in that slot — so the tests below feed the
`.ubjx` directly, no `unbundle` needed. `methscope inspect $CT` shows its
framework mark and labels.

### Test 1 — cell-type prediction (cross-atlas concordance)

The query is four Loyfer 2023 WGBS cells of known type; `predict` runs the human
Zhou2025 classifier bundled with its MRMP (`.ubjx`). Each call names the correct
cell *biology* even though the two atlases label cells differently:

```sh
$MS predict human_hg38_celltypes.cg $CT
# cell             prediction_label  confidence
# Oligodendrocyte  ODC               0.915134   # oligodendrocyte
# Pancreas-Beta    Beta              0.930708   # pancreatic islet beta cell
# Blood-NK         NK CD16           0.812749   # natural killer cell
# Blood-Monocytes  Mono              0.835970   # monocyte
# cell name = Loyfer ground truth, prediction = Zhou2025 label  ->  4/4 concordant
```

### Test 2 — deconvolution (self-identity)

Treat each of the four cells as its own "cell type": build a `.refx` reference
(signature + MRMP, self-contained) with `matrix --refx`, then each cell must
deconvolve to itself. Show each cell's proportions as percentages, dropping 0%:

```sh
$MS matrix --refx -o self.refx human_hg38_celltypes.cg $CT   # signature + MRMP
$MS deconv human_hg38_celltypes.cg self.refx \
  | awk 'NR==1{for(j=2;j<=NF;j++)t[j]=$j; next}
         {printf "%-16s", $1; for(j=2;j<=NF;j++) if($j+0>0) printf " %s=%.0f%%", t[j], $j*100; print ""}'
# Oligodendrocyte  Oligodendrocyte=100%
# Pancreas-Beta    Pancreas-Beta=100%
# Blood-NK         Blood-NK=100%
# Blood-Monocytes  Blood-Monocytes=100%
```

For a real deconvolution, build the reference from per-cell-type pseudobulks
(`yame subset -l ids.txt x.cg | yame rowop -o musum -` per type) and run
`matrix --refx -o panel.refx pseudobulks.cg ref.mrmp`.

### Test 3 — cell-type deconvolution (simulated whole-body mixture)

`human_hg38_immune_mixture.cg` (fetched above) is a simulated methylome — pooled single cells
of **70% macrophage + 30% monocyte** (downsampled). Deconvolving it against the
shipped 65-cell-type whole-body reference `hg38_65celltypes.refx` (58 Zhou
single-cell + 7 Loyfer organ/blood types: liver, kidney tubular, kidney
podocyte, adipose, neutrophil, erythroid, thyroid) recovers that composition — the other ~63 cell types get ~0:

```sh
$MS deconv human_hg38_immune_mixture.cg $DX \
  | awk 'NR==1{for(j=2;j<=NF;j++)t[j]=$j; next}
         {printf "%-8s", $1; for(j=2;j<=NF;j++) if($j+0>0.02) printf " %s=%.0f%%", t[j], $j*100; print ""}'
# 1        Macrophage=70% Mono=29%       (truth: Macrophage 70%, Mono 30%)
```

`methscope inspect $DX` prints `kind refx` and the full 65-cell-type list. The
reference uses a deterministic (reproducible) MRMP; build recipe in the MethScope
lab journal (`20251216_methscope.org`, `hg38_65celltypes.refx`).

### Test 4 — upscaling: recover dense methylation from sparse input

A trained "upscale" block decoder reconstructs CpG-level methylation for one
10k-CpG block from a sparse measurement:

| file | what it is |
|------|-----------|
| `$UP` (`hg38_10k1.updecx`, downloaded above) | block-10k1 decoder + its MRMP + the imputed-CpG locations (`bundle -m mrmp100.cm -O outcpg.cm …`) |
| `human_hg38_test.cg` | one hg38 cell, **downsampled to ~0.1% of CpGs** (`yame subset` one sample, then `yame dsample`) — the sparse input |
| `human_hg38_test.truth.cg` | the same cell **un-downsampled** (ground truth) |

`upscale` featurizes the sparse `.cg` against the bundled MRMP, runs the decoder,
and — because the bundle carries the output-CpG locations (`outcpg.cm`) — writes a
**whole-genome `.cg`** (format 6): the block's 10 000 CpGs called, the rest NA. So
all three (truth, input, reconstruction) are whole-genome `.cg`s and line up.

```sh
# ($MS, $UP and the .cg fixtures are from the setup block above; `yame` on PATH)

# upscale the 0.1% input -> whole-genome .cg (~1-2 min; genome-wide featurization)
$MS upscale -o recon.cg $UP human_hg38_test.cg
# -> stderr: [methscope] upscaled 1 sample(s) x 29401795 CpGs (genome .cg)

# accuracy: block-10k1 calls vs ground truth (skip NA = 2)
paste <(yame rowsub -I 1_10000 recon.cg                  | yame unpack -a - | cut -f4) \
      <(yame rowsub -I 1_10000 human_hg38_test.truth.cg | yame unpack -a - | cut -f4) \
  | awk '$2!=2{t++; if($1==$2)c++} END{printf "Test 4 upscale accuracy: %d/%d (%.1f%%)\n",c,t,100*c/t}'
# expected: Test 4 upscale accuracy: 9211/9794 (94.0%)
```

So from a measurement covering ~0.1% of CpGs, the decoder reconstructs block-10k1
methylation at **94%** accuracy.

### Train the whole-genome upscale model

`upscale-train` trains one unified whole-genome `UPDEC2` model. The top 1,000
MRMP averages are deterministic inputs. An optional learned 512-dimensional
decoder trunk can be shared by every membership-first processing unit; it is
downstream of MRMP aggregation, not a CpG-to-MRMP encoder. Beta-only,
beta-plus-missing, and beta-plus-count inputs are supported. PyTorch is not
used.

```sh
$MS upscale-train \
  -i training.msur \
  --index whole_genome_membership_16k.msui \
  --mrmp mrmp1000.cm \
  -o hg38_upscale.updecx \
  --work-dir ~/tmp/hg38_upscale_train \
  --features beta \
  --pure-bottleneck 16 \
  --mixed-bottleneck 32 \
  --activation leaky \
  --device 0
# -> one self-contained .updecx plus a training manifest
```

Build training support with
`make CUDA=1 CUDA_HOME=/path/to/cuda CUDA_ARCH=sm_80`. Each unit is checkpointed
under `--work-dir`, so interrupted runs resume completed units. The distributed
model still runs through the pure-C CPU inference path without CUDA or BLAS.

Preparation and reference-index construction remain experimental internal
tools under `methscope _upscale`; the Zhou 2018 evaluator is invoked by the
non-public `analysis/zhou2018_upscale_eval.sh` script. See
[`docs/upscale-train.md`](docs/upscale-train.md) and
[`docs/upscale-unified-updec2.md`](docs/upscale-unified-updec2.md).

**Visualize it.** Because the tracks are all whole-genome `.cg`, just stack them,
slice a 50-CpG window with one `rowsub -B <beg0>_<end1>` (genome rows; block 10k1
starts at row 10000, so 11625–11674 is a window inside it with two observed input
CpGs), and let `yame hprint` colour the calls (`1`=methylated, `0`=unmethylated,
`2`=NA — colour on by default in a recent YAME; pass `-c` to disable):

```sh
cat human_hg38_test.truth.cg human_hg38_test.cg recon.cg \
  | yame rowsub -B 11625_11675 - | yame hprint -
# truth  10101111111111111101111000000100000000000000000000    dense 0/1
# input  22222222222212222222222222222222222222202222222222    2 = NA; only 2 CpGs observed
# recon  10101111111111111101111000000100000000000000000000    matches truth
```

From two observed CpGs in this window, the reconstruction matches the truth
position-for-position. `--probs` emits per-CpG probabilities as TSV instead of a
`.cg`. (Rebuild the bundle: `export_upscale_model.py … -o 10k1.updec`, then
`bundle -m mrmp100.cm -O outcpg.cm -o 10k1.updecx 10k1.updec`, where
`outcpg.cm` is a genome-wide YAME mask marking the block's CpGs.)

No-download smoke (self-contained, no torch): build a tiny toy `.updec`
(`n_in=3, n_hidden=2, n_out=4`, identity preprocessing/BatchNorm) and run it:

```sh
python3 - <<'PY'
import struct, array
def f32(a): return array.array('f', a).tobytes()
with open("toy.updec","wb") as f:
    f.write(b"UPDEC1\x00\x00")
    f.write(struct.pack("<iii", 3, 2, 4)); f.write(struct.pack("<f", 1e-5))
    f.write(f32([0,0,0])); f.write(f32([0,0,0])); f.write(f32([1,1,1]))  # identity pre
    f.write(f32([1,0,0, 0,1,0])); f.write(f32([0,0]))                    # W1 -> h=[x1,x2]
    f.write(f32([1,1])); f.write(f32([0,0])); f.write(f32([0,0])); f.write(f32([1,1]))  # BN id
    f.write(f32([1,0, 0,1, 1,1, -1,0])); f.write(f32([0,0,0,0]))         # W2, b2
PY
printf 'feat_1\tfeat_2\tfeat_3\n2\t-1\t5\nNA\t0\t1\n' > toy_feats.tsv
$MS upscale --probs toy.updec toy_feats.tsv
# 0.880796  0.5  0.880796  0.119204    # row1: h=[relu(2),relu(-1)]=[2,0]
# 0.5       0.5  0.5       0.5          # row2: NA imputed to 0 -> all sigmoid(0)
```

### Note on row order

`predict`/`matrix`/`deconv` emit one row per query record **in query-file order**
(the MethScope R package instead sorts rows by cell name). When you supply labels
to `train`, give them in that same query-record order.

### Advanced: parity against the R package

`predict`/`matrix` reproduce the R `PredictCellType`/`GenerateInput` outputs (the
small residual in the matrix is R's 3-decimal text rounding; C uses full
precision). See `test/parity.sh`, which needs a MethScope checkout + R.

## License

GNU Affero General Public License v3.0 or later. See `LICENSE`.
Copyright (c) 2025 Hongxiang Fu and Wanding Zhou.

Vendored: `src/nnls.c` — Lawson–Hanson NNLS (C. Lawson & R. Hanson, JPL/SIAM;
[netlib lawson-hanson](https://www.netlib.org/lawson-hanson/)), f2c-translated,
self-contained.
