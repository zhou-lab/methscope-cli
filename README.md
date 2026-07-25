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

Pretrained models are hosted on HuggingFace
([zhou-lab/methscope](https://huggingface.co/zhou-lab/methscope)) — too large for
git. `methscope fetch` carries the catalog and downloads them; the
[methscope_data](https://github.com/zhou-lab/methscope_data) repo holds the query
`.cg` test fixtures (`test/`) and the reproducibility archive.

```sh
methscope fetch                       # list the catalog + what is already local
methscope fetch hg38_celltype.ubjx    # entries are named by their file
methscope fetch models                # every model
methscope fetch data                  # every example .cg fixture
# -> $METHSCOPE_DATA_DIR, else ~/.cache/methscope (--store DIR overrides)
```

The catalog covers both the models and the query `.cg` fixtures the examples
run against. Entries are named by their file, so what you ask for is what lands in the
store. Human-facing lines go to stderr and stdout is one absolute path per
requested file, so fetching also composes when a script wants a path:

```sh
mkdir -p ~/tmp/methscope && cd ~/tmp/methscope
export METHSCOPE_DATA_DIR="."          # fetch into the working directory
methscope fetch hg38_celltype.ubjx human_hg38_celltypes.cg
methscope classify human_hg38_celltypes.cg hg38_celltype.ubjx
```

It is idempotent — the first run downloads, every run after just resolves the
path — and a `.cg`'s `.cg.idx` sibling rides along without being a second name
to remember. Every entry carries a pinned SHA-256; a download that misses it is
discarded, and `--verify` re-checks files already in the store.

`fetch` never prompts, so it is safe inside a container build or a workflow
step, and it is the only command that touches the network. Downloads land on a
`.part` sibling and are renamed only once the byte count matches the catalog.
libcurl is optional: without it the build still lists the catalog and prints the
URL to download by hand.

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

## Runnable examples

Runnable smoke tests — cell-type prediction (cross-atlas concordance),
deconvolution (self-identity and a simulated whole-body mixture), and upscaling
from ~0.1% coverage, each with fetch commands and expected outputs — are on the
**docs page: <https://zhou-lab.github.io/methscope-cli/>**. Models are fetched from
[HuggingFace](https://huggingface.co/zhou-lab/methscope); the query `.cg` fixtures
from [methscope_data](https://github.com/zhou-lab/methscope_data) (`test/`).

## Training & internals

### Train the whole-genome upscale model

`upscale-train` trains one unified whole-genome `UPDEC2` model. The top 1,000
MRMP averages are deterministic inputs. An optional learned 512-dimensional
decoder trunk can be shared by every membership-first processing unit; it is
downstream of MRMP aggregation, not a CpG-to-MRMP encoder. Beta-only,
beta-plus-missing, and beta-plus-count inputs are supported. PyTorch is not
used.

The `.mrmp` artifact is the build pipeline's currency: `upscale-featurize`,
`upscale-set-units`, and `upscale-train` all read the same one, so the sidecar's
per-CpG group map and the mask the model ships cannot drift apart. The `.cm` is
the *runtime* form — `upscale-train` materializes it into `--work-dir` and packs
it into the bundle. `mrmp-export` stays available for inspection and for
feeding the `.cm`-based commands, but it is no longer a pipeline step. (An
already-exported `.cm` is still accepted wherever a `.mrmp` is.)

```sh
$MS upscale-train \
  -i training.msur \
  --units processing_units_16k.msui \
  --mrmp zhou_major_p1000.mrmp \
  -o hg38_upscale.updecx \
  --work-dir ~/tmp/hg38_upscale_train \
  --features beta \
  --pure-bottleneck 16 \
  --mixed-bottleneck 32 \
  --activation leaky \
  --device 0
# -> one self-contained .updecx plus a training manifest
```

By default the source cells are shuffled by `--seed` and cut 70/15/15. Pass
`--split FILE` to pin the assignment instead — rows of
`<cell_index>TAB<train|val|test>`, one per cell, indexed by the sample order of
the truth `.cg` the sidecar was prepared from (a trailing sample-name column is
ignored, and one non-numeric header row is allowed). Use it when a random cut
would strand a whole cell type outside training, or to train against the exact
held-out cells an external baseline used. The split is validated before CUDA is
claimed, is recorded in the training manifest, and is folded into the checkpoint
run checksum, so one work directory cannot resume across two different splits.
Pass the same file to `_upscale trunk-train` when a frozen trunk is involved.

Build training support with
`make CUDA=1 CUDA_HOME=/path/to/cuda CUDA_ARCH=sm_80`. Each unit is checkpointed
under `--work-dir`, so interrupted runs resume completed units. The distributed
model still runs through the pure-C CPU inference path without CUDA or BLAS.

The three build steps are public commands: `upscale-featurize` (MSURAW2
sidecar), `upscale-set-units` (MSUIDX1 unit index), then `upscale-train`. Only
the research trunk trainer and the Zhou 2018 evaluator remain under
`methscope _upscale`; the latter is invoked by the
non-public `analysis/zhou2018_upscale_eval.sh` script. See the MethScope lab journal (`20251216_methscope.org`) and the
[docs page](https://zhou-lab.github.io/methscope-cli/).

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

`predict`/`build-reference`/`deconv` emit one row per query record **in query-file order**
(the MethScope R package instead sorts rows by cell name). When you supply labels
to `train`, give them in that same query-record order.

### Advanced: parity against the R package

`predict` / `build-reference --matrix` reproduce the R `PredictCellType`/`GenerateInput` outputs (the
small residual in the matrix is R's 3-decimal text rounding; C uses full
precision). See `test/parity.sh`, which needs a MethScope checkout + R.

## License

GNU Affero General Public License v3.0 or later. See `LICENSE`.
Copyright (c) 2025 Hongxiang Fu and Wanding Zhou.

Vendored: `src/nnls.c` — Lawson–Hanson NNLS (C. Lawson & R. Hanson, JPL/SIAM;
[netlib lawson-hanson](https://www.netlib.org/lawson-hanson/)), f2c-translated,
self-contained.
