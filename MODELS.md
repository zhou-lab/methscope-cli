# Publishing a model set to HuggingFace

One checklist, top to bottom. This is the model-artifact counterpart of
[`RELEASE.md`](RELEASE.md), which covers the *binary*. The two are independent:
a code release does not require a model tag, and a model tag does not require a
code release. They meet only at step 8, where YAME's registry pins the tag every
tool downloads.

Models live at **`huggingface.co/zhou-lab/methscope`**, are addressed by a tag
`vN`, and are fetched by `yame fetch` through a pin compiled into the binary —
not by a URL a user types. So publishing is not "upload a file": it is
*upload, then move the pin, then rebuild the tools that carry it*.

## the three places a model lives

| location | what it is | git? |
| -------- | ---------- | ---- |
| `/mnt/isilon/zhou_lab/projects/20260726_HuggingFace_Models/` | **staging** — what the NEXT tag will contain | no, a plain directory |
| `huggingface.co/zhou-lab/methscope` | **published** — LFS, tagged `vN` | yes, `git@hf.co:zhou-lab/methscope.git` |
| `/mnt/isilon/zhou_lab/projects/20260726_HuggingFace_Models_git/` | **pointer clone** of the published side | yes, `GIT_LFS_SKIP_SMUDGE=1` |

The pointer clone is 1.8 MB for the whole history, because an LFS pointer
already carries each file's sha256 and size. It answers offline what the
registry cache cannot — which tags exist, when each was cut, what two differ by:

```sh
G=/mnt/isilon/zhou_lab/projects/20260726_HuggingFace_Models_git
git -C $G tag
git -C $G show v8:SHA256SUMS
git -C $G ls-tree --name-only v8
git -C $G diff --stat v7 v8
```

It is a *reading* clone. Do not push model bytes from it; `git lfs pull` a blob
only to RUN an old model.

## 1. Stage the artifacts

Copy the finished models into the staging directory. Staging is the source of
truth for the next tag — whatever sits there is what gets published, so it must
be byte-identical to the artifact you validated, not a rebuild.

```sh
S=/mnt/isilon/zhou_lab/projects/20260726_HuggingFace_Models
cp <validated>.updecx "$S"/
```

**Check the diff against what is already published before going further.** A
tag is usually a pure addition; a rename or a withdrawal is a different
conversation and needs saying out loud in the README and the org.

```sh
diff <(sort "$S/SHA256SUMS") <(git -C $G show v8:SHA256SUMS | sort)
```

## 2. Regenerate `SHA256SUMS`

```sh
cd "$S" && sha256sum *.updecx *.clfx *.msdref > SHA256SUMS
```

Every published file, hashed from the bytes on disk. This file is what the
registry's manifest cache is built from, so an error here propagates into every
tool's compiled pin.

## 3. Update `README.md`

The HuggingFace README is the model card and the page a user lands on. It
carries YAML front-matter (`license: agpl-3.0`, `library_name: methscope`,
tags) followed by one section per model. A new model needs: what it is, what it
was trained on, its size, and the command that runs it. A model that changes
role — superseding another, or being withdrawn — needs that stated, because the
file list alone cannot say it.

## 4. Upload

Work in a **full** clone, not the pointer clone:

```sh
export PATH=~/software/bin:$PATH          # git-lfs 3.5.1, user-local
git clone git@hf.co:zhou-lab/methscope.git /tmp/hf-methscope   # or ~/tmp/...
cd /tmp/hf-methscope
cp "$S"/<new files> "$S"/SHA256SUMS "$S"/README.md .
git lfs track "*.updecx" "*.clfx" "*.msdref"     # if the pattern is new
git add .gitattributes <new files> && git commit -m "<file>: <one line>"
git add SHA256SUMS && git commit -m "SHA256SUMS: add <file>"
git add README.md  && git commit -m "README: <what changed>"
git push
```

Three commits, in that order — model, sums, README — is the established shape
(v8: `3bcc289`, `909bbe2`, `0aabcd7`). It makes the history readable and lets a
reviewer see the bytes land before the claims about them.

## 5. Tag

```sh
git tag vN && git push origin vN
```

The tag is what the registry pins, so it must point at the commit where
SHA256SUMS and README already describe the models — the last of the three, not
the first.

## 6. Refresh the pointer clone

```sh
git -C $G fetch --tags origin && git -C $G tag | tail -3
```

So the offline view knows the new tag exists.

## 7. Verify from the published side, not from staging

Staging and published being identical is the thing most likely to be assumed
and not true:

```sh
diff <(sort "$S/SHA256SUMS") <(git -C $G show vN:SHA256SUMS | sort) &&
  echo "staging == published vN"
```

## 8. Move the registry pin, and rebuild what carries it

The tag is invisible to users until YAME's catalog points at it. This is the
step that makes `yame fetch` serve the new models.

```sh
cd ~/repo/YAME
$EDITOR tools/registry/TAGS          # methscope_models   vN   <url unchanged>
tools/make_registry.sh --refresh --tag=vN    # re-caches the manifest; ONLY network mode
tools/make_registry.sh --tool=yame -o src/registry.h
make && ./yame fetch -l | grep -c methscope  # the new set is listed
```

`--refresh` caches the manifest under `tools/registry/sums/methscope_models/vN/`
so emission stays offline and CI regenerates byte-identically. Commit the TAGS
line, the cached manifest and the regenerated `registry.h` **together** — a pin
without its manifest is a digest nobody can check.

A pin bump reaches users through a YAME release, so it follows YAME's own
release path from there.

## 9. Log it

Add a dated entry to the RESEARCH_LOG of
`~/repo/labjournal/zhouw3/2025/20251216_methscope.org`: which files the tag
adds or changes, their sha256, the HuggingFace commits and tag, and the
registry pin. Update the staging-vs-published table in OVERVIEW, which is what
a reader consults to know what the next tag will contain.

## what not to do

- **Do not publish a model that only exists in scratch.** `~/tmp` is deletable;
  an artifact that is about to become a shipped model belongs somewhere that
  survives, and its build must be reproducible from a git-tracked driver.
- **Do not rebuild to publish.** Upload the bytes you validated. A rebuilt
  artifact is a different artifact until proven otherwise.
- **Do not push LFS content from the pointer clone.** It exists to read history
  cheaply; pushing from it defeats that.
- **Do not bump the pin before the tag is pushed.** The registry would point at
  a tag that does not resolve, and every tool that regenerates would bake it in.
