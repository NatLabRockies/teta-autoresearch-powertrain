#!/usr/bin/env bash
# Create an isolated experiment tree from the autoresearch harness.
#
# Each tree is a separate git repo in its own sibling directory. The tree
# has exactly one prior commit (the scaffold) so `git log --all`,
# `git tag -l`, and `git branch -a` reveal nothing from the template's
# history or from sibling trees. Bias channels (learnings.md, prior tags,
# session branches, committed results/plans) are wiped; the framework
# (program.md, fixed_utils.py, tools/, optimizers/, templates/) and the
# chosen domain's scaffold (domain.md, domain.json, train.py) are copied.
#
# Trees are always flattened: the selected domain's files (train.py,
# domain.md, seed.md, learnings.md, data/) live at the tree root, not
# under `domains/<name>/`. This preserves the LLM agent's UX across
# domains. Optimizer trees additionally get `optimizers/` and keep the
# domain's `search/` package under `domains/<name>/search/` for imports.
#
# See tools/README-trees.md for the surrounding workflow.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
TEMPLATE_DIR="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"

NAME=""
DOMAIN=""
MODE=""
PARTITION=""
OPTIMIZER=""
SEED_MD=""
SEED_LEARNINGS=""
TARGET_DIR=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --name <tree> --domain <name> --mode <llm|optimizer> --partition <val> [options]

Create an isolated experiment tree.

Required:
  --name <tree>             Tree name (used for dir and registry entry)
  --domain <name>           Domain name (resolves to domains/<name>/)
  --mode <mode>             llm | optimizer — experiment driver
  --partition <val>         Partition value (must appear in domain.json)

Required when --mode optimizer:
  --optimizer <name>        tpe | cmaes | random — sampler for the tree

Optional:
  --seed-md <file>          File to use as seed.md (default: empty)
  --seed-learnings <file>   File to use as learnings.md (default: empty)
  --target-dir <path>       Where to create the tree
                            (default: ~/repos/routee-autoresearch-trees/<name>)
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)           NAME="$2"; shift 2;;
    --domain)         DOMAIN="$2"; shift 2;;
    --mode)           MODE="$2"; shift 2;;
    --partition)      PARTITION="$2"; shift 2;;
    --optimizer)      OPTIMIZER="$2"; shift 2;;
    --seed-md)        SEED_MD="$2"; shift 2;;
    --seed-learnings) SEED_LEARNINGS="$2"; shift 2;;
    --target-dir)     TARGET_DIR="$2"; shift 2;;
    -h|--help)        usage; exit 0;;
    *) echo "unknown flag: $1" >&2; usage; exit 1;;
  esac
done

[[ -n "$NAME" ]]      || { echo "--name is required" >&2; exit 1; }
[[ -n "$DOMAIN" ]]    || { echo "--domain is required" >&2; exit 1; }
[[ -n "$MODE" ]]      || { echo "--mode is required" >&2; exit 1; }
[[ -n "$PARTITION" ]] || { echo "--partition is required" >&2; exit 1; }

[[ "$MODE" == "llm" || "$MODE" == "optimizer" ]] \
  || { echo "--mode must be llm or optimizer" >&2; exit 1; }

if [[ "$MODE" == "optimizer" ]]; then
  [[ -n "$OPTIMIZER" ]] || { echo "--optimizer is required when --mode optimizer" >&2; exit 1; }
  [[ "$OPTIMIZER" =~ ^(tpe|cmaes|random)$ ]] \
    || { echo "--optimizer must be tpe|cmaes|random" >&2; exit 1; }
fi

DOMAIN_DIR="$TEMPLATE_DIR/domains/$DOMAIN"
DOMAIN_CONFIG="$DOMAIN_DIR/domain.json"
[[ -d "$DOMAIN_DIR" ]]    || { echo "domain not found: $DOMAIN_DIR" >&2; exit 1; }
[[ -f "$DOMAIN_CONFIG" ]] || { echo "domain.json missing: $DOMAIN_CONFIG" >&2; exit 1; }

# Validate --partition against the domain's declared values, and pick up
# the train.py constant name to stamp.
VALID_PARTITIONS="$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["partition"]["values"]))' "$DOMAIN_CONFIG")"
SELECTOR_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["partition"]["train_py_selector"])' "$DOMAIN_CONFIG")"
# shellcheck disable=SC2206
VALID_ARR=($VALID_PARTITIONS)
found=0
for v in "${VALID_ARR[@]}"; do
  [[ "$v" == "$PARTITION" ]] && found=1
done
if [[ $found -eq 0 ]]; then
  echo "--partition '$PARTITION' not in domain's declared values: ${VALID_PARTITIONS}" >&2
  exit 1
fi

if [[ -z "$TARGET_DIR" ]]; then
  TARGET_DIR="${HOME}/repos/routee-autoresearch-trees/${NAME}"
fi

if [[ -e "$TARGET_DIR" ]]; then
  echo "target already exists: $TARGET_DIR" >&2; exit 1
fi

if [[ -n "$SEED_MD" && ! -f "$SEED_MD" ]]; then
  echo "--seed-md file not found: $SEED_MD" >&2; exit 1
fi
if [[ -n "$SEED_LEARNINGS" && ! -f "$SEED_LEARNINGS" ]]; then
  echo "--seed-learnings file not found: $SEED_LEARNINGS" >&2; exit 1
fi

# Resolve seed paths to absolute before we cd around.
[[ -n "$SEED_MD" ]] && SEED_MD="$(cd "$(dirname "$SEED_MD")" && pwd)/$(basename "$SEED_MD")"
[[ -n "$SEED_LEARNINGS" ]] && SEED_LEARNINGS="$(cd "$(dirname "$SEED_LEARNINGS")" && pwd)/$(basename "$SEED_LEARNINGS")"

TEMPLATE_SHA="$(git -C "$TEMPLATE_DIR" rev-parse HEAD)"
TEMPLATE_SHA_SHORT="$(git -C "$TEMPLATE_DIR" rev-parse --short HEAD)"

mkdir -p "$TARGET_DIR"

# --- Framework files (top-level, copied as-is) ---
FRAMEWORK_FILES=(
  program.md
  fixed_utils.py
  pixi.toml
  pixi.lock
  pyproject.toml
  Dockerfile
  dprint.json
  .gitignore
  .gitattributes
  README.md
  EXTENDING.md
)
for f in "${FRAMEWORK_FILES[@]}"; do
  if [[ -f "$TEMPLATE_DIR/$f" ]]; then
    cp -p "$TEMPLATE_DIR/$f" "$TARGET_DIR/$f"
  fi
done

_copy_dir() {
  local src="$1" dst="$2"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude '__pycache__' --exclude '*.pyc' "$src/" "$dst/"
  else
    cp -R "$src" "$dst"
    find "$dst" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
    find "$dst" -type f -name '*.pyc' -delete 2>/dev/null
  fi
}

# --- Framework directories ---
for d in tools templates; do
  if [[ -d "$TEMPLATE_DIR/$d" ]]; then
    mkdir -p "$TARGET_DIR/$d"
    _copy_dir "$TEMPLATE_DIR/$d" "$TARGET_DIR/$d"
  fi
done

# --- Domain scaffold files, flattened to tree root ---
DOMAIN_FLAT_FILES=(domain.md domain.json train.py)

for f in "${DOMAIN_FLAT_FILES[@]}"; do
  [[ -f "$DOMAIN_DIR/$f" ]] || { echo "domain file missing: $DOMAIN_DIR/$f" >&2; exit 1; }
  cp -p "$DOMAIN_DIR/$f" "$TARGET_DIR/$f"
done

# Data: symlink (saves disk; data is read-only per protocol).
ln -s "$DOMAIN_DIR/data" "$TARGET_DIR/data"

# seed.md / learnings.md: wiped (or caller-seeded).
if [[ -n "$SEED_MD" ]]; then
  cp "$SEED_MD" "$TARGET_DIR/seed.md"
else
  : > "$TARGET_DIR/seed.md"
fi
if [[ -n "$SEED_LEARNINGS" ]]; then
  cp "$SEED_LEARNINGS" "$TARGET_DIR/learnings.md"
else
  : > "$TARGET_DIR/learnings.md"
fi

# Results skeleton — one subdir per declared partition.
for v in "${VALID_ARR[@]}"; do
  mkdir -p "$TARGET_DIR/results/$v"
  : > "$TARGET_DIR/results/$v/.gitkeep"
done
: > "$TARGET_DIR/results/.gitkeep"

# --- Optimizer mode: copy the optimizer framework and the domain's search package ---
if [[ "$MODE" == "optimizer" ]]; then
  mkdir -p "$TARGET_DIR/optimizers"
  _copy_dir "$TEMPLATE_DIR/optimizers" "$TARGET_DIR/optimizers"
  if [[ -d "$DOMAIN_DIR/search" ]]; then
    mkdir -p "$TARGET_DIR/domains/$DOMAIN/search"
    _copy_dir "$DOMAIN_DIR/search" "$TARGET_DIR/domains/$DOMAIN/search"
    # Package markers so `from domains.<d>.search import ...` resolves.
    : > "$TARGET_DIR/domains/__init__.py"
    : > "$TARGET_DIR/domains/$DOMAIN/__init__.py"
  else
    echo "warning: domain has no search/ package; optimizer mode may fail" >&2
  fi
fi

# --- Stamp the partition selector in train.py ---
python3 - "$TARGET_DIR/train.py" "$SELECTOR_NAME" "$PARTITION" <<'PY'
import re, pathlib, sys
path, name, val = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
src = path.read_text()
pattern = rf'^{re.escape(name)}\s*=.*$'
new, n = re.subn(
    pattern,
    f'{name} = "{val}"  # session selector — the only domain-related line to change',
    src, count=1, flags=re.M,
)
if n != 1:
    sys.exit(f"selector {name!r} not found in {path}")
path.write_text(new)
PY

# --- Tree metadata ---
NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SEED_MD_REF="${SEED_MD:-<empty>}"
SEED_LEARN_REF="${SEED_LEARNINGS:-<empty>}"

python3 - "$TARGET_DIR/.tree-meta.json" \
  "$NAME" "$TEMPLATE_SHA" "$TEMPLATE_SHA_SHORT" \
  "$DOMAIN" "$PARTITION" "$MODE" "$OPTIMIZER" \
  "$NOW_UTC" "$SEED_MD_REF" "$SEED_LEARN_REF" <<'PY'
import json, sys
(out, name, sha, sha_short, domain, partition, mode, optimizer,
 now, seed_md, seed_learn) = sys.argv[1:12]
with open(out, "w") as f:
    json.dump({
        "name": name,
        "template_sha": sha,
        "template_sha_short": sha_short,
        "domain": domain,
        "partition": partition,
        "mode": mode,
        "optimizer": optimizer or None,
        "layout": "flat",
        "created_utc": now,
        "seed_md_source": seed_md,
        "seed_learnings_source": seed_learn,
    }, f, indent=2)
    f.write("\n")
PY

cd "$TARGET_DIR"
git init -q -b main
git config advice.detachedHead false
git config advice.addIgnoredFile false
git add -A
git -c user.name="autoresearch-harness" \
    -c user.email="autoresearch-harness@local" \
    commit -q -m "initial scaffold (from template@${TEMPLATE_SHA_SHORT})"

REGISTRY_DIR="${HOME}/repos/routee-autoresearch-trees"
mkdir -p "$REGISTRY_DIR"
REGISTRY="${REGISTRY_DIR}/registry.jsonl"
python3 - "$REGISTRY" \
  "$NAME" "$TARGET_DIR" "$TEMPLATE_SHA" \
  "$DOMAIN" "$PARTITION" "$MODE" "$OPTIMIZER" \
  "$NOW_UTC" "$SEED_MD_REF" "$SEED_LEARN_REF" <<'PY'
import json, sys
(reg, name, target, sha, domain, partition, mode, optimizer,
 now, seed_md, seed_learn) = sys.argv[1:12]
with open(reg, "a") as f:
    f.write(json.dumps({
        "name": name,
        "target_dir": target,
        "template_sha": sha,
        "domain": domain,
        "partition": partition,
        "mode": mode,
        "optimizer": optimizer or None,
        "layout": "flat",
        "created_utc": now,
        "seed_md_source": seed_md,
        "seed_learnings_source": seed_learn,
    }) + "\n")
PY

cat <<EOF

created tree:
  name:       $NAME
  path:       $TARGET_DIR
  template:   ${TEMPLATE_SHA_SHORT}
  domain:     $DOMAIN
  partition:  $PARTITION
  mode:       $MODE${OPTIMIZER:+ ($OPTIMIZER)}
  seed.md:    ${SEED_MD_REF}
  learnings:  ${SEED_LEARN_REF}

registry:     ${REGISTRY}

start an agent from within the tree dir.
EOF
