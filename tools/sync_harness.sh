#!/usr/bin/env bash
# Sync framework + optimizer files from the template into an existing tree.
#
# Use when a harness fix (program.md, fixed_utils.py, an optimizer bug fix,
# pixi config, etc.) must reach a live tree mid-experiment. Writes harness
# files from the template's HEAD into the tree and commits them as a
# single "harness: sync" commit.
#
# Mirrors the flattening rule used by new_experiment_tree.sh: the tree's
# domain files live at the tree root, so `domains/<d>/domain.md` in the
# template syncs to `<tree>/domain.md`. The domain's `search/` package
# stays under its full subpath (only present in optimizer-mode trees).
#
# Never synced (tree-owned state): train.py, learnings.md, seed.md,
# results/, plans/, data/, .tree-meta.json.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
TEMPLATE_DIR="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"

TREE=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --tree <path>

Sync framework + optimizer files from the template ($TEMPLATE_DIR) into an
existing tree. Reads the tree's .tree-meta.json to learn the domain and
syncs that domain's framework files too.

  --tree <path>   Path to the tree to sync (required)
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tree) TREE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown flag: $1" >&2; usage; exit 1;;
  esac
done

[[ -n "$TREE" && -d "$TREE/.git" ]] \
  || { echo "--tree must point at a git repo" >&2; exit 1; }

META="$TREE/.tree-meta.json"
[[ -f "$META" ]] || { echo "missing .tree-meta.json in $TREE" >&2; exit 1; }

read -r DOMAIN MODE < <(python3 - "$META" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
print(f"{meta['domain']} {meta['mode']}")
PY
)
[[ -n "$DOMAIN" ]] || { echo "could not read domain from $META" >&2; exit 1; }

DOMAIN_DIR="$TEMPLATE_DIR/domains/$DOMAIN"
[[ -d "$DOMAIN_DIR" ]] || { echo "domain not found in template: $DOMAIN_DIR" >&2; exit 1; }

TEMPLATE_SHA_SHORT="$(git -C "$TEMPLATE_DIR" rev-parse --short HEAD)"

# --- Framework files (top-level) ---
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
    mkdir -p "$TREE/$(dirname "$f")"
    cp -p "$TEMPLATE_DIR/$f" "$TREE/$f"
  fi
done

_sync_dir() {
  local src="$1" dst="$2"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' "$src/" "$dst/"
  else
    rm -rf "$dst"
    cp -R "$src" "$dst"
    find "$dst" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
    find "$dst" -type f -name '*.pyc' -delete 2>/dev/null
  fi
}

# --- Framework directories ---
for d in tools templates; do
  if [[ -d "$TEMPLATE_DIR/$d" ]]; then
    mkdir -p "$TREE/$d"
    _sync_dir "$TEMPLATE_DIR/$d" "$TREE/$d"
  fi
done

# --- Domain scaffold files: flatten to tree root, matching new-tree layout ---
DOMAIN_FLAT_FILES=(domain.md domain.json)
for f in "${DOMAIN_FLAT_FILES[@]}"; do
  if [[ -f "$DOMAIN_DIR/$f" ]]; then
    cp -p "$DOMAIN_DIR/$f" "$TREE/$f"
  fi
done

# --- Optimizer framework + domain search package (optimizer-mode trees only) ---
if [[ "$MODE" == "optimizer" ]]; then
  if [[ -d "$TEMPLATE_DIR/optimizers" ]]; then
    mkdir -p "$TREE/optimizers"
    _sync_dir "$TEMPLATE_DIR/optimizers" "$TREE/optimizers"
  fi
  if [[ -d "$DOMAIN_DIR/search" ]]; then
    mkdir -p "$TREE/domains/$DOMAIN/search"
    : > "$TREE/domains/__init__.py"
    : > "$TREE/domains/$DOMAIN/__init__.py"
    _sync_dir "$DOMAIN_DIR/search" "$TREE/domains/$DOMAIN/search"
  fi
fi

cd "$TREE"
if git diff --quiet && git diff --cached --quiet; then
  echo "no harness changes to sync; tree is already at template@${TEMPLATE_SHA_SHORT}"
  exit 0
fi

git add -A
git -c user.name="autoresearch-harness" \
    -c user.email="autoresearch-harness@local" \
    commit -q -m "harness: sync from template@${TEMPLATE_SHA_SHORT}"

echo "synced harness into $TREE at template@${TEMPLATE_SHA_SHORT}"
