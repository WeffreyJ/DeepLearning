#!/usr/bin/env bash
set -euo pipefail
SRC_PATH="${1:?Usage: tools/sync_from_path.sh </path/to/zip-or-folder>}"

BR="chore/assistant-sync-$(date +%Y%m%d)"
SYNC_DIR="../ydl-sync"
TMP="/tmp/assistant_zip"

git checkout main
git add -A && git commit -m "wip: save local changes" || true
git push || true
git checkout -b "$BR" || git checkout "$BR"
git worktree add "$SYNC_DIR" "$BR" || true

if [ -d "$SRC_PATH" ]; then
  SRC_DIR="$SRC_PATH"
else
  rm -rf "$TMP"
  mkdir -p "$TMP"
  unzip -q "$SRC_PATH" -d "$TMP"
  SRC_DIR="$(find "$TMP" -maxdepth 1 -type d -name 'your-dl-repo*' | head -n1)"
fi

rsync -a --delete "$SRC_DIR"/ "$SYNC_DIR"/ \
  --exclude '.git/' --exclude '.venv/' --exclude 'data/' \
  --exclude 'saved_models/' --exclude '.ipynb_checkpoints/'

cd "$SYNC_DIR"
git add -A
git commit -m "chore(sync): apply assistant update"
git push -u origin "$BR"
echo "Done. Open a PR from $BR."
