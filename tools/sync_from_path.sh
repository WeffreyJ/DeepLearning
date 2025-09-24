# 0) Move the auto-unzipped folder OUTSIDE the repo (if needed)
mkdir -p "$HOME/Downloads/assistant_updates"
mv "$HOME/Downloads/P.Projects/DeepLearning/your-dl-repoV4" \
   "$HOME/Downloads/assistant_updates/" 2>/dev/null || true

SRC="$HOME/Downloads/assistant_updates/your-dl-repoV4"
BR="chore/assistant-sync-$(date +%Y%m%d)"
SYNC_DIR="../ydl-sync"

# 1) Ensure main is clean and up to date
git checkout main
git pull --rebase
git add -A && git commit -m "wip: save local changes" || true
git push || true

# 2) Create a new worktree with a NEW branch that starts from origin/main
git fetch origin
git worktree add "$SYNC_DIR" -b "$BR" origin/main

# 3) Mirror the update into the worktree (and delete files that were removed in update)
rsync -a --delete "$SRC"/ "$SYNC_DIR"/ \
  --exclude '.git/' --exclude '.venv/' \
  --exclude 'data/' --exclude 'saved_models/' --exclude '.ipynb_checkpoints/'

# 4) Commit, push, PR
cd "$SYNC_DIR"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q

git add -A
git commit -m "chore(sync): apply assistant update (V4)"
git push -u origin "$BR"
# → Open PR from $BR to main
