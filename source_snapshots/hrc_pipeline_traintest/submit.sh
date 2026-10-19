#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

BRANCH="$(git branch --show-current)"
MSG="${1:-update: $(date '+%Y-%m-%d %H:%M:%S')}"

git add -A

if git diff --cached --quiet; then
    echo "No new changes to commit."
else
    git status --short
    git commit -m "$MSG"
fi

git push origin "$BRANCH"
echo "Pushed to origin/$BRANCH"

