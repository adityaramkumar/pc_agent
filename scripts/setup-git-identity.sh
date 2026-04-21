#!/usr/bin/env bash
# Scope this repo's git identity to adityaramkumar <ramkumar.aditya@gmail.com>.
# Writes only to .git/config (never --global), so your ~/.gitconfig is untouched.
# Also activates the in-tree hooks under .githooks/ via core.hooksPath.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

git config --local user.name  "adityaramkumar"
git config --local user.email "ramkumar.aditya@gmail.com"
git config --local core.hooksPath .githooks

echo "Identity scoped to this repo:"
echo "  user.name  = $(git config --local --get user.name)"
echo "  user.email = $(git config --local --get user.email)"
echo "  hooksPath  = $(git config --local --get core.hooksPath)"
