#!/usr/bin/env bash
# Copy this suite into an existing /home/tahiti/MultiFairGDT project.
# This script deliberately does not install packages and does not modify raw data.

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${1:-/home/tahiti/MultiFairGDT}"

mkdir -p "$DST/multifair" "$DST/scripts"

cp -f "$SRC_DIR"/multifair/*.py "$DST/multifair/"
cp -f "$SRC_DIR"/scripts/journal_experiments.py "$DST/scripts/"
cp -f "$SRC_DIR"/requirements-core.txt "$DST/"
cp -f "$SRC_DIR"/requirements-deep.txt "$DST/"
cp -f "$SRC_DIR"/README_RUNS.md "$DST/"

echo "Installed experiment suite into $DST"
echo "No raw/processed data were modified."
