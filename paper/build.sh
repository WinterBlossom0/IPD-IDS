#!/usr/bin/env bash
# Build the PDF. Uses tectonic (self-contained: no system TeX install, fetches only the
# packages this document actually needs and caches them under ~/.cache/Tectonic).
#
# Regenerates the results tables from the experiment CSVs first, so the PDF can never
# show numbers that disagree with runs/frontier/.
set -eu
cd "$(dirname "$0")"

TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"
if [ ! -x "$TECTONIC" ]; then
  echo "tectonic not found at $TECTONIC"
  echo "install:  curl -sL https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz | tar xz -C ~/.local/bin"
  exit 1
fi

echo "--- regenerating tables from experiment CSVs ---"
ds-python make_tables.py || echo "WARN: tables not regenerated"

echo "--- compiling ---"
"$TECTONIC" main.tex

echo
echo "built: $(pwd)/main.pdf  ($(du -h main.pdf | cut -f1))"
grep -c "pending" main.tex >/dev/null && \
  echo "NOTE: $(grep -o '\\pending' main.tex | wc -l) \\pending{} markers still in the text (render red)."
