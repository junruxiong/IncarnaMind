#!/bin/sh
# Prepares a work folder outside the repository for the #288 spike: a Rust toolchain
# (rustup with --no-modify-path, so no shell profile or global config changes), the CSL
# styles and locales, and the citeproc.js that Zotero ships, all at pinned commits.
#
# citeproc.js is CPAL-1.0 or AGPL-3.0. It is only a test oracle standing in for Zotero's
# output; it lives in the work folder and must never be copied into this repository.
#
# usage: WORK=/tmp/csl-spike sh setup.sh
set -eu
WORK=${WORK:?set WORK to an empty folder outside the repository}
STYLES_COMMIT=f2c83bb7374107239366dd168b616c4e75ffa36a   # citation-style-language/styles, 2026-10-09
LOCALES_COMMIT=a89adece41013402236e2c9020972d7e931fbab8  # citation-style-language/locales, 2026-09-10
ZOTERO_COMMIT=08880f7a0e51021cc13b6aff248fdcbc96234c91   # zotero/zotero, last change to citeproc.js (2026-07-30)

mkdir -p "$WORK"
export RUSTUP_HOME="$WORK/rustup" CARGO_HOME="$WORK/cargo"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$WORK/rustup-init.sh"
sh "$WORK/rustup-init.sh" -y --no-modify-path --profile minimal --default-toolchain stable \
  --target wasm32-unknown-unknown
"$CARGO_HOME/bin/cargo" install --locked wasm-bindgen-cli --version 0.2.129
"$CARGO_HOME/bin/cargo" install --locked cargo-license
mkdir -p "$WORK/tools"
(cd "$WORK/tools" && npm init -y >/dev/null && npm install --no-audit --no-fund binaryen@132.0.0)

fetch() { # repo commit dir
  mkdir -p "$3"
  git -C "$3" init -q
  git -C "$3" fetch -q --depth 1 "https://github.com/$1.git" "$2"
  git -C "$3" checkout -q FETCH_HEAD
}
fetch citation-style-language/styles "$STYLES_COMMIT" "$WORK/styles"
fetch citation-style-language/locales "$LOCALES_COMMIT" "$WORK/locales"
curl -sSfL -o "$WORK/citeproc.js" \
  "https://raw.githubusercontent.com/zotero/zotero/$ZOTERO_COMMIT/chrome/content/zotero/xpcom/citeproc.js"
echo "Ready: $WORK"
