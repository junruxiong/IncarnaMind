#!/bin/sh
# Builds the citation-style engine (ADR-0016) to WebAssembly, reproducibly, into
# resources/citation-styles/engine.wasm, with its SHA-256 beside it.
#
# The built file is committed, so `npm ci` and the app's build need no Rust. Run this only when
# wasm/citation-styles changes (its source, Cargo.toml or Cargo.lock), and commit both files.
# CI's "Citation-style engine" workflow runs it with --check whenever they change, to show the
# committed file is what this source builds to.
#
# Needs rustup and Node. The pinned Rust toolchain is installed through rustup if it is missing;
# binaryen's wasm-opt comes from npm (set WASM_OPT to use another copy of the same version).
# Nothing here changes a shell profile or global configuration.
#
# usage: sh wasm/citation-styles/build.sh           build, and replace the committed file
#        sh wasm/citation-styles/build.sh --check   build, and fail if it differs
set -eu

RUST_VERSION=1.99.0
BINARYEN_VERSION=132.0.0

MODE=build
[ "${1:-}" = "--check" ] && MODE=check

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT="$ROOT/resources/citation-styles/engine.wasm"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

rustup toolchain install "$RUST_VERSION" --profile minimal --target wasm32-unknown-unknown
CARGO_HOME=${CARGO_HOME:-"$HOME/.cargo"}

# Paths on this machine would otherwise end up in the module (in panic locations), and make it
# differ from one machine to the next.
RUSTFLAGS="--remap-path-prefix=$CARGO_HOME=/cargo --remap-path-prefix=$HERE=/build" \
  CARGO_TARGET_DIR="$WORK/target" \
  cargo "+$RUST_VERSION" build --locked --release --lib --target wasm32-unknown-unknown \
  --manifest-path "$HERE/Cargo.toml"

if [ -n "${WASM_OPT:-}" ]; then
  set -- "$WASM_OPT"
else
  set -- npm exec --yes --package="binaryen@$BINARYEN_VERSION" -- wasm-opt
fi
"$@" -O3 --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext \
  --enable-mutable-globals --enable-reference-types --enable-multivalue \
  -o "$WORK/engine.wasm" "$WORK/target/wasm32-unknown-unknown/release/citation_styles.wasm"

SUM=$(node -e 'const { createHash } = require("node:crypto");
process.stdout.write(createHash("sha256").update(require("node:fs").readFileSync(process.argv[1])).digest("hex"));' \
  "$WORK/engine.wasm")

if [ "$MODE" = check ]; then
  EXPECTED=$(cut -d' ' -f1 "$OUT.sha256")
  if [ "$SUM" != "$EXPECTED" ]; then
    echo "This source builds an engine with SHA-256 $SUM, but the committed one is $EXPECTED." >&2
    echo "Run sh wasm/citation-styles/build.sh, and commit resources/citation-styles/engine.wasm and its .sha256." >&2
    exit 1
  fi
  echo "The committed engine is what this source builds to: $SUM"
else
  cp "$WORK/engine.wasm" "$OUT"
  echo "$SUM  engine.wasm" >"$OUT.sha256"
  echo "Built $OUT: $(wc -c <"$OUT" | tr -d ' ') bytes, SHA-256 $SUM"
fi
