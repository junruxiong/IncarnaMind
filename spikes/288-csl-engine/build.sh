#!/bin/sh
# Builds the binding to wasm with the toolchain setup.sh put in $WORK, and writes the
# package (wasm-bindgen glue for Node and for the web, wasm-opt applied) to $WORK/pkg.
# usage: WORK=/tmp/csl-spike sh build.sh            (size-optimised: opt-level s, wasm-opt -Oz)
#        WORK=/tmp/csl-spike PROFILE=release-o3 OPT=-O3 sh build.sh   (speed-optimised)
set -eu
WORK=${WORK:?set WORK to the folder setup.sh prepared}
HERE=$(cd "$(dirname "$0")" && pwd)
export RUSTUP_HOME="$WORK/rustup" CARGO_HOME="$WORK/cargo" CARGO_TARGET_DIR="$WORK/target"
export PATH="$CARGO_HOME/bin:$PATH"
PROFILE=${PROFILE:-release}
OPT=${OPT:--Oz}
OUT=${OUT:-"$WORK/pkg"}

cargo build --lib --target wasm32-unknown-unknown --profile "$PROFILE" \
  --manifest-path "$HERE/binding/Cargo.toml"
RAW="$CARGO_TARGET_DIR/wasm32-unknown-unknown/$PROFILE/csl_spike.wasm"

# Node glue for the comparison; web glue for what a renderer or worker would load.
wasm-bindgen --target nodejs --out-dir "$OUT/node" "$RAW"
wasm-bindgen --target web --out-dir "$OUT/web" "$RAW"
for dir in "$OUT/node" "$OUT/web"; do
  node "$WORK/tools/node_modules/.bin/wasm-opt" $OPT --enable-bulk-memory \
    --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals \
    --enable-reference-types --enable-multivalue \
    -o "$dir/csl_spike_bg.opt.wasm" "$dir/csl_spike_bg.wasm"
  mv "$dir/csl_spike_bg.wasm" "$dir/csl_spike_bg.unopt.wasm"
  mv "$dir/csl_spike_bg.opt.wasm" "$dir/csl_spike_bg.wasm"
done
ls -l "$RAW" "$OUT/web/csl_spike_bg.wasm" "$OUT/web/csl_spike.js"
