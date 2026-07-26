#!/usr/bin/env bash
# Copy the browser dependencies out of node_modules into web/vendor/.
#
# The vendored files are not committed: they are 26 MB of third-party build
# output reproducible from the pinned versions in package.json. The HF Space
# upload runs this first so the deployed page is self-contained and needs no CDN.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d node_modules ] || { echo "run 'npm install' first" >&2; exit 1; }
mkdir -p vendor/mujoco vendor/three vendor/ort

cp node_modules/mujoco-js/dist/mujoco_wasm.js vendor/mujoco/
# three.module.js imports three.core.js at runtime; both are required.
cp node_modules/three/build/three.module.js node_modules/three/build/three.core.js vendor/three/
# Only the plain single-threaded wasm backend: policy.js pins numThreads = 1, and
# the asyncify/jsep/jspi builds add 66 MB for capabilities this page never uses.
cp node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs \
   node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs \
   node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm vendor/ort/

du -sh vendor/*
