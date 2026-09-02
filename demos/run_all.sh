#!/bin/sh
# Builds and runs every demo. From the repo root:  sh demos/run_all.sh
set -e
mkdir -p demos/out
for d in mandelbrot svd_compress eigenmodes heat_expm fft_denoise; do
    printf '\n=== %s ===\n' "$d"
    g++ -std=c++17 -O2 -fopenmp -I. "demos/$d.cpp" -o "/tmp/demo_$d"
    "/tmp/demo_$d"
done
printf '\nfigures in demos/out/\n'
