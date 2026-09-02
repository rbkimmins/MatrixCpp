#!/bin/sh
# Emits the compiler flags needed to embed Julia. Use it inline:
#
#     g++ -std=c++17 -O2 -fopenmp myplot.cpp -o myplot $(plotting/juliaflags.sh)
#
# Everything it prints comes from asking the installed julia where it lives, so
# it follows juliaup version switches without editing.
BINDIR=$(julia -e 'print(Sys.BINDIR)')
echo "-I${BINDIR}/../include/julia -L${BINDIR}/../lib -Wl,-rpath,${BINDIR}/../lib -ljulia"
