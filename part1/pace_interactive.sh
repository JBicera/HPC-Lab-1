#!/bin/sh
salloc -Jlab1_openmp_interactive --partition=ice-cpu,coc-cpu -N1 --ntasks-per-node=16 -C"intel&core24" --mem-per-cpu=16G
exit 0
