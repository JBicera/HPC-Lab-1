#!/bin/sh
salloc -Jlab1_gpu_interactive --partition=ice-gpu,coc-gpu -N1 --ntasks-per-node=8 -C"intel&core24&nvidia-gpu" --gres=gpu:1 --mem-per-cpu=16G
exit 0
