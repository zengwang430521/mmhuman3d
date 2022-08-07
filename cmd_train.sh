#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
srun -p pat_earth -x SH-IDC1-10-198-4-[90-91,100-103,116-119] \
srun -p pat_earth  \
srun -p pat_earth -x SH-IDC1-10-198-4-[80-99] \
srun -p pat_dev  \
srun -p 3dv-share --quotatype=spot \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p mm_human --quotatype=auto \
   --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \

srun -p mm_research --quotatype=spot \
   --ntasks=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=1 --kill-on-bad-exit=1 \
    --job-name=mesh python -u tools/train.py  --launcher="slurm" \
    configs/tcformer.py --work-dir=work_dirs/tcfrormer
