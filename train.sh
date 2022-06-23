#!/bin/sh
set -x

GPUS_PER_NODE=8

PARTITION=$1
GPUS=$2
config=/mnt/lustre/zhangfeng4/PRSA-Net/config/thumos_i3d_PRSA.yaml

declare -u expname

expname=prsa
echo $expname

if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}

currenttime=`date "+%Y%m%d%H%M%S"`
mkdir -p  results/${expname}/running_log
g=$(($2<16?$2:16))

srun -p ${PARTITION} \
   --job-name=${expname} \
   --gres=gpu:$g \
   --ntasks=${GPUS} \
   --ntasks-per-node=$g \
   --cpus-per-task=${CPUS_PER_TASK} \
   --kill-on-bad-exit=1 \
   ${SRUN_ARGS} \
   python -u -W ignore main.py --mode train --cfg $config \
   2>&1 | tee results/${expname}/running_log/train_${currenttime}.log