#!/usr/bin/env bash

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
CONFIG_VIDEO=$1
CONFIG_AUDIO=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS \
    --master_port=$PORT $(dirname "$0")/train_mk.py $CONFIG_VIDEO $CONFIG_AUDIO --launcher pytorch ${@:4}
# Any arguments from the third one are captured by ${@:3}
