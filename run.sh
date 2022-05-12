#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=13583
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4
export OMP_NUM_THREADS=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_DIR=/data2/private/luofuwen
CODE_BASE_DIR=${BASE_DIR}/DYLE_repro

python3 -m torch.distributed.run ${DISTRIBUTED_ARGS} ${CODE_BASE_DIR}/train.py
