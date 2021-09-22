#!/bin/bash

CHECKPOINT_PATH=$1
# Large
# MPSIZE=2
# NLAYERS=32
# NHIDDEN=2560
# NATT=32
# MAXSEQLEN=1024

# Small
MPSIZE=1
NLAYERS=12
NHIDDEN=768
NATT=12
MAXSEQLEN=1024

CMD="python -m torch.distributed.launch --nproc_per_node 8 zero-shot-cls.py \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --seq-length $MAXSEQLEN \
       --max-position-embeddings 1024 \
       --fp16 \
       --cache-dir cache \
       --eval-data-path $2 \
       --tokenizer-path bpe_3w_new/ \
       --vocab-size 30000 \
       --task tnews "

$CMD
