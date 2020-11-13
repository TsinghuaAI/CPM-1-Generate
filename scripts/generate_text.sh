#!/bin/bash

CHECKPOINT_PATH=$1
MPSIZE=2
NLAYERS=32
NHIDDEN=2560
NATT=32
MAXSEQLEN=1024

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0

python -m torch.distributed.launch --nproc_per_node 2 generate_samples.py \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --seq-length $MAXSEQLEN \
       --max-position-embeddings 1024 \
       --fp16 \
       --cache-dir cache \
       --out-seq-length 512 \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP \
       --tokenizer-path bpe_3w_new/ \
       --vocab-size 30000 

