#!/bin/bash

BERT_BASE_DIR=pretrain/chinese_L-12_H-768_A-12

python3 convert_tf_to_pytorch/convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin