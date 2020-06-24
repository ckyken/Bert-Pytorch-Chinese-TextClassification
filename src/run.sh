DATA_DIR=../../dl-course-final-competition
BERT_BASE_DIR=../pretrain/chinese_L-12_H-768_A-12
BERT_PYTORCH_DIR=../pretrain/chinese_L-12_H-768_A-12

python3 run_classifier_word.py \
  --task_name DLCompetition \
  --do_train \
  --do_eval \
  --do_test \
  --data_dir $DATA_DIR \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
  --max_seq_length 256 \
  --train_batch_size 24 \
  --learning_rate 2e-5 \
  --num_train_epochs 50.0 \
  --output_dir ./DLCompetition_output/ \
  --local_rank 3 \
  --no_cuda