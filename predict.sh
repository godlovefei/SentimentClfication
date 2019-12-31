export BERT_BASE_DIR="./chinese_L-12_H-768_A-12"
export SENTIMENT_DIR="./data/"

python bert-master/run_classifier.py \
  --task_name=Sentiment \
  --do_predict=True \
  --data_dir=$SENTIMENT_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --output_dir=./output/