export BERT_BASE_DIR="./chinese_L-12_H-768_A-12"
export SENTIMENT_DIR="./data/"


python bert-master/run_classifier.py \
  --task_name=Sentiment \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --data_dir=$SENTIMENT_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=2 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=512 \
  --output_dir=./output/
