export CUDA_VISIBLE_DEVICES=0,1,2,3

python run_pretrain.py \
--model_type bert \
--model_name_or_path ~/lyl/bert_base_zh/ \
--do_lower_case \
--train_language zh \
--do_train \
--do_eval \
--data_dir ../data/ \
--output_dir ../outputs/ \
--max_seq_length 100 \
--eval_steps 150 \
--per_gpu_train_batch_size 128 \
--gradient_accumulation_steps 1 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 16 \
--learning_rate 1e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 5500
