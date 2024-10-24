python ./src/train.py \
 --do_train \
 --per_device_train_batch_size 16 \
 --per_device_eval_batch_size 16 \
 --num_train_epochs 10 \
 --seed 2024 \
 --output_dir ./models/train_dataset \
 --dataset_name ./data/raw/train_dataset/ \
 --overwrite_output_dir \
 --overwrite_cache