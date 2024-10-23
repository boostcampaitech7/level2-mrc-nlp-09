python ./src/inference.py \
 --do_predict \
 --per_device_eval_batch_size 32 \
 --seed 2024 \
 --output_dir ./outputs/test_dataset/ \
 --dataset_name ./data/raw/test_dataset/ \
 --model_name_or_path ./models/train_dataset/ \
 --overwrite_output_dir \
 --overwrite_cache  