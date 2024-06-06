accelerate launch  --config_file accelerate_config.yaml \
    train.py \
    --data_dict './dataset' \
    --model_type 'microsoft/deberta-v3-base' \
    --batch_size 32 \
    --num_epochs 10 \
    --optimizer 'adamw' \
    --distill \
    --teacher_model_type 'microsoft/deberta-v3-base' \
    --teacher_model_path './saved_model/teacher' \
    --temperature 0.2 \
    