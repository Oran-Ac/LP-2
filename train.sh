accelerate launch  --config_file accelerate_config.yaml \
    train.py \
    --data_dict './data' \
    --model_type 'roberta-base' \
    --batch_size 18 \
    --num_epochs 10 \
    --max_length 256\
    --lr 2e-5\
    --optimizer 'adamw'\
    --weight_decay 1e-5\
    --hybrid_flag True\
    