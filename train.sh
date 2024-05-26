accelerate launch  --config_file accelerate_config.yaml \
    train.py \
    --data_dict './dataset' \
    --model_type 'bert-base-uncased' \
    --batch_size 16 \
    --num_epochs 10 \
    
    