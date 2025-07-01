uv run python3 src/train_bpe.py \
    --input_path data/owt_train.txt \
    --output_dir tokenizers/owt_train_tokenizer \
    --vocab_size 50000 \
    --special_tokens "<|endoftext|>" "<|pad|>" \
    --verbose