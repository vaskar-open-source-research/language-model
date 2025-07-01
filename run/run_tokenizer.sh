uv run python3 src/tokenizer.py \
    --tokenizer_path tokenizers/owt_valid_tokenizer \
    --input_path data/TinyStoriesV2-GPT4-valid.txt \
    --output_path data/tiny_stories_valid_token_ids.pt \
    --show_progress