import json
import os

vocab_txt_path = "models/blip_processor/vocab.txt"
vocab_json_path = "models/blip_processor/vocab.json"

# Read the vocab.txt file
with open(vocab_txt_path, 'r', encoding='utf-8') as f:
    tokens = [line.strip() for line in f if line.strip()]

# Create proper JSON format
vocab = {token: idx for idx, token in enumerate(tokens)}

# Save as vocab.json
with open(vocab_json_path, 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"Converted {vocab_txt_path} to proper {vocab_json_path}")
print(f"Contains {len(vocab)} tokens")