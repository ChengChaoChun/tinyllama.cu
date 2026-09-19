from transformers import AutoTokenizer


MODEL_PATH = "models/TinyLlama-1.1B-Chat-v1.0"


# ============================================================
# Same messages as the official TinyLlama example
# ============================================================
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "我正在學習 CUDA。我已經了解 thread、block 和 grid，但還不太理解 warp。請先解釋 warp 是什麼，然後舉一個實際的 CUDA 範例。",
    }, 
]


# ============================================================
# Load tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# ============================================================
# Apply TinyLlama Chat Template
# ============================================================

result = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
)


# ============================================================
# Get input IDs
# ============================================================

tokens = result["input_ids"]


# ============================================================
# Print messages
# ============================================================

print("============================================")
print(" TinyLlama Chat Template Reference")
print("============================================")

print("\nMessages:")

for message in messages:
    print(f"role    : {message['role']}")
    print(f"content : {message['content']}")
    print()


# ============================================================
# Token IDs
# ============================================================

print("Token IDs:")
print(tokens)


# ============================================================
# Tokens
# ============================================================

print("\nTokens:")

token_strings = tokenizer.convert_ids_to_tokens(tokens)

for i, (token_id, token_string) in enumerate(
    zip(tokens, token_strings)
):
    print(f"{i:3d}: {token_id:5d}  {repr(token_string)}")


# ============================================================
# Decode
# ============================================================

decoded = tokenizer.decode(tokens)

print("\nDecoded:")
print(repr(decoded))


# ============================================================
# Chat Template as text
# ============================================================

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("\nChat Template (raw):")
print("--------------------------------------------")
print(prompt, end="")
print("--------------------------------------------")

print("\nChat Template:")
print(repr(prompt))


# ============================================================
# Special Token IDs
# ============================================================

print("\nSpecial Token IDs:")

print(
    "<|system|>    :",
    tokenizer.convert_tokens_to_ids("<|system|>")
)

print(
    "<|user|>      :",
    tokenizer.convert_tokens_to_ids("<|user|>")
)

print(
    "<|assistant|> :",
    tokenizer.convert_tokens_to_ids("<|assistant|>")
)

print(
    "</s>          :",
    tokenizer.eos_token_id
)