import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_PATH = "models/TinyLlama-1.1B-Chat-v1.0"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DUMP_DIR = os.path.join(SCRIPT_DIR, "dumps")

CPP_TOKENS_PATH = os.path.join(
    DUMP_DIR,
    "cpp_tokens.txt",
)

MAX_NEW_TOKENS = 100
EOS_TOKEN_ID = 2

def get_prompt():
    if len(sys.argv) != 3 or sys.argv[1] != "--prompt":
        raise RuntimeError(
            'Usage: python3 test/verify_generation.py --prompt "your prompt"'
        )

    return sys.argv[2]


# Load C++ generated tokens
def load_cpp_tokens():
    if not os.path.exists(CPP_TOKENS_PATH):
        raise RuntimeError(
            f"\n[Error] Token dump file not found:\n"
            f"{CPP_TOKENS_PATH}\n\n"
            f"Please run the C++/CUDA inference program first."
        )

    with open(CPP_TOKENS_PATH, "r") as f:
        tokens = [
            int(line.strip())
            for line in f
            if line.strip()
        ]

    return tokens

# PyTorch autoregressive generation
def generate_pytorch_tokens(
    model,
    input_ids,
    max_new_tokens,
    eos_token_id,
):
    generated_tokens = []

    current_input_ids = input_ids

    past_key_values = None

    with torch.no_grad():
        # Prefill
        outputs = model(
            current_input_ids,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
        ).item()

        # Autoregressive decoding
        for _ in range(max_new_tokens):

            if next_token == eos_token_id:
                break

            generated_tokens.append(next_token)

            current_input_ids = torch.tensor(
                [[next_token]],
                device=input_ids.device,
            )

            outputs = model(
                current_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
            ).item()

    return generated_tokens

def compare_tokens(cpp_tokens, pytorch_tokens):

    print("\n================================")
    print("Generation Verification")
    print("================================")

    print(f"C++ token count     : {len(cpp_tokens)}")
    print(f"PyTorch token count : {len(pytorch_tokens)}")

    compare_count = min(
        len(cpp_tokens),
        len(pytorch_tokens),
    )

    if compare_count == 0:
        print("\n[Error] No generated tokens to compare.")
        return False

    matched = 0

    print("\nToken comparison:")

    for step in range(compare_count):

        cpp_token = cpp_tokens[step]
        pytorch_token = pytorch_tokens[step]

        match = cpp_token == pytorch_token

        if match:
            matched += 1

        status = "MATCH" if match else "MISMATCH"

        print(
            f"Step {step:3d}: "
            f"C++ = {cpp_token:6d}, "
            f"PyTorch = {pytorch_token:6d} "
            f"[{status}]"
        )

    match_rate = (
        matched / compare_count * 100.0
    )

    print("\n--------------------------------")
    print(
        f"Token Match Rate: "
        f"{matched}/{compare_count} "
        f"({match_rate:.2f}%)"
    )

    if len(cpp_tokens) != len(pytorch_tokens):

        print("\n[Warning] Generated token counts differ.")

    if matched == compare_count and \
       len(cpp_tokens) == len(pytorch_tokens):

        print("\nResult: PASS")
        return True

    print("\nResult: FAIL")
    return False

def main():

    print("\n================================")
    print("Starting PyTorch generation verification.")
    print("================================")

    prompt = get_prompt()

    print(f"Prompt: {prompt}")

    cpp_tokens = load_cpp_tokens()

    print(
        f"C++ generated tokens: "
        f"{len(cpp_tokens)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
    ).to(device)

    model.eval()

    messages = [ 
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)

    print(
        f"Chat Template prompt length: "
        f"{input_ids.shape[1]} tokens"
    )

    print("\nRunning PyTorch autoregressive generation...")

    pytorch_tokens = generate_pytorch_tokens(
        model=model,
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=EOS_TOKEN_ID,
    )

    passed = compare_tokens(
        cpp_tokens,
        pytorch_tokens,
    )

    print("\n================================")

    if passed:
        print("PyTorch generation verification passed.")
    else:
        print("PyTorch generation verification failed.")

    print("================================")

    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())