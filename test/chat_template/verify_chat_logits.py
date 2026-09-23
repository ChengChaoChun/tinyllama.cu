import os
import glob
import sys

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


# Configuration
MODEL_PATH = "models/TinyLlama-1.1B-Chat-v1.0"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DUMP_DIR = os.path.join(SCRIPT_DIR, "dumps")


def get_prompt():
    if len(sys.argv) != 3 or sys.argv[1] != "--prompt":
        raise RuntimeError(
            'Usage: python3 test/verify_logits.py --prompt "your prompt"'
        )

    return sys.argv[2]


def compute_metrics(
    ref_logits: torch.Tensor,
    cpp_logits: torch.Tensor,
    name: str,
):
    """
    Compare C++ logits against PyTorch reference logits.

    Metrics:
        - Max Absolute Error
        - Mean Absolute Error
        - Cosine Similarity
        - Argmax Match Rate
    """

    if isinstance(cpp_logits, np.ndarray):
        cpp_logits = torch.from_numpy(cpp_logits)

    cpp_logits = cpp_logits.to(ref_logits.device)

    ref = ref_logits.to(torch.float32)
    cpp = cpp_logits.to(torch.float32)

    if ref.shape != cpp.shape:
        raise RuntimeError(
            f"Shape mismatch! "
            f"PyTorch: {list(ref.shape)}, "
            f"C++: {list(cpp.shape)}"
        )

    # Absolute error
    abs_diff = torch.abs(ref - cpp)

    max_err = torch.max(abs_diff).item()
    mean_err = torch.mean(abs_diff).item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(
        ref.view(-1, ref.size(-1)),
        cpp.view(-1, cpp.size(-1)),
        dim=-1,
    ).mean().item()

    # Argmax comparison
    ref_argmax = torch.argmax(ref, dim=-1)
    cpp_argmax = torch.argmax(cpp, dim=-1)

    argmax_match = (
        (ref_argmax == cpp_argmax)
        .float()
        .mean()
        .item()
        * 100
    )

    # Print results
    print(f"\n--- {name} ---")
    print(f"Shape               : {list(ref.shape)}")
    print(f"Max Absolute Error  : {max_err:.6f}")
    print(f"Mean Absolute Error : {mean_err:.6f}")
    print(f"Cosine Similarity   : {cos_sim:.8f}")
    print(f"Argmax Match Rate   : {argmax_match:.2f}%")

    print(
        f"PyTorch Argmax Token: "
        f"{ref_argmax.reshape(-1)[-1].item()}"
    )

    print(
        f"C++ Argmax Token    : "
        f"{cpp_argmax.reshape(-1)[-1].item()}"
    )

    # Only print Top-5 when argmax differs
    if ref_argmax.flatten()[-1].item() != cpp_argmax.flatten()[-1].item():
        ref_flat = ref.reshape(-1)
        cpp_flat = cpp.reshape(-1)

        ref_top5 = torch.topk(ref_flat, k=5)
        cpp_top5 = torch.topk(cpp_flat, k=5)

        print("\n  Top-5 PyTorch:")
        for rank, (token_id, logit) in enumerate(
            zip(
                ref_top5.indices.tolist(),
                ref_top5.values.tolist(),
            ),
            start=1,
        ):
            print(
                f"    {rank}. token {token_id:5d} : "
                f"{logit:.6f}"
            )

        print("\n  Top-5 C++:")
        for rank, (token_id, logit) in enumerate(
            zip(
                cpp_top5.indices.tolist(),
                cpp_top5.values.tolist(),
            ),
            start=1,
        ):
            print(
                f"    {rank}. token {token_id:5d} : "
                f"{logit:.6f}"
            )


def main():
    # Get prompt
    prompt = get_prompt()

    print(f"Target Dump Directory: {DUMP_DIR}")

    if not os.path.exists(DUMP_DIR):
        print(
            f"\n[Error] Dump directory not found: {DUMP_DIR}"
        )
        print(
            "Please make sure you have run the C++ program "
            "first to generate dump files."
        )
        return 1

    print(f"Prompt: {prompt}")

    # Load PyTorch model and tokenizer
    print("\nLoading PyTorch model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"PyTorch device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
    ).to(device)

    model.eval()

    # Tokenize prompt with chat template
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

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)

    prompt_len = input_ids.shape[1]

    print(f"Chat Template prompt length: {prompt_len} tokens")

    # Prefill verification
    cpp_prefill_path = os.path.join(
        DUMP_DIR,
        "cpp_prefill_logits.npy",
    )

    if not os.path.exists(cpp_prefill_path):
        print(
            f"\n[Error] Prefill dump file not found:"
            f"\n{cpp_prefill_path}"
        )
        return 1

    print("\nRunning PyTorch Prefill...")

    with torch.no_grad():

        outputs = model(
            input_ids,
            use_cache=True,
        )

        py_prefill_last_logits = outputs.logits[:, -1, :]

        past_key_values = outputs.past_key_values

    cpp_prefill_logits = np.load(
        cpp_prefill_path
    )

    compute_metrics(
        py_prefill_last_logits,
        cpp_prefill_logits,
        "Prefill Stage (Last Token Logits)",
    )

    # =========================================================================
    # Decode verification
    #
    # PyTorch is force-fed the token IDs generated by the C++ implementation.
    # This ensures that both implementations receive the same input token
    # at every decoding step, allowing their logits to be compared directly.
    # =========================================================================
    cpp_tokens_path = os.path.join(
        DUMP_DIR,
        "cpp_tokens.txt",
    )

    decode_npy_files = sorted(
        glob.glob(
            os.path.join(
                DUMP_DIR,
                "cpp_decode_logits_*.npy",
            )
        ),
        key=lambda x: int(
            x.split("_")[-1].split(".")[0]
        ),
    )

    if not os.path.exists(cpp_tokens_path):
        print(
            f"\n[Error] Token dump file not found:"
            f"\n{cpp_tokens_path}"
        )
        return 1

    if not decode_npy_files:
        print(
            f"\n[Error] No decode logits found in:"
            f"\n{DUMP_DIR}"
        )
        return 1

    print(
        "\nRunning PyTorch Decode Alignment "
        "(Step-by-Step Force Feeding)..."
    )

    # Load generated token IDs from C++
    with open(cpp_tokens_path, "r") as f:
        cpp_generated_tokens = [
            int(line.strip())
            for line in f
            if line.strip()
        ]

    # ========================================================================
    # Decode Argmax Statistics
    #
    # Count how many decode steps have the same argmax token between
    # PyTorch and C++.
    # ========================================================================
    decode_match_count = 0
    decode_total_count = 0

    # Compare every decode step
    for step, npy_file in enumerate(decode_npy_files):

        if step >= len(cpp_generated_tokens):
            break

        # --------------------------------------------------------------------
        # C++ generated token becomes the next PyTorch input token.
        #
        # This is force feeding:
        #
        # C++ generated token
        #          ↓
        # PyTorch decode
        #          ↓
        # PyTorch logits
        # --------------------------------------------------------------------
        input_token_id = cpp_generated_tokens[step]

        current_input_id = torch.tensor(
            [[input_token_id]],
            device=device,
        )

        # PyTorch one-token decode
        with torch.no_grad():

            outputs = model(
                current_input_id,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # outputs.logits: [batch, 1, vocab]
            # Remove batch dimension: [1, vocab]
            py_decode_logits = outputs.logits.squeeze(0)

            past_key_values = outputs.past_key_values

        # Load C++ decode logits
        cpp_decode_logits = np.load(
            npy_file
        )

        # Compare
        compute_metrics(
            py_decode_logits,
            cpp_decode_logits,
            (
                f"Decode Step {step} "
                f"(Input Token ID: {input_token_id})"
            ),
        )

        # --------------------------------------------------------------------
        # Accumulate overall Decode Argmax accuracy
        # --------------------------------------------------------------------
        ref_argmax = torch.argmax(
            py_decode_logits,
            dim=-1,
        ).item()

        cpp_argmax = torch.argmax(
            torch.from_numpy(cpp_decode_logits).to(device),
            dim=-1,
        ).item()

        if ref_argmax == cpp_argmax:
            decode_match_count += 1

        decode_total_count += 1

    # ========================================================================
    # Overall Decode Argmax Summary
    # ========================================================================
    if decode_total_count > 0:
        decode_match_rate = (
            decode_match_count
            / decode_total_count
            * 100
        )

        print("\n================================")
        print("Decode Argmax Summary")
        print("================================")
        print(
            f"Argmax Match Rate : "
            f"{decode_match_rate:.2f}% "
            f"({decode_match_count}/{decode_total_count})"
        )

    print("\n================================")
    print("PyTorch verification completed.")
    print("================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())  