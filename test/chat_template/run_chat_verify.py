import os
import subprocess


SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)
  
DUMP_DIR = os.path.join(
    SCRIPT_DIR,
    "dumps"
)

os.makedirs(
    DUMP_DIR,
    exist_ok=True
)

PROMPT = ("You discover a bank where people can deposit and withdraw time instead of money. You have 37 years in your account, but someone has just withdrawn 6 minutes from your childhood. What do you do?")

subprocess.run(
    [
        "build/gpu-chat-logits-test",
        "--prompt",
        PROMPT,
    ],
    check=True,
)

subprocess.run(
    [
        "python3",
        "test/chat_template/verify_chat_logits.py",
        "--prompt",
        PROMPT,
    ],
    check=True,
)

subprocess.run(
    [
        "python3",
        "test/chat_template/verify_chat_generation.py",
        "--prompt",
        PROMPT,
    ],
    check=True,
)