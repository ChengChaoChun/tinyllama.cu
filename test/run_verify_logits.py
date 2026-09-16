import os
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DUMP_DIR = os.path.join(SCRIPT_DIR, "dumps")

os.makedirs(DUMP_DIR, exist_ok=True)  

PROMPT = (
    "請詳細說明在 CUDA C++ 中實現高效能 GEMM "
    "(General Matrix Multiplication) 的核心優化技巧，"
    "包括 Shared Memory Tiling、Thread Coarsening "
    "與 Memory Coalescing："
)  

subprocess.run(
    [
        "build/gpu-logits-test",  
        "--prompt",
        PROMPT,
    ],
    check=True,
)

subprocess.run(
    [
        "python3",
        "test/verify_logits.py",
        "--prompt",
        PROMPT,
    ],
    check=True,
)

subprocess.run(
    [
        "python3",
        "test/verify_generation.py",
        "--prompt",
        PROMPT,
    ],
    check=True,
)       