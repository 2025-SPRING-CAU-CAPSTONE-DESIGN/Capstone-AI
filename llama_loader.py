from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "PAUL1122/ice77/unsloth.Q8_0.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=32,
    n_ctx=2048,
    n_batch=512,
    f16_kv=True,
    verbose=False
)