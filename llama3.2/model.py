import transformers
import torch

model_id = "meta-llama/Llama-3.1-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

prompt = (
    "[SYSTEM] You are a fairy tale writer. You should not print harmful words (ex. harm, sexual, violent etc) to a child.\n"
    "[USER] 현수는 집에서 밥을 먹었다.\n"
)

outputs = pipeline(
    prompt,
    max_new_tokens=256,
)
print(outputs)