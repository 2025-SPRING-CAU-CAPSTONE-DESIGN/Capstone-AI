from typing import Union
import uvicorn

from fastapi import FastAPI
from llama_cpp import Llama
from pydantic import BaseModel

app = FastAPI()

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="PAUL1122/ice", filename="unsloth.Q8_0.gguf")
print(path)

llm = Llama(
    model_path=path,
    n_gpu_layers=32,
    n_ctx=2048,
    n_batch=512,
    f16_kv=True,
    verbose=False
)

alpaca_prompt = """### Instruction:
{instruction}

### Input:
{input}

### Response:"""



class PromptRequest(BaseModel):
    instruction: str
    input: Union[str, None] = ""  # optional
    max_new_tokens: int = 128

@app.get("/")
def read_root():
    return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

@app.post("/generate")
async def generate_text(request: PromptRequest):
    
    formatted_prompt = alpaca_prompt.format(
        instruction=request.instruction.strip(),
        input=request.input.strip() if request.input else ""
    )

    print("\n📥 입력된 Prompt:\n", formatted_prompt)

    response = llm(
        formatted_prompt,
        max_tokens=request.max_new_tokens,
        temperature=0.8,
        top_p=0.95,
        stop=["###", "</s>", "<|endoftext|>"]  # 응답 깔끔하게 자르기 위한 stop token
    )

    output_text = response["choices"][0]["text"].strip()
    return {"response": output_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)