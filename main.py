from typing import Union
import uvicorn

from fastapi import FastAPI
from llama_cpp import Llama
from pydantic import BaseModel
from googleapiclient import discovery
from dotenv import load_dotenv
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

API_KEY = os.environ.get('API_KEY')
app = FastAPI()

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import hf_hub_download

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

system_prompt = 'Please change the input sentence to a sentence that is pureed for children.'

path = hf_hub_download(repo_id="hyeonsooKim/Llama_finetuning", filename="unsloth.Q8_0.gguf")
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

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

class PromptRequest(BaseModel):
    instruction: Union[str, None] = 'You are a kind and creative fairy tale writer for children. Your goal is to write heartwarming, imaginative, and age-appropriate stories that are safe for kids. Do not include any harmful, violent, sexual, threatening, or inappropriate language. Your stories must always be suitable for young children. All your responses must be written in Korean. You must respond with only 1 or 2 sentences per answer, no more. Continue the story naturally based on the user’s input.'
    input: Union[str, None] = ""  # optional
    max_new_tokens: int = 512

@app.get("/")
def read_root():
    return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

@app.post("/generate")
async def generate_text(request: PromptRequest):
    
    analyze_request = {
        'comment': { 'text' : request.input},
        'requestedAttributes': {'TOXICITY': {}},
        'languages': ['ko']
    }

    response = client.comments().analyze(body=analyze_request).execute()
    if response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value'] > 0.5:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': request.input}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        print('\n====================================순화된 내용용\n')
        print(tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True))
        request.input = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
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