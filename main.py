from service import get_api_key, llama_8B_instruct, generate

from typing import Union

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from llama_cpp import Llama

# from googleapiclient import discovery


API_KEY = get_api_key()

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model, tokenizer = llama_8B_instruct()

system_prompt = 'Please change the input sentence to a sentence that is pureed for children.'

llm = Llama(
    model_path='./model/llama/unsloth.Q8_0.gguf',
    n_gpu_layers=32,
    n_ctx=2048,
    n_batch=512,
    f16_kv=True,
    verbose=False
)


# client = discovery.build(
#   "commentanalyzer",
#   "v1alpha1",
#   developerKey=API_KEY,
#   discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
#   static_discovery=False,
# )

class PromptRequest(BaseModel):
    instruction: Union[str, None] = 'You are a kind and creative fairy tale writer for children. Your goal is to write heartwarming, imaginative, and age-appropriate stories that are safe for kids. Do not include any harmful, violent, sexual, threatening, or inappropriate language. Your stories must always be suitable for young children. All your responses must be written in Korean. You must respond with only 1 or 2 sentences per answer, no more. Continue the story naturally based on the user’s input.'
    input: Union[str, None] = ""  # optional
    max_new_tokens: int = 512

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
async def generate_text(request: PromptRequest):
    
    output_text = generate(request, llm)
    return {"response": output_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
