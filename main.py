from service import get_api_key, llama_8B_instruct, generate

from typing import Union
import torch

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import transformers

from llama_cpp import Llama

from googleapiclient import discovery


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



llm = transformers.pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-3B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

class PromptRequest(BaseModel):
    instruction: Union[str, None] = 'All answers must be written in Korean. You are a kind and creative fairy tale writer for children. Do not include harmful, violent, sexually threatening, or inappropriate language. Your story should always be appropriate for young children. All answers must be kind and creative fairy tale writers for children. All answers must be written in Korean. Follow user\'s input and continue the story naturally.'
    input: Union[str, None] = ""  # optional
    max_new_tokens: int = 512

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
async def generate_text(request: PromptRequest):
    
    output_text = generate(request, llm, client)
    return {"response": output_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
