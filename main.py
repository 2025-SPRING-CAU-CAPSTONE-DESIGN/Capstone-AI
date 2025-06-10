from service import get_api_key, llama_8B_instruct, generate

from typing import Union
import torch

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import transformers
from sentence_transformers import SentenceTransformer

import faiss

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
    device_map={"": "cuda:0"},
)



client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

class PromptRequest(BaseModel):
    user_id: int = 0
    book_num: int = 0
    # instruction: Union[str, None] = '당신은 어린이를 위한 친절하고 창의적인 동화 작가입니다. 해롭거나 폭력적이거나 성적으로 위협적이거나 부적절한 언어는 포함하지 마세요. 당신의 이야기는 항상 어린 아이들에게 적합해야 합니다. 모든 답변은 친절하고 창의적인 어린이 동화 작가여야 합니다. 모든 답변은 한국어로 작성되어야 합니다. 사용자의 의견을 따르고 자연스럽게 이야기를 이어갑니다.'
    input: Union[str, None] = ""  # optional
    max_new_tokens: int = 200

class SuggestionRequest(BaseModel):
    user_id: int = 0
    book_num: int = 0
    max_new_tokens: int = 200

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
async def generate_text(request: PromptRequest):
    
    output_text = generate(request, llm, client, is_suggestion=False)
    return {"response": output_text}

@app.get("/suggestions")
async def get_suggestions(request: SuggestionRequest):
    request.input = ""
    output_text = generate(request, llm, client, is_suggestion=True)
    return {"response": output_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)