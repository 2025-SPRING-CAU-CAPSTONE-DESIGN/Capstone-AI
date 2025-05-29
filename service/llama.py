from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Union
from pydantic import BaseModel
from api import api
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def llama_8B_instruct():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    return model, tokenizer

def alpaca_prompt():
    template = """

### Instruction:
{instruction}

### Input:
{input}

### Response:

"""
    return template
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexIDMap(faiss.IndexFlatL2(384))
chunks = []
all_chunk_texts = {}
all_indexes = {}
all_deques = {}
chunk_texts = {}
def add_chunk(user_id, book_num, text):
    key = (user_id, book_num)
    if key not in all_chunk_texts:
        all_chunk_texts[key] = {}
        all_indexes[key] = faiss.IndexIDMap(faiss.IndexFlatL2(384))
    chunk_texts = all_chunk_texts[key]
    index = all_indexes[key]
    if text in chunk_texts.values():
        return
    chunk_id = len(chunk_texts)
    chunk_embedding = embedder.encode([text])
    index.add_with_ids(chunk_embedding, np.array([chunk_id]))
    chunk_texts[chunk_id] = text

def retrieve_relevant_chunks(user_id, book_num, query, top_k=3):
    key = (user_id, book_num)
    if key not in all_indexes:
        return []
    index = all_indexes[key]
    chunk_texts = all_chunk_texts[key]
    query_embedding = embedder.encode([query]).astype("float32")
    D, I = index.search(query_embedding, top_k)
    sorted_ids = sorted(I[0])
    return [chunk_texts[i] for i in sorted_ids if i in chunk_texts]
class PromptRequest(BaseModel):
    user_id: int = 0
    book_num: int = 0
    instruction: Union[str, None] = 'All answers must be written in Korean. You are a kind and creative fairy tale writer for children. Do not include harmful, violent, sexually threatening, or inappropriate language. Your story should always be appropriate for young children. All answers must be kind and creative fairy tale writers for children. All answers must be written in Korean. Follow user\'s input and continue the story naturally.'
    input: Union[str, None] = ""  # optional
    max_new_tokens: int = 200

def generate(request: PromptRequest, llm, client):
    retrieve_chunks = retrieve_relevant_chunks(request.user_id, request.book_num, request.input)
    print(retrieve_chunks)
    add_chunk(request.user_id, request.book_num, request.input)
    for i in retrieve_chunks:
        request.input += i
    formatted_prompt = [
        {"role": "system", "content": request.instruction},
        {"role": "user", "content": request.input}
    ]
    response = llm(
        formatted_prompt,
        max_new_tokens=request.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    output_text = response[0]["generated_text"][-1]['content']
    sentences = output_text.strip().split(". ")
    output_text = ". ".join(sentences[:2]).strip()
    if output_text == '':
        output_text = '서버에 문제가 생겼습니다.'
    else:
        # output_text = api(client, llm, output_text)
        print('a')
    add_chunk(request.user_id, request.book_num, output_text)
    

    return output_text