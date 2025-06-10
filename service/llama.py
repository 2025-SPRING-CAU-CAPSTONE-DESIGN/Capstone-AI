from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Union
from pydantic import BaseModel
from api import api
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from deep_translator import GoogleTranslator
import openai


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
    # instruction: Union[str, None] = '당신은 어린이를 위한 친절하고 창의적인 동화 작가입니다. 해롭거나 폭력적이거나 성적으로 위협적이거나 부적절한 언어는 포함하지 마세요. 당신의 이야기는 항상 어린 아이들에게 적합해야 합니다. 모든 답변은 친절하고 창의적인 어린이 동화 작가여야 합니다. 모든 답변은 한국어로 작성되어야 합니다. 사용자의 의견을 따르고 자연스럽게 이야기를 이어갑니다.'
    input: Union[str, None] = ""  # optional
    max_new_tokens: int = 200

openai.api_key = ''

def get_completion(prompt, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": "You do not change the content and change the prompt to make it feel naturally assimilated."},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def generate(request: PromptRequest, llm, client, is_suggestion=False):
    retrieve_chunks = retrieve_relevant_chunks(request.user_id, request.book_num, request.input)
    print(retrieve_chunks)
    add_chunk(request.user_id, request.book_num, request.input)
    input_conv = ""
    for i in retrieve_chunks:
        input_conv += i
    input_conv += request.input
    formatted_prompt = []
    if is_suggestion:
        formatted_prompt = [
            {"role": "system", "content": 'You are a kind and creative fairy tale writer for children. Do not include harmful, violent, sexually threatening, or inappropriate language. Your story should always be appropriate for young children. All answers must be kind and creative fairy tale writers for children. All answers must be written in Korean.'},
            {"role": "user", "content": 'Please recommend the first fairy tale in 2 sentences.'}
        ]
    else:
        formatted_prompt = [
            {"role": "system", "content": 'You are a kind and creative fairy tale writer for children. Do not include harmful, violent, sexually threatening, or inappropriate language. Your story should always be appropriate for young children. All answers must be kind and creative fairy tale writers for children. All answers must be written in Korean. Follow user\'s input and continue the story naturally.'},
            {"role": "user", "content": input_conv}
        ]
    print(input_conv)
    response = llm(
        formatted_prompt,
        max_new_tokens=request.max_new_tokens,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )

    output_text = response[0]["generated_text"][-1]['content']
    sentences = output_text.strip().split(".")
    if sentences[0] == sentences[1]:
        output_text = sentences[0]
    else:
        output_text = ". ".join(sentences[:2]).strip()
    if output_text == '':
        output_text = '서버에 문제가 생겼습니다.'
    output_text += "."
    output_text = get_completion(output_text)
    print(output_text)
    add_chunk(request.user_id, request.book_num, output_text)
    

    return output_text