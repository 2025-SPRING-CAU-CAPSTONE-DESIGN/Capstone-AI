from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Union
from pydantic import BaseModel
from api import api

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

class PromptRequest(BaseModel):
    instruction: Union[str, None] = 'You are a kind and creative fairy tale writer for children. Your goal is to write heartwarming, imaginative, and age-appropriate stories that are safe for kids. Do not include any harmful, violent, sexual, threatening, or inappropriate language. Your stories must always be suitable for young children. All your responses must be written in Korean. You must respond with only 1 or 2 sentences per answer, no more. Continue the story naturally based on the user’s input.'
    input: Union[str, None] = ""  # optional
    max_new_tokens: int = 512

def generate(request: PromptRequest, llm, client):
    
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

    return output_text