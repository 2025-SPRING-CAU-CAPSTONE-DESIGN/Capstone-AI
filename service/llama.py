from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Union
from pydantic import BaseModel

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

def generate(request: PromptRequest, llm):
    
    # analyze_request = {
    #     'comment': { 'text' : request.input},
    #     'requestedAttributes': {'TOXICITY': {}},
    #     'languages': ['ko']
    # }

    # response = client.comments().analyze(body=analyze_request).execute()
    # if response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value'] > 0.5:
    #     messages = [
    #         {'role': 'system', 'content': system_prompt},
    #         {'role': 'user', 'content': request.input}
    #     ]
    #     prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    #     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    #     input_length = inputs["input_ids"].shape[1]

    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=512,
    #         do_sample=True,
    #         temperature=0.7,
    #         pad_token_id=tokenizer.eos_token_id,
    #     )
    #     print('\n====================================순화된 내용용\n')
    #     print(tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True))
    #     request.input = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    formatted_prompt = alpaca_prompt().format(
        instruction=request.instruction.strip(),
        input=request.input.strip() if request.input else ""
    )
    
    response = llm(
        formatted_prompt,
        max_tokens=request.max_new_tokens,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.15,
        stop=["###", "</s>", "<|endoftext|>"]  # 응답 깔끔하게 자르기 위한 stop token
    )

    output_text = response["choices"][0]["text"].strip()
    return output_text