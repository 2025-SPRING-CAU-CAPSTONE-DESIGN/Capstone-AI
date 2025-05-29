
def api(client, model, output_text):
    system_prompt = 'A kind and creative children\'s storyteller for children. The goal is to translate it into a warm, imaginative, age-appropriate story that is safe for children. Do not include language that is harmful, violent, sexually threatening, or inappropriate. Stories should always be appropriate for young children. All answers should be in Korean. You should answer in only one to two sentences per answer and do not answer any more. Please change your input sentence to a pureed one for children.'
    
    analyze_request = {
        'comment': { 'text' : output_text},
        'requestedAttributes': {'TOXICITY': {}},
        'languages': ['ko']
    }
    toxicity_res = client.comments().analyze(body=analyze_request).execute()
    if toxicity_res['attributeScores']['TOXICITY']['spanScores'][0]['score']['value'] > 0.5:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': output_text}
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
        print('\n====================================순화된 내용\n')
        print(tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True))
        output_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    else: output_text = output_text
    return output_text