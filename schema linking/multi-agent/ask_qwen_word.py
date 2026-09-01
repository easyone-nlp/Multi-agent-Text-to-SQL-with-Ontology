import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--word", default="적요")
parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
).to("cuda").eval()

messages = [
    {
        "role": "system",
        "content": "당신은 한국어 사전 보조자다. 단어를 오타로 단정하지 말고 동음이의어와 한자를 구분해 설명해라.",
    },
    {
        "role": "user",
        "content": f"한국어 단어 '{args.word}'의 뜻을 알려줘. 품사, 한자, 서로 다른 뜻과 짧은 예문을 한국어로 답해줘.",
    },
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

answer_ids = output_ids[0, inputs.input_ids.shape[1]:]
print(tokenizer.decode(answer_ids, skip_special_tokens=True).strip())