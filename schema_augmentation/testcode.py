from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-4B",
    trust_remote_code=True,
)

print("loaded")