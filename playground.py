from llm.llm_api import LLM


model = "meta-llama/Llama-3.3-70B-Instruct"
# model = "openGPT-X/Teuken-7B-instruct-research-v0.4"
# model = "deepseek-ai/DeepSeek-R1"
# model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
# model = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

llm = LLM(model=model)

topic = ("Hallucination Mitigation in Large Language Models")

answer, bibtex = llm.request(topic=topic)
print(answer)
print(bibtex)
