from llm.llm_api import LLM
from llm.prompts import RelatedWork
from experiments import run_task
from analysis import calculate_biases
from dataset import load_results, get_bias_values


model = "meta-llama/Llama-3.3-70B-Instruct"
# model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

llm = LLM(model=model)
prompt = RelatedWork()
iterations = 1

topics = ["Record Linkage", "Spam Detection", "Stance Detection", "Named Entity Recognition", "German Reunification",
          "Earthquake Detection", "Surrealism", "Quantum Cryptography", "Brain-Computer Interfaces",
          "Mediterranean Diet"]
bias = ("Publication-Year")

calculate_biases(prompt=prompt, topics=topics, model=model, bias=bias)
# get_bias_values(topics=topics, bias=bias, plot=True)

# for topic in topics:
#     biases = None
#     run_task(topic=topic, prompt=prompt, model=model, biases=biases, iterations=iterations)
#     biases = ["Citation-Count"]
#     run_task(topic=topic, prompt=prompt, model=model, biases=biases, iterations=iterations)
#     biases = ["Paper-Type"]
#     run_task(topic=topic, prompt=prompt, model=model, biases=biases, iterations=iterations)
#     biases = ["Publication-Year"]
#     run_task(topic=topic, prompt=prompt, model=model, biases=biases, iterations=iterations)
#     biases = ["Country"]
#     run_task(topic=topic, prompt=prompt, model=model, biases=biases, iterations=iterations)
