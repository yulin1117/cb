from llm.llm_api import LLM
from llm.prompts import RelatedWork
from experiments import run_task
from analysis import calculate_biases
from dataset import load_results, get_bias_values
from openalex import get_openalex_topics, get_works_for_topic, load_topic_title_abstract
from topic_clustering import cluster_topic, run_topic_gpt


model = "meta-llama/Llama-3.3-70B-Instruct"
# model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

llm = LLM(model=model)
prompt = RelatedWork()
iterations = 1

topics = ["Record Linkage", "Spam Detection", "Stance Detection", "Named Entity Recognition", "German Reunification",
          "Earthquake Detection", "Surrealism", "Quantum Cryptography", "Brain-Computer Interfaces",
          "Mediterranean Diet"]
bias = ("Publication-Year")

# calculate_biases(prompt=prompt, topics=topics, model=model, bias=bias)
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

#oa_topics = get_openalex_topics()

topic_url = "https://openalex.org/T10181"
# cluster_topic(topic_url=topic_url, n=20000)
labels = run_topic_gpt(topic_url)


# topic_url = "https://openalex.org/T10764"
# cluster_topic(topic_url=topic_url, n=20000)
#
# topic_url = "https://openalex.org/T12380"
# cluster_topic(topic_url=topic_url, n=20000)

# oa_works = get_works_for_topic(topic_url=topic_url, n=20000)
# works = load_topic_title_abstract(topic_url=topic_url, n=20000)

