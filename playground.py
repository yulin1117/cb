from analysis import calculate_biases
from openalex.data_analysis import dataset_to_csv, eda
from openalex.openalex import get_openalex_topics, get_works_for_topic, load_topic_title_abstract
from openalex.topic_clustering import create_openalex_dataset, select_openalex_topics

model = "meta-llama/Llama-3.3-70B-Instruct"
# model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# llm = LLM(model=model)
# prompt = RelatedWork()
# iterations = 1

# topics = ["Record Linkage", "Spam Detection", "Stance Detection", "Named Entity Recognition", "German Reunification",
#           "Earthquake Detection", "Surrealism", "Quantum Cryptography", "Brain-Computer Interfaces",
#           "Mediterranean Diet"]
# bias = ("Publication-Year")

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

# oa_topics = get_openalex_topics()

# topics = select_openalex_topics(n=10)
# topic_url = "https://openalex.org/T10181"
# get_works_for_topic(topic_url=topic_url, n=5000)
# cluster_topic(topic_url=topic_url)
# labels = run_topic_gpt(topic_url)
# selected_cluster, selected_payload = select_topic_from_topicgpt(topic_url=topic_url)
# print(selected_cluster)
# print(selected_payload)

# TODO Tobias: Adjust position bias mitigation -> from 20 runs to 5 runs -> queue +4 with seeded random shuffle per run -> ensure 1-4, 5-8, 9-12, 13-16, 17-20

create_openalex_dataset(n_per_field=10, works_n=8000, random_state=42)
dataset_to_csv()
# 3. 測試自動繪製統計圖表與缺失值 EDA 報告
# 將下面這一行的註解解開：
eda()