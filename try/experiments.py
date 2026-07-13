from llm.llm_api import LLM
from CitationBias.test.dataset import Dataset

import random
import os
import json


def run_task(topic: str, prompt, model, biases: list[str] | None = None, iterations: int = 1):
    """
    Run the same task multiple times using a queueing system to rotate
    document positions, avoiding position bias. The total number of runs
    will be len(context_docs) * iterations.

    :param topic: Topic string for the LLM request
    :param prompt: Prompt class instance
    :param model: LLM client instance or model name
    :param biases: List with biases
    :param iterations: Number of full queue cycles (each doc in each position once per cycle)
    """
    results = []
    dataset = Dataset()
    context_docs = dataset.load(topic=topic, biases=biases)

    llm = LLM(model=model)

    # Shuffle once
    doc_items = list(context_docs.items())
    random.shuffle(doc_items)

    n_docs = len(doc_items)
    total_runs = n_docs * iterations

    print(f"Running {total_runs} total iterations ({iterations} full queue cycles)")

    for i in range(total_runs):
        # Create current context order
        current_order = doc_items.copy()
        shuffled_context_docs = dict(current_order)

        # Call LLM
        answer, citations = llm.request(
            topic=topic,
            prompt=prompt,
            context_docs=shuffled_context_docs
        )

        results.append({
            "iteration": i + 1,
            "order": [k for k, _ in current_order],
            "answer": answer,
            "citations": citations
        })

        # After each iteration, rotate queue by 1
        doc_items = doc_items[1:] + doc_items[:1]

    # Prepare output directory
    safe_model = str(model).replace("/", "_").replace(":", "_")
    model_dir = os.path.join("out", safe_model)
    os.makedirs(model_dir, exist_ok=True)

    safe_topic = topic.replace(" ", "-")

    # Filter biases, remove Title and Abstract
    filtered_biases = [b for b in (biases or []) if b.lower() not in ["title", "abstract"]]

    # Build filename
    bias_suffix = f"_{'_'.join(filtered_biases)}" if filtered_biases else ""
    filename = f"{prompt.name}_{safe_topic}{bias_suffix}.json"

    file_path = os.path.join(model_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved all {total_runs} iterations to {file_path}")
