import json
import re
import os
from typing import Any

from llm.llm_api import LLM
from llm.prompts import TopicGPTPrompt
from openalex import get_openalex_topics


class TopicGPT:
    """
    TopicGPT-style topic labeling using an LLM.
    Wraps the existing LLM class without modifying it.
    """

    def __init__(self, model: str, temperature: float = 0.2):
        """
        :param model: LLM model identifier
        :param temperature: Sampling temperature (low = stable labels)
        """
        self.llm = LLM(model=model, temperature=temperature)
        self.prompt = TopicGPTPrompt()

    @staticmethod
    def _extract_json_topic(text: str) -> dict[str, Any] | None:
        """
        Extract JSON from ```json_topic ...``` block.

        :param text: LLM response text
        :return: Parsed dict or None
        """
        match = re.search(r"```json_topic\s*\n(.*?)```", text, re.DOTALL)
        if not match:
            return None

        try:
            data = json.loads(match.group(1).strip())
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def run_topic_gpt(
            topic_url: str,
            model: str = "meta-llama/Llama-3.3-70B-Instruct",
            temperature: float = 0.2,
            representative_abstracts: int = 20,
            out_root: str = os.path.join("out", "topics"),
    ) -> dict[int, dict[str, Any]]:
        """
        Run TopicGPT labeling on clustered representative abstracts.

        Loads umbrella topic metadata (display_name + description) from data/openalex/topics.json.
        If the file is missing, calls get_openalex_topics() to create it.

        :param topic_url: OpenAlex topic URL
        :param model: LLM model identifier
        :param temperature: Sampling temperature
        :param representative_abstracts: Number of reps per cluster (must match filename)
        :param out_root: Output root directory
        :return: Dict mapping cluster_id -> TopicGPT result
        """
        # -----------------------------
        # Resolve topic_id and paths
        # -----------------------------
        topic_id: str = topic_url.rstrip("/").split("/")[-1]  # e.g. "T10346"
        save_dir: str = os.path.join(out_root, topic_id)

        reps_path: str = os.path.join(save_dir, f"representatives_top{representative_abstracts}.txt")
        if not os.path.exists(reps_path):
            raise FileNotFoundError(f"Missing representatives file: {reps_path}")

        # -----------------------------
        # Load OpenAlex umbrella topic metadata
        # -----------------------------
        topics_path: str = os.path.join("data", "openalex", "topics.json")

        if not os.path.exists(topics_path):
            # Create it, then try again
            get_openalex_topics()
            if not os.path.exists(topics_path):
                raise FileNotFoundError(
                    f"{topics_path} not found even after calling get_openalex_topics()."
                )

        with open(topics_path, "r", encoding="utf-8") as f:
            topics_data = json.load(f)

        # topics.json may be a list[dict] or dict[str, dict] depending on your pipeline.
        topic_obj: dict[str, Any] | None = None

        if isinstance(topics_data, list):
            # Each element has e.g. {"id": "https://openalex.org/T10346", ...}
            for t in topics_data:
                if isinstance(t, dict) and (t.get("id") == topic_url or t.get("id", "").endswith(f"/{topic_id}")):
                    topic_obj = t
                    break
        elif isinstance(topics_data, dict):
            # Could be keyed by topic_id or topic_url
            topic_obj = topics_data.get(topic_id) or topics_data.get(topic_url)

            # Or could be {"results": [...]} depending on your fetch format
            if topic_obj is None and "results" in topics_data and isinstance(topics_data["results"], list):
                for t in topics_data["results"]:
                    if isinstance(t, dict) and (t.get("id") == topic_url or t.get("id", "").endswith(f"/{topic_id}")):
                        topic_obj = t
                        break

        if topic_obj is None:
            raise KeyError(
                f"Topic {topic_url} ({topic_id}) not found in {topics_path}. "
                f"Check how topics.json is structured."
            )

        umbrella_display_name: str = str(topic_obj.get("display_name", "")).strip()
        umbrella_description: str = str(topic_obj.get("description", "")).strip()

        if not umbrella_display_name:
            # If OpenAlex metadata is incomplete, still proceed, but labels may get broader.
            umbrella_display_name = topic_id

        # -----------------------------
        # Parse representatives file
        # -----------------------------
        clusters: dict[int, list[str]] = {}
        current_cluster: int | None = None

        header_re = re.compile(r"^Cluster\s+(\d+)\s+\(n=\d+\):")
        rep_re = re.compile(r"^\s*\[\d+\]\s+(.*)$")

        with open(reps_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")

                header = header_re.match(line)
                if header:
                    current_cluster = int(header.group(1))
                    clusters[current_cluster] = []
                    continue

                rep = rep_re.match(line)
                if rep and current_cluster is not None:
                    clusters[current_cluster].append(rep.group(1))

        if not clusters:
            raise ValueError("No clusters found in representatives file.")

        # -----------------------------
        # Run TopicGPT
        # -----------------------------
        topic_gpt = TopicGPT(model=model, temperature=temperature)

        results: dict[int, dict[str, Any]] = {}
        for cluster_id in sorted(clusters):
            results[cluster_id] = topic_gpt.label_cluster(
                cluster_id=cluster_id,
                abstracts=clusters[cluster_id],
                umbrella_display_name=umbrella_display_name,
                umbrella_description=umbrella_description,
            )

        # -----------------------------
        # Save results
        # -----------------------------
        out_path = os.path.join(save_dir, "topicgpt_labels.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "topic_id": topic_id,
                    "topic_url": topic_url,
                    "umbrella_display_name": umbrella_display_name,
                    "umbrella_description": umbrella_description,
                    "model": model,
                    "temperature": temperature,
                    "representative_abstracts": representative_abstracts,
                    "clusters": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("Saved TopicGPT labels to:", out_path)
        return results
