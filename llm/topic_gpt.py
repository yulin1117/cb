import json
import re
import os
from typing import Any

from llm.llm_api import LLM
from llm.prompts import TopicGPTPrompt


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

    def label_cluster(self, cluster_id: int, abstracts: list[str], umbrella_display_name: str,
                      umbrella_description: str) -> dict[str, Any]:
        """
        Generate a TopicGPT label for one cluster.
        :param cluster_id: Cluster identifier
        :param abstracts: Representative abstracts for this cluster
        :param umbrella_display_name: OpenAlex umbrella topic display_name
        :param umbrella_description: OpenAlex umbrella topic description
        :return: Parsed TopicGPT JSON result (or fallback with raw_output)
        """
        topic_lines = [
            "Umbrella topic (OpenAlex display_name):",
            umbrella_display_name.strip(),
            "",
            "Umbrella description (context only):",
            umbrella_description.strip(),
            "",
            f"Cluster {cluster_id} representative abstracts:",
        ]
        for i, a in enumerate(abstracts, start=1):
            topic_lines.append(f"[{i}] {a.strip()}")

        answer, _ = self.llm.request(
            prompt=self.prompt,
            topic="\n".join(topic_lines),
            use_metadata=False,
            context_docs=None,
        )

        parsed = self._extract_json_topic(answer)
        if parsed is not None:
            parsed.setdefault("topic_name", "")
            parsed.setdefault("description", "")
            parsed.setdefault("keywords", [])
            parsed.setdefault("confidence", "medium")
            return parsed

        return {
            "topic_name": "",
            "description": "",
            "keywords": [],
            "confidence": "low",
            "raw_output": answer,
        }
