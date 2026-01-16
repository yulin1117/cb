import json
import re
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

    def label_cluster(self, cluster_id: int, abstracts: list[str]) -> dict[str, Any]:
        """
        Generate a TopicGPT label for one cluster.

        :param cluster_id: Cluster identifier
        :param abstracts: Representative abstracts
        :return: TopicGPT output dict
        """
        topic_text = [f"Cluster {cluster_id} representative abstracts:"]
        for i, text in enumerate(abstracts, start=1):
            topic_text.append(f"[{i}] {text.strip()}")

        answer, _ = self.llm.request(
            prompt=self.prompt,
            topic="\n".join(topic_text),
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

        # Fallback (never crash the pipeline)
        return {
            "topic_name": "",
            "description": "",
            "keywords": [],
            "confidence": "low",
            "raw_output": answer,
        }
