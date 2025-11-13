from llm.prompts import Prompt

import re
import os
import csv
import json
from openai import OpenAI


class LLM:
    def __init__(self, model: str, temperature: float = 0.3):
        """
        Initialize the LLM client
        :param model: The model identifier (e.g., "meta-llama/Llama-3.3-70B-Instruct")
        :param temperature: Sampling temperature
        """
        api_key_path = "api_keys/scads_llm.txt"
        try:
            with open(api_key_path, "r") as keyfile:
                api_key = keyfile.readline().strip()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: The file '{api_key_path}' was not found. Please make sure it exists and contains your API key."
            )

        base_url = "https://llm.scads.ai/v1"
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

    def request(self, prompt: Prompt, topic: str, use_metadata: bool = False,
                context_docs: dict[str, dict[str, str]] | None = None) -> tuple[str, list[str]]:
        """
        Send a prompt to the model and return two outputs:
        - answer text
        - citations (list)

        Uses the class prompt and, if context is enabled, adds structured context documents.
        Adds explanations for all metadata fields used in the context, read from data/metadata/metadata.csv,
        only if use_metadata=True. Metadata descriptions preserve the same order as in context_docs.

        :param prompt: Specify prompt / task for LLM request
        :param topic: The subject/topic to append to the prompt
        :param use_metadata: Include metadata description in prompt (if True)
        :param context_docs: dict of {doc_id: {feature: text}}
        """
        base_prompt = prompt.prompt
        include_context = getattr(prompt, "context", False)

        metadata_text = ""
        metadata_descriptions = {}

        # --- Load metadata descriptions only if explicitly requested ---
        if use_metadata:
            metadata_path = os.path.join("data", "metadata", "metadata.csv")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = row.get("Metadata")
                        desc = row.get("Description")
                        if key and desc:
                            metadata_descriptions[key.strip()] = desc.strip()

        # --- Build structured context if needed ---
        context_text = ""
        if include_context and context_docs:
            # Determine metadata order from first paper
            first_paper_features = next(iter(context_docs.values()))
            ordered_metadata = list(first_paper_features.keys())

            # Add explanations for each metadata field in original order (only if metadata loaded)
            if use_metadata and metadata_descriptions:
                explanations = []
                for meta in ordered_metadata:
                    desc = metadata_descriptions.get(meta, "")
                    explanations.append(f"- {meta}: {desc}" if desc else f"- {meta}: (no description available)")
                metadata_text = "Metadata Definitions:\n" + "\n".join(explanations)

            # Add structured paper context
            context_parts = []
            for doc_id, features in context_docs.items():
                context_parts.append(f"Paper-ID: {doc_id}")
                for feature, text in features.items():
                    context_parts.append(f"  - {feature}: {text.strip()}")
                context_parts.append("")  # blank line between docs
            context_text = "Papers:\n" + "\n".join(context_parts).strip()

        # --- Construct full prompt ---
        full_prompt = f"{base_prompt}\n\nTopic:\n{topic}"
        if metadata_text:
            full_prompt += f"\n\n{metadata_text}"
        if context_text:
            full_prompt += f"\n\n{context_text}"

        # --- Send request to model ---
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Always return two outputs:\n"
                        "1. Answer text.\n"
                        "2. A JSON list of all paper IDs cited in the generated text, "
                        "inside a fenced code block labeled 'json'."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
        )

        content = response.choices[0].message.content.strip()

        # --- Extract cited paper IDs from ```json [...] ``` ---
        citations_match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
        if citations_match:
            try:
                citations = json.loads(citations_match.group(1).strip())
            except json.JSONDecodeError:
                citations = []
        else:
            citations = []

        # --- Remove the JSON block from answer text ---
        answer = re.sub(r"```json\n.*?```", "", content, flags=re.DOTALL).strip()

        return answer, citations

