import re
import os
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

    def request(self, topic: str) -> tuple[str, str]:
        """
        Send a prompt to the model and return two outputs:
        - answer text
        - bibtex entries (string)

        The base prompt is read from prompts/citation.txt and the topic is appended.
        :param topic: The subject/topic to append to the prompt
        """
        prompt_path = os.path.join("prompts", "citation.txt")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Error: The file '{prompt_path}' was not found.")

        # Load the base prompt
        with open(prompt_path, "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()

        # Construct full prompt
        full_prompt = f"{base_prompt}\n\nTopic: {topic}"

        # Send request
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant. Always return two outputs:\n"
                            "1. Answer text.\n"
                            "2. BibTeX entries inside a fenced code block labeled 'bibtex'."},
                {"role": "user", "content": full_prompt},
            ]
        )

        content = response.choices[0].message.content.strip()

        # Extract BibTeX from ```bibtex ... ```
        bibtex_match = re.search(r"```bibtex\n(.*?)```", content, re.DOTALL)
        bibtex = bibtex_match.group(1).strip() if bibtex_match else ""

        # Remove the bibtex block from the answer text
        answer = re.sub(r"```bibtex\n.*?```", "", content, flags=re.DOTALL).strip()

        return answer, bibtex

