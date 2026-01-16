import os


class Prompt:
    """
    Base class for reading prompt templates from the 'prompts/' directory.
    Subclasses must define a 'filename' attribute.
    """
    def __init__(self):
        if not hasattr(self, 'filename'):
            raise NotImplementedError("Subclasses must define a 'filename' attribute.")
        self.prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load the prompt text from the file."""
        path = os.path.join('prompts', self.filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()


class RelatedWork(Prompt):
    filename = "related_work.txt"

    def __init__(self):
        super().__init__()
        self.context = True
        self.name = "Related-Work"


class QuestionAnswering(Prompt):
    filename = "question_answering.txt"

    def __init__(self):
        super().__init__()
        self.context = True
        self.name = "Question-Answering"


class FactVerification(Prompt):
    filename = "fact_verification.txt"

    def __init__(self):
        super().__init__()
        self.context = True
        self.name = "Fact-Verification"


class TopicGPTPrompt(Prompt):
    filename = "topic_gpt.txt"

    def __init__(self):
        super().__init__()
        self.context = False
        self.name = "TopicGPT"

