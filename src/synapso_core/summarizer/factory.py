from .instruct_summarizer import InstructSummarizer
from .interface import Summarizer


class SummarizerFactory:
    """
    A factory for creating summarizers.
    """

    @staticmethod
    def create_summarizer(summarizer_type: str) -> Summarizer:
        available_summarizers = {
            "instruct": InstructSummarizer,
        }
        if summarizer_type not in available_summarizers:
            raise ValueError(f"Invalid summarizer type: {summarizer_type}")
        return available_summarizers[summarizer_type]()
