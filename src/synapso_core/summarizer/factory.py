from .instruct_summarizer import InstructSummarizer
from .interface import Summarizer


class SummarizerFactory:
    """
    A factory for creating summarizers.
    """

    @staticmethod
    def create_summarizer(summarizer_type: str) -> Summarizer | None:
        available_summarizers = {
            "instruct": InstructSummarizer,
        }
        return available_summarizers[summarizer_type]()
