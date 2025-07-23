from .interface import Summarizer

class SummarizerFactory:
    """
    A factory for creating summarizers.
    """

    @staticmethod
    def create_summarizer(summarizer_type: str) -> Summarizer:
        pass