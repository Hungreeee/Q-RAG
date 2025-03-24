from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from ragas.metrics import NonLLMContextRecall, NonLLMContextPrecisionWithReference
from ragas.dataset_schema import SingleTurnSample


class ContextRecall(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        self.metric = NonLLMContextRecall()

    @property
    def __name__(self):
        return "Context Recall"

    def generate_score(self, test_case: LLMTestCase):
        sample = SingleTurnSample(
            retrieved_contexts=test_case.retrieval_context,
            reference_contexts=test_case.context,
        )
        score = self.metric.single_turn_score(sample)
        return score

    async def async_generate_score(self, test_case: LLMTestCase):
        return self.generate_score(test_case)

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            self.score = self.generate_score(test_case)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        try:
            self.score = await self.async_generate_score(test_case)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success
        

class ContextPrecision(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        self.metric = NonLLMContextPrecisionWithReference()

    @property
    def __name__(self):
        return "Context Precision"

    def generate_score(self, test_case: LLMTestCase):
        sample = SingleTurnSample(
            retrieved_contexts=test_case.retrieval_context,
            reference_contexts=test_case.context,
        )
        score = self.metric.single_turn_score(sample)
        return score

    async def async_generate_score(self, test_case: LLMTestCase):
        return self.generate_score(test_case)

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            self.score = self.generate_score(test_case)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        try:
            self.score = await self.async_generate_score(test_case)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success