from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


# Template for custom metrics
class CustomMetric(BaseMetric):
    @property
    def __name__(self):
        return "My Custom Metric"

    def generate_hypothetical_score(self):
        pass

    async def async_generate_hypothetical_score(self):
        return self.generate_hypothetical_score()

    def measure(self, test_case: LLMTestCase) -> float:
        # Although not required, we recommend catching errors
        # in a try block
        try:
            self.score = self.generate_hypothetical_score(test_case)
            if self.include_reason:
                self.reason = self.generate_hypothetical_reason(test_case)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            # set metric error and re-raise it
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        # Although not required, we recommend catching errors
        # in a try block
        try:
            self.score = await self.async_generate_hypothetical_score(test_case)
            if self.include_reason:
                self.reason = await self.async_generate_hypothetical_reason(test_case)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            # set metric error and re-raise it
            self.error = str(e)
            raise

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success