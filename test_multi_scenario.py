from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True
)
test_case_01 = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the output from your LLM app
    actual_output="We offer a 30-day full refund at no extra cost."
)
test_case_02 = LLMTestCase(
    input="What is adidas?",
    # Replace this with the output from your LLM app
    actual_output="We offer a 30-day full refund at no extra cost."
)

dataset = EvaluationDataset()
dataset.add_test_case(test_case_01)
dataset.add_test_case(test_case_02)

#evaluate(test_cases=[test_case_01, test_case_02], metrics=[metric])
evaluate(test_cases=dataset.test_cases, metrics=[metric])