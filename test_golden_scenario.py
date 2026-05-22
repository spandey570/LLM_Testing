from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.dataset.utils import convert_goldens_to_test_cases

golden_Scenario = Golden(
    input="What if these shoes don't fit?",
    expected_output="We offer a 30-day full refund at no extra cost.",
    actual_output="We offer a 30-day full refund at no extra cost.",  # output from your LLM app
    context=["All customers are eligible for a 30 day full refund at no extra cost."],
)

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True
)
# test_case_01 = LLMTestCase(
#     input="What if these shoes don't fit?",
#     # Replace this with the output from your LLM app
#     actual_output="We offer a 30-day full refund at no extra cost."
# )
# test_case_02 = LLMTestCase(
#     input="What is adidas?",
#     # Replace this with the output from your LLM app
#     actual_output="We offer a 30-day full refund at no extra cost."
# )

dataset = EvaluationDataset()
dataset.add_golden(golden_Scenario)
# dataset.add_test_case(test_case_01)
# dataset.add_test_case(test_case_02)

test_cases = convert_goldens_to_test_cases(dataset.goldens)
evaluate(test_cases=test_cases, metrics=[metric])
