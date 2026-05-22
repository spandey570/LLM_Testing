import os

import requests
from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

CHAT_API_URL = os.getenv("CHAT_API_URL")
TOP_K = int(os.getenv("EVAL_TOP_K"))

# Local judge for DeepEval metrics (requires Ollama running: ollama serve)
JUDGE_MODEL = OllamaModel(
    model=os.getenv("EVAL_JUDGE_MODEL"),
    temperature=float(os.getenv("EVAL_JUDGE_TEMPERATURE")),
)

evaluation_dataset = [
    {
        "input": "What is Space X",
        "expected_output": (
            "SpaceX is an American aerospace company founded by Elon Musk."
        ),
    },
    {
        "input": "Who founded Tesla?",
        "expected_output": (
            "Tesla was founded by Martin Eberhard and Marc Tarpenning."
        ),
    }
]

# metric_precision = ContextualPrecisionMetric(
#     threshold=0.7,
#     model=JUDGE_MODEL,
#     include_reason=True,
# )

metric_recall = ContextualRecallMetric(
    threshold=0.6,
    model=JUDGE_MODEL,
    include_reason=True,
)

test_cases = []
for item in evaluation_dataset:
    question = item["input"]
    expected_output = item["expected_output"]

    response = requests.post(
        CHAT_API_URL,
        json={"question": question, "top_k": TOP_K},
        timeout=60,
    ).json()

    print(f"Question: {question}")
    print(f"Generated answer: {response['generated_answer']}\n")

    test_cases.append(
        LLMTestCase(
            input=response["question"],
            actual_output=response["generated_answer"],
            retrieval_context=response["retrieved_chunks"],
            expected_output=expected_output,
        )
    )

evaluate(test_cases=test_cases, metrics= [metric_recall])
