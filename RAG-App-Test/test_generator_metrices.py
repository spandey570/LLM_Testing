import os

import requests
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

CHAT_API_URL = os.getenv("CHAT_API_URL")
TOP_K = int(os.getenv("EVAL_TOP_K"))

evaluation_dataset = [
    {
        "input": "What is the relationship between QA Labs For You and QA SAMVAAD?",
        "expected_output": "QA SAMVAAD is a knowledge-sharing talk show initiative under QA Labs For You. It provides a platform where QA professionals, automation engineers, AI enthusiasts, and industry experts share their experiences, career journeys, automation strategies, AI testing insights, and industry trends to help the QA community learn and grow.",
    },
    {
        "input": "How does QA SAMVAAD inspire the QA community?",
        "expected_output": "QA SAMVAAD inspires the QA community by sharing real-world experiences, career journeys, automation practices, AI adoption strategies, enterprise testing challenges, mentorship, and practical learning insights from industry professionals and experts. It encourages continuous learning and community-driven growth.",
    },
    {
        "input": "Is QA Labs For You a gaming platform?",
        "expected_output": "No, QA Labs For You is not a gaming platform. It is a learning-focused QA and AI community initiative that helps software testing professionals learn modern testing technologies, automation, Generative AI, RAG systems, Playwright, API testing, and AI-powered QA practices.",
    },
    {
        "input": "Why QBot is developed?",
        "expected_output": "QBot was developed as an AI-powered knowledge assistant under QA Labs For You to help QA engineers and AI practitioners understand real-world Retrieval-Augmented Generation (RAG) systems. It provides context-aware responses, semantic search, document-based question answering, retrieval inspection, DeepEval metric analysis, hallucination analysis, and AI evaluation workflows for practical learning.",
    },
]

metric_relevancy = AnswerRelevancyMetric(
    threshold=0.6,
    model="gpt-4o-mini",
    include_reason=True,
)

metric_faithfulness = FaithfulnessMetric(
    threshold=0.6,
    model="gpt-4o-mini",
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

evaluate(test_cases=test_cases, metrics=[metric_relevancy, metric_faithfulness])
