from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv();

CHAT_MODEL = ChatOllama(model="llama3.1:8b", temperature=0.6)

PROMPT = "What is the capital of India"
chat_response = CHAT_MODEL.invoke(PROMPT).content
print("Chat model output:", chat_response)

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True
)
test_case = LLMTestCase(
    input=PROMPT,
    actual_output=chat_response,
)

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

evaluate(test_cases=[test_case], metrics=[metric])