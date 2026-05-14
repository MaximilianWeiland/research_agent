import threading
from openai import OpenAI
from pydantic import BaseModel, Field
from langfuse import get_client

langfuse_client = get_client()
openai_client = OpenAI()

JUDGE_PROMPT = """You are an evaluation judge. Score the assistant response on the following metrics.

Definitions:
- answer_relevance: Does the response directly address the user's question? (0=off-topic, 1=fully addresses it)
- faithfulness: Is every claim in the response supported by the retrieved context? (0=hallucinated, 1=fully grounded)
- conciseness: Is the response focused without unnecessary padding? (0=very padded, 1=tight and focused)

User question: {question}

Retrieved context:
{context}

Assistant response:
{answer}"""


class EvaluationScores(BaseModel):
    answer_relevance: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    conciseness: float = Field(ge=0.0, le=1.0)


def _run_evaluation(trace_id: str, question: str, answer: str, context: str) -> None:
    prompt = JUDGE_PROMPT.format(question=question, context=context or "(no context retrieved)", answer=answer)

    response = openai_client.responses.parse(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0,
        text_format=EvaluationScores,
    )

    scores = response.output_parsed

    for name, value in scores.model_dump().items():
        langfuse_client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
        )


def evaluate_async(trace_id: str, question: str, answer: str, context: str) -> None:
    """Post evaluation scores to Langfuse in a background thread."""
    thread = threading.Thread(
        target=_run_evaluation,
        args=(trace_id, question, answer, context),
        daemon=True,
    )
    thread.start()
