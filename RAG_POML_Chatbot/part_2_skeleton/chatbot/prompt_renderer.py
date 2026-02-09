"""
POML Prompt Renderer

This file handles using parameters passed to it in conjunction with the POML prompt files 
to build prompts to be sent to the LLMs.
"""

from pathlib import Path
from poml import poml

ROUTING_POML = Path(__file__).parent / "prompts" / "route.poml"
ANSWER_POML = Path(__file__).parent / "prompts" / "answer.poml"
JUDGE_POML = Path(__file__).parent / "prompts" / "judge.poml"

ANSWER_TECHNIQUES = {
    "zero_shot": "Zero-Shot",
    "few_shot": "Few-Shot",
    "cot": "Chain of Thought",
    "advanced": "Task Decomposition" #: fill in the information on your selected advanced prompt
}


def render_routing_prompt(question: str) -> str:
    poml_rendered = poml(
        str(ROUTING_POML),
        chat=True,
        format="langchain",
        # : fill in the context that should be passed to the router prompt
        context={
            "question": question
        }
    )

    return poml_rendered["messages"][0]["data"]["content"]


def render_answer_prompt(technique: str, question: str, 
                         context: str, language: str,
                         conversation_context: str = "") -> str:
    if technique not in ANSWER_TECHNIQUES:
        raise ValueError(
            f"Invalid technique: {technique}. Must be one of {list(ANSWER_TECHNIQUES.keys())}"
        )

    poml_rendered = poml(
        str(ANSWER_POML),
        chat=True,
        format="langchain",
        # : fill in the context that should be passed to the answer prompt
        context={
            "technique": technique,
            "question": question,
            "context": context,
            "language": language,
            "conversation_context": conversation_context
        }
    )

    # : correct the return statement to return just the content of the rendered poml prompt.
    return poml_rendered["messages"][0]["data"]["content"]


def render_judge_prompt(question: str, responses: list, comparison_type: str) -> str:
    # Build a list of dicts for each LLM response with it's label and response content
    responses_list = []
    for idx, item in enumerate(responses, 1):
        label = item.get("model_name") or item.get("technique_name") or f"Item {idx}"
        responses_list.append(
            {
                "label": label,
                "response": item.get("response", ""),
            }
        )

    poml_rendered = poml(
        str(JUDGE_POML),
        chat=True,
        format="langchain",
        context={
            "question": question,
            "comparison_type": comparison_type,
            "responses_list": responses_list,
        }
    )

    return poml_rendered["messages"][0]["data"]["content"]


def get_answer_techniques() -> dict:
    """Returns a dictionary of the different answer prompting techniques"""
    return ANSWER_TECHNIQUES
