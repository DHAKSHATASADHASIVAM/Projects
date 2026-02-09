"""
Main CLI for Code Documentation Q&A (RAG + POML)

Options:
1. Single model & single technique
2. Compare multiple models (same technique)
3. Compare multiple techniques (same model)
"""

import subprocess
import sys
from pathlib import Path

import chromadb
import json
import re
from langchain_groq import ChatGroq
from prompt_renderer import get_answer_techniques, render_judge_prompt
from rag_pipeline import (
    run_rag_pipeline,
    run_model_comparison,
    run_technique_comparison,
    COMPARISON_MODELS,
    JUDGE_MODEL,
)


def display_menu(title: str, options: list) -> int:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print("=" * 60)

    while True:
        try:
            choice = int(input("\nEnter choice: ").strip())
            if 1 <= choice <= len(options):
                return choice
            print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")


def select_prompt_technique(title: str, techniques: dict) -> str:
    technique_list = list(techniques.items())

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for i, (key, name) in enumerate(technique_list, 1):
        print(f"  {i}. {name}")
    print("=" * 60)

    while True:
        try:
            choice = int(input("\nEnter choice: ").strip())
            if 1 <= choice <= len(technique_list):
                return technique_list[choice - 1][0]
            print(f"Please enter a number between 1 and {len(technique_list)}")
        except ValueError:
            print("Please enter a valid number")


def select_model() -> dict:
    print("\n" + "=" * 60)
    print("Select Model")
    print("=" * 60)
    for i, model in enumerate(COMPARISON_MODELS, 1):
        print(f"  {i}. {model['name']}")
    print("=" * 60)

    while True:
        try:
            choice = int(input("\nEnter choice: ").strip())
            if 1 <= choice <= len(COMPARISON_MODELS):
                return COMPARISON_MODELS[choice - 1]
            print(f"Please enter a number between 1 and {len(COMPARISON_MODELS)}")
        except ValueError:
            print("Please enter a valid number")


def get_question() -> str:
    print("\n" + "-" * 60)
    question = input("Enter your question: ").strip()
    if not question:
        print("Question cannot be empty")
        return get_question()
    return question


def display_results(result: dict):
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    route = "unknown"
    if result.get("routing"):
        routing = result["routing"]
        route = routing.get("route", route)
        print("\n[Routing]")
        print(f"  Routed to: {route}")
        if routing.get("message"):
            print(f"  Message: {routing['message']}")

    if result.get("chunks"):
        print(f"\n[Retrieved Context] ({len(result['chunks'])} chunks)")
        for chunk in result["chunks"]:
            preview = chunk["content"][:100] + "..." if len(chunk["content"]) > 100 else chunk["content"]
            distance = chunk.get("distance", None)
            chunk_index = chunk.get("metadata", {}).get("chunk_index", "unknown")
            if distance is not None:
                print(f"\n--Chunk {chunk_index} (collection: {route}, distance: {distance:.4f}):")
            else:
                print(f"\n--Chunk {chunk_index} (collection: {route}):")
            print(f"    {preview}")

    if result.get("error"):
        print(f"\n[Error] {result['error']}")

    print("\n" + "-" * 60)
    print("[Answer]")
    print("-" * 60)
    print(result.get("response", "No response generated"))

    print("\n" + "=" * 60)


def display_model_comparison(model_responses: list):
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for resp in model_responses:
        print(f"\n{'─' * 60}")
        print(f"📌 {resp['model_name']}")
        print("─" * 60)
        if resp["error"]:
            print(f"  ❌ Error: {resp['error']}")
        else:
            print(resp["response"])

    print("\n" + "=" * 60)


def display_technique_comparison(technique_responses: list):
    print("\n" + "=" * 60)
    print("PROMPTING TECHNIQUE COMPARISON")
    print("=" * 60)

    for resp in technique_responses:
        print(f"\n{'─' * 60}")
        print(f"📝 {resp['technique_name']}")
        print("─" * 60)
        if resp["error"]:
            print(f"  ❌ Error: {resp['error']}")
        else:
            print(resp["response"])

    print("\n" + "=" * 60)


def display_judge_evaluation(judge_result: dict):
    print("\n" + "=" * 60)
    print("LLM JUDGE EVALUATION")
    print("=" * 60)

    if judge_result.get("error"):
        print(f"\n  Error: {judge_result['error']}")
        if "raw_response" in judge_result:
            raw = judge_result["raw_response"]
            print("\n  Raw response:")
            print(raw if raw else "<empty>")
        print("\n" + "=" * 60)
        return

    rankings = judge_result.get("rankings", [])
    if rankings:
        print("\nRANKINGS:")
        rank_labels = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
        for i, item in enumerate(rankings):
            label = rank_labels[i] if i < len(rank_labels) else f"{i + 1}th"
            print(f"  {label}: {item}")

    feedback = judge_result.get("feedback", {})
    if feedback:
        print("\nFEEDBACK:")
        for item, comment in feedback.items():
            print(f"\n  {item}:")
            print(f"    {comment}")

    summary = judge_result.get("summary", "")
    if summary:
        print("\nSUMMARY:")
        print(f"  {summary}")

    print("\n" + "=" * 60)


def evaluate_responses(question: str, responses: list, comparison_type: str, judge_model: str) -> dict:
    """Evaluate and rank responses using an LLM judge."""
    if not responses:
        return {"error": "No responses to evaluate"}

    #  use the correct function from the prompt_renderer.py file
    prompt = (question, responses, comparison_type)
    prompt = render_judge_prompt(question, responses, comparison_type)

    try:
        llm = ChatGroq(
            model=judge_model,
            temperature=0.0,
        )
        response = llm.invoke(prompt)
        content = response.content.strip()
        if not content:
            return {
                "error": "Judge returned an empty response",
                "raw_response": "",
            }

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            cleaned = content.replace("```json", "").replace("```", "").strip()
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    sanitized = re.sub(r"[\r\n]+", " ", match.group(0)).strip()
                    try:
                        return json.loads(sanitized)
                    except json.JSONDecodeError:
                        pass
            return {
                "error": "Judge did not return valid JSON",
                "raw_response": content,
            }

    except Exception as e:
        return {"error": str(e)}


def ensure_vector_store() -> bool:
    """Check for existing vector store; optionally build it if missing."""
    chroma_path = Path(__file__).parent.parent / "rag_ingestion" / "chroma_db"
    expected_collections = {"python_book", "java_book", "javascript_book", "C_book"}

    if chroma_path.exists() and any(chroma_path.iterdir()):
        client = chromadb.PersistentClient(path=str(chroma_path))
        existing = {c.name for c in client.list_collections()}
        if expected_collections.issubset(existing):
            return True

    print("\nNo vector store found. You need to build it before running the chatbot.")
    choice = input("Build vector store now? (y/n): ").strip().lower()
    if choice != "y":
        return False

    ingest_script = Path(__file__).parent.parent / "rag_ingestion" / "ingest_documents.py"
    print("\nBuilding vector store...")
    result = subprocess.run([sys.executable, str(ingest_script)])
    return result.returncode == 0


def main():
    if not ensure_vector_store():
        print("Exiting without running chatbot.")
        return

    print("\nCode Documentation Chatbot")

    mode = display_menu(
        "Select Response Mode",
        [
            "Single Model & Single Prompting Technique",
            "Compare Multiple Models using Single Prompting Technique",
            "Compare Multiple Prompting Techniques using Single Model",
        ],
    )

    if mode == 1:
        response_technique = select_prompt_technique(
            "Select Prompting Technique for Response Generation",
            get_answer_techniques(),
        )
        question = get_question()
        result = run_rag_pipeline(question, response_technique)
        display_results(result)

    elif mode == 2:
        response_technique = select_prompt_technique(
            "Select Prompting Technique for Response Generation",
            get_answer_techniques(),
        )
        question = get_question()
        result = run_model_comparison(question, response_technique)
        if result.get("error"):
            print(f"\nError: {result['error']}")
            return
        display_model_comparison(result["model_responses"])
        judge_result = evaluate_responses(
            question=question,
            responses=[r for r in result["model_responses"] if not r["error"]],
            comparison_type="models",
            judge_model=JUDGE_MODEL,
        )
        display_judge_evaluation(judge_result)

    elif mode == 3:
        model = select_model()
        question = get_question()
        result = run_technique_comparison(question, model_id=model["id"])
        if result.get("error"):
            print(f"\nError: {result['error']}")
            return
        display_technique_comparison(result["technique_responses"])
        judge_result = evaluate_responses(
            question=question,
            responses=[r for r in result["technique_responses"] if not r["error"]],
            comparison_type="techniques",
            judge_model=JUDGE_MODEL,
        )
        display_judge_evaluation(judge_result)


if __name__ == "__main__":
    main()
