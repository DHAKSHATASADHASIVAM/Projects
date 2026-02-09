"""
RAG Pipeline for Code Documentation Q&A

Flow:
1. Route question to correct collection using selected prompting technique
2. Query ChromaDB collection
3. Generate response using an LLM with retrieved context
4. Optional comparisons (models or techniques) and LLM-as-a-Judge evaluation
"""

import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import chromadb
from chromadb.utils import embedding_functions
import logging
from transformers import logging as transformers_logging




from prompt_renderer import (
    render_routing_prompt,
    render_answer_prompt,
    get_answer_techniques,
)

load_dotenv()

# Load Groq API Key from .env file
#  load in your Groq API key from your .env file.
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is required. Add it to your .env file.")


# Define where the chroma db vector collections are.
CHROMA_PATH = Path(__file__).parent.parent / "rag_ingestion" / "chroma_db"
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
transformers_logging.set_verbosity_error()

# The specific embedding model we will use
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

COLLECTIONS = ["python_book", "java_book", "javascript_book", "C_book"]


COMPARISON_MODELS = [
    {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B"},
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B"},
    {"id": "qwen/qwen3-32b", "name": "Qwen 3 32B"},
]
ROUTER_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL = "llama-3.3-70b-versatile"


def extract_json(text: str) -> dict:
    """Extract JSON object from LLM response."""
    json_pattern = r"\{[^{}]*\}"
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            return json.loads(match) 
        except json.JSONDecodeError:
            continue

    try:
        return json.loads(match) # load the JSON formatted string, 'matches' into a python dictionary
    except json.JSONDecodeError:
        return {"route": "CLARIFY", "message": "Could not parse routing response"}


def route_question(question: str) -> dict:
    """Route the question to the appropriate collection."""

    # use the prompt_renderer.py file to build the prompt using the route.poml prompt
    #  use the corrert function from prompt_renderer.py and pass it the question
    prompt = render_routing_prompt(question)

    llm = ChatGroq(
        model=ROUTER_MODEL,
        temperature=  0.0
    )
    response = llm.invoke(prompt)

    llm_response = response.content.strip()
    result = extract_json(llm_response)
    route = result.get("route") or result.get("collection", "CLARIFY")

    return {
        "route": route,
        "message": result.get("message", ""),
        "raw_response": llm_response,
    }


def query_collection(collection_name: str, question: str, top_k: int = 5) -> list:
    """Query ChromaDB collection to look for similar chunks."""
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )

        #  query the specified collection. 
        results = collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        # build a list of information on the retrieved chunks
        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            chunks.append(
                {
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                }
            )

        return chunks

    except Exception as e:
        print(f"Error querying collection: {e}")
        return []


def generate_response(question: str, context_chunks: list, collection_name: str,
                      conversation_context: str = "", response_technique: str = "zero_shot",
                      model_id: str = "llama-3.3-70b-versatile") -> str:
    """Generate a response using an LLM with retrieved context."""

    language_map = {
        "python_book": "Python",
        "java_book": "Java",
        "javascript_book": "JavaScript",
        "C_book": "C",
    }
    language = language_map.get(collection_name, "programming")


    # Call the correct function from prompt_renderer.py
    prompt = render_answer_prompt(
        technique=response_technique,
        question=question,
        context=context_chunks,
        language=language,
        conversation_context=conversation_context,
    )

    llm = ChatGroq(
        model=model_id,
        temperature= 0.5#  set temperature
    )
    #  Invoke the LLM with the prompt
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_multi_model_responses(question: str, context_chunks: list, collection_name: str,
                                   conversation_context: str = "",
                                   response_technique: str = "zero_shot") -> list:
    """Generate responses from multiple models for comparison."""
    responses = []

    for model in COMPARISON_MODELS:
        try:
            answer = generate_response(
                question=question,
                context_chunks=context_chunks,
                collection_name=collection_name,
                conversation_context=conversation_context,
                response_technique=response_technique,
                model_id=model["id"],
            )
            responses.append({"model_name": model["name"], "response": answer, "error": None})
        except Exception as e:
            responses.append({"model_name": model["name"], "response": None, "error": str(e)})

    return responses


def generate_multi_technique_responses(question: str, context_chunks: list, collection_name: str,
                                       conversation_context: str = "",
                                       model_id: str = "llama-3.3-70b-versatile") -> list:
    """Generate responses from multiple prompting techniques for comparison."""
    responses = []
    techniques = get_answer_techniques()

    for technique_key, technique_name in techniques.items():
        try:
            answer = generate_response(
                question=question,
                context_chunks=context_chunks,
                collection_name=collection_name,
                conversation_context=conversation_context,
                response_technique=technique_key,
                model_id=model_id,
            )
            responses.append(
                {
                    "technique_key": technique_key,
                    "technique_name": technique_name,
                    "response": answer,
                    "error": None,
                }
            )
        except Exception as e:
            responses.append(
                {
                    "technique_key": technique_key,
                    "technique_name": technique_name,
                    "response": None,
                    "error": str(e),
                }
            )

    return responses


def run_rag_pipeline(question: str, response_technique: str,
                     conversation_context: str = "") -> dict:
    """Run the full RAG pipeline (route -> query -> generate)."""
    result = {
        "routing": None,
        "chunks": None,
        "response": None,
        "error": None,
    }

    try:
        routing = route_question(question)
        result["routing"] = routing

        if routing["route"] == "CLARIFY":
            result["response"] = routing.get("message") or "Please specify the programming language."
            return result

        if routing["route"] not in COLLECTIONS:
            result["error"] = f"Invalid route: {routing['route']}"
            return result

        chunks = query_collection(routing["route"], question)
        result["chunks"] = chunks

        response = generate_response(
            question=question,
            context_chunks=chunks,
            collection_name=routing["route"],
            conversation_context=conversation_context,
            response_technique=response_technique,
        )
        result["response"] = response

    except Exception as e:
        result["error"] = str(e)

    return result


def run_model_comparison(question: str, response_technique: str) -> dict:
    """Route once, then compare multiple models on the same question."""

    routing = route_question(question)

    if routing["route"] == "CLARIFY":
        return {"error": routing.get("message") or "Please specify the programming language."}

    if routing["route"] not in COLLECTIONS:
        return {"error": f"Invalid route: {routing['route']}"}

    chunks = query_collection(routing["route"], question)

    model_responses = generate_multi_model_responses(
        question=question,
        context_chunks=chunks,
        collection_name=routing["route"],
        response_technique=response_technique,
    )

    return {
        "routing": routing,
        "chunks": chunks,
        "model_responses": model_responses,
    }


def run_technique_comparison(question: str, model_id: str) -> dict:
    """Route once, then compare multiple prompting techniques using one model."""
    routing = route_question(question)

    if routing["route"] == "CLARIFY":
        return {"error": routing.get("message") or "Please specify the programming language."}

    if routing["route"] not in COLLECTIONS:
        return {"error": f"Invalid route: {routing['route']}"}

    chunks = query_collection(routing["route"], question)

    technique_responses = generate_multi_technique_responses(
        question=question,
        context_chunks=chunks,
        collection_name=routing["route"],
        model_id=model_id,
    )

    return {
        "routing": routing,
        "chunks": chunks,
        "technique_responses": technique_responses,
    }
