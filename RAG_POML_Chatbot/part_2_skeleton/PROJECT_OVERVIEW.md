# Individual Project

You will build and evaluate an LLM-powered assistant using different prompting techniques and a RAG. The project has two main parts:

- **Part 1: Prompting + LLM-as-a-judge** (completed last week)
- **Part 2: RAG + POML + prompt techniques + LLM-as-a-judge** 


## Part 2: RAG + POML + Prompt Techniques + LLM-as-a-Judge

This part builds a code-documentation Q&A chatbot using a RAG pipeline. It answers questions about **Python**, **Java**, and **JavaScript** by pulling information from documentation PDFs.

We will be using ChromaDB for our vector stores. Take a look at this page to learn more about it: https://docs.trychroma.com/docs/overview/getting-started

### Pipeline:
1. **Build Vector Collections**: Initial check to see if the vector collections exist, if they don't, build them.
2. **Routing**: A POML prompt classifies the question based on what programming language it is referring to.
3. **Retrieval**: ChromaDB searches the vector collection corresponding to that programming language for relevant chunks.
4. **Answering**: A POML prompt generates an answer using the retrieved context from the documentation PDFs, and the LLM call is made through LangChain (ChatGroq).
5. **Comparison (conditional)**: If the comparison option was selected, use model graded evaluation to compare the responses.

### File Structure
- `part_2_skeleton/ingest_files/` — PDFs used for retrieval
- `part_2_skeleton/rag_ingestion/ingest_documents.py` — builds the ChromaDB vector store
- `part_2_skeleton/chatbot/main.py` — CLI entry point and judge evaluation
- `part_2_skeleton/chatbot/rag_pipeline.py` — routing, retrieval, generation, comparisons
- `part_2_skeleton/chatbot/prompts/` — POML prompt templates (routing, answering, judging)

---

## Setup

Make sure you have a `.env` file with your Groq API key in your directory for this class (used by LangChain's ChatGroq).
```
GROQ_API_KEY=your_api_key_here
```

Install Python dependencies:

```bash
pip install -r individual_project/part_2_skeleton/requirements.txt
```

---

## Running the code
1. **Build the vector store** (can either do this manually, or call it when you run the main.py file for the first time):
   ```bash
   python Gen_AI_Course_Site_Spring_2026/individual_project/part_2_skeleton/rag_ingestion/ingest_documents.py
   ```

2. **Run the chatbot CLI**:
   ```bash
   python Gen_AI_Course_Site_Spring_2026/individual_project/part_2_skeleton/chatbot/main.py
   ```

3. **Choose a mode**:
   - Single Model & Single Prompting Technique
   - Compare Multiple Models using Single Prompting Technique
   - Compare Multiple Prompting Techniques using Single Model

---

## What you need to do:
- Fill in TODOs throughout the code
- Edit prompt templates in `chatbot/prompts`
- Set which models are used in `chatbot/rag_pipeline.py`

---

## Deliverables
- .zip file of your code
- Short PDF write-up explaining what you did and any takeaways you have relating to chunking, retrieval, model selection, prompting techniques, or POML. Include either screenshots of copy/pasted outputs of test conversations with the chatbot.

If you prefer to build off of your implementation of part 1, feel free to do so. Otherwise you can directly edit this repo for your submission.


## Ways to improve the system
#### If you are able to finish the requirements and want to take things further, here are a few ideas:
- Right now the routing and judge POML prompts only use a single prompting technique. You could add multiple options for prompting techniques and compare results.
- Try out different retrieval methods. We will cover some of these later in the class, but if you want to give them a try you can. Do some research on ChromaDBs options for cosine similarity, L2 Distance, keyword, and hybrid search.
- Implement multi-turn conversations with chat memory.
- Switch out the ingestion documents to different topics you find interesting.
- Any other ideas that you can come up with that you would like to include!
