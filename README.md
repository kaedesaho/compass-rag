# Cornell College Compass Q&A
> Built: August 2025

A local RAG (Retrieval-Augmented Generation) application that lets you ask questions about Cornell College's student policies using the Compass handbook.

## How it works

1. The app loads and chunks the Compass PDF handbook
2. Chunks are embedded with `nomic-embed-text` and stored in a local FAISS vector index
3. When you ask a question, a `MultiQueryRetriever` generates multiple query variations to retrieve the most relevant context
4. `llama3.1` (via Ollama) answers your question based strictly on the retrieved context

## Requirements

- [Ollama](https://ollama.com) running locally with the following models pulled:
  ```
  ollama pull llama3.1
  ollama pull nomic-embed-text
  ```
- Python 3.11+

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

On first run, the app builds the FAISS vector index from `data/compass.pdf` and saves it to `faiss-db/`. Subsequent runs load the saved index directly.

## Project structure

```
Compass-RAG/
├── app.py              # Main Streamlit app
├── data/
│   └── compass.pdf     # Cornell College Compass handbook
├── faiss-db/           # Generated FAISS vector index (auto-created)
├── requirements.txt
└── .devcontainer/
    └── devcontainer.json
```
