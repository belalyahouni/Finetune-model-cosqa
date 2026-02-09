# Semantic Search Engine with SentenceTransformers and CoSQA Fine-Tuning
This project implements a semantic search engine that allows users to upload documents (PDF or text) and perform natural language searches using a fine-tuned model.

Note: Readme made with AI

## Built With
- 🧩 SentenceTransformers for model training and embeddings
- ⚡ FastAPI for serving the search API
- 🔍 USearch for fast vector-similarity retrieval
- 📊 CoSQA dataset for model fine-tuning and evaluation

## 🚀 Overview
This project combines model fine-tuning with a search backend and an API interface.

**Workflow:**
1. Fine-tune a SentenceTransformers model on the CoSQA dataset (`code/finetuning_multiple.py`, `code/part_3.ipynb`).
2. Index documents (PDF or TXT) via the FastAPI service (`code/main.py`, `code/search_engine.py`).
3. Perform semantic search over indexed content using natural language queries.
4. Evaluate the model using Recall@10, MRR@10, and nDCG@10 (`code/evaluate.py`, `code/part_2.ipynb`).

## 🏗️ Project Structure
```
ML-for-Context-in-AI-Assistant/
├── code/
│   ├── evaluate.py
│   ├── finetuning_multiple.py
│   ├── finetuning_pair.py
│   ├── finetuning_triple.py
│   ├── main.py
│   ├── part_1.ipynb
│   ├── part_2.ipynb
│   ├── part_3.ipynb
│   ├── search_engine.py
│   ├── documents.json
│   └── search_index.usearch
├── documents/
│   ├── cryptocurrency.pdf
│   ├── great_barrier_reef.pdf
│   ├── history_of_internet.txt
│   ├── industrial_revolution.txt
│   └── nature_of_black_holes.txt
├── report/
│   ├── part_1.ipynb
│   └── part_2.ipynb
├── requirements.txt
└── README.md
```

## ⚙️ Installation
1) Clone the repository
```bash
git clone https://github.com/belalyahouni/ML-for-Context-in-AI-Assistant
cd ML-for-Context-in-AI-Assistant
```

2) Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

3) Install dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Running the API
Start the FastAPI server (required for `code/part_1.ipynb`):
```bash
cd code
uvicorn main:app --reload --port 8080
```

## 📥 Index Documents
Upload PDF or TXT files to build the search index via the API.

## 🔍 Search
Once documents are indexed, send natural-language queries to retrieve semantically related text.

## 📈 Evaluation
Evaluate retrieval performance on CoSQA:
```python
evaluate_model(model_name)
```
Metrics reported:
- Recall@10: how often a relevant document appears in the top 10
- MRR@10: how high the first relevant document ranks
- nDCG@10: ranking quality considering all relevant documents

## 🛠️ Fine-Tuning
Run the fine-tuning routine:
```python
fine_tune_cosqa()
```
This function:
- Loads and preprocesses the dataset
- Creates query–document pairs
- Uses MultipleNegativesRankingLoss to fine-tune embeddings
- Logs and plots loss using a custom callback
- Outputs training logs and a loss graph for visualization

## 🧪 Notebooks Summary
| Notebook      | Purpose                                                         |
|---------------|-----------------------------------------------------------------|
| `part_1.ipynb` | Client demo for uploading and searching documents using the API |
| `part_2.ipynb` | Evaluation of search quality using CoSQA metrics                |
| `part_3.ipynb` | Full fine-tuning + evaluation experiment                        |

### ▶️ Running the Notebooks
Use Jupyter Lab to go through the tasks step by step with the notebooks. Run them in order: `part_1.ipynb`, then `part_2.ipynb`, then `part_3.ipynb`.

1) Launch Jupyter Lab from the project root
```bash
jupyter lab
```

2) Open and run `code/part_1.ipynb`
- Ensure the API is running before executing cells in part 1:
```bash
cd code
uvicorn main:app --reload --port 8080
```

3) Open and run `code/part_2.ipynb`
- Evaluate retrieval metrics on the CoSQA setup.
- Note: This notebook does not use the API. It imports and uses classes/methods directly from modules in `code/`.

4) Open and run `code/part_3.ipynb`
- Perform fine-tuning and end-to-end evaluation.
- Note: This notebook does not use the API. It imports and uses classes/methods directly from modules in `code/`.
