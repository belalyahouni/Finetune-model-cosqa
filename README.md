# Semantic Search Engine with Sentence Transformers & CoSQA Fine-Tuning
This project implements a semantic search engine that allows users to upload documents (PDF or text) and perform natural language searches a using fine-tuned model.

## It’s built using:
🧩 SentenceTransformers for model training and embeddings

⚡ FastAPI for serving the search API

🔍 Usearch for fast vector similarity retrieval

📊 CoSQA dataset for model fine-tuning and evaluation

## 🚀 Overview
This project combines machine learning model fine-tuning with a search backend and API interface.

Workflow

Fine-tune a Sentence Transformer model on the CoSQA dataset (finetuning_multiple.py / part_3.ipynb)

Index documents (PDF or TXT) using FastAPI (main.py + search_engine.py)

Search semantically related text based on natural language queries

Evaluate the model using Recall@10, MRR@10, and nDCG@10 (evaluate.py, part_2.ipynb)


## 🏗️ Project Structure

📂 ML-for-Context-in-AI-Assistant/

│

├── code

│      ├──finetuning_multiple.py     # Fine-tunes SentenceTransformer on CoSQA

│      ├── search_engine.py           # Encodes, indexes, and retrieves documents

│      ├── main.py                    # FastAPI backend for indexing/search endpoints

│      ├──evaluate.py                # Evaluation script (Recall, MRR, nDCG)

│      │

│      ├── part_1.ipynb               # API demo: uploading and querying documents

│      ├── part_2.ipynb               # Evaluation notebook for retrieval metrics

│      └── part_3.ipynb               # Full fine-tuning + evaluation pipeline

│

├── documents/                 # sample PDFs or text files (for part2.ipynb)

├── README.md    

├──requirements.txt

└── README.md 

## ⚙️ Installation
1. Clone the Repository

git clone https://github.com/belalyahouni/ML-for-Context-in-AI-Assistant

cd semantic-search-engine

3. Create and Activate a Virtual Environment

python -m venv venv

source venv/bin/activate

5. Install Dependencies

pip install -r requirements.txt

## Running the API

Start the FastAPI server:

This step will be needed for part1.ipynb.

You must be in the code repository.

uvicorn main:app --reload --port 8080

## 📥 Index Documents

You can upload PDF or TXT files to build the search index.

## 🔍 Search

Once documents are indexed, send a query.

## Evaluation

Evaluate retrieval performance on CoSQA:

evaluate_model(model_name)

Metrics reported:

Recall@10 – how often a relevant doc appears in top 10

MRR@10 – how high the first relevant doc ranks

nDCG@10 – measures ranking quality considering all relevant docs

## Fine-tuning

fine_tune_cosqa()

This function:

Loads and preprocesses the dataset.

Creates query-document pairs.

Uses MultipleNegativesRankingLoss to fine-tune embeddings.

Logs and plots loss using the custom callback.

Outputs training logs and a loss graph for visualization.

## 🧪 Notebooks Summary

Notebook	Purpose

part_1.ipynb	Client demo for uploading and searching documents using the API

part_2.ipynb	Evaluation of search quality using CoSQA metrics

part_3.ipynb	Full fine-tuning + evaluation experiment
