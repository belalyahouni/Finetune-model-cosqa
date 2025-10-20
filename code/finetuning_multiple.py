from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.trainer import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from transformers import TrainerCallback

import random
from torch.utils.data import DataLoader
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Loss logger callback
class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.loss_values = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.loss_values.append(logs["loss"])
            self.steps.append(state.global_step)
            # Optional: print periodically
            if state.global_step % 100 == 0:
                print(f"Step {state.global_step}: Loss = {logs['loss']:.4f}")
                

# Load datasets in
def fine_tune_cosqa():
    corpus_dataset = load_dataset("CoIR-Retrieval/cosqa", "corpus", split="corpus")
    queries_dataset = load_dataset("CoIR-Retrieval/cosqa", "queries", split="queries")
    train_dataset_raw = load_dataset("CoIR-Retrieval/cosqa", name="default", split="train")
    print(f"Corpus loaded with {len(corpus_dataset)} documents.")
    print(f"Queries loaded with {len(queries_dataset)} queries.")
    print(f"Train 'test' split loaded with {len(train_dataset_raw)} query-document pairs.")

    # Load model in
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded.")

    # Linking corpus to id and query to id
    print("Building corpus and query lookup maps...")
    corpus_map = {item["_id"]: item["text"] for item in corpus_dataset}
    query_map = {item["_id"]: item["text"] for item in queries_dataset}

    # Building train pairs, linking query (id) to correct corpus (id)
    print("Building positive (query, code) pairs...")
    train_pairs = []
    for item in train_dataset_raw:
        query_text = query_map[item["query-id"]]
        code_text = corpus_map[item["corpus-id"]]
        
        train_pairs.append((query_text, code_text))
    print(f"Built {len(train_pairs)} training pairs.")

    # Defining corpus ids from the corpus map
    corpus_ids = list(corpus_map.keys())

    # Creating negative samples, and creating triplets
    print("Creating triplets...")
    triplets = []
    for item in train_dataset_raw:
        qid = item["query-id"]
        did = item["corpus-id"]
        query = query_map[qid]
        pos = corpus_map[did]
        # sample one random negative
        while True:
            neg_id = random.choice(corpus_ids)
            neg = corpus_map[neg_id]
            if neg != pos:
                break
        triplets.append({
            "anchor": query,
            "positive": pos,
            "negative": neg
        })
    print(f"Created {len(triplets)} sample triplets :")

    # Converting list of triples to dataset for trainer 
    train_dataset = Dataset.from_list(triplets)
    print(f"Created train_dataset with {len(train_dataset)} triplets.")

    # Define training loss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    print("MultipleNegativesRankingLoss defined.")

    # Add loss callback for plotting loss
    loss_logger = LossLoggerCallback()

    # Define training args
    training_args = SentenceTransformerTrainingArguments(
        output_dir="fine_tuned_model_multiple",
        num_train_epochs=2,
        learning_rate=1e-5,
        warmup_steps=1000,
        per_device_train_batch_size=16,

        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        router_mapping={"texts": {"0": "anchor", "1": "positive", "2": "negative"}}
    )

    # Define trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        loss=train_loss,
        callbacks=[loss_logger]
    )

    # Train
    trainer.train()
    print("Fine-tuning complete! Model saved to 'fine_tuned_model_multiple'.")

    # Plot graph using loss from logger.
    sns.set(style="whitegrid", context="talk")

    # Convert to DataFrame for Seaborn
    loss_df = pd.DataFrame({
        "Step": loss_logger.steps,
        "Loss": loss_logger.loss_values
    })

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=loss_df, x="Step", y="Loss", color="blue", linewidth=2)
    plt.title("Training Loss per Step", fontsize=16, weight='bold')
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.tight_layout()
    plt.savefig("training_loss_multiple.png", dpi=300)
    plt.show()
    print("Loss curve saved to 'training_loss_multiple.png'.")

def main():
    fine_tune_cosqa()

if __name__ == "__main__":
    main()