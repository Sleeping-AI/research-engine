import os
import pickle
import pandas as pd
import time
from model2vec import StaticModel
import sys

model = StaticModel.from_pretrained("minishlab/potion-base-8M")

def get_paper_embedding(csv_file, output_dir, batch_size=30, rows_per_file=2000):
    print("Model loaded successfully.")

    df = pd.read_csv(csv_file)

    if 'Title' not in df.columns or 'Abstract' not in df.columns:
        raise ValueError("CSV file must contain 'Title' and 'Abstract' columns")

    total_rows = len(df)
    print(f"Total {total_rows} rows to process.")

    total_batches = (total_rows // batch_size) + (1 if total_rows % batch_size > 0 else 0)
    total_files = (total_rows // rows_per_file) + (1 if total_rows % rows_per_file > 0 else 0)

    print(f"Total {total_batches} batches needed.")
    print(f"Each batch will contain {batch_size} rows.")
    print(f"Each Pickle file will have up to {rows_per_file} rows.")

    os.makedirs(output_dir, exist_ok=True)

    batch_counter = 0
    file_counter = 0

    start_embedding_time = time.time()

    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size]

        sys.stdout.write(f"\rProcessing batch {batch_counter + 1}/{total_batches}... ")
        sys.stdout.flush()

        titles = [row['Title'].strip() for _, row in batch.iterrows()]
        abstracts = [row['Abstract'].strip() for _, row in batch.iterrows()]

        embeddings_titles = model.encode(titles)
        embeddings_abstracts = model.encode(abstracts)

        batch_dict = {
            "Title": batch["Title"].tolist(),
            "Abstract": batch["Abstract"].tolist(),
            "Categories": batch.get("Categories", "").tolist(),
            "Title_Embedding": embeddings_titles.tolist(),
            "Abstract_Embedding": embeddings_abstracts.tolist()
        }

        if (batch_counter + 1) * batch_size >= (file_counter + 1) * rows_per_file:
            output_path = os.path.join(output_dir, f"{file_counter:04d}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(batch_dict, f)
            file_counter += 1
            print(f"\rSaved {len(batch)} rows to {output_path}...     ", end="")

        batch_counter += 1

    end_embedding_time = time.time()
    embedding_time_taken = (end_embedding_time - start_embedding_time) / 60
    print(f"\nEmbedding generation completed.")
    print(f"Time taken for embedding generation: {embedding_time_taken:.2f} minutes.")

if __name__ == "__main__":
    csv_file = "arxiv-csv.csv"
    output_dir = "paper"
    get_paper_embedding(csv_file, output_dir)
