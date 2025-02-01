import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from model2vec import StaticModel
import sys

model = StaticModel.from_pretrained("minishlab/potion-base-8M")

def get_paper_embedding(csv_file, output_dir, batch_size=10000):
    df = pd.read_csv(csv_file)

    if 'Title' not in df.columns or 'Abstract' not in df.columns:
        raise ValueError("CSV file must contain 'Title' and 'Abstract' columns")

    os.makedirs(output_dir, exist_ok=True)

    total_rows = len(df)
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size]

        sys.stdout.write(f"\rProcessing {len(batch)} rows... ")
        sys.stdout.flush()

        data_for_embedding = [f"Title: {row['Title'].strip()} ; Abstract: {row['Abstract'].strip()}" for _, row in batch.iterrows()]
        embeddings = model.encode(data_for_embedding)

        batch_dict = {
            "Title": batch["Title"].tolist(),
            "Abstract": batch["Abstract"].tolist(),
            "Categories": batch.get("Categories", "").tolist(),
            "Embedding": embeddings.tolist()
        }

        table = pa.Table.from_pydict(batch_dict)
        output_path = os.path.join(output_dir, f"{i//batch_size:04d}.parquet")
        pq.write_table(table, output_path)

        sys.stdout.write(f"\rSaved {len(batch)} rows to {output_path}    \n")
        sys.stdout.flush()

if __name__ == "__main__":
    csv_file = "arxiv-csv.csv"
    output_dir = "paper"
    get_paper_embedding(csv_file, output_dir)
