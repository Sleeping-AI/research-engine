import pickle
import os
from model2vec import StaticModel

model = StaticModel.from_pretrained("minishlab/potion-base-8M")

def is_title_already_in_data(doc, data):
    title = doc['title'].lower().strip()
    for d in data:
        if title == d['title'].lower().strip():
            return False
    return True

def embed_and_save_papers_with_model2vec(papers, query, output_dir):
    data = []
    for paper in papers:
        if paper['title'] and paper['abstract'] and is_title_already_in_data(paper, data):
            data.append(paper)

    print(f"Embedding {len(data)} papers about {query}")

    data_for_embedding = [f"Title: {d['title'].strip()} ; Abstract: {d['abstract'].strip()}" for d in data]

    embeddings = model.encode(data_for_embedding)

    text_embedding_tuplist = [(text['title'], text['abstract'], text['link'], embedding) for text, embedding in zip(data, embeddings)]

    # Ensure the directory exists
    output_path = f'{output_dir}/{query}'
    os.makedirs(output_path, exist_ok=True)

    pickle_output_path = f'{output_path}/{query}_embeddings.pkl'
    with open(pickle_output_path, 'wb') as f:
        pickle.dump(text_embedding_tuplist, f)

    print(f"Text Embedding Done!")
    return text_embedding_tuplist

if __name__ == "__main__":
    query = 'AI Research'
    output_dir = '/content'
    papers = [
        {'title': 'AI in Healthcare', 'abstract': 'Exploring AI in the healthcare industry', 'link': 'https://example.com/1'},
        {'title': 'Deep Learning Advances', 'abstract': 'Recent progress in deep learning', 'link': 'https://example.com/2'}
    ]
    embed_and_save_papers_with_model2vec(papers, query, output_dir)
