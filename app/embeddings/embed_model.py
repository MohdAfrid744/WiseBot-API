import numpy as np
import faiss

def generate_embeddings(datasets, model):
    """Generate embeddings for all verses."""
    verses = []
    embeddings = []
    sources = []

    for book, verses_list in datasets.items():
        for entry in verses_list:
            verses.append(entry["verse"])
            embeddings.append(model.encode(entry["verse"]))
            sources.append(book)

    return np.array(embeddings), verses, sources

def search_similar(query, model, embeddings, verses, sources, selected_books, datasets):
    """Search for similar verses using FAISS."""
    query_embedding = model.encode(query).reshape(1, -1)

    # Create FAISS index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    # Search for the top 3 results
    distances, indices = index.search(query_embedding, 3)

    results = []
    for idx in indices[0]:
        if sources[idx] in selected_books:
            matching_entry = next(
                (entry for entry in datasets[sources[idx]] if entry["verse"] == verses[idx]),
                {"verse": "No verse available", "meaning": "No meaning available", "source": "Unknown Source"}
            )
            matching_entry["source"] = sources[idx]
            results.append(matching_entry)

    return results
