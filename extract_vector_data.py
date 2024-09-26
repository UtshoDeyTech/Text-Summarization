import chromadb
from chromadb.config import Settings
import numpy as np

def extract_vector_data(collection_name):
    # Initialize the ChromaDB client
    client = chromadb.HttpClient(host="localhost", port=8000)

    # Get the collection
    collection = client.get_collection(name=collection_name)

    # Query the collection to get all items
    results = collection.query(
        query_texts=[""],
        n_results=100,  # Adjust this number based on your collection size
        include=["documents", "metadatas", "distances", "embeddings"]
    )

    # Extract and process the results
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    embeddings = results['embeddings'][0]
    ids = results['ids'][0]

    print(f"Extracted {len(documents)} documents from collection '{collection_name}'")

    # Print some sample data
    for i in range(min(3, len(documents))):
        print(f"\nDocument {i+1}:")
        print(f"ID: {ids[i]}")
        print(f"Text: {documents[i][:100]}...")  # Print first 100 characters
        print(f"Metadata: {metadatas[i]}")
        print(f"Embedding shape: {np.array(embeddings[i]).shape}")

    return documents, metadatas, embeddings, ids

# Example usage
if __name__ == "__main__":
    collection_name = "my_collection"
    docs, metas, embeds, ids = extract_vector_data(collection_name)