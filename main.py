import chromadb
from chromadb.config import Settings

def store_vector_data(collection_name, documents, metadatas, ids):
    # Initialize the ChromaDB client
    client = chromadb.HttpClient(host="localhost", port=8000)

    # Get or create a collection
    collection = client.get_or_create_collection(name=collection_name)

    # Add documents to the collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Added {len(documents)} documents to collection '{collection_name}'")

# Example usage
if __name__ == "__main__":
    collection_name = "my_collection"
    documents = [
        "This is the first document",
        "This is the second document",
        "This is the third document"
    ]
    metadatas = [
        {"source": "document1.txt"},
        {"source": "document2.txt"},
        {"source": "document3.txt"}
    ]
    ids = ["doc1", "doc2", "doc3"]

    store_vector_data(collection_name, documents, metadatas, ids)