"""
Example script showing how to use the retrieval functionality.
"""

from ingest_pdfs import load_chunks_from_jsonl, retrieve_chunks

# Example: Load chunks and search
if __name__ == "__main__":
    # Load chunks from JSONL file
    chunks = load_chunks_from_jsonl("outputs/chunks.jsonl")
    print(f"Loaded {len(chunks)} chunks")
    
    # Example queries
    queries = [
        "supplier code of conduct",
        "human rights",
        "child labor",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)
        
        results = retrieve_chunks(query, chunks, top_k=3)
        
        for i, chunk in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {chunk['metadata']['filename']}")
            print(f"Page(s): {chunk['metadata']['page_numbers']}")
            if chunk['metadata']['title']:
                print(f"Title: {chunk['metadata']['title']}")
            print(f"Text preview: {chunk['text'][:2000]}...")

