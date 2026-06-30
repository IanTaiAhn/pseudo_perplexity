import sys
sys.path.insert(0, ".")
from ingestion.indexer import get_collection

c = get_collection()
print("Total chunks indexed:", c.count())

all_docs = c.get(include=["documents", "metadatas"])
for i, (doc, meta) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
    print(f"\n--- chunk {i} | source={meta['source']} idx={meta['chunk_index']} ---")
    print(doc[:200])