from rag_core import RAGSystem

rag = RAGSystem()
rag.ingest_documents()

print(rag.ask("what happened to the server?"))