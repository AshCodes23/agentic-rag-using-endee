from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
from endee import Endee, Precision
import os
import uuid


class RAGSystem:
    def __init__(self):
        print("Initializing models...")

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.llm = AutoModelForCausalLM.from_pretrained(
            "models",
            model_file="mistral.gguf",
            model_type="mistral"
        )

        # Connect to Endee
        self.client = Endee()
        self.client.set_base_url("http://localhost:8080/api/v1")

        self.index_name = "rag_index"
        self.dimension = 384

        self._create_index()

        self.index = self.client.get_index(name=self.index_name)

        self.conversation_memory = []

 
   
   
    def _create_index(self):
        try:
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                space_type="cosine",
                precision=Precision.FLOAT32
            )
            print("Index created.")
        except Exception:
            print("Index already exists.")

    def ingest_documents(self):
        if self.documents_loaded:
            return

        full_text = ""
        for file in os.listdir("data/docs"):
            if file.endswith(".pdf"):
                reader = PdfReader(os.path.join("data/docs", file))
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text

        chunks = self._chunk_text(full_text)
        embeddings = self.embed_model.encode(chunks)

        vectors = []
        for i, emb in enumerate(embeddings):
            vectors.append({
                "id": str(uuid.uuid4()),
                "vector": emb.tolist(),
                "meta": {"text": chunks[i]}
            })

        self.index.upsert(vectors)
        self.documents_loaded = True
        print("Documents inserted into Endee.")


    def _validate_answer(self, context, answer):
        prompt = f"""
    You are a strict validation agent.

    Check whether the answer is fully supported by the context.
    If the answer contains information not present in context, mark NOT_SUPPORTED.

    Context:
    {context}

    Answer:
    {answer}

    Reply ONLY in one of these formats:
    SUPPORTED
    NOT_SUPPORTED
    """

        verdict = self.llm(prompt, max_new_tokens=10).strip().upper()

        if verdict.startswith("SUPPORTED"):
            return True
        return False


    def _rewrite_query(self, question):
        prompt = f"""
    You are a query rewriting agent for semantic search.
    Rewrite the user question into ONE clear, specific search query.
    Do NOT answer the question.
    Return ONLY the rewritten query.

    User Question:
    {question}

    Rewritten Query:
    """
        rewritten = self.llm(prompt, max_new_tokens=30)
        return rewritten.strip()
  
    def search(self, query, top_k=2):
        query_embedding = self.embed_model.encode([query])[0]

        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k
        )

        contexts = []
        for r in results:
            contexts.append(r["meta"]["text"])

        return "\n".join(contexts)

   
    def _chunk_text(self, text, chunk_size=400, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    # ---------------------------
    # Ask
    # ---------------------------
    def ask(self, question):
      
        rewritten_query = self._rewrite_query(question)
        print("Rewritten Query:", rewritten_query)


        context = self.search(rewritten_query)

        memory_context = "\n".join(
            [f"Q: {m['question']}\nA: {m['answer']}"
             for m in self.conversation_memory[-3:]]
        )

        prompt = f"""
You are an assistant that answers ONLY using the provided context.
If the answer is not present, say "I don't know based on the document".

Previous conversation:
{memory_context}

Context:
{context}

Question:
{question}

Answer:
"""

        answer = self.llm(prompt, max_new_tokens=120)

        is_valid = self._validate_answer(context, answer)

        if not is_valid:
            final_answer = "I don't know based on the document."
        else:
            final_answer = answer

        self.conversation_memory.append({
            "question": question,
            "answer": final_answer
        })

        return final_answer
   
