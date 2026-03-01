from flask import Flask, render_template, request
from rag_core import RAGSystem
import os

app = Flask(__name__)

# Initialize RAG once
rag = RAGSystem()

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    message = None

    if request.method == "POST":

        # Handle PDF upload
        if "document" in request.files:
            file = request.files["document"]
            if file.filename != "":
                save_path = os.path.join("data/docs", file.filename)
                file.save(save_path)

                rag.documents_loaded = False
                rag.ingest_documents()

                message = "Document uploaded and indexed successfully."

        # Handle question
        elif "question" in request.form:
            question = request.form["question"]
            answer = rag.ask(question)

    return render_template("index.html",
                           answer=answer,
                           message=message,
                           history=rag.conversation_memory)

if __name__ == "__main__":
    app.run(debug=True)