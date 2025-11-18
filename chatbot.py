import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------
# Load models
# --------------------------------------------------------
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading DeepSeek model...")
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# --------------------------------------------------------
# Load and chunk Frankenstein
# --------------------------------------------------------
def load_chunks(path, chunk_size=500):
    with open(path, "r", encoding="utf8") as f:
        text = f.read()

    words = text.split()
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    return chunks

chunks = load_chunks("frankenstein.txt")
print(f"Loaded {len(chunks)} chunks.")


# --------------------------------------------------------
# Build FAISS vector index
# --------------------------------------------------------
print("Embedding chunks...")
embeddings = embedder.encode(chunks, convert_to_numpy=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print("Index built.")


# --------------------------------------------------------
# Retrieval function
# --------------------------------------------------------
def search(query, k=3):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, k)

    return [chunks[i] for i in indices[0]]


# --------------------------------------------------------
# Chat loop
# --------------------------------------------------------
def ask_deepseek(question, context):
    prompt = f"""
You are a helpful assistant answering questions about the novel *Frankenstein*.
Use only the context below and do not invent details.

Context:
{context}

Question: {question}
Answer:
"""

    output = generator(prompt, max_new_tokens=250)[0]["generated_text"]
    return output.split("Answer:", 1)[-1].strip()


print("\nChatbot ready! Ask anything about Frankenstein.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break

    passages = search(user_input)
    context = "\n\n".join(passages)

    answer = ask_deepseek(user_input, context)
    print(f"\nBot: {answer}\n")
