import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_vector_store():
    with open("faiss_index.pkl", "rb") as f:
        index, texts = pickle.load(f)
    return index, texts

def get_answer(question, top_k=3):
    index, texts = load_vector_store()

    #embed question
    q_embedding = model.encode([question], convert_to_numpy=True)

    #search
    distances, indices = index.search(q_embedding, top_k)

    #Get relevant chunks
    context = "\n\n".join([texts[i] for i in indices[0]])

    prompt = ChatPromptTemplate.from_template(
    "You are a helpful college assistant. Answer the question using only the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

    chain = prompt | llm

    # Answer
    answer = chain.invoke({"context": context, "question": question})
    return answer