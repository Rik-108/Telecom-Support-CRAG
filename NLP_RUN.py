import chromadb

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./telecom_db")  # Saves data persistently

# Create collections for different datasets
faq_collection = chroma_client.get_or_create_collection("telecom_faqs")
support_collection = chroma_client.get_or_create_collection("customer_support_logs")
knowledge_collection = chroma_client.get_or_create_collection("general_knowledge_base")

faq_data = [
    {"question": "How can I check my telecom bill?", "answer": "You can check your bill through the telecom provider's website or mobile app."},
    {"question": "What should I do if my internet is slow?", "answer": "Restart your router, check your data limit, or contact customer support."},
    {"question": "How do I activate international roaming?", "answer": "You can activate it through the telecom app or by dialing the customer service number."}
]

# Insert into ChromaDB
for idx, faq in enumerate(faq_data):
    faq_collection.add(
        ids=[str(idx)],  # Unique ID
        documents=[faq["question"]],  # Text for similarity search
        metadatas=[{"answer": faq["answer"]}]  # Store answer
    )

support_logs = [
    {"query": "My SIM card is not working", "response": "Try reinserting the SIM or contact customer support."},
    {"query": "I am unable to make calls", "response": "Check if your bill is paid and if network coverage is available."}
]

# Insert into ChromaDB
for idx, log in enumerate(support_logs):
    support_collection.add(
        ids=[str(idx)],
        documents=[log["query"]],
        metadatas=[{"response": log["response"]}]
    )
knowledge_docs = [
    {"title": "5G Network Benefits", "content": "5G networks provide faster speeds and lower latency."},
    {"title": "How to Port Your Number", "content": "You can port your number by sending a port request SMS to your new provider."}
]

# Insert into ChromaDB
for idx, doc in enumerate(knowledge_docs):
    knowledge_collection.add(
        ids=[str(idx)],
        documents=[doc["title"]],
        metadatas=[{"content": doc["content"]}]
    )

query = "How do I check my telecom bill?"
results = faq_collection.query(query_texts=[query], n_results=1)

# Get the best-matched FAQ
if results["documents"]:
    print("Answer:", results["metadatas"][0][0]["answer"])
else:
    print("No relevant FAQ found.")

from huggingface_hub import InferenceClient

# Your Hugging Face API token
api_token = "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"

# Initialize the Inference Client
client = InferenceClient(model="google/flan-t5-small", token=api_token)

# Define input text
input_text = "What is 5G technology?"

# Generate text
response = client.text_generation(input_text, max_new_tokens=200, temperature=0.9, do_sample=True)

# Print the output9
print(response)

import chromadb
from transformers import pipeline

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./telecom_db")
faq_collection = chroma_client.get_or_create_collection("telecom_faqs")
support_collection = chroma_client.get_or_create_collection("customer_support_logs")
knowledge_collection = chroma_client.get_or_create_collection("general_knowledge_base")

# Load a pre-trained text generation model (FLAN-T5 or any preferred model)
text_generator = pipeline("text2text-generation", model="google/flan-t5-small")

def retrieve_documents(query, collection, n_results=2):
    """Retrieve relevant documents from a ChromaDB collection."""
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["metadatas"] if results["documents"] else []

def generate_response(user_query):
    """Retrieve context and generate a response."""
    # Retrieve from all collections
    faq_results = retrieve_documents(user_query, faq_collection)
    support_results = retrieve_documents(user_query, support_collection)
    knowledge_results = retrieve_documents(user_query, knowledge_collection)
    
    # Combine relevant information
    context = "\n".join([entry[0]["answer"] for entry in faq_results] +
                          [entry[0]["response"] for entry in support_results] +
                          [entry[0]["content"] for entry in knowledge_results])
    
    if not context:
        context = "No relevant information found. Please try rephrasing your question."
    
    # Generate an AI-assisted response
    prompt = f"Context: {context}\nUser Query: {user_query}\nResponse:"
    response = text_generator(prompt, max_length=100, do_sample=True)
    
    return response[0]["generated_text"]

# Example usage
user_input = "How can I check my telecom bill?"
bot_response = generate_response(user_input)
print("Chatbot:", bot_response)

import chromadb
from transformers import pipeline

def initialize_chromadb():
    """Initialize ChromaDB collections."""
    chroma_client = chromadb.PersistentClient(path="./telecom_db")
    return {
        "faq": chroma_client.get_or_create_collection("telecom_faqs"),
        "support": chroma_client.get_or_create_collection("customer_support_logs"),
        "knowledge": chroma_client.get_or_create_collection("general_knowledge_base"),
    }

def retrieve_documents(query, collection, n_results=2):
    """Retrieve relevant documents from a ChromaDB collection."""
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["metadatas"] if results["documents"] else []

def generate_response(user_query, collections, model):
    """Retrieve context and generate a response."""
    faq_results = retrieve_documents(user_query, collections["faq"])
    support_results = retrieve_documents(user_query, collections["support"])
    knowledge_results = retrieve_documents(user_query, collections["knowledge"])
    
    context = "\n".join(
        [entry[0].get("answer", "") for entry in faq_results] +
        [entry[0].get("response", "") for entry in support_results] +
        [entry[0].get("content", "") for entry in knowledge_results]
    )
    
    if not context.strip():
        return "No relevant information found. Please try rephrasing your question."
    
    prompt = f"Context: {context}\nUser Query: {user_query}\nResponse:"
    response = model(prompt, max_length=100, do_sample=True)
    
    return response[0]["generated_text"]

if __name__ == "__main__":
    collections = initialize_chromadb()
    text_generator = pipeline("text2text-generation", model="google/flan-t5-small")
    
    user_input = input("Enter your query: ")
    bot_response = generate_response(user_input, collections, text_generator)
    print("Chatbot:", bot_response)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.schema import HumanMessage, AIMessage
import os

# Set up Hugging Face API Token (Ensure this is set as an environment variable)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load LLM from Hugging Face Hub
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1")

# Load and process documents
loader = TextLoader("Telecom_text.txt")  # Replace with actual file
raw_documents = loader.load()

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(raw_documents)

# Embed and store documents in FAISS vector database
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever()

# Set up memory with explicit output key

memory = ConversationBufferMemory(memory_key="chat_history", output_key="result")


# Define QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    memory=memory,
    return_source_documents=True,  # Keep this, but handle it separately
    output_key="result"  # Ensures only 'result' is stored in memory
)


# Function to interact with chatbot

def chat_with_bot(query):
    response = qa_chain.invoke({"query": query})  # Ensure query is passed correctly
    answer = response.get("result", "No answer found.")  # Extract result safely
    sources = response.get("source_documents", [])  # Extract source documents safely
    return answer, sources


# Example Query
query = "What is 5G technology?"
answer, sources = chat_with_bot(query)

print("Bot Answer:", answer)
print("\nSources:", sources)

from langchain.llms import HuggingFaceHub
from langchain.embeddings import SentenceTransformerEmbeddings

print("Imports successful!")

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", output_key="result")  # Store only 'result'
