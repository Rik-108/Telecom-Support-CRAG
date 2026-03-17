# Telecom Intelligent Support: Corrective RAG (CRAG) Chatbot 🤖📡

## 1. Project Overview
A specialized AI support agent designed for the telecommunications industry. Unlike standard chatbots that can hallucinate, this system uses **Corrective RAG (CRAG)** to retrieve, evaluate, and refine knowledge from a private database before generating a response.

## 2. The Architecture: RAG vs. CRAG
While standard RAG simply retrieves and answers, this system adds a **Self-Correction Loop**:
1. **Retrieval:** Fetches relevant documents from a **ChromaDB** or **FAISS** vector store.
2. **Evaluation:** An LLM-based grader evaluates if the retrieved documents are truly relevant to the query.
3. **Refinement:** If the information is insufficient or irrelevant, the system triggers a refinement process to ensure the final answer is grounded in fact.



## 3. Key Features
- **Vector Search:** Utilizes **HuggingFace Embeddings** for semantic understanding of telecom jargon (e.g., "VoLTE," "Porting," "Roaming").
- **Hybrid Storage:** Implemented both **ChromaDB** for persistent knowledge and **FAISS** for high-speed similarity search.
- **Contextual Memory:** Uses `ConversationBufferMemory` to maintain the flow of support tickets across multiple turns.
- **Domain Focus:** Optimized for Billing, Network Troubleshooting, and Service Activation queries.

## 4. Technical Stack
- **Framework:** LangChain
- **Vector Databases:** ChromaDB, FAISS
- **Embeddings:** HuggingFace (Sentence-Transformers)
- **LLM Integration:** OpenAI / HuggingFace Endpoint
- **Data Processing:** Regular Expressions (Regex) for parsing unstructured support logs

## 5. Usage
1. **Prepare Database:**
   ```bash
   python src/ingestion.py
