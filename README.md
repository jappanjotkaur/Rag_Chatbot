# Loan Approval RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** based intelligent chatbot built using **LangChain**, **Ollama**, and **ChromaDB** that assists users in understanding **loan eligibility** and decisions based on real applicant data.

## Overview

This project uses a fine-tuned local LLM (`tinyllama`) and a structured dataset to simulate a chatbot that answers questions about loan approval. It retrieves the most relevant data points from past loan applications and uses them as context for answering user queries.

> Built with Streamlit UI for user-friendly web experience, and a CLI version for development/testing.

---

## Key Features

- **LLM + RAG Pipeline** using LangChain & Ollama  
- Parses and embeds structured CSV data (loan applications)  
- Stores embeddings using **Chroma Vector DB**  
- Retrieves top 5 similar cases using vector similarity  
- Dual interface:  
  - Web App (via Streamlit)  
  - CLI-based terminal chat  
- Stylish UI with avatars and smooth animations  
- Automatically persists and reuses vector database  

---

## Project Structure

```
Loan_Approval_RAG_Chatbot
├── frontend.py        # Streamlit-based chatbot UI
├── main.py            # CLI interface to chat with model
├── vector.py          # Loads and embeds CSV data into ChromaDB
├── Training Dataset.csv  # Structured loan data used for retrieval
├── static/
│   ├── user.jpg       # Avatar for user
│   └── bot.png        # Avatar for bot
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/jappanjotkaur/Rag_Chatbot.git
cd Rag_Chatbot
```

### 2. Create Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> 📌 Make sure you have [Ollama](https://ollama.com/) installed and running for local LLMs.

### 4. Start Vector Store Creation
```bash
python vector.py
```
This will process `Training Dataset.csv`, create embeddings using `mxbai-embed-large`, and store them using Chroma.

---

## Run the Chatbot

### Option 1: Web Interface
```bash
streamlit run frontend.py
```
Open in browser at `http://localhost:8501`

### Option 2: Terminal Interface
```bash
python main.py
```
You can ask questions like:
- `"What causes loan rejection?"`
- `"Will a self-employed person get loan easily?"`
- `"What if applicant income is very low?"`

---

## Dataset Details

The dataset contains loan applications with fields like:

- Gender, Marital Status  
- Income (Applicant & Coapplicant)  
- Loan Amount & Term  
- Education, Self Employment  
- Property Area, Credit History  
- Loan Status (Approved/Rejected)  

> Embedded using Ollama’s `mxbai-embed-large` model and stored in ChromaDB.

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)  
- [Ollama](https://ollama.com/)  
- [Chroma](https://www.trychroma.com/)  
- [Streamlit](https://streamlit.io)  
