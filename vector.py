from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df=pd.read_csv("Training Dataset.csv")
embeddings=OllamaEmbeddings(model="mxbai-embed-large")

db_location="./chrome_langchain_db"
add_documents=not os.path.exists(db_location)

if add_documents:
    documents=[]
    ids=[]



    for i, row in df.iterrows():
            # Prepare document content
            page_content = f"""
            Gender: {row.get('Gender', '')},
            Married: {row.get('Married', '')},
            Dependents: {row.get('Dependents', '')},
            Education: {row.get('Education', '')},
            Self Employed: {row.get('Self_Employed', '')},
            Applicant Income: {row.get('ApplicantIncome', '')},
            Coapplicant Income: {row.get('CoapplicantIncome', '')},
            Loan Amount: {row.get('LoanAmount', '')},
            Loan Term: {row.get('Loan_Amount_Term', '')},
            Credit History: {row.get('Credit_History', '')},
            Property Area: {row.get('Property_Area', '')},
            Loan Status: {row.get('Loan_Status', '')}
            """

            doc = Document(
                page_content=page_content.strip(),
                metadata={"Loan_ID": row.get("Loan_ID", "")},
                id=str(i)
            )
            documents.append(doc)
            ids.append(str(i))

    # Create or load the Chroma vector store
vector_store = Chroma(
        collection_name="loan_applications",
        persist_directory=db_location,
        embedding_function=embeddings
)

# Add documents if store doesn't exist
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Create retriever (you can use this with a QA chain or chatbot)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})