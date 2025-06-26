import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import argparse

# Add command line argument parsing for device selection
parser = argparse.ArgumentParser(description='PDF QA with device selection')
parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', 
                   help='Choose device: cpu or gpu (default: gpu)')
args = parser.parse_args()

# Device configuration
if args.device == 'gpu' and torch.backends.mps.is_available():
    device = "mps"
    print(f"Using GPU (MPS): {device}")
elif args.device == 'gpu' and torch.cuda.is_available():
    device = "cuda"
    print(f"Using GPU (CUDA): {device}")
else:
    device = "cpu"
    print(f"Using CPU: {device}")

# Set up the local language model - Using flan-t5-large for better accuracy
model_name = "google/flan-t5-large"  # Much better reasoning than base version
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to selected device
if device != "cpu":
    model = model.to(device)

llm_pipeline = pipeline(
    "text2text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_length=256,  # Conservative limit
    truncation=True,
    do_sample=True,
    temperature=0.7,
    device=device
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Load the PDF
pdf_path = "/Users/girija/documents/air-india-coc.pdf"  # Replace with your PDF file path
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Reduced from 1000
    chunk_overlap=100,  # Reduced from 200
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': device},  # Use the same device as LLM
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Reduced from 3

# Define the prompt template - Back to the simple, working version
template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer: """
prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain - Back to the simple, working version
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Function to ask questions
def ask_question(question):
    response = rag_chain.invoke(question)
    print(f"Question: {question}")
    print(f"Answer: {response}\n")

# Example questions
if __name__ == "__main__":
    print("\nPDF QA Chat. Type your question and press Enter. Type 'exit' to quit.\n")
    while True:
        question = input("Your question: ")
        if question.strip().lower() == 'exit':
            print("Exiting. Goodbye!")
            break
        ask_question(question)