import os
import torch
import logging
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import argparse

# Suppress verbose logging and progress bars
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
tqdm.disabled = True

# Add command line argument parsing for device selection
parser = argparse.ArgumentParser(description='PDF Summarizer with device selection')
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

# Set up the local language model - Using BART-Large-CNN for excellent summarization
model_name = "facebook/bart-large-cnn"  # Specifically designed for summarization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to selected device
if device != "cpu":
    model = model.to(device)

llm_pipeline = pipeline(
    "summarization",  # Use summarization pipeline instead of text2text-generation
    model=model, 
    tokenizer=tokenizer, 
    max_length=512,  # Longer summaries
    min_length=100,  # Minimum summary length
    do_sample=True,
    temperature=0.7,
    device=device
)
llm = HuggingFacePipeline(pipeline=llm_pipeline, verbose=False)

# Load the PDF
pdf_path = "/Users/girija/documents/air-india-coc.pdf"  # Replace with your PDF file path
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split documents into chunks - Optimized for comprehensive summarization
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # Even larger chunks for better context
    chunk_overlap=400,  # More overlap for continuity
    length_function=len,
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]  # Better paragraph and sentence-aware splitting
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': device},  # Use the same device as LLM
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})  # Get many more chunks for comprehensive coverage

# Define the prompt template - Optimized for summarization
template = """Summarize the following text in a clear and concise manner. Focus on the key points, main ideas, and important details. Make the summary informative and well-structured.

Text to summarize: {context}

Summary:"""
prompt = ChatPromptTemplate.from_template(template)

# Create the summarization chain
summarization_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Function to generate different types of summaries
def generate_summary(prompt_text):
    try:
        # Get all chunks and combine them properly
        all_docs = retriever.get_relevant_documents("")
        
        if not all_docs:
            return "No content found to summarize."
        
        # Combine all chunks into a single text
        full_text = "\n\n".join([doc.page_content for doc in all_docs])
        
        # Clean up the text
        full_text = full_text.replace('\n\n\n', '\n\n').strip()
        
        # For better summaries, we'll process the document in sections
        # Split the full text into manageable chunks for BART
        max_chunk_size = 1024
        chunks_for_summary = []
        
        # Split text into overlapping chunks
        for i in range(0, len(full_text), max_chunk_size - 200):  # 200 char overlap
            chunk = full_text[i:i + max_chunk_size]
            if len(chunk) > 100:  # Only process chunks with substantial content
                chunks_for_summary.append(chunk)
        
        # Determine summary type based on prompt
        prompt_lower = prompt_text.lower()
        
        if "executive" in prompt_lower or "brief" in prompt_lower:
            # Executive summary - very concise
            max_len = 150
            min_len = 30
        elif "detailed" in prompt_lower or "comprehensive" in prompt_lower:
            # Detailed summary - longer
            max_len = 500
            min_len = 150
        elif "key points" in prompt_lower or "main points" in prompt_lower:
            # Key points summary
            max_len = 300
            min_len = 100
        else:
            # Default summary
            max_len = 400
            min_len = 100
        
        # Process each chunk and combine summaries
        chunk_summaries = []
        for chunk in chunks_for_summary:
            try:
                summary = llm_pipeline(
                    chunk,
                    max_length=max_len // len(chunks_for_summary) + 50,  # Distribute length
                    min_length=30,
                    do_sample=False,
                    temperature=0.3,
                    num_beams=4,
                    early_stopping=True
                )
                chunk_summaries.append(summary[0]['summary_text'])
            except:
                continue
        
        # Combine all chunk summaries
        combined_summary = " ".join(chunk_summaries)
        
        # If we have multiple summaries, create a final summary
        if len(chunk_summaries) > 1:
            # Create a final summary from all chunk summaries
            final_summary = llm_pipeline(
                combined_summary,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                temperature=0.3,
                num_beams=4,
                early_stopping=True
            )
            return final_summary[0]['summary_text']
        else:
            return combined_summary
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Function to ask for summaries
def ask_for_summary(prompt_text):
    summary = generate_summary(prompt_text)
    print(f"\n📄 Summary:\n{summary}\n")
    print("-" * 80)

# Main summarization interface
if __name__ == "__main__":
    print("\n" + "="*80)
    print("📚 PDF SUMMARIZER - Powered by BART-Large-CNN")
    print("="*80)
    print("\n💡 Example prompts you can try:")
    print("   • 'summarize the document' (default summary)")
    print("   • 'give me an executive summary' (brief overview)")
    print("   • 'provide a detailed summary' (comprehensive)")
    print("   • 'what are the key points?' (main points)")
    print("   • 'summarize the main policies' (focused)")
    print("   • 'brief summary' (concise)")
    print("\nType your request and press Enter. Type 'exit' to quit.\n")
    
    while True:
        prompt_text = input("📝 Your request: ")
        if prompt_text.strip().lower() == 'exit':
            print("👋 Exiting. Goodbye!")
            break
        ask_for_summary(prompt_text)