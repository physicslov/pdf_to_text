import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, ValidationError

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Make sure you set the GOOGLE_API_KEY environment variable.")
else:
    genai.configure(api_key=api_key)

# Define a Pydantic model to validate some configuration
class AppConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int

    @validator('chunk_size', 'chunk_overlap')
    def validate_chunk_size(cls, value, values, field):
        if field.name == 'chunk_size' and value <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if field.name == 'chunk_overlap' and value < 0:
            raise ValueError("chunk_overlap must not be negative")
        if 'chunk_size' in values and field.name == 'chunk_overlap' and value >= values['chunk_size']:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return value

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, chunk_size=10000, chunk_overlap=1000):
    # Validate the configuration using AppConfig model
    try:
        config = AppConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except ValidationError as e:
        st.error(f"Configuration error: {e}")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Safely load the FAISS index
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except ValueError as e:
        st.error("Error loading FAISS index. Please ensure the index is correctly saved and try again.")
        return
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:  # Proceed only if text_chunks is not empty
                        get_vector_store(text_chunks)
                        st.success("Done")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
