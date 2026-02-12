import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    # FIXED: Updated model to text-embedding-004
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )
    db = FAISS.from_texts(text_chunks, embeddings)
    db.save_local("faiss_index")

def user_input(question):
    if not os.path.exists("faiss_index"):
        st.error("Please upload and process PDFs first.")
        return

    # FIXED: Updated model to text-embedding-004 (Must match the one used for saving)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the provided context.
    If answer is not in context, say:
    "Answer not available in the context."

    Context:
    {context}

    Question:
    {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)

    st.write("Reply:")
    st.write(response)

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF")

    question = st.text_input("Ask a Question from the PDF Files")

    if question:
        user_input(question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("Done")
                else:
                    st.warning("Please upload a PDF file first.")

if __name__ == "__main__":
    main()