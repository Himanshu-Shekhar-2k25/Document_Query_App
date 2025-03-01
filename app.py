import os
from dotenv import load_dotenv
import google.generativeai as generativeai
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
import streamlit as st

def load_document(pdf_docs):
    file_path = "user_doc.pdf"
    with open(file_path, "wb") as f:
        f.write(pdf_docs.getbuffer())
                
        loader = PyPDFLoader(file_path)
        pdf_doc = loader.load()
        return pdf_doc

def create_chunks(pdf_doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    pdf_chunks = text_splitter.split_documents(pdf_doc)

    chunk_data = []
    for chunk in pdf_chunks:
        chunk_data.append(chunk.page_content)
    return chunk_data

def get_embeddings(chunk_data):
    embeddings = []
    status_text = st.text("Creating Embeddings")
    progress_bar = st.progress(0)

    for i, chunk in enumerate(chunk_data):
        result = generativeai.embed_content(model="models/text-embedding-004",content=chunk)
        embeddings.append({'text':chunk, 'vector':result['embedding']})
        progress = (i + 1) / len(chunk_data)
        progress_bar.progress(progress)

    status_text.text("Embeddings created succesfully.")
    return embeddings


def pinecone_store(embeddings):
    records = []
    id=0
    for embedding in embeddings:
        records.append({
        "id": str(id),
        "values": embedding['vector'],
        "metadata":{'text':embedding['text']}})
        id = id+1

    stats = st.session_state.index.describe_index_stats()
    if "namespaces" in stats and "ProjectVectorStore" in stats["namespaces"]:
        st.session_state.index.delete(delete_all=True, namespace="ProjectVectorStore")

    st.session_state.index.upsert(
        vectors=records,
        namespace="ProjectVectorStore")
    
    return st.session_state.index

def get_answer(query):
    query_embedding =   generativeai.embed_content(model="models/text-embedding-004",content=query)['embedding']
    
    query_result = st.session_state.index.query(
    vector=query_embedding,  
    top_k=5,  
    namespace="ProjectVectorStore",
    include_metadata=True)


    context = ""
    for match in query_result['matches']:
        information = match['metadata']['text']
        context = context + information + '\n\n'


    prompt = "The user will ask some question. The context of the information will be provided. Answer from the given context only. DO NOT ANSWER FROM YOUR KNOWLEDGE OR TRY TO MAKE UP SOME ANSWER. IF ANSWER IS NOT PRESENT REPLY 'I DO NOT KNOW'"

    final_query = prompt + '\n\n' + context +'\n\n' + query


    response = st.session_state.client.models.generate_content(
    model="gemini-1.5-flash", contents=final_query)

    return response.text

def main():
    load_dotenv()
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
    generativeai.configure(api_key=os.environ["GEMINI_API_KEY"])

    st.session_state.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    st.session_state.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    st.session_state.index = index = st.session_state.pc.Index("project-index")


    with st.sidebar:
        st.subheader("Your document")
        pdf_docs = st.file_uploader("Upload your PDF here", type=["pdf"])

        if pdf_docs is not None:
            if st.button("Process document"):
                pdf_doc = load_document(pdf_docs)
                chunk_data = create_chunks(pdf_doc)
                embeddings = get_embeddings(chunk_data)
                st.session_state.index = pinecone_store(embeddings)

    st.title("Document Query App")  
    query = st.text_input(label="Enter your query")
    get_llm_answer = st.button("Get Answer")
    if(get_llm_answer):
        answer = get_answer(query)
        st.write(answer)
    

if __name__ == '__main__':
    main()